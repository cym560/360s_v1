"""
感知管线 (Perception Pipeline)。

7阶段处理: 原始点云 → 四层 z-bin 扇区距离数组。

阶段:
  1. 置信度过滤 — Mid-360 tag 字段, 丢雨/雾/尘低置信点
  2. 姿态补偿   — pitch/roll 旋转矩阵修正
  3. 空间过滤   — 距离裁剪 + 高度裁剪 + CropBox 自屏蔽
  4. 降采样     — VoxelGrid
  5. 去噪       — RadiusOutlier
  6. 扇区分仓   — 72×5°, 20th percentile, 自适应 min_points
  7. 时域+膨胀  — 不对称平滑 + 角度膨胀 + 扇区置信度 EMA

纯 numpy, 零 ROS 依赖。
"""

import math
import numpy as np
from collections import deque


class One360sPipeline:
    """7阶段感知管线。实例化时加载参数。"""

    def __init__(self, params: dict):
        # 距离范围
        rf = params.get("range_filter", {})
        self.min_range_m = rf.get("min_range_m", 0.25)
        self.max_range_m = rf.get("max_range_m", 40.0)
        self.z_bins = rf.get("z_bins", {
            "L0": [-3.0, -0.4],
            "L1": [-0.4, 0.5],
            "L2": [0.5, 2.0],
            "up": [2.0, 6.0],
        })
        self.z_bin_keys = list(self.z_bins.keys())  # ["L0", "L1", "L2", "up"]

        # 体素
        vf = params.get("voxel_filter", {})
        self.voxel_enable = vf.get("enable", True)
        self.voxel_leaf = vf.get("leaf_size_m", 0.05)

        # 去噪
        ro = params.get("radius_outlier", {})
        self.outlier_enable = ro.get("enable", True)
        self.outlier_radius = ro.get("radius_m", 0.15)
        self.outlier_min_neighbors = ro.get("min_neighbors", 2)

        # 扇区
        sec = params.get("sector", {})
        self.num_sectors = sec.get("num_sectors", 72)
        self.sector_deg = sec.get("sector_deg", 5.0)
        self.percentile = sec.get("percentile", 20)
        self.min_points_near = sec.get("min_points_near", 3)
        self.min_points_mid = sec.get("min_points_mid", 2)
        self.min_points_far = sec.get("min_points_far", 1)

        # 时域滤波
        tf = params.get("temporal_filter", {})
        self.temporal_enable = tf.get("enable", True)
        self.receding_alpha = tf.get("receding_alpha", 0.4)
        self.clear_frames = tf.get("clear_frames", 3)

        # 障碍置信度 EMA
        oc = params.get("obstacle_confidence", {})
        self.conf_decay = oc.get("decay", 0.9)
        self.conf_threshold = oc.get("threshold", 0.5)
        self.conf_conservative_dist = oc.get("conservative_dist_m", 3.0)

        # 膨胀
        inf = params.get("inflation", {})
        self.inflate_enable = inf.get("enable", True)
        self.vehicle_radius_m = inf.get("vehicle_radius_m", 0.45)
        self.safety_extra_m = inf.get("safety_extra_m", 0.25)
        self.max_inflate_bins = inf.get("max_inflate_bins", 6)

        # 姿态补偿
        ac = params.get("attitude_compensation", {})
        self.attitude_enable = ac.get("enable", True)
        self.attitude_min_angle = ac.get("min_angle_deg", 5.0)

        # 置信度过滤
        tg = params.get("tag_filter", {})
        self.tag_enable = tg.get("enable", True)
        self.min_confidence = tg.get("min_confidence", 1)

        # CropBox 自屏蔽 (排除无人机自身点云)
        cb = params.get("crop_box", {})
        self.crop_enable = cb.get("enable", False)
        self.crop_x_min = cb.get("x_min", -0.35)
        self.crop_x_max = cb.get("x_max", 0.30)
        self.crop_y_min = cb.get("y_min", -0.40)
        self.crop_y_max = cb.get("y_max", 0.40)
        self.crop_z_min = cb.get("z_min", -0.55)
        self.crop_z_max = cb.get("z_max", None)  # None = 不过滤上方

        # ── 内部状态 ──
        self._prev_distances = {k: np.full(72, 40.0) for k in self.z_bin_keys}
        self._clear_counters = {k: np.zeros(72, dtype=np.int32)
                                for k in self.z_bin_keys}
        self._obstacle_confidence = {k: np.zeros(72) for k in self.z_bin_keys}

    # ── 主入口 ──

    def process(
        self,
        points: np.ndarray,
        tags: np.ndarray = None,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> dict:
        """
        一帧处理。

        参数:
          points: (N, 3) xyz 点云 (body 坐标系)
          tags: (N,) uint8 置信度 tag (可选)
          pitch_deg: 当前俯仰角
          roll_deg: 当前横滚角

        返回:
          {
              'L0': ndarray[72],
              'L1': ndarray[72],
              'L2': ndarray[72],
              'up': ndarray[72],
              'filtered_points': int,
              'filtered_cloud': ndarray(N,3),
              'stats': {
                  'input_points': int,
                  'filtered_points': int,
                  'sectors_with_data': dict,
                  'processing_time_ms': float,
              },
          }
        """
        original_count = len(points)
        pts = points.copy()

        # ── 阶段 1: 置信度过滤 ──
        if self.tag_enable and tags is not None:
            mask = tags >= self.min_confidence
            pts = pts[mask]

        # ── 阶段 2: 姿态补偿 ──
        if self.attitude_enable:
            if abs(pitch_deg) > self.attitude_min_angle or \
               abs(roll_deg) > self.attitude_min_angle:
                pts = self._attitude_compensate(pts, pitch_deg, roll_deg)

        # ── 阶段 3: 空间过滤 ──
        pts = self._spatial_filter(pts)

        if len(pts) == 0:
            return self._empty_result(original_count)

        # ── 阶段 4: VoxelGrid 降采样 ──
        if self.voxel_enable:
            pts = self._voxel_downsample(pts)

        # ── 阶段 5: RadiusOutlier 去噪 ──
        if self.outlier_enable:
            pts = self._radius_outlier(pts)

        if len(pts) == 0:
            return self._empty_result(original_count)

        # ── 保存过滤后点云 (调试用) ──
        filtered_cloud = pts.copy()

        # ── 阶段 6: 四层扇区分仓 ──
        distances_raw = {}
        for key in self.z_bin_keys:
            z_min, z_max = self.z_bins[key]
            layer_mask = (pts[:, 2] >= z_min) & (pts[:, 2] < z_max)
            layer_pts = pts[layer_mask]

            if len(layer_pts) == 0:
                distances_raw[key] = np.full(self.num_sectors, self.max_range_m)
            else:
                distances_raw[key] = self._extract_sectors(layer_pts)

        # ── 阶段 7: 时域滤波 + 膨胀 + 置信度 EMA ──
        distances_out = {}
        for key in self.z_bin_keys:
            d = distances_raw[key]

            # 7a. 障碍置信度 EMA 更新
            self._update_confidence(key, d)

            # 7b. 置信度改写 (障碍"闪烁"抑制)
            d = self._apply_confidence(key, d)

            # 7c. 时域不对称平滑
            if self.temporal_enable:
                d = self._temporal_filter(key, d)

            # 7d. 角度膨胀
            if self.inflate_enable:
                d = self._inflate_angles(d)

            distances_out[key] = d
            self._prev_distances[key] = d

        return {
            **distances_out,
            "filtered_points": len(pts),
            "filtered_cloud": filtered_cloud,
            "stats": {
                "input_points": original_count,
                "filtered_points": len(pts),
                "sectors_with_data": {
                    k: int(np.sum(distances_out[k] < self.max_range_m))
                    for k in self.z_bin_keys
                },
                "processing_time_ms": 0.0,  # 由 processor_node 填入
            },
        }

    # ── 阶段 1: 置信度过滤 (在 process 中内联) ──

    # ── 阶段 2: 姿态补偿 ──

    def _attitude_compensate(
        self, pts: np.ndarray, pitch_deg: float, roll_deg: float,
    ) -> np.ndarray:
        """
        对 body 系点云做俯仰/横滚补偿。

        补偿后点云近似在"水平面"坐标系。
        只修 pitch/roll, 不修 yaw (扇区分仓以机头为 0° 参考)。
        """
        pitch = math.radians(-pitch_deg)  # 绕 y 轴
        roll = math.radians(-roll_deg)    # 绕 x 轴

        # 旋转矩阵 R = Ry(pitch) × Rx(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # Ry
        x1 = pts[:, 0] * cp + pts[:, 2] * sp
        y1 = pts[:, 1]
        z1 = -pts[:, 0] * sp + pts[:, 2] * cp

        # Rx
        x2 = x1
        y2 = y1 * cr - z1 * sr
        z2 = y1 * sr + z1 * cr

        return np.column_stack([x2, y2, z2])

    # ── 阶段 3: 空间过滤 ──

    def _spatial_filter(self, pts: np.ndarray) -> np.ndarray:
        """距离裁剪 + 高度裁剪 + CropBox 自屏蔽。"""
        # 水平距离
        horiz = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        mask = (horiz >= self.min_range_m) & (horiz <= self.max_range_m)

        # 高度 — 取所有 z-bin 的并集 (最宽的 z 范围)
        all_z_min = min(z[0] for z in self.z_bins.values())
        all_z_max = max(z[1] for z in self.z_bins.values())
        mask &= (pts[:, 2] >= all_z_min) & (pts[:, 2] <= all_z_max)

        # CropBox 自屏蔽 — 排除无人机机身占据的空间
        if self.crop_enable:
            crop_mask = np.ones(len(pts), dtype=bool)
            # x 范围
            crop_mask &= (pts[:, 0] < self.crop_x_min) | (pts[:, 0] > self.crop_x_max)
            # y 范围
            crop_mask &= (pts[:, 1] < self.crop_y_min) | (pts[:, 1] > self.crop_y_max)
            # z 范围 (z_max=None 则不过滤上方)
            crop_mask &= pts[:, 2] >= self.crop_z_min
            if self.crop_z_max is not None:
                crop_mask &= pts[:, 2] <= self.crop_z_max
            mask &= crop_mask

        return pts[mask]

    # ── 阶段 4: VoxelGrid ──

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        """体素降采样: 每个 3D 体素保留一个点(质心)。"""
        if len(pts) == 0:
            return pts

        leaf = self.voxel_leaf
        voxel_indices = np.floor(pts / leaf).astype(np.int64)

        # 用 unique 找每个体素的第一个点
        dims = voxel_indices.max(axis=0) - voxel_indices.min(axis=0) + 1
        flat = (
            voxel_indices[:, 0] * dims[1] * dims[2] +
            voxel_indices[:, 1] * dims[2] +
            voxel_indices[:, 2]
        )
        _, unique_idx = np.unique(flat, return_index=True)
        return pts[unique_idx]

    # ── 阶段 5: RadiusOutlier ──

    def _radius_outlier(self, pts: np.ndarray) -> np.ndarray:
        """半径去噪: 某点半径内邻居 < min_neighbors → 删除。"""
        if len(pts) <= self.outlier_min_neighbors:
            return pts

        n = len(pts)
        keep = np.ones(n, dtype=bool)

        # 暴力搜索 O(N²) — Nano 上 2000 点约 5-15ms
        # 未来切换到 cKDTree 可降到 ~2ms
        radius_sq = self.outlier_radius ** 2

        for i in range(n):
            diff = pts - pts[i]
            dist_sq = np.sum(diff ** 2, axis=1)
            neighbors = np.sum(dist_sq < radius_sq) - 1  # 减自己
            if neighbors < self.outlier_min_neighbors:
                keep[i] = False

        return pts[keep]

    # ── 阶段 6: 扇区分仓 ──

    def _extract_sectors(self, layer_pts: np.ndarray) -> np.ndarray:
        """单层点云 → 72 扇区距离数组。"""
        if len(layer_pts) == 0:
            return np.full(self.num_sectors, self.max_range_m)

        # 计算每个点的方位角和水平距离
        azimuth = np.degrees(np.arctan2(-layer_pts[:, 1], layer_pts[:, 0]))
        azimuth = (azimuth + 360) % 360  # 0-360, 0=机头, 逆时针
        horiz_dist = np.sqrt(layer_pts[:, 0]**2 + layer_pts[:, 1]**2)

        sector_indices = np.round(azimuth / self.sector_deg).astype(int) % self.num_sectors

        distances = np.full(self.num_sectors, self.max_range_m)

        for s in range(self.num_sectors):
            mask = sector_indices == s
            if np.sum(mask) < self._adaptive_min_points(10.0):  # 初始用远距离阈值
                continue  # 点数不够 → MAX

            sector_dists = horiz_dist[mask]

            # 自适应 min_points (用该扇区中位距离)
            med_dist = np.median(sector_dists)
            min_pts = self._adaptive_min_points(med_dist)

            if len(sector_dists) < min_pts:
                continue

            # 20th percentile
            sorted_dists = np.sort(sector_dists)
            idx = max(0, int(len(sorted_dists) * self.percentile / 100.0))
            distances[s] = sorted_dists[idx]

        return distances

    def _adaptive_min_points(self, sector_distance_m: float) -> int:
        """min_points 自适应: 远处宽容。"""
        if sector_distance_m < 5:
            return self.min_points_near
        elif sector_distance_m < 10:
            return self.min_points_mid
        else:
            return self.min_points_far

    # ── 阶段 7a: 障碍置信度 EMA ──

    def _update_confidence(self, key: str, distances: np.ndarray):
        """更新扇区障碍置信度。"""
        conf = self._obstacle_confidence[key]
        for s in range(self.num_sectors):
            if distances[s] < self.max_range_m * 0.5:  # < 20m → 有障碍
                conf[s] = 1.0
            else:
                conf[s] *= self.conf_decay  # 缓慢衰减

    def _apply_confidence(self, key: str, distances: np.ndarray) -> np.ndarray:
        """置信度改写: 高置信度但当前帧无数据的扇区 → 保守距离。"""
        conf = self._obstacle_confidence[key]
        result = distances.copy()
        for s in range(self.num_sectors):
            if conf[s] > self.conf_threshold and distances[s] >= self.max_range_m * 0.5:
                # 该扇区"应该"有障碍, 但当前帧没看到 → 保守值
                result[s] = min(distances[s], self.conf_conservative_dist)
        return result

    # ── 阶段 7b: 时域滤波 ──

    def _temporal_filter(self, key: str, distances: np.ndarray) -> np.ndarray:
        """不对称时域平滑。每扇区独立计数。"""
        prev = self._prev_distances[key]
        result = distances.copy()
        counters = self._clear_counters[key]

        for s in range(self.num_sectors):
            d_new = distances[s]
            d_old = prev[s]

            if d_new >= self.max_range_m * 0.5:
                counters[s] += 1
                if counters[s] >= self.clear_frames:
                    result[s] = self.max_range_m
                else:
                    result[s] = d_old
            else:
                counters[s] = 0
                if d_new < d_old:
                    result[s] = d_new
                else:
                    result[s] = self.receding_alpha * d_new + \
                                (1 - self.receding_alpha) * d_old

        return result

    # ── 阶段 7c: 角度膨胀 ──

    def _inflate_angles(self, distances: np.ndarray) -> np.ndarray:
        """角度膨胀: 飞机物理尺寸 → 扇区扩张。"""
        R = self.vehicle_radius_m + self.safety_extra_m
        result = distances.copy()

        for s in range(self.num_sectors):
            d = distances[s]
            if d >= self.max_range_m * 0.5:
                continue

            # 该扇区障碍物对飞机的张角
            inflate_angle = math.degrees(math.atan(R / max(d, 0.01)))
            inflate_bins = min(self.max_inflate_bins,
                               int(math.ceil(inflate_angle / self.sector_deg)))

            # 向两侧膨胀: 将障碍距离传播到相邻无障碍扇区
            for offset in range(-inflate_bins, inflate_bins + 1):
                t = (s + offset) % self.num_sectors
                result[t] = min(result[t], d)

        return result

    # ── 辅助 ──

    def _empty_result(self, input_count: int) -> dict:
        """空点云结果。"""
        empty = {k: np.full(self.num_sectors, self.max_range_m)
                 for k in self.z_bin_keys}
        empty["filtered_points"] = 0
        empty["filtered_cloud"] = np.empty((0, 3), dtype=np.float32)
        empty["stats"] = {
            "input_points": input_count,
            "filtered_points": 0,
            "sectors_with_data": {},
            "processing_time_ms": 0.0,
        }
        return empty
