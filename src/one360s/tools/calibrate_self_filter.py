"""
机身自屏蔽标定工具 (CropBox Calibration Tool)。

用法:
  1. 将飞机放在开阔场地 (周围 5m 无任何物体), 上电
  2. 启动 Livox LiDAR 驱动
  3. 运行本脚本, 录制 10 秒点云:
       python calibrate_self_filter.py --duration 10 --output body_mask.npz
  4. 脚本输出:
       - 机身在各扇区/各层的最近距离 (body_mask)
       - 建议的 CropBox 尺寸
       - 扇区级的保守遮罩 (哪些扇区被机身永久遮挡)

原理:
  开阔场地中, 所有点云应该都是 40m (MAX) 或至少 >5m。
  如果某个扇区持续出现 <2m 的点, 那就是机身反射。
"""

import argparse
import math
import time
import struct
import json
from collections import defaultdict
import numpy as np

# ── 如果没有 ROS/LiDAR 实时数据, 也可以从录好的 rosbag 或 pcd 文件分析 ──

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    HAS_ROS = True
except ImportError:
    HAS_ROS = False


# ── 默认 Mid-360 机身包围盒 (需按实际机型调整) ──
# LiDAR 通常安装在机身上方中央, 坐标系: x=前, y=左, z=上
DEFAULT_BODY_BOX = {
    "x": [-0.35, 0.30],   # 前后: 机头30cm前, 机尾35cm后
    "y": [-0.40, 0.40],   # 左右: 各40cm (含机臂)
    "z": [-0.55, None],   # 下方: LiDAR到脚架底部 ~55cm
    # z_max=None 表示不裁剪上方 (上方天空不应有障碍)
}


def points_from_rosbag(pcd_file: str) -> np.ndarray:
    """从录好的 rosbag 导出 pcd 文件分析 (离线模式)。"""
    # 简化: 读取 PCD ASCII
    pts = []
    with open(pcd_file, "r") as f:
        in_data = False
        for line in f:
            if in_data:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
            if line.startswith("DATA"):
                in_data = True
    return np.array(pts, dtype=np.float32)


def analyze_body_points(
    points: np.ndarray,
    body_box: dict = None,
    sector_deg: float = 5.0,
    num_sectors: int = 72,
    max_range: float = 40.0,
) -> dict:
    """
    分析所有点云帧, 找出永远有近距点的扇区 → 机身遮挡。

    参数:
      points: (N, 3) 累积的所有点云 (可多帧拼接)
      body_box: 预期的机身包围盒 {x: [min, max], y: [min, max], z: [min, max]}
      sector_deg: 扇区宽度
      num_sectors: 扇区数
      max_range: 最大有效距离

    返回:
      {
          'body_sectors': [扇区号列表],      # 被机身永久遮挡的扇区
          'body_distances': [72],            # 每个扇区机身最远距离
          'suggested_crop_box': {x,y,z},     # 建议 CropBox 尺寸
          'self_points_ratio': float,        # 自身点占比
      }
    """
    n_total = len(points)
    if n_total == 0:
        return {"body_sectors": [], "body_distances": [max_range] * num_sectors,
                "suggested_crop_box": body_box or DEFAULT_BODY_BOX,
                "self_points_ratio": 0.0}

    # 水平距离
    horiz = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    azimuth = np.degrees(np.arctan2(-points[:, 1], points[:, 0]))
    azimuth = (azimuth + 360) % 360
    sector_idx = np.round(azimuth / sector_deg).astype(int) % num_sectors

    # 每扇区统计
    sector_min_dist = np.full(num_sectors, max_range)
    sector_self_count = np.zeros(num_sectors, dtype=int)
    sector_total_count = np.zeros(num_sectors, dtype=int)

    for s in range(num_sectors):
        mask = sector_idx == s
        sector_total_count[s] = np.sum(mask)
        if sector_total_count[s] > 0:
            sector_min_dist[s] = np.min(horiz[mask])
            # 距离 < 2m 的点视为可疑机身点
            sector_self_count[s] = np.sum(horiz[mask] < 2.0)

    # 怀疑准则: 扇区最小距离 < 2m 且该扇区有足够多点
    body_sectors = []
    for s in range(num_sectors):
        if sector_min_dist[s] < 2.0 and sector_total_count[s] >= 5:
            body_sectors.append(s)

    # 机身点占比
    self_ratio = np.sum(horiz < 2.0) / max(n_total, 1)

    # 建议 CropBox: 取所有 <2m 点的 xyz 范围
    near_mask = horiz < 2.0
    if np.sum(near_mask) > 10:
        near_pts = points[near_mask]
        suggested = {
            "x": [float(np.percentile(near_pts[:, 0], 1)),
                  float(np.percentile(near_pts[:, 0], 99))],
            "y": [float(np.percentile(near_pts[:, 1], 1)),
                  float(np.percentile(near_pts[:, 1], 99))],
            "z_min": float(np.percentile(near_pts[:, 2], 1)),
            "z_max": None,  # 上方不过滤
        }
        # 加 10cm 余量
        suggested["x"][0] -= 0.10
        suggested["x"][1] += 0.10
        suggested["y"][0] -= 0.10
        suggested["y"][1] += 0.10
        suggested["z_min"] -= 0.10
    else:
        suggested = body_box if body_box else DEFAULT_BODY_BOX

    return {
        "body_sectors": body_sectors,
        "body_distances": sector_min_dist.tolist(),
        "suggested_crop_box": suggested,
        "self_points_ratio": round(self_ratio, 3),
    }


def print_report(result: dict):
    """打印标定报告。"""
    print()
    print("=" * 60)
    print("  机身自屏蔽标定报告")
    print("=" * 60)
    print(f"  疑似机身扇区数: {len(result['body_sectors'])}")
    if result["body_sectors"]:
        sectors_str = ", ".join(str(s) for s in result["body_sectors"])
        angles = [f"{s*5}°" for s in result["body_sectors"]]
        print(f"  扇区号: {sectors_str}")
        print(f"  对应角度: {', '.join(angles)}")
    print(f"  近距点占比 (<2m): {result['self_points_ratio']:.1%}")
    print()
    print("  建议 CropBox 参数 (写入 params.yaml):")
    box = result["suggested_crop_box"]
    print(f"    crop_box:")
    print(f"      enable: true")
    print(f"      x_min: {box['x'][0]:.2f}")
    print(f"      x_max: {box['x'][1]:.2f}")
    print(f"      y_min: {box['y'][0]:.2f}")
    print(f"      y_max: {box['y'][1]:.2f}")
    print(f"      z_min: {box['z_min']:.2f}")
    print(f"      # z_max: null  # 不过滤上方")
    print()
    print("  ⚠ 以上参数需根据实飞验证微调!")
    print("=" * 60)


# ── ROS 实时采集模式 ──

if HAS_ROS:
    class BodyCalibrationNode(Node):
        """实时订阅点云 → 累积分析。"""

        def __init__(self, duration_s: float = 10.0):
            super().__init__("body_calibration")
            self.duration_s = duration_s
            self.all_points = []
            self.start_time = time.time()

            self.sub = self.create_subscription(
                PointCloud2, "/livox/lidar/pointcloud",
                self.cloud_cb, 100)

            self.timer = self.create_timer(duration_s + 0.5, self.done_cb)
            self.get_logger().info(
                f"开始采集 {duration_s}s... 保持飞机静止, 周围无障碍!")

        def cloud_cb(self, msg: PointCloud2):
            if time.time() - self.start_time > self.duration_s:
                return
            pts = self._unpack_xyz(msg)
            if len(pts) > 0:
                self.all_points.append(pts)

        def _unpack_xyz(self, msg: PointCloud2) -> np.ndarray:
            fields = {f.name: f.offset for f in msg.fields}
            x_off = fields.get("x", 0)
            y_off = fields.get("y", 4)
            z_off = fields.get("z", 8)
            step = msg.point_step
            data = msg.data
            n = len(data) // step
            pts = np.zeros((n, 3), dtype=np.float32)
            for i in range(n):
                base = i * step
                pts[i, 0] = struct.unpack_from("f", data, base + x_off)[0]
                pts[i, 1] = struct.unpack_from("f", data, base + y_off)[0]
                pts[i, 2] = struct.unpack_from("f", data, base + z_off)[0]
            return pts

        def done_cb(self):
            if not self.all_points:
                self.get_logger().error("未收到任何点云!")
                rclpy.shutdown()
                return

            all_pts = np.vstack(self.all_points)
            self.get_logger().info(
                f"采集完成: {len(self.all_points)} 帧, {len(all_pts)} 点")

            result = analyze_body_points(all_pts)
            print_report(result)

            # 保存
            out_path = "/tmp/one360s_body_mask.npz"
            np.savez(out_path,
                     body_sectors=result["body_sectors"],
                     suggested_box=json.dumps(result["suggested_crop_box"]))
            self.get_logger().info(f"结果已保存到 {out_path}")
            rclpy.shutdown()


# ── 离线模式 (从 pcd 文件分析) ──

def offline_mode(pcd_files: list, output: str = None):
    """从录好的 PCD 文件分析。"""
    all_pts = []
    for f in pcd_files:
        pts = points_from_rosbag(f)
        if len(pts) > 0:
            all_pts.append(pts)
        print(f"  {f}: {len(pts)} 点")

    if not all_pts:
        print("错误: 无有效点云数据")
        return

    merged = np.vstack(all_pts)
    print(f"总点数: {len(merged)}")
    result = analyze_body_points(merged)
    print_report(result)

    if output:
        np.savez(output,
                 body_sectors=result["body_sectors"],
                 suggested_box=json.dumps(result["suggested_crop_box"]))
        print(f"结果已保存到 {output}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="one360s 机身自屏蔽标定工具")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="ROS 实时采集时长(秒)")
    parser.add_argument("--pcd", nargs="+", default=None,
                        help="离线模式: 从 PCD 文件分析")
    parser.add_argument("--output", type=str,
                        default="/tmp/one360s_body_mask.npz",
                        help="输出路径")
    args = parser.parse_args()

    if args.pcd:
        offline_mode(args.pcd, args.output)
    elif HAS_ROS:
        rclpy.init()
        node = BodyCalibrationNode(args.duration)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("错误: 需要 ROS 2 (实时模式) 或 --pcd 文件 (离线模式)")
        print()
        print("离线模式用法:")
        print("  1. ros2 bag record /livox/lidar/pointcloud -o body_calib")
        print("  2. ros2 run pcl_ros bag_to_pcd body_calib/ body_calib.pcd")
        print("  3. python calibrate_self_filter.py --pcd body_calib.pcd")


if __name__ == "__main__":
    main()
