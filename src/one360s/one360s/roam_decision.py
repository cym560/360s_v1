"""
方向选择引擎。

融合夹点(深度) + 伪压缩(宽度)评分, 选最优飞行方向。
包含: 滞后保护、时序确认、夹点趋势检测、垂直决策触发。
"""

import math
import time
from collections import deque
import numpy as np

from .pinch_point import pinch_point, scan_pinch_map
from .pseudo_compress import expand_margin, width_score as _width_score
from .climb_eval import can_climb, detect_slope, slope_follow_target
from .topo_memory import TopoMemory


class RoamDecision:
    """
    每帧决策引擎。

    内部状态:
      - current_bearing:  当前方向
      - current_pinch:    当前夹点
      - pinch_history:    最近 N 帧夹点 (趋势检测)
      - low_pinch_count:  连续低夹点帧数 (时序确认)
      - in_climb:         是否处于爬升模式
    """

    def __init__(self, params: dict, topo: TopoMemory):
        pp = params.get("pinch", {})
        pc = params.get("pseudo_compress", {})
        sp = params.get("speed", {})
        cp = params.get("climb", {})

        # 夹点参数
        self.effective_R = pp.get("effective_R", 0.70)
        self.max_range = pp.get("max_range", 40.0)
        self.scan_half_deg = pp.get("scan_half_deg", 90)
        self.step_deg = pp.get("step_deg", 5)
        self.hysteresis_m = pp.get("hysteresis_m", 1.0)
        self.all_blocked_m = pp.get("all_blocked_m", 1.5)
        self.decline_ratio = pp.get("decline_ratio", 0.3)
        decline_window_s = pp.get("decline_window_s", 3.0)
        self.confirm_frames = pp.get("confirm_frames", 3)

        # 伪压缩参数
        self.expand_factors = pc.get("expand_factors", [1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        self.width_full_score = pc.get("width_full_score", 3.0)

        # 速度参数
        self.max_decel_ms2 = sp.get("max_decel_ms2", 3.0)
        self.safety_factor = sp.get("safety_factor", 0.5)
        self.max_speed_ms = sp.get("max_speed_ms", 3.0)
        self.min_speed_ms = sp.get("min_speed_ms", 0.5)
        self.cautious_step_m = sp.get("cautious_step_m", 2.0)
        self.normal_step_m = sp.get("normal_step_m", 5.0)

        # 爬升参数
        self.max_climb_rate_ms = cp.get("max_climb_rate_ms", 1.2)
        self.window_factor = cp.get("window_factor", 0.7)
        self.slope_min_sectors = cp.get("slope_min_sectors", 5)
        self.slope_safe_height_m = cp.get("slope_safe_height_m", 2.0)

        # 拓扑记忆
        self.topo = topo

        # 任务参数
        mp = params.get("mission", {})
        self.mission_mode = mp.get("mode", "roam")
        self.goal_bearing_deg = mp.get("goal_bearing_deg", -1.0)
        if self.mission_mode == "inspect":
            self.goal_weight = mp.get("inspect_weight", 0.6)
        else:
            self.goal_weight = mp.get("goal_weight", 0.3)
        if self.mission_mode == "roam":
            self.goal_bearing_deg = -1.0  # 强制关闭方向偏置
        self.roam_alt_m = sp.get("roam_alt_m", 10.0)

        # 状态
        self.current_bearing = 0.0
        self.current_pinch = 40.0
        self.pinch_history = deque(maxlen=int(decline_window_s * 10))  # 10Hz
        self.low_pinch_count = 0
        self.in_climb = False
        self.climb_target_alt = 0.0

    # ── 主入口 ──

    def decide(
        self,
        distances_all: dict,
        heading_deg: float,
        groundspeed_ms: float,
        current_alt_m: float,
        up_laser_m: float = 0.0,
        laser_alt_m: float = 0.0,
    ) -> dict | None:
        """
        一帧决策。

        参数:
          distances_all: {'L0': [72], 'L1': [72], 'L2': [72], 'up': [72]}
          heading_deg: 当前机头朝向
          groundspeed_ms: 当前地速
          current_alt_m: 当前高度 (气压/融合)
          up_laser_m: 上视激光测距值(m), 0=未连接
          laser_alt_m: 下视激光测距值(m), 0=未连接

        返回:
          {
              'action': 'fly'|'climb'|'descend'|'backtrack'|'hover',
              'bearing_deg': float,
              'step_m': float,
              'speed_ms': float,
              'target_alt_m': float|None,
              'reason': str,
          }
          或 None → 全堵无退路 → 上层应触发告警
        """
        distances_L1 = np.array(distances_all["L1"], dtype=np.float64)
        distances_L2 = np.array(distances_all.get("L2", distances_all["L1"]),
                                dtype=np.float64)
        distances_up = np.array(distances_all.get("up", distances_all["L1"]),
                                dtype=np.float64)
        distances_L0 = np.array(distances_all.get("L0", distances_all["L1"]),
                                dtype=np.float64)

        # 更新拓扑记忆朝向
        self.topo.set_current_bearing(heading_deg)

        # ── 模式: 爬升中 ──
        if self.in_climb:
            return self._handle_climb(
                distances_L1, distances_L2, distances_up,
                heading_deg, groundspeed_ms, current_alt_m,
                up_laser_m,
            )

        # ── 正常水平飞行 ──
        # 1. 扫描 L1 层
        scan = scan_pinch_map(
            distances_L1,
            self.scan_half_deg, self.step_deg,
            heading_deg, self.effective_R, self.max_range,
        )

        # 2. 对所有候选方向算融合评分
        scores = self._score_all(scan["bearings"], scan["pinches"], distances_L1)

        # 3. 选最优方向 (含滞后保护)
        best = self._select_best(scores, heading_deg)
        if best is None:
            # 4. 所有方向夹点 < all_blocked_m → 垂直评估或回溯
            return self._handle_blocked(
                distances_L1, distances_L2, distances_up, distances_L0,
                heading_deg, groundspeed_ms, current_alt_m, up_laser_m, laser_alt_m,
            )

        best_bearing, best_score, best_pinch = best

        # 5. 时序确认
        self._update_pinch_history(best_pinch)
        if best_pinch < self.all_blocked_m:
            self.low_pinch_count += 1
        else:
            self.low_pinch_count = 0

        if self.low_pinch_count >= self.confirm_frames:
            # 连续多帧低夹点 → 确认全堵
            self.low_pinch_count = 0
            return self._handle_blocked(
                distances_L1, distances_L2, distances_up, distances_L0,
                heading_deg, groundspeed_ms, current_alt_m, up_laser_m, laser_alt_m,
            )

        # 6. 正常飞行
        step_m = (
            self.normal_step_m
            if best_pinch >= 7.0
            else self.cautious_step_m
        )
        speed_ms = compute_speed(
            best_pinch,
            self.max_decel_ms2, self.safety_factor,
            self.max_speed_ms, self.min_speed_ms,
        )

        # 7. 更新当前方向
        self.current_bearing = best_bearing
        self.current_pinch = best_pinch

        # 8. 检测分叉口 → 记录到拓扑记忆
        fork_min_angle = getattr(self.topo, 'fork_min_angle_sep', 30.0)
        good_exits = [
            b for b, _, p in scores
            if p > self.topo.fork_min_pinch_m and abs(b - best_bearing) > fork_min_angle
        ]
        if len(good_exits) >= 1:
            exits = [int(round(b / 5.0)) % 72 for b in [best_bearing] + good_exits]
            chosen = int(round(best_bearing / 5.0)) % 72
            self.topo.record_fork(exits, chosen)

        return {
            "action": "fly",
            "bearing_deg": best_bearing,
            "step_m": step_m,
            "speed_ms": speed_ms,
            "target_alt_m": self.roam_alt_m,
            "reason": f"pinch={best_pinch:.1f}m, score={best_score:.2f}",
        }

    # ── 融合评分 ──

    def _score_all(
        self, bearings: np.ndarray, pinches: np.ndarray, distances: np.ndarray,
    ) -> list:
        """对所有候选方向计算融合评分。返回 [(bearing, score, pinch), ...] 降序。"""
        results = []
        for b, p in zip(bearings, pinches):
            score = self._score_direction(distances, float(b), float(p))
            results.append((float(b), score, float(p)))
        results.sort(key=lambda x: -x[1])  # 按分数降序
        return results

    def _score_direction(
        self, distances: np.ndarray, bearing: float, pinch: float,
    ) -> float:
        """
        单方向融合评分 = depth_score × (0.5 + 0.5 × width_score) × goal_bias。

        depth_score  = pinch / 30m (封顶 1.0)
        width_score  = 膨胀裕度 / 3.0 (封顶 1.0)
        goal_bias    = 1.0 + goal_weight × cos(bearing - goal_bearing)
                       goal方向最高, 反方向最低

        死胡同方向 → 降权。
        """
        # 深度
        depth = min(1.0, pinch / 30.0)

        # 宽度 (伪压缩)
        margin = expand_margin(
            distances, bearing, pinch,
            self.effective_R, self.expand_factors,
        )
        width = _width_score(margin, self.width_full_score)

        score = depth * (0.5 + 0.5 * width)

        # 目标方向偏置
        if self.goal_bearing_deg >= 0:
            angle_diff = abs(bearing - self.goal_bearing_deg)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            goal_bias = 1.0 + self.goal_weight * math.cos(math.radians(angle_diff))
            score *= goal_bias

        # 死胡同降权
        if self.topo.is_dead_end(bearing):
            score *= 0.3

        return score

    # ── 方向选择(含滞后) ──

    def _select_best(
        self, scores: list, heading_deg: float,
    ) -> tuple | None:
        """
        从评分列表中选择最优方向。

        滞后保护:
          - 当前方向有 ≥1m 夹点优势 → 保持
          - 新方向需显著优于当前方向才切换

        返回: (bearing, score, pinch) 或 None
        """
        if not scores:
            return None

        best_bearing, best_score, best_pinch = scores[0]

        # 检查是否存在任何可通过的方向
        if best_pinch < self.all_blocked_m:
            return None

        # 滞后: 在 scores 里找匹配当前朝向的夹点
        current_pinch_in_scan = self._find_closest_pinch(
            scores, self.current_bearing)
        if current_pinch_in_scan is not None and \
           current_pinch_in_scan > self.all_blocked_m:
            if best_pinch - current_pinch_in_scan < self.hysteresis_m:
                return (self.current_bearing, best_score,
                        current_pinch_in_scan)

        return (best_bearing, best_score, best_pinch)

    def _find_closest_pinch(self, scores: list, bearing: float) -> float | None:
        """在 scores 列表中找最接近 bearing 的方向的夹点。"""
        if not scores:
            return None
        best = min(scores, key=lambda x: abs(x[0] - bearing))
        return best[2]  # pinch

    # ── 全堵处理 ──

    def _handle_blocked(
        self,
        distances_L1, distances_L2, distances_up, distances_L0,
        heading_deg, groundspeed_ms, current_alt_m,
        up_laser_m: float = 0.0,
        laser_alt_m: float = 0.0,
    ) -> dict | None:
        """
        水平全堵时的决策。

        优先级:
        1. 坡度检测 → 坡度跟随
        2. 爬升评估 → 爬升越障 (L2+up 层)
        3. 下降评估 → 下降钻底 (L0 层)
        4. 拓扑回溯 → 退到分叉口选未探索出口
        5. 悬停告警
        """

        # 1. 坡度检测
        slope = detect_slope(distances_L1, self.slope_min_sectors)
        if slope["is_slope"]:
            target = slope_follow_target(
                distances_L1, distances_L2,
                current_alt_m, self.slope_safe_height_m,
            )
            if target["safe"]:
                # 爬升 1m 试试
                climb = can_climb(
                    distances_L1, distances_L2, distances_up,
                    current_alt_m, current_alt_m + 1.0,
                    groundspeed_ms, self.max_climb_rate_ms, self.window_factor,
                    up_laser_m, self.effective_R,
                )
                if climb["feasible"]:
                    self.in_climb = True
                    self.climb_target_alt = target["target_alt_m"]
                    return {
                        "action": "climb",
                        "bearing_deg": target["bearing_deg"],
                        "step_m": self.cautious_step_m,
                        "speed_ms": self.min_speed_ms,
                        "target_alt_m": target["target_alt_m"],
                        "reason": f"slope_follow, slope={slope['slope_angle_deg']}°",
                    }

        # 2. 爬升评估
        for test_dh in [1.0, 2.0, 3.0]:
            climb = can_climb(
                distances_L1, distances_L2, distances_up,
                current_alt_m, current_alt_m + test_dh,
                groundspeed_ms, self.max_climb_rate_ms, self.window_factor,
                up_laser_m, self.effective_R,
            )
            if climb["feasible"]:
                self.in_climb = True
                self.climb_target_alt = current_alt_m + test_dh
                return {
                    "action": "climb",
                    "bearing_deg": heading_deg,
                    "step_m": self.cautious_step_m,
                    "speed_ms": self.min_speed_ms,
                    "target_alt_m": current_alt_m + test_dh,
                    "reason": f"climb_{test_dh}m, available={climb['available_distance_m']:.1f}m",
                }

        # 3. 下降评估 — 检查 L0 层能否钻底
        ground_clearance = laser_alt_m if laser_alt_m > 0.1 else current_alt_m
        if ground_clearance > 1.5:  # 至少 1.5m 地面空间才考虑下降
            scan_L0 = scan_pinch_map(
                distances_L0, self.scan_half_deg, self.step_deg,
                heading_deg, self.effective_R, self.max_range)
            if scan_L0["best_pinch"] > self.all_blocked_m:
                return {
                    "action": "descend",
                    "bearing_deg": scan_L0["best_bearing"],
                    "step_m": self.cautious_step_m,
                    "speed_ms": self.min_speed_ms,
                    "target_alt_m": max(0.5, ground_clearance * 0.5 + current_alt_m - ground_clearance),
                    "reason": f"L0_clear_pinch={scan_L0['best_pinch']:.1f}m",
                }

        # 4. 夹点趋势检测 — 提前回溯
        if self._is_declining():
            self.topo.record_narrowing(self.current_pinch)
            bt = self.topo.backtrack()
            if bt:
                return {
                    "action": "backtrack",
                    "bearing_deg": bt["from_bearing_deg"],
                    "step_m": self.normal_step_m,
                    "speed_ms": 1.0,
                    "target_alt_m": None,
                    "reason": "early_backtrack_decline",
                }

        # 5. 全堵 — 拓扑回溯
        self.topo.record_dead_end(self.current_pinch)
        bt = self.topo.backtrack()
        if bt:
            # 标记当前方向为死胡同
            self.topo.mark_dead_end(heading_deg)
            return {
                "action": "backtrack",
                "bearing_deg": bt["from_bearing_deg"],
                "step_m": self.normal_step_m,
                "speed_ms": 1.0,
                "target_alt_m": None,
                "reason": "backtrack_dead_end",
            }

        # 6. 悬停告警
        return {
            "action": "hover",
            "bearing_deg": heading_deg,
            "step_m": 0.0,
            "speed_ms": 0.0,
            "target_alt_m": None,
            "reason": "all_blocked_no_escape",
        }

    # ── 爬升模式处理 ──

    def _handle_climb(
        self,
        distances_L1, distances_L2, distances_up,
        heading_deg, groundspeed_ms, current_alt_m,
        up_laser_m: float = 0.0,
    ) -> dict:
        """爬升模式: 到达目标高度后验证 L1 是否真的通了, 否则继续爬。"""
        at_target = current_alt_m >= self.climb_target_alt - 0.3

        if at_target:
            # 验证 L1 是否有可通过方向
            scan = scan_pinch_map(
                distances_L1, self.scan_half_deg, self.step_deg,
                heading_deg, self.effective_R, self.max_range)
            if scan["best_pinch"] > self.all_blocked_m:
                self.in_climb = False
                return {
                    "action": "fly",
                    "bearing_deg": heading_deg,
                    "step_m": self.normal_step_m,
                    "speed_ms": 1.5,
                    "target_alt_m": None,
                    "reason": "climb_complete_L1_clear",
                }
            else:
                # L1 仍然全堵 → 继续爬升 1m
                self.climb_target_alt += 1.0

        return {
            "action": "climb",
            "bearing_deg": heading_deg,
            "step_m": self.cautious_step_m,
            "speed_ms": self.min_speed_ms,
            "target_alt_m": self.climb_target_alt,
            "reason": "climbing",
        }

    # ── 趋势检测 ──

    def _update_pinch_history(self, pinch: float):
        self.pinch_history.append(pinch)

    def _is_declining(self) -> bool:
        """夹点趋势检测: 3秒内夹点减半 → 通道在收窄。"""
        if len(self.pinch_history) < self.pinch_history.maxlen:
            return False

        recent = list(self.pinch_history)
        now_pinch = recent[-1]
        past_pinch = recent[0]

        if past_pinch <= 0 or now_pinch <= 0:
            return False

        return now_pinch < past_pinch * self.decline_ratio and now_pinch < 8.0


# ── 速度计算 (独立函数, 可复用于其他模块) ──

def compute_speed(
    pinch_m: float,
    max_decel_ms2: float = 3.0,
    safety_factor: float = 0.5,
    max_speed_ms: float = 3.0,
    min_speed_ms: float = 0.5,
) -> float:
    """
    夹点距离 → 安全飞行速度 (物理刹车约束)。

    v = √(2 × a × (pinch - 1.0)) × safety_factor

    margin=1.0m: 刹车后距离障碍的安全余量。
    """
    v_max = math.sqrt(2 * max_decel_ms2 * max(0.0, pinch_m - 1.0))
    speed = v_max * safety_factor
    return max(min_speed_ms, min(max_speed_ms, speed))
