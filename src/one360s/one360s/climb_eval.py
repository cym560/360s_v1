"""
爬升评估 + 坡度检测。

回答两个问题:
  1. 水平全堵时, 能否爬升越障? (can_climb)
  2. 前方是墙壁还是坡度?        (detect_slope)
"""

import math
import numpy as np
from .pinch_point import pinch_point


def can_climb(
    distances_L1: np.ndarray,
    distances_L2: np.ndarray,
    distances_up: np.ndarray,
    current_alt_m: float,
    target_alt_m: float,
    groundspeed_ms: float,
    max_climb_rate_ms: float = 1.2,
    window_factor: float = 0.7,
    up_laser_m: float = 0.0,
    R: float = 0.70,
) -> dict:
    """
    评估爬升是否可行。

    步骤:
    1. 计算水平可用距离 (前方 ±30° 内 L1 最近夹点)
    2. 爬升时间 = 高度差 / 爬升率
    3. 前飞距离 = 爬升时间 × 地速
    4. if 前飞距离 < 可用距离 × window_factor → 检查上方
    5. 检查 L2/up 层数据充分性 (>10 个有效扇区)
    6. L2 和 up 层通畅 → 可以爬升

    参数:
      distances_L1/L2/up: 三层扇区距离数组
      current_alt_m: 当前高度(米)
      target_alt_m: 目标高度(米)
      groundspeed_ms: 当前地速(m/s)
      max_climb_rate_ms: 最大爬升率(保守值)
      window_factor: 爬升窗口系数 (前飞距离/可用距离 上限)
      up_laser_m: 上视激光测距(m), 0=未连接, >0=上方可用距离

    返回:
      {
          'feasible': bool,
          'climb_time_s': float,
          'forward_during_climb_m': float,
          'available_distance_m': float,
          'reason': str,
      }
    """
    height_diff = target_alt_m - current_alt_m
    if height_diff <= 0:
        return {
            "feasible": False,
            "climb_time_s": 0.0,
            "forward_during_climb_m": 0.0,
            "available_distance_m": 0.0,
            "reason": "no_climb_needed",
        }

    # 1. 水平可用距离 — 前方 ±30° 内 L1 最近夹点
    available_m = float("inf")
    for offset in range(-30, 31, 5):
        b = float(offset % 360)
        pp = pinch_point(distances_L1, b, R=R, max_range=40.0)
        if pp < available_m:
            available_m = pp

    if available_m < 1.0:
        return {
            "feasible": False,
            "climb_time_s": 0.0,
            "forward_during_climb_m": 0.0,
            "available_distance_m": available_m,
            "reason": "no_horizontal_space",
        }

    # 2-3. 爬升时间和前飞距离
    climb_time_s = height_diff / max_climb_rate_ms
    forward_m = groundspeed_ms * climb_time_s

    # 4. 爬升窗口检查
    if forward_m > available_m * window_factor:
        return {
            "feasible": False,
            "climb_time_s": climb_time_s,
            "forward_during_climb_m": forward_m,
            "available_distance_m": available_m,
            "reason": "no_time",
        }

    # 5. 上方数据充分性检查 — L2 和 up 层必须足够稠密
    #    如果点云太稀疏, 可能是开阔天空 (安全)
    #    但如果有少数可疑点, 上层数据不足 ⇒ 不冒险爬升
    min_valid_sectors = 10  # 前方 ±30° 范围 (12 个扇区) 至少 10 个有数据
    l2_valid = int(np.sum(distances_L2 < 39.0))
    up_valid = int(np.sum(distances_up < 39.0))

    if l2_valid < min_valid_sectors and up_valid < min_valid_sectors:
        # 两层都稀疏 → 可能是开阔天空, 允许爬升
        pass
    elif l2_valid < min_valid_sectors:
        return {
            "feasible": False,
            "climb_time_s": climb_time_s,
            "forward_during_climb_m": forward_m,
            "available_distance_m": available_m,
            "reason": "L2_sparse_cannot_verify",
        }
    elif up_valid < min_valid_sectors:
        return {
            "feasible": False,
            "climb_time_s": climb_time_s,
            "forward_during_climb_m": forward_m,
            "available_distance_m": available_m,
            "reason": "up_sparse_cannot_verify",
        }

    # 6. 上视激光优先检查 — 如果安装了, 直接验证头顶空间
    height_needed_m = target_alt_m - current_alt_m
    if up_laser_m > 0.0 and up_laser_m < height_needed_m + 0.5:
        return {
            "feasible": False,
            "climb_time_s": climb_time_s,
            "forward_during_climb_m": forward_m,
            "available_distance_m": available_m,
            "reason": f"up_laser_blocked_{up_laser_m:.1f}m",
        }

    # 7. 上方空间检查 — L2 层 (前方 ±30°)
    for offset in range(-30, 31, 5):
        b = float(offset % 360)
        s = int(round(b / 5.0)) % 72
        if distances_L2[s] < available_m:
            return {
                "feasible": False,
                "climb_time_s": climb_time_s,
                "forward_during_climb_m": forward_m,
                "available_distance_m": available_m,
                "reason": "L2_blocked",
            }

    # 8. 树冠顶部检查 — up 层 (仅无上视激光时用 LiDAR up 层)
    if up_laser_m <= 0.0:
        for offset in range(-30, 31, 5):
            b = float(offset % 360)
            s = int(round(b / 5.0)) % 72
            if distances_up[s] < available_m:
                return {
                    "feasible": False,
                    "climb_time_s": climb_time_s,
                    "forward_during_climb_m": forward_m,
                    "available_distance_m": available_m,
                    "reason": "up_blocked",
                }

    return {
        "feasible": True,
        "climb_time_s": climb_time_s,
        "forward_during_climb_m": forward_m,
        "available_distance_m": available_m,
        "reason": "ok",
    }


def detect_slope(
    distances_L1: np.ndarray,
    min_sectors: int = 5,
    sector_deg: float = 5.0,
) -> dict:
    """
    检测前方是否为坡度 (而非墙壁/离散障碍)。

    区分逻辑:
      连续 min_sectors 个相邻扇区 pinch 递减 → 坡度
      单一扇区 pinch 突变            → 墙壁

    算法:
      扫描前方 ±60°, 找递减序列。
      对每个候选递减序列, 线性拟合斜率 → 坡面角度。

    返回:
      {
          'is_slope': bool,
          'slope_angle_deg': float,       # 估算坡度角
          'slope_start_m': float,         # 坡面起始距离(最远扇区距离)
          'slope_direction_deg': float,   # 坡面中心朝向
      }
    """
    num_sectors = len(distances_L1)

    # 扫描前方 ±60°
    best_run_len = 0
    best_start = 0
    slope_angle = 0.0
    slope_start_m = 0.0

    for start in range(-12, 13):  # -60° ~ +60°, 步长5°
        s0 = start % num_sectors
        if distances_L1[s0] >= 40.0:
            continue  # 无障碍, 不可能是坡面起点

        run_len = 1
        for j in range(1, 12):  # 往后检查最多 12 个扇区 (60°)
            prev = (start + j - 1) % num_sectors
            curr = (start + j) % num_sectors

            # 递减: 下一扇区更近 (或接近)
            if distances_L1[curr] <= distances_L1[prev] + 1.0:
                run_len += 1
            else:
                break

        if run_len >= min_sectors:
            # 线性拟合估算坡度
            xs = np.arange(run_len) * sector_deg  # 角度偏移
            ys = np.array([
                distances_L1[(start + j) % num_sectors]
                for j in range(run_len)
            ])

            # 角度 → 水平距离 → 高度变化
            # d_horiz = d × cos(angle), d_vert = d × sin(angle)
            # 坡度 = atan(d_vert_max / d_horiz_max)?

            # 简单估算: 用扇区间的距离衰减率
            dist_start = ys[0]
            dist_end = ys[-1]
            if dist_start > dist_end:
                # 坡度角 ≈ atan(高度差 / 水平距离)
                angular_width = run_len * sector_deg  # 扇区覆盖的角度
                mid_dist = (dist_start + dist_end) / 2
                horiz_width = 2 * mid_dist * math.sin(math.radians(angular_width / 2))
                # 高度差 ≈ 2m (L1 层高 0.9m, 这是坡面进入 L1 层的范围)
                height_range = 1.8  # 近似: L1 层高度范围

                if horiz_width > 0.5:
                    angle = math.degrees(math.atan(height_range / horiz_width))
                    if run_len > best_run_len:
                        best_run_len = run_len
                        best_start = start
                        slope_angle = angle
                        slope_start_m = dist_start

    if best_run_len >= min_sectors:
        direction = (best_start + best_run_len / 2) * sector_deg
        return {
            "is_slope": True,
            "slope_angle_deg": round(slope_angle, 1),
            "slope_start_m": round(slope_start_m, 1),
            "slope_direction_deg": round(direction, 1),
        }

    return {
        "is_slope": False,
        "slope_angle_deg": 0.0,
        "slope_start_m": 0.0,
        "slope_direction_deg": 0.0,
    }


def slope_follow_target(
    distances_L1: np.ndarray,
    distances_L2: np.ndarray,
    current_alt_m: float,
    safe_height_m: float = 2.0,
) -> dict:
    """
    坡度跟随: 计算沿坡面上升的目标高度和方向。

    原则:
      - 维持离坡面 ~safe_height_m 的安全高度
      - L1 最近的障碍距离 = 坡面位置
      - 目标高度 = 当前高度 + (L1_min_dist 对应的坡面上升量)

    返回:
      {
          'bearing_deg': float,    # 坡面方向
          'target_alt_m': float,   # 目标高度
          'safe': bool,            # 是否安全(上方空间充足)
      }
    """
    slope = detect_slope(distances_L1)
    if not slope["is_slope"]:
        return {
            "bearing_deg": 0.0,
            "target_alt_m": current_alt_m,
            "safe": True,
        }

    direction = slope["slope_direction_deg"]
    sector = int(round(direction / 5.0)) % 72

    # 检查上方空间
    safe = distances_L2[sector] > 3.0

    # 沿坡面微调高度: 保持离坡面 safe_height_m
    l1_min = float(np.min(distances_L1[max(0, sector - 6): sector + 7]))
    if l1_min < safe_height_m:
        # 太近 → 需要爬升
        target_alt = current_alt_m + (safe_height_m - l1_min) * 0.5
    else:
        target_alt = current_alt_m

    return {
        "bearing_deg": direction,
        "target_alt_m": round(target_alt, 1),
        "safe": safe,
    }
