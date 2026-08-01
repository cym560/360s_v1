"""
夹点探测 (Pinch Point Detection)。

核心问题: 飞机沿某个方向飞行, 最远能飞多少米才被障碍卡住?
本质: 飞机几何投影锥与障碍物扇区的最近交会距离。
"""

import math
import numpy as np


def pinch_point(
    distances: np.ndarray,
    bearing_deg: float,
    R: float = 0.70,
    max_range: float = 40.0,
    sector_deg: float = 5.0,
    num_sectors: int = 72,
) -> float:
    """
    飞机沿 bearing 方向飞行, 返回夹点距离(米)。

    算法:
      从 d=1m 递增到 max_range。
      每一步: 飞机在距离 d 处的半张角 = atan(R/d)
              需要的扇区数 = ceil(半张角/5°) × 2 + 1
              检查这些扇区的距离是否都 ≥ d
      第一个不满足的 d-1 = 夹点。

    参数:
      distances: [72] 扇区距离数组(米), MAX=无障碍
      bearing_deg: 方向角 0-360, 0=机头, 逆时针
      R: 飞机有效半径(机身+安全余量), 默认 0.70m
      max_range: 最大探测距离
      sector_deg: 每扇区度数, 默认5°
      num_sectors: 扇区总数, 默认72

    返回:
      夹点距离(米)。max_range = 全程无阻挡。
    """
    for d in range(1, int(max_range) + 1):
        half_angle = math.degrees(math.atan(R / d))
        n_sectors = math.ceil(half_angle / sector_deg)
        center = int(round(bearing_deg / sector_deg)) % num_sectors

        for offset in range(-n_sectors, n_sectors + 1):
            s = (center + offset) % num_sectors
            if distances[s] < d:
                return float(d - 1)  # 上一米还能过, 这米卡住了

    return max_range  # 全程无阻挡


def scan_pinch_map(
    distances: np.ndarray,
    scan_half_deg: float = 90.0,
    step_deg: float = 5.0,
    current_heading: float = 0.0,
    R: float = 0.70,
    max_range: float = 40.0,
    sector_deg: float = 5.0,
) -> dict:
    """
    扫描前方 ±scan_half_deg, 返回所有候选方向的夹点。

    参数:
      distances: [72] 扇区距离数组
      scan_half_deg: 扫描半角, 默认90(前方±90°)
      step_deg: 扫描步长, 默认5°
      current_heading: 当前机头朝向
      R: 有效半径
      max_range: 最大探测距离

    返回:
      {
          'bearings': ndarray[N],     # 候选方向(度)
          'pinches': ndarray[N],      # 对应夹点(米)
          'best_bearing': float,      # 夹点最大的方向
          'best_pinch': float,        # 最大夹点
      }
    """
    offsets = np.arange(-scan_half_deg, scan_half_deg + step_deg, step_deg)
    bearings = [(current_heading + o) % 360 for o in offsets]
    pinches = np.array([
        pinch_point(distances, b, R, max_range, sector_deg)
        for b in bearings
    ])

    best_idx = int(np.argmax(pinches))
    return {
        "bearings": np.array(bearings),
        "pinches": pinches,
        "best_bearing": float(bearings[best_idx]),
        "best_pinch": float(pinches[best_idx]),
    }
