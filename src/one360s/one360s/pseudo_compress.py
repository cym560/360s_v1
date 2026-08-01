"""
伪压缩法 (Pseudo-Compress) — 通道宽度评估。

核心理念: 逐步膨胀障碍物, 检查通道是否仍可通过。
膨胀裕度越高 = 通道越宽 = 方向越安全。
"""

import math
import numpy as np

# 默认膨胀系数序列
EXPAND_FACTORS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def expand_margin(
    distances: np.ndarray,
    bearing_deg: float,
    pinch: float,
    R: float = 0.70,
    factors: list = None,
    sector_deg: float = 5.0,
    num_sectors: int = 72,
) -> float:
    """
    逐步膨胀障碍物, 返回最大可承受的膨胀系数。

    对每个膨胀系数 e:
      Re = R × e (膨胀后的障碍半径)
      在夹点距离 pinch 处, 膨胀后障碍对飞机的张角 = 2 × asin(Re / pinch)
      检查投影锥内扇区是否都 ≥ pinch
      第一个堵住的 e = 膨胀裕度上限

    额外检查: 在 pinch/2 和 pinch*0.75 处各抽查一次,
    防止通道在更近处收窄而漏检。

    参数:
      distances: [72] 扇区距离数组
      bearing_deg: 方向角
      pinch: 该方向的夹点距离 (从 pinch_point 传入)
      R: 飞机有效半径
      factors: 膨胀系数列表, 默认 [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
      sector_deg: 每扇区度数
      num_sectors: 扇区总数

    返回:
      最大膨胀系数 (1.0 ~ 5.0)。
      1.0 = 刚好能过(窄缝)。
      5.0 = 极宽通道。
      如果连 1.5 都过不去, 返回 1.0 (标称尺寸刚好能过)。
    """
    if factors is None:
        factors = EXPAND_FACTORS

    if pinch <= R:
        return 1.0  # 夹点太近, 无法评估宽度

    margin = 1.0  # 标称尺寸至少能过 (否则 pinch 会更近)

    # 需要在多个距离处检查的膨胀系数 (只用最大的几个系数做抽查, 省算力)
    check_distances = [d for d in [pinch * 0.5, pinch * 0.75] if d > R * 2]

    for e in factors:
        Re = R * e

        if pinch <= Re:
            # 膨胀半径已超过夹点距离 → 障碍物已在膨胀圈内
            break

        # ── 夹点距离处检查 ──
        half_angle = math.degrees(math.asin(Re / pinch))
        n_sectors = math.ceil(half_angle / sector_deg)
        center = int(round(bearing_deg / sector_deg)) % num_sectors

        blocked = False
        for offset in range(-n_sectors, n_sectors + 1):
            s = (center + offset) % num_sectors
            if distances[s] < pinch:
                blocked = True
                break

        if blocked:
            break

        # ── 中间距离抽查 (仅对大膨胀系数, 省算力) ──
        if e >= 3.0:
            mid_blocked = False
            for chk_dist in check_distances:
                if chk_dist <= Re:
                    continue
                chk_half = math.degrees(math.asin(Re / chk_dist))
                chk_n = math.ceil(chk_half / sector_deg)
                for offset in range(-chk_n, chk_n + 1):
                    s = (center + offset) % num_sectors
                    if distances[s] < chk_dist:
                        mid_blocked = True
                        break
                if mid_blocked:
                    break
            if mid_blocked:
                break

        margin = e

    return margin


def width_score(
    margin: float,
    full_score_at: float = 3.0,
) -> float:
    """
    膨胀裕度 → 宽度评分 (0-1)。

    margin=1.0 → score=0.33 (窄缝)
    margin=3.0 → score=1.0 (宽通道,满分)
    margin=5.0 → score=1.0 (极宽,封顶)
    """
    return min(1.0, margin / full_score_at)


def scan_widths(
    distances: np.ndarray,
    bearings: np.ndarray,
    pinches: np.ndarray,
    R: float = 0.70,
    factors: list = None,
) -> np.ndarray:
    """
    批量计算一组方向的膨胀裕度。

    参数:
      distances: [72] 扇区距离数组
      bearings: [N] 方向角数组
      pinches: [N] 对应夹点数组
      R: 有效半径
      factors: 膨胀系数列表

    返回:
      [N] 膨胀裕度数组
    """
    return np.array([
        expand_margin(distances, float(b), float(p), R, factors)
        for b, p in zip(bearings, pinches)
    ])
