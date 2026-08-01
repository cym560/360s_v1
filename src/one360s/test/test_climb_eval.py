"""
爬升评估 + 坡度检测单元测试。
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one360s.climb_eval import can_climb, detect_slope


def test_can_climb_sufficient_space():
    """足够空间 → 可以爬升。"""
    L1 = np.full(72, 40.0)
    L1[0:5] = 15.0  # 前方15m处有障碍

    L2 = np.full(72, 40.0)  # 上方全通
    up = np.full(72, 40.0)  # 树冠全通

    result = can_climb(
        L1, L2, up,
        current_alt_m=5.0,
        target_alt_m=6.0,
        groundspeed_ms=3.0,
        max_climb_rate_ms=2.0,
    )

    assert result["feasible"], f"Should be feasible: {result['reason']}"
    assert result["climb_time_s"] == 0.5  # 1m / 2m/s


def test_can_climb_no_horizontal_space():
    """前方 0.5m 紧贴障碍 → 不能爬升。"""
    L1 = np.full(72, 0.5)  # 极近障碍 (<1m 阈值)
    L2 = np.full(72, 40.0)
    up = np.full(72, 40.0)

    result = can_climb(L1, L2, up, 5.0, 6.0, 3.0)
    assert not result["feasible"]
    # 可用距离 < 1.0 → no_horizontal_space; 1.0-可用距离间 → no_time
    assert result["reason"] in ("no_horizontal_space", "no_time")


def test_can_climb_L2_blocked():
    """上方空间被堵 → 不能爬升。"""
    L1 = np.full(72, 40.0)
    L1[0:5] = 15.0

    L2 = np.full(72, 40.0)
    L2[0] = 5.0  # 上方有障碍

    up = np.full(72, 40.0)

    result = can_climb(L1, L2, up, 5.0, 6.0, 3.0)
    assert not result["feasible"]
    assert result["reason"] == "L2_blocked"


def test_detect_slope_positive():
    """连续递减扇区 → 检测为坡度。"""
    distances = np.full(72, 40.0)
    # 模拟前方坡度: 距离逐步递减
    for i in range(10):
        distances[i] = 15.0 - i * 1.2  # 15m → 4.2m

    result = detect_slope(distances, min_sectors=5)
    assert result["is_slope"], "Should detect slope"


def test_detect_slope_wall():
    """单一扇区突变 → 不是坡度 (墙壁)。"""
    distances = np.full(72, 40.0)
    distances[0] = 3.0  # 单一突变

    result = detect_slope(distances, min_sectors=5)
    assert not result["is_slope"], "Single sector jump should not be slope"


if __name__ == "__main__":
    test_can_climb_sufficient_space()
    test_can_climb_no_horizontal_space()
    test_can_climb_L2_blocked()
    test_detect_slope_positive()
    test_detect_slope_wall()
    print("All climb_eval tests passed!")
