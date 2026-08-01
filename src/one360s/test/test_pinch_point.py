"""
夹点探测单元测试。

纯 Python + numpy, 无需 ROS。
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one360s.pinch_point import pinch_point, scan_pinch_map


def test_pinch_clear():
    """完全无障碍 → pinch = max_range。"""
    distances = np.full(72, 40.0)
    pp = pinch_point(distances, 0.0, max_range=40.0)
    assert pp == 40.0, f"Expected 40.0, got {pp}"


def test_pinch_wall_front():
    """正前方 4.9m 有墙 → 夹点应在 4m (≤4.9m 处刹停)。"""
    distances = np.full(72, 40.0)
    distances[0] = 4.9  # 正前方
    distances[1] = 4.9
    distances[71] = 4.9

    pp = pinch_point(distances, 0.0)
    # 墙在 4.9m → d=4 时 4.9>=4 通过, d=5 时 4.9<5 卡住 → pinch=4
    assert pp == 4.0, f"Expected 4.0, got {pp}"


def test_pinch_narrow_gap():
    """窄缝: 只有 3 个扇区通。"""
    distances = np.full(72, 3.0)  # 全是墙
    distances[0] = 30.0  # 正前方通
    distances[1] = 30.0
    distances[71] = 30.0  # 3扇区窄缝

    pp = pinch_point(distances, 0.0)
    # 3m 前需要 ±35° ≈ 15 扇区 → 不够 → 夹点很浅
    assert pp < 5.0, f"Expected shallow pinch, got {pp}"


def test_pinch_wide_channel():
    """宽通道: 15 扇区通。"""
    distances = np.full(72, 10.0)
    # 前方 15 扇区 (75°) 全通
    for i in range(-7, 8):
        distances[i % 72] = 40.0

    pp = pinch_point(distances, 0.0)
    # 宽通道 → 夹点应该较深
    assert pp >= 10.0, f"Expected deep pinch, got {pp}"


def test_scan_pinch_map_returns_all_directions():
    """scan_pinch_map 返回 36 个方向。"""
    distances = np.full(72, 40.0)
    result = scan_pinch_map(distances, scan_half_deg=90, step_deg=5)
    assert len(result["bearings"]) >= 36
    assert len(result["pinches"]) == len(result["bearings"])
    assert result["best_pinch"] == 40.0


def test_scan_all_blocked():
    """全堵场景: 所有方向夹点 < 3m。"""
    distances = np.full(72, 2.0)  # 全是 2m 墙
    result = scan_pinch_map(distances, scan_half_deg=90, step_deg=5)
    assert result["best_pinch"] < 3.0, "Should be all blocked"


if __name__ == "__main__":
    test_pinch_clear()
    test_pinch_wall_front()
    test_pinch_narrow_gap()
    test_pinch_wide_channel()
    test_scan_pinch_map_returns_all_directions()
    test_scan_all_blocked()
    print("All pinch_point tests passed!")
