"""
方向选择单元测试。

纯 Python + numpy, 无需 ROS。
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one360s.roam_decision import RoamDecision, compute_speed
from one360s.topo_memory import TopoMemory
from one360s.config import DEFAULT_PARAMS


def test_decide_clear_path():
    """前方全通 → fly。"""
    L1 = np.full(72, 40.0)
    distances_all = {"L0": L1, "L1": L1, "L2": L1, "up": L1}

    topo = TopoMemory()
    rd = RoamDecision(DEFAULT_PARAMS, topo)

    result = rd.decide(distances_all, heading_deg=0.0,
                       groundspeed_ms=2.0, current_alt_m=5.0)

    assert result is not None
    assert result["action"] == "fly"
    assert result["speed_ms"] > 0


def test_decide_wall_in_front():
    """前方有墙(1m), 侧方通 → 必须转向或爬升。"""
    L1 = np.full(72, 40.0)
    # 前方 ±3 扇区全堵在 1m
    for i in range(-3, 4):
        L1[i % 72] = 1.5
    # 侧方(60°)通
    L1[12] = 30.0

    distances_all = {"L0": L1, "L1": L1, "L2": L1, "up": L1}

    topo = TopoMemory()
    rd = RoamDecision(DEFAULT_PARAMS, topo)
    # 先让当前方向的夹点变低 (模拟前一帧状态)
    rd.current_pinch = 1.0

    result = rd.decide(distances_all, heading_deg=0.0,
                       groundspeed_ms=2.0, current_alt_m=5.0)

    assert result is not None
    assert result["action"] in ("fly", "climb", "backtrack"), \
        f"Expected fly/climb/backtrack, got {result['action']}"
    if result["action"] == "fly":
        assert abs(result["bearing_deg"]) > 5, \
            f"Should turn away from wall, bearing={result['bearing_deg']}"


def test_decide_all_blocked():
    """全堵 → hover 或 backtrack (pinch≈1.0 < all_blocked_m=1.5)。"""
    L1 = np.full(72, 1.0)  # 全是墙, pinch≈1.0 < 1.5
    distances_all = {"L0": L1, "L1": L1, "L2": L1, "up": L1}

    topo = TopoMemory()
    rd = RoamDecision(DEFAULT_PARAMS, topo)

    result = rd.decide(distances_all, heading_deg=0.0,
                       groundspeed_ms=2.0, current_alt_m=5.0)

    assert result is not None
    assert result["action"] in ("hover", "backtrack", "climb")


def test_prefer_wide_over_narrow():
    """宽通道应比窄缝得分高 (近场场景, 角度宽度差异显著)。"""
    # 窄缝: 3扇区通, 但夹点只有 6m
    L1_narrow = np.full(72, 3.0)
    L1_narrow[0] = 8.0
    L1_narrow[1] = 8.0
    L1_narrow[71] = 8.0  # 3扇区窄缝, pinch=8m

    # 宽通道: 15扇区通, 夹点 8m
    L1_wide = np.full(72, 3.0)
    for i in range(-7, 8):
        L1_wide[i % 72] = 8.0  # 15扇区宽通道, pinch=8m

    topo = TopoMemory()
    rd = RoamDecision(DEFAULT_PARAMS, topo)

    score_narrow = rd._score_direction(L1_narrow, 0.0, 6.0)
    score_wide = rd._score_direction(L1_wide, 0.0, 6.0)

    # 相同深度, 宽通道得分应高于窄缝
    assert score_wide > score_narrow, \
        f"Wide({score_wide:.3f}) should beat narrow({score_narrow:.3f})"


def test_compute_speed():
    """速度计算: pinch 越远速度越快, 有上下限。"""
    v_near = compute_speed(2.0)   # pinch=2m → 低速
    v_mid = compute_speed(5.0)    # pinch=5m → 中速
    v_far = compute_speed(15.0)   # pinch=15m → 近全速

    assert v_near < v_mid, f"near({v_near:.1f}) < mid({v_mid:.1f})"
    assert v_mid < v_far, f"mid({v_mid:.1f}) < far({v_far:.1f})"
    assert 0.5 <= v_near <= 3.0
    assert 0.5 <= v_far <= 3.0


if __name__ == "__main__":
    test_decide_clear_path()
    test_decide_wall_in_front()
    test_decide_all_blocked()
    test_prefer_wide_over_narrow()
    test_compute_speed()
    print("All roam_decision tests passed!")
