"""
伪压缩法单元测试。

纯 Python + numpy, 无需 ROS。
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one360s.pseudo_compress import expand_margin, width_score


def test_narrow_slit():
    """窄缝: 只有 3 扇区通 → 膨胀裕度低。"""
    distances = np.full(72, 3.0)
    distances[0] = 25.0
    distances[1] = 25.0
    distances[71] = 25.0  # 3扇区窄缝

    pinch = 20.0  # 夹点 20m
    margin = expand_margin(distances, 0.0, pinch, R=0.70)
    # 窄缝 → 连 1.5 倍膨胀可能都过不去
    assert margin <= 2.0, f"Narrow slit should have low margin, got {margin}"


def test_wide_channel():
    """宽通道: 15 扇区通 → 膨胀裕度高。"""
    distances = np.full(72, 10.0)
    for i in range(-7, 8):
        distances[i % 72] = 40.0

    pinch = 20.0
    margin = expand_margin(distances, 0.0, pinch, R=0.70)
    # 宽通道 → 膨胀裕度 > 2.0
    assert margin >= 2.0, f"Wide channel should have high margin, got {margin}"


def test_width_score_narrow():
    """窄缝 → width_score 低。"""
    assert width_score(1.0) < 0.4, f"margin=1.0 should give low score"


def test_width_score_wide():
    """宽通道 → width_score 满分。"""
    assert width_score(3.0) == 1.0
    assert width_score(5.0) == 1.0  # 封顶


if __name__ == "__main__":
    test_narrow_slit()
    test_wide_channel()
    test_width_score_narrow()
    test_width_score_wide()
    print("All pseudo_compress tests passed!")
