"""
one360s v1 — 单LiDAR夹点规划低算力自主导航。

纯算法包, 零ROS依赖。
核心理念: 不建图、不定位、在传感器空间内直接做规划。
"""

from .pinch_point import pinch_point, scan_pinch_map
from .pseudo_compress import expand_margin, width_score
from .roam_decision import RoamDecision, compute_speed
from .climb_eval import can_climb, detect_slope, slope_follow_target
from .topo_memory import TopoMemory, TopoNode
from .config import load_params, validate_params, derive_params

__all__ = [
    "pinch_point",
    "scan_pinch_map",
    "expand_margin",
    "width_score",
    "RoamDecision",
    "compute_speed",
    "can_climb",
    "detect_slope",
    "slope_follow_target",
    "TopoMemory",
    "TopoNode",
    "load_params",
]
