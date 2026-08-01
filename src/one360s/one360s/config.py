"""
参数加载模块。
YAML → 类型化参数字典。零 ROS 依赖。
"""

import os
import yaml

# ── 默认参数 (当 YAML 缺失时使用) ──

DEFAULT_PARAMS = {
    # ── 无人机物理参数 (换飞机改这里) ──
    "drone": {
        "body_length_m": 0.65,
        "body_width_m": 0.80,
        "body_height_m": 0.55,
        "lidar_z_offset": 0.15,
        "safety_extra_m": 0.25,
        "max_speed_ms": 3.0,
        "max_decel_ms2": 3.0,
        "min_speed_ms": 0.5,
    },
    "range_filter": {
        "min_range_m": 0.25,
        "max_range_m": 40.0,
        "z_bins": {
            "L0": [-3.0, -0.4],
            "L1": [-0.4, 0.5],
            "L2": [0.5, 2.0],
            "up": [2.0, 6.0],
        },
    },
    "voxel_filter": {
        "enable": True,
        "leaf_size_m": 0.05,
    },
    "radius_outlier": {
        "enable": True,
        "radius_m": 0.15,
        "min_neighbors": 2,
    },
    "sector": {
        "num_sectors": 72,
        "sector_deg": 5.0,
        "percentile": 20,
        "min_points_near": 3,   # <5m
        "min_points_mid": 2,    # 5-10m
        "min_points_far": 1,    # >10m
    },
    "temporal_filter": {
        "enable": True,
        "receding_alpha": 0.4,
        "clear_frames": 3,
    },
    "obstacle_confidence": {
        "decay": 0.9,
        "threshold": 0.5,
        "conservative_dist_m": 3.0,
    },
    "inflation": {
        "enable": True,
        "vehicle_radius_m": 0.45,
        "safety_extra_m": 0.25,
        "max_inflate_bins": 6,
    },
    "attitude_compensation": {
        "enable": True,
        "min_angle_deg": 5.0,
    },
    "tag_filter": {
        "enable": True,
        "min_confidence": 1,
    },
    "crop_box": {
        "enable": False,
        "x_min": -0.35,
        "x_max": 0.30,
        "y_min": -0.40,
        "y_max": 0.40,
        "z_min": -0.55,
        "z_max": None,       # None = 不过滤上方
    },
    "pinch": {
        "vehicle_radius_m": 0.45,
        "safety_extra_m": 0.25,
        "effective_R": 0.70,
        "max_range": 40.0,
        "scan_half_deg": 90,
        "step_deg": 5,
        "hysteresis_m": 1.0,
        "all_blocked_m": 1.5,
        "decline_ratio": 0.3,
        "decline_window_s": 3.0,
        "confirm_frames": 3,
    },
    "pseudo_compress": {
        "expand_factors": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "width_full_score": 3.0,
    },
    "speed": {
        "max_decel_ms2": 3.0,
        "safety_factor": 0.5,
        "max_speed_ms": 3.0,
        "min_speed_ms": 0.5,
        "cautious_step_m": 2.0,
        "normal_step_m": 5.0,
        "roam_alt_m": 10.0,          # 定高漫游目标高度(米)
    },
    "mission": {
        "mode": "roam",                   # "roam"=定高漫游 | "traverse"=定向穿越 | "inspect"=巡检穿越
        "goal_bearing_deg": -1.0,         # 目标方向 0-360°, -1=无目标
        "goal_weight": 0.3,               # 定向穿越权重
        "inspect_weight": 0.6,            # 巡检穿越权重 (更强调方向)
    },
    "climb": {
        "max_climb_rate_ms": 1.2,
        "window_factor": 0.7,
        "slope_min_sectors": 5,
        "slope_safe_height_m": 2.0,
    },
    "topo": {
        "save_path": "/tmp/one360s_topo.json",
        "save_interval_s": 60.0,
        "fork_min_pinch_m": 15.0,          # 高阈值防误触发: 只有真正开阔岔路才记
        "fork_min_angle_sep": 30.0,        # 两个出口至少隔30°才算不同方向
        "backtrack_tolerance_m": 5.0,
        "dead_end_ttl_s": 30.0,            # 短TTL: 死路标记30秒过期
        "max_nodes": 20,                   # 最多保留20个节点, 超过删最旧的
    },
    "mavlink": {
        "device": "/dev/ttyTHS1",
        "baud": 921600,
        "rate_hz": 10,
    },
    "debug": {
        "publish_filtered_cloud": False,
    },
}


def load_params(yaml_path: str = None) -> dict:
    """
    加载 params.yaml → 嵌套字典。
    如果文件不存在，返回默认参数。
    """
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            user_params = yaml.safe_load(f) or {}
    else:
        user_params = {}

    # 深度合并: 用户参数覆盖默认值
    return _deep_merge(DEFAULT_PARAMS, user_params)


def get_pipeline_params(params: dict) -> dict:
    """提取 pipeline 相关参数"""
    return {
        k: params.get(k, DEFAULT_PARAMS.get(k, {}))
        for k in [
            "range_filter",
            "voxel_filter",
            "radius_outlier",
            "sector",
            "temporal_filter",
            "obstacle_confidence",
            "inflation",
            "attitude_compensation",
            "tag_filter",
            "crop_box",
        ]
    }


def derive_params(params: dict) -> dict:
    """
    从 drone 物理参数自动推导所有算法参数。
    调用时机: load_params 之后。
    """
    drone = params.get("drone", DEFAULT_PARAMS["drone"])
    body_w = drone["body_width_m"]
    body_l = drone["body_length_m"]
    body_h = drone["body_height_m"]
    l_z = drone["lidar_z_offset"]
    safety = drone["safety_extra_m"]

    # 车辆有效半径 = 最大半宽 + 安全余量
    vehicle_radius = max(body_w, body_l) / 2.0
    effective_R = vehicle_radius + safety

    # ── 自动推导 crop_box ──
    # LiDAR 在机身中心上方 lidar_z_offset 处
    # crop_box 坐标系: LiDAR 为原点, x=前, y=左, z=上
    if "crop_box" not in params:
        params["crop_box"] = {}
    params["crop_box"].setdefault("x_min", -body_l / 2 + l_z * 0.1)
    params["crop_box"].setdefault("x_max", body_l / 2)
    params["crop_box"].setdefault("y_min", -body_w / 2)
    params["crop_box"].setdefault("y_max", body_w / 2)
    params["crop_box"].setdefault("z_min", -body_h)
    params["crop_box"].setdefault("z_max", None)

    # ── 同步到 inflation ──
    params.setdefault("inflation", {})
    params["inflation"].setdefault("vehicle_radius_m", vehicle_radius)
    params["inflation"].setdefault("safety_extra_m", safety)

    # ── 同步到 pinch ──
    params.setdefault("pinch", {})
    params["pinch"].setdefault("vehicle_radius_m", vehicle_radius)
    params["pinch"].setdefault("safety_extra_m", safety)
    params["pinch"].setdefault("effective_R", effective_R)

    # ── 同步到 speed ──
    params.setdefault("speed", {})
    params["speed"].setdefault("max_speed_ms", drone["max_speed_ms"])
    params["speed"].setdefault("max_decel_ms2", drone["max_decel_ms2"])
    params["speed"].setdefault("min_speed_ms", drone["min_speed_ms"])

    return params


def validate_params(params: dict) -> list[str]:
    """
    Pre-arm 参数校验。
    返回错误列表，空列表 = 通过。
    """
    errors = []

    try:
        pinch = params.get("pinch", {})
        if pinch.get("effective_R", 0) <= 0.05:
            errors.append("pinch.effective_R 过小 (<0.05m)")
        if pinch.get("max_range", 0) < 5.0:
            errors.append("pinch.max_range 过小 (<5m)")
        if pinch.get("all_blocked_m", 0) <= 0.3:
            errors.append("pinch.all_blocked_m 过小 (<0.3m)")

        drone = params.get("drone", {})
        if drone.get("body_width_m", 0) <= 0.1:
            errors.append("drone.body_width_m 过小 (<0.1m)")
        if drone.get("body_length_m", 0) <= 0.1:
            errors.append("drone.body_length_m 过小 (<0.1m)")

        speed = params.get("speed", {})
        if speed.get("max_speed_ms", 0) <= 0.2:
            errors.append("speed.max_speed_ms 过小 (<0.2m/s)")
        if speed.get("max_decel_ms2", 0) <= 0.5:
            errors.append("speed.max_decel_ms2 过小 (<0.5m/s²)")

        climb = params.get("climb", {})
        if climb.get("max_climb_rate_ms", 0) <= 0.1:
            errors.append("climb.max_climb_rate_ms 过小 (<0.1m/s)")

        # 一致性检查
        if pinch.get("effective_R", 0) > drone.get("body_width_m", 99):
            errors.append("pinch.effective_R > drone.body_width_m (可能填反了)")

        # 速度逻辑检查
        if speed.get("max_speed_ms", 0) > 10.0:
            errors.append("speed.max_speed_ms > 10m/s (不合理)")
        if speed.get("min_speed_ms", 0) >= speed.get("max_speed_ms", 0):
            errors.append("speed.min_speed_ms >= speed.max_speed_ms")

    except Exception as e:
        errors.append(f"参数校验异常: {e}")

    return errors


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个字典。override 的值覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
