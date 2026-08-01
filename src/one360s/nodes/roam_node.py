"""
roam_node — 漫游决策节点。

ROS 2 薄包装: 订阅 distances + position → RoamDecision.decide() → 发布 roam_target。
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8

import numpy as np

from one360s.roam_decision import RoamDecision
from one360s.topo_memory import TopoMemory
from one360s.config import load_params, derive_params


class RoamNode(Node):
    """distances → 方向选择 → roam_target。"""

    def __init__(self):
        super().__init__("roam_node")

        # 参数
        self.declare_parameter("config_path", "")
        config_path = self.get_parameter(
            "config_path").get_parameter_value().string_value
        self.params = load_params(config_path if config_path else None)
        self.params = derive_params(self.params)  # 从 drone 物理参数推导算法参数

        # 拓扑记忆
        self.topo = TopoMemory(**self.params.get("topo", {}))
        self.topo.load_snapshot()

        # 决策引擎
        self.decision = RoamDecision(self.params, self.topo)

        # 订阅
        self.distances_sub = self.create_subscription(
            Float32MultiArray, "/one360s/distances", self.distances_cb, 10)
        self.position_sub = self.create_subscription(
            Float32MultiArray, "/one360s/position", self.position_cb, 10)

        # 发布
        self.target_pub = self.create_publisher(
            Float32MultiArray, "/one360s/roam_target", 10)

        # 状态
        self.latest_heading = 0.0
        self.latest_speed = 0.0
        self.latest_alt = 0.0
        self.latest_laser_alt = 0.0
        self.latest_up_laser = 0.0  # 上视激光测距

        # 定时保存拓扑快照
        self.save_timer = self.create_timer(
            self.params.get("topo", {}).get("save_interval_s", 60.0),
            self.save_cb,
        )

        self.get_logger().info("roam_node started")

    def position_cb(self, msg: Float32MultiArray):
        """[heading_deg, groundspeed_ms, alt_m, laser_alt_m, up_laser_m]"""
        data = msg.data
        if len(data) >= 4:
            self.latest_heading = data[0]
            self.latest_speed = data[1]
            self.latest_alt = data[2]
            self.latest_laser_alt = data[3]
        if len(data) >= 5:
            self.latest_up_laser = data[4]

    def distances_cb(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float64)

        if len(data) != 288:
            self.get_logger().warn(
                f"Expected 288 values, got {len(data)}", throttle_duration_sec=5)
            return

        # 拆回四层
        distances_all = {
            "L0": data[0:72],
            "L1": data[72:144],
            "L2": data[144:216],
            "up": data[216:288],
        }

        # 决策
        result = self.decision.decide(
            distances_all,
            heading_deg=self.latest_heading,
            groundspeed_ms=self.latest_speed,
            current_alt_m=self.latest_alt,
            up_laser_m=self.latest_up_laser,
            laser_alt_m=self.latest_laser_alt,
        )

        if result is None:
            self.get_logger().error("全堵无退路 — 需要人工介入!")
            # 发布 hover 指令
            hover_msg = Float32MultiArray()
            hover_msg.data = [
                self.latest_heading, 0.0, 0.0,
                self.latest_alt, 4.0,  # action=hover
            ]
            self.target_pub.publish(hover_msg)
            return

        # 发布 roam_target
        target_msg = Float32MultiArray()
        action_code = {
            "fly": 0, "climb": 1, "descend": 2,
            "backtrack": 3, "hover": 4,
        }.get(result["action"], 0)

        target_alt = result.get("target_alt_m")
        target_msg.data = [
            result["bearing_deg"],
            result["step_m"],
            result["speed_ms"],
            target_alt if target_alt is not None else self.latest_alt,
            float(action_code),
        ]
        self.target_pub.publish(target_msg)

        # 日志
        self.get_logger().info(
            f"action={result['action']} "
            f"bearing={result['bearing_deg']:.0f}° "
            f"speed={result['speed_ms']:.1f}m/s "
            f"reason={result['reason']}",
            throttle_duration_sec=1.0,
        )

    def save_cb(self):
        self.topo.periodic_save()


def main():
    rclpy.init()
    node = RoamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.topo.periodic_save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
