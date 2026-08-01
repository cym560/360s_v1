"""
watchdog_node — 健康监控节点。

监控: 点云到达间隔 / 处理延迟。
状态: OK → DEGRADED → FAIL。
降级: 按飞行模式差异化。
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray, UInt8

import time


class WatchdogNode(Node):
    """健康监控 + 降级。"""

    def __init__(self):
        super().__init__("watchdog_node")

        # 参数
        self.cloud_timeout_ms = 300
        self.proc_timeout_ms = 400
        self.startup_grace_ms = 20000
        self.start_time = time.time() * 1000

        # 订阅
        self.status_sub = self.create_subscription(
            String, "/one360s/status", self.status_cb, 10)

        # 额外的帧到达监控 (通过 distances topic)
        self.distances_sub = self.create_subscription(
            Float32MultiArray, "/one360s/distances", self.distances_cb, 10)

        # 发布
        self.health_pub = self.create_publisher(
            UInt8, "/one360s/health", 10)

        # 状态
        self.last_cloud_ms = 0.0
        self.last_proc_ms = 0.0
        self.health = 0  # 0=OK, 1=DEGRADED, 2=FAIL
        self.degraded_since_ms = 0.0

        # 定时检查
        self.check_timer = self.create_timer(0.1, self._check_health)  # 10Hz

        self.get_logger().info("watchdog_node started")

    def distances_cb(self, msg: Float32MultiArray):
        self.last_cloud_ms = time.time() * 1000

    def status_cb(self, msg: String):
        self.last_proc_ms = time.time() * 1000

    def _check_health(self):
        now_ms = time.time() * 1000

        # 冷启动宽限期
        if now_ms - self.start_time < self.startup_grace_ms:
            self._set_health(0)
            return

        # 点云超时检查
        cloud_dt = now_ms - self.last_cloud_ms if self.last_cloud_ms else float("inf")
        if cloud_dt > self.cloud_timeout_ms:
            self._degrade("cloud_timeout")
            return

        # 处理延迟检查
        proc_dt = now_ms - self.last_proc_ms if self.last_proc_ms else float("inf")
        if proc_dt > self.proc_timeout_ms:
            self._degrade("proc_lag")
            return

        # 恢复正常
        if self.health != 0:
            if now_ms - self.degraded_since_ms > 2000:  # 2秒恢复
                self._set_health(0)

    def _degrade(self, reason: str):
        now_ms = time.time() * 1000

        if self.health == 0:
            self.degraded_since_ms = now_ms
            self.get_logger().warn(f"降级: {reason}")

        if now_ms - self.degraded_since_ms > 5000:
            self._set_health(2)  # FAIL
        else:
            self._set_health(1)  # DEGRADED

    def _set_health(self, level: int):
        if level != self.health:
            self.health = level
            msg = UInt8(data=level)
            self.health_pub.publish(msg)

            labels = {0: "OK", 1: "DEGRADED", 2: "FAIL"}
            self.get_logger().info(f"Health → {labels[level]}")


def main():
    rclpy.init()
    node = WatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
