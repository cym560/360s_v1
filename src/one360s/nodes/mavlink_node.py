"""
mavlink_node — MAVLink 桥接节点。

上行 (Nano → 飞控):
  1. OBSTACLE_DISTANCE @10Hz  (来自 /one360s/distances)
  2. SET_POSITION_TARGET_GLOBAL_INT  (来自 /one360s/roam_target)
  3. DISTANCE_SENSOR  (来自激光测距, 飞控原生支持)

下行 (飞控 → Nano):
  ATTITUDE / GLOBAL_POSITION_INT / DISTANCE_SENSOR
  → 发布 /one360s/position (heading, speed, alt, laser_alt)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8

import time
import math
import struct

try:
    from pymavlink import mavutil
    HAS_MAVLINK = True
except ImportError:
    HAS_MAVLINK = False


class MavlinkNode(Node):
    """MAVLink 桥接。"""

    def __init__(self):
        super().__init__("mavlink_node")

        if not HAS_MAVLINK:
            self.get_logger().error("pymavlink not installed!")
            return

        # 参数
        self.declare_parameter("device", "/dev/ttyTHS1")
        self.declare_parameter("baud", 921600)
        self.declare_parameter("rate_hz", 10)

        device = self.get_parameter("device").value
        baud = self.get_parameter("baud").value
        self.rate_hz = self.get_parameter("rate_hz").value

        # MAVLink 连接
        self.get_logger().info(f"Connecting to {device} @ {baud}...")
        self.mav = mavutil.mavlink_connection(device, baud=baud)
        self.get_logger().info("MAVLink connected")

        # 订阅
        self.distances_sub = self.create_subscription(
            Float32MultiArray, "/one360s/distances",
            self.distances_cb, 10)
        self.target_sub = self.create_subscription(
            Float32MultiArray, "/one360s/roam_target",
            self.target_cb, 10)
        self.health_sub = self.create_subscription(
            UInt8, "/one360s/health",
            self.health_cb, 10)

        # 发布
        self.position_pub = self.create_publisher(
            Float32MultiArray, "/one360s/position", 10)

        # 定时发送 (10Hz)
        self.send_timer = self.create_timer(1.0 / self.rate_hz, self._send_loop)

        # 状态
        self.health = 0
        self.emergency_land_sent = False
        self.latest_distances = None
        self.latest_target = None
        self.last_distance_send = 0.0

        # 飞控状态
        self.heading_deg = 0.0
        self.groundspeed_ms = 0.0
        self.alt_m = 0.0
        self.laser_alt_m = 0.0
        self.up_laser_m = 0.0  # 上视激光测距 (0=未连接)

        # 请求飞控数据流
        self._request_data_streams()

    def _request_data_streams(self):
        """请求飞控下发所需数据。"""
        # 请求 ATTITUDE @10Hz
        self.mav.mav.command_long_send(
            1, 1,  # target_system, target_component
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            100000,  # 100ms = 10Hz
            0, 0, 0, 0, 0,
        )
        # 请求 GLOBAL_POSITION_INT @5Hz
        self.mav.mav.command_long_send(
            1, 1,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            200000,  # 5Hz
            0, 0, 0, 0, 0,
        )

    # ── 订阅回调 ──

    def distances_cb(self, msg: Float32MultiArray):
        self.latest_distances = msg.data

    def target_cb(self, msg: Float32MultiArray):
        self.latest_target = msg.data

    def health_cb(self, msg: UInt8):
        prev = self.health
        self.health = msg.data
        if self.health == 2 and prev != 2 and not self.emergency_land_sent:
            self.get_logger().error("watchdog FAIL — 强制 LAND!")
            self._send_emergency_land()

    # ── 紧急降落 ──

    def _send_emergency_land(self):
        """发送 LAND 模式指令到飞控。"""
        try:
            # MAV_CMD_DO_SET_MODE: 切换到 LAND
            self.mav.mav.command_long_send(
                1, 1,  # target_system, target_component
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                9,  # PX4: LAND = 9 (MAIN_STATE_AUTO_LAND)
                0, 0, 0, 0, 0,
            )
            self.emergency_land_sent = True
            self.get_logger().info("LAND 指令已发送")
        except Exception as e:
            self.get_logger().error(f"发送 LAND 失败: {e}")

    # ── 定时发送 ──

    def _send_loop(self):
        """每帧发送。"""
        # 1. 处理飞控下行消息
        self._process_mavlink_incoming()

        # 2. 发送 OBSTACLE_DISTANCE
        if self.latest_distances and len(self.latest_distances) >= 144:
            self._send_obstacle_distance()

        # 3. 发送 GUIDED 指令
        if self.latest_target and len(self.latest_target) >= 5:
            self._send_guided_target()

        # 4. 发布位置信息
        pos_msg = Float32MultiArray()
        pos_msg.data = [
            self.heading_deg,
            self.groundspeed_ms,
            self.alt_m,
            self.laser_alt_m,
            self.up_laser_m,   # 上视激光测距 (0=未连接)
        ]
        self.position_pub.publish(pos_msg)

    def _process_mavlink_incoming(self):
        """处理飞控下行的 MAVLink 消息。"""
        msg = self.mav.recv_match(blocking=False)
        while msg is not None:
            msg_type = msg.get_type()

            if msg_type == "ATTITUDE":
                # 从四元数提取 yaw
                q0, q1, q2, q3 = msg.q
                siny = 2.0 * (q0 * q3 + q1 * q2)
                cosy = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
                self.heading_deg = math.degrees(math.atan2(siny, cosy)) % 360

            elif msg_type == "GLOBAL_POSITION_INT":
                self.alt_m = msg.relative_alt / 1000.0  # mm → m
                self.groundspeed_ms = math.sqrt(
                    msg.vx**2 + msg.vy**2) / 100.0  # cm/s → m/s

            elif msg_type == "DISTANCE_SENSOR":
                if msg.orientation == 25:  # MAV_SENSOR_ROTATION_PITCH_270 = 朝下
                    self.laser_alt_m = msg.current_distance / 100.0
                elif msg.orientation == 24:  # MAV_SENSOR_ROTATION_PITCH_90 = 朝上
                    self.up_laser_m = msg.current_distance / 100.0

            elif msg_type == "HEARTBEAT":
                pass  # 心跳维持

            msg = self.mav.recv_match(blocking=False)

    def _send_obstacle_distance(self):
        """
        distances[72] (L1层) → OBSTACLE_DISTANCE MAVLink 消息。
        格式: uint16[72] 厘米, 65535 = 无障碍。
        """
        now = time.time()
        if now - self.last_distance_send < 0.09:  # 最多 ~11Hz
            return
        self.last_distance_send = now

        # 取 L1 层: distances[72:144]
        l1 = self.latest_distances[72:144]
        dist_cm = []
        for d in l1:
            if d >= 39.0:
                dist_cm.append(65535)  # MAX
            else:
                dist_cm.append(min(65534, int(d * 100)))  # m → cm

        self.mav.mav.obstacle_distance_send(
            int(time.time() * 1e6),  # time_usec
            0,    # sensor_type: MAV_DISTANCE_SENSOR_LASER
            dist_cm,  # distances[72]
            5.0,  # increment: 5°/扇区 (72扇区 × 5° = 360°)
            5.0,  # min_distance
            40.0, # max_distance
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 未使用
        )

    def _send_guided_target(self):
        """
        roam_target → SET_POSITION_TARGET_LOCAL_NED (MAV_FRAME_BODY_NED)。
        [bearing_deg, step_m, speed_ms, alt_m, action_code]

        BODY_NED: x=前, y=右, z=下
        x,y = 水平步进偏移
        z   = 高度差 (target_alt - current_alt, NED反转)
        yaw = 飞行方向 (机头朝前)
        """
        bearing, step, speed, target_alt, action = self.latest_target

        # 水平: 体轴系步进
        bearing_rad = math.radians(bearing)
        x_m = step * math.cos(bearing_rad)
        y_m = step * math.sin(bearing_rad)

        # 垂直: 高度差 → NED z
        alt_diff = target_alt - self.alt_m if target_alt > 0.1 else 0.0
        z_m = -alt_diff  # NED: 上=负

        # type_mask: 忽略速度/加速度/yaw_rate, 控制位置(x,y,z)+yaw
        # bit: 10=YAW_RATE 9=YAW 8=AZ 7=AY 6=AX 5=VZ 4=VY 3=VX 2=Z 1=Y 0=X
        type_mask = 0b0000010111111000  # IGNORE: VX,VY,VZ,AX,AY,AZ,YAW_RATE; USE: X,Y,Z,YAW

        self.mav.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            1, 1,  # target_system, target_component
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            x_m, y_m, z_m,   # x, y, z
            0.0, 0.0, 0.0,   # vx, vy, vz (ignored)
            0.0, 0.0, 0.0,   # ax, ay, az (ignored)
            bearing_rad, 0.0, # yaw = 飞行方向, yaw_rate (ignored)
        )


def main():
    rclpy.init()
    node = MavlinkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
