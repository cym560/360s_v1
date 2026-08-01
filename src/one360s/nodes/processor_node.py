"""
processor_node — 感知处理节点。

ROS 2 薄包装: 订阅点云 → pipeline.process() → 发布 distances[288]。
不包含算法逻辑。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from std_msgs.msg import Float32MultiArray, String

import math
import numpy as np
import struct
import time

from one360s.pipeline import One360sPipeline
from one360s.config import load_params, get_pipeline_params, derive_params, validate_params


class ProcessorNode(Node):
    """点云 → 四层扇区距离。"""

    def __init__(self):
        super().__init__("processor_node")

        # 参数
        self.declare_parameter("config_path", "")
        config_path = self.get_parameter(
            "config_path").get_parameter_value().string_value
        params = load_params(config_path if config_path else None)
        params = derive_params(params)  # 从 drone 物理参数推导算法参数
        pipe_params = get_pipeline_params(params)

        self.pipeline = One360sPipeline(pipe_params)
        self.publish_filtered = params.get("debug", {}).get(
            "publish_filtered_cloud", False)

        # 订阅
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/livox/lidar/pointcloud", self.cloud_cb, 10)
        self.imu_sub = self.create_subscription(
            Imu, "/mavros/imu/data", self.imu_cb, 10)

        # 发布
        self.distances_pub = self.create_publisher(
            Float32MultiArray, "/one360s/distances", 10)
        self.status_pub = self.create_publisher(
            String, "/one360s/status", 10)

        if self.publish_filtered:
            self.filtered_pub = self.create_publisher(
                PointCloud2, "/one360s/filtered_cloud", 10)

        # 状态
        self.latest_pitch = 0.0
        self.latest_roll = 0.0
        self.frame_count = 0

        # ── Pre-arm 参数校验 ──
        errors = validate_params(params)
        if errors:
            for e in errors:
                self.get_logger().error(f"PREARM FAIL: {e}")
            self.get_logger().fatal(
                f"参数校验失败 ({len(errors)}项), 禁止起飞! 请检查 params.yaml")
        else:
            self.get_logger().info("pre-arm 参数校验通过")

        self.get_logger().info("processor_node started")

    def imu_cb(self, msg: Imu):
        """从四元数提取 pitch/roll。"""
        q = msg.orientation
        # 简化: 直接从 mavros IMU 拿欧拉角
        # 如果没有, 用四元数转换
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self.latest_pitch = math.copysign(90.0, sinp)
        else:
            self.latest_pitch = math.degrees(math.asin(sinp))

        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.latest_roll = math.degrees(math.atan2(sinr, cosr))

    def cloud_cb(self, msg: PointCloud2):
        t0 = time.perf_counter()
        self.frame_count += 1

        # 解析 PointCloud2 → numpy
        points, tags = self._unpack_pointcloud(msg)

        # 调用管线
        result = self.pipeline.process(
            points, tags, self.latest_pitch, self.latest_roll)

        dt_ms = (time.perf_counter() - t0) * 1000
        result["stats"]["processing_time_ms"] = round(dt_ms, 1)

        # 打包 distances[288]
        dist_msg = Float32MultiArray()
        flat = []
        for key in self.pipeline.z_bin_keys:
            flat.extend(result[key].tolist())
        dist_msg.data = flat
        self.distances_pub.publish(dist_msg)

        # 状态 (每10帧发布一次)
        if self.frame_count % 10 == 0:
            stats = result["stats"]
            status = String()
            status.data = (
                f"frame={self.frame_count} "
                f"in={stats['input_points']} "
                f"out={stats['filtered_points']} "
                f"dt={stats['processing_time_ms']:.1f}ms"
            )
            self.status_pub.publish(status)

        # 调试: 过滤后点云
        if self.publish_filtered and hasattr(self, "filtered_pub"):
            filtered_pts = result.get("filtered_cloud")
            if filtered_pts is not None and len(filtered_pts) > 0:
                filtered_msg = self._pack_pointcloud(filtered_pts, msg.header)
                if filtered_msg:
                    self.filtered_pub.publish(filtered_msg)

    def _unpack_pointcloud(self, msg: PointCloud2):
        """
        PointCloud2 → (points, tags)。

        Mid-360 点云字段: x, y, z, intensity, tag, line
        解析 xyz 和 tag。
        """
        # 获取字段偏移
        fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
        x_offset = fields.get("x", (0, 7))[0]
        y_offset = fields.get("y", (4, 7))[0]
        z_offset = fields.get("z", (8, 7))[0]
        # tag 字段可能叫 "tag" 也可能不存在
        tag_info = fields.get("tag")

        point_step = msg.point_step
        data = msg.data
        n = len(data) // point_step

        points = np.zeros((n, 3), dtype=np.float32)
        tags = None

        for i in range(n):
            base = i * point_step
            points[i, 0] = struct.unpack_from("f", data, base + x_offset)[0]
            points[i, 1] = struct.unpack_from("f", data, base + y_offset)[0]
            points[i, 2] = struct.unpack_from("f", data, base + z_offset)[0]

        if tag_info is not None:
            tag_offset = tag_info[0]
            tags = np.zeros(n, dtype=np.uint8)
            for i in range(n):
                base = i * point_step
                tags[i] = data[base + tag_offset]

        return points, tags

    def _pack_pointcloud(self, pts: np.ndarray, header) -> PointCloud2 | None:
        """numpy (N,3) float32 → PointCloud2。手动打包，无需 sensor_msgs_py。"""
        if pts is None or len(pts) == 0:
            return None
        from sensor_msgs.msg import PointField
        import sys

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(pts)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12  # 3 × float32
        msg.row_step = msg.point_step * msg.width
        msg.is_bigendian = (sys.byteorder == "big")
        msg.is_dense = True

        # numpy → bytes
        buf = pts.astype(np.float32).tobytes()
        msg.data = bytearray(buf)
        return msg


def main():
    rclpy.init()
    node = ProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
