"""
one360s 启动文件。

启动全部 4 个节点 + livox_ros_driver2 (如果已安装)。

用法:
  ros2 launch one360s one360s.launch.py
  ros2 launch one360s one360s.launch.py config_path:=/path/to/params.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = LaunchConfiguration("config_path")
    config_arg = DeclareLaunchArgument(
        "config_path",
        default_value="",
        description="Path to params.yaml (空则使用默认参数)",
    )

    # processor_node — 感知处理
    processor = Node(
        package="one360s",
        executable="processor_node",
        name="processor_node",
        parameters=[{"config_path": config_path}],
        output="screen",
    )

    # roam_node — 漫游决策
    roam = Node(
        package="one360s",
        executable="roam_node",
        name="roam_node",
        parameters=[{"config_path": config_path}],
        output="screen",
    )

    # mavlink_node — MAVLink 桥接
    mavlink = Node(
        package="one360s",
        executable="mavlink_node",
        name="mavlink_node",
        output="screen",
    )

    # watchdog_node — 健康监控
    watchdog = Node(
        package="one360s",
        executable="watchdog_node",
        name="watchdog_node",
        output="screen",
    )

    return LaunchDescription([
        config_arg,
        processor,
        roam,
        mavlink,
        watchdog,
    ])
