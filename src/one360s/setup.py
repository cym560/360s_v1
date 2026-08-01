from setuptools import setup, find_packages
import os

package_name = "one360s"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/one360s.launch.py"]),
        ("share/" + package_name + "/config", ["config/params.yaml"]),
    ],
    install_requires=["numpy", "pyyaml", "pymavlink"],
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "processor_node = nodes.processor_node:main",
            "roam_node = nodes.roam_node:main",
            "mavlink_node = nodes.mavlink_node:main",
            "watchdog_node = nodes.watchdog_node:main",
        ],
    },
    zip_safe=True,
    author="one360s",
    description="单LiDAR夹点规划低算力自主导航",
    python_requires=">=3.10",
)
