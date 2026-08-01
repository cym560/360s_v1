"""
拓扑记忆单元测试。

纯 Python, 无需 ROS。
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from one360s.topo_memory import TopoMemory, TopoNode


def test_record_fork():
    """记录分叉口 → 可以回溯。"""
    tm = TopoMemory()
    tm.set_current_bearing(0.0)

    exits = [0, 10, 20]  # 三个出口
    tm.record_fork(exits, chosen=0)

    assert len(tm) == 1
    assert tm.nodes[0].pattern == "fork"
    assert tm.nodes[0].unexplored == [10, 20]


def test_backtrack_returns_unexplored():
    """回溯 → 返回未探索出口。"""
    tm = TopoMemory()
    tm.set_current_bearing(0.0)

    tm.record_fork([0, 10, 20], chosen=0)
    result = tm.backtrack()

    assert result is not None
    assert result["next_exit_deg"] in [50.0, 100.0]  # 10*5 或 20*5


def test_backtrack_exhausted():
    """所有出口都探索完 → 返回 None。"""
    tm = TopoMemory()
    tm.set_current_bearing(0.0)

    tm.record_fork([0, 10], chosen=0)
    tm.backtrack()  # 消耗 10
    tm.record_fork([0], chosen=0)
    tm.backtrack()  # 消耗完

    result = tm.backtrack()
    assert result is None, "All exits explored, should return None"


def test_mark_dead_end():
    """标记死胡同 → is_dead_end 返回 True。"""
    tm = TopoMemory(dead_end_ttl_s=999)
    tm.set_current_bearing(45.0)

    tm.record_dead_end(2.5)
    assert tm.is_dead_end(45.0, ttl_s=999)


def test_dead_end_ttl_expires():
    """死胡同标记过期 → is_dead_end 返回 False。"""
    tm = TopoMemory(dead_end_ttl_s=0)  # 立刻过期
    tm.set_current_bearing(90.0)

    tm.record_dead_end(2.0)
    time.sleep(0.1)
    assert not tm.is_dead_end(90.0, ttl_s=0)


def test_clean_old_nodes():
    """清理超旧节点。"""
    tm = TopoMemory()
    tm.set_current_bearing(0.0)
    tm.record_fork([0, 10], chosen=0)
    tm.clear_old_nodes(max_age_s=0)  # 立刻清理
    assert len(tm) == 0


if __name__ == "__main__":
    test_record_fork()
    test_backtrack_returns_unexplored()
    test_backtrack_exhausted()
    test_mark_dead_end()
    test_dead_end_ttl_expires()
    test_clean_old_nodes()
    print("All topo_memory tests passed!")
