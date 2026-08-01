"""
拓扑记忆 (Topological Memory)。

在关键时刻记录分叉口/死胡同/通道收窄。
支持回溯: 全堵时退回最近分叉口的未探索出口。
内存 <2KB, 每60秒持久化。
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TopoNode:
    """拓扑节点 — 记录一个关键位置的结构信息。"""
    pattern: str           # "fork" | "dead_end" | "narrowing"
    entry_bearing: int     # 进入方向(扇区号)
    exits: list = field(default_factory=list)       # 出口方向列表
    chosen: Optional[int] = None                    # 选了哪个出口
    unexplored: list = field(default_factory=list)  # 还没走过的出口
    pinch_at_entry: float = 0.0                     # 进入时的夹点
    timestamp: float = 0.0


class TopoMemory:
    """
    拓扑记忆管理器。

    写操作: record_fork / record_dead_end / record_narrowing / mark_dead_end
    读操作: backtrack / is_dead_end
    持久化: periodic_save / load_snapshot
    """

    def __init__(
        self,
        save_path: str = "/tmp/one360s_topo.json",
        save_interval_s: float = 60.0,
        fork_min_pinch_m: float = 15.0,
        fork_min_angle_sep: float = 30.0,
        backtrack_tolerance_m: float = 5.0,
        dead_end_ttl_s: float = 30.0,
        max_nodes: int = 20,
    ):
        self.nodes: list[TopoNode] = []
        self.save_path = save_path
        self.save_interval_s = save_interval_s
        self.fork_min_pinch_m = fork_min_pinch_m
        self.fork_min_angle_sep = fork_min_angle_sep
        self.backtrack_tolerance_m = backtrack_tolerance_m
        self.dead_end_ttl_s = dead_end_ttl_s
        self.max_nodes = max_nodes
        self._last_save = time.time()
        self._current_bearing = 0

    # ── 写操作 ──

    def set_current_bearing(self, bearing_deg: float):
        """更新当前朝向（在各 record 方法之前调用）"""
        self._current_bearing = int(round(bearing_deg / 5.0)) % 72

    def _add_node(self, node: TopoNode):
        """添加节点并执行容量控制。"""
        self.nodes.append(node)
        while len(self.nodes) > self.max_nodes:
            self.nodes.pop(0)

    def record_fork(self, exits: list, chosen: int):
        """
        记录分叉口。
        调用时机: 存在 ≥2 个夹点 > fork_min_pinch_m 且角度分离 > fork_min_angle_sep 的方向。
        """
        # 去重: 太近的出口合并
        filtered = [exits[0]]
        for e in exits[1:]:
            if all(abs(e - f) >= self.fork_min_angle_sep / 5.0 for f in filtered):
                filtered.append(e)
        if len(filtered) < 2:
            return  # 不够两个真正分开的方向, 不记

        self._add_node(TopoNode(
            pattern="fork",
            entry_bearing=self._current_bearing,
            exits=list(filtered),
            chosen=chosen,
            unexplored=[e for e in filtered if e != chosen],
            timestamp=time.time(),
        ))

    def record_dead_end(self, pinch: float = 0.0):
        """记录死胡同。调用时机: 全方向夹点 < all_blocked_m。"""
        self._add_node(TopoNode(
            pattern="dead_end",
            entry_bearing=self._current_bearing,
            pinch_at_entry=pinch,
            timestamp=time.time(),
        ))

    def record_narrowing(self, pinch: float = 0.0):
        """记录通道收窄。调用时机: pinch 3秒内减半。"""
        self._add_node(TopoNode(
            pattern="narrowing",
            entry_bearing=self._current_bearing,
            pinch_at_entry=pinch,
            timestamp=time.time(),
        ))

    def mark_dead_end(self, bearing_deg: float):
        """
        标记某方向通向死胡同。
        回溯完成后调用, 标记刚退出的那个方向。
        """
        sector = int(round(bearing_deg / 5.0)) % 72
        # 在最近的分叉口中标记该出口为死胡同
        for node in reversed(self.nodes):
            if node.pattern == "fork" and sector in node.unexplored:
                node.unexplored.remove(sector)
                break

    # ── 读操作 ──

    def backtrack(self) -> Optional[dict]:
        """
        找最近的有未探索出口的分叉口。
        从后往前遍历 → 最近优先。

        返回:
          {
              'from_bearing_deg': float,    # 来路反方向
              'next_exit_deg': float,        # 下一个要探索的出口
              'tolerance_m': float,          # 到达判定容差
          }
          或 None → 全探索完 → 悬停告警
        """
        now = time.time()
        for node in reversed(self.nodes):
            if node.pattern == "fork" and node.unexplored:
                # 检查 TTL (死胡同标记的有效期)
                if now - node.timestamp > self.dead_end_ttl_s * 3:
                    continue  # 太老了, 可能场景已变化

                next_sector = node.unexplored.pop()
                node.chosen = next_sector

                # 来路反方向
                from_bearing = (node.entry_bearing + 36) % 72 * 5.0
                next_bearing = next_sector * 5.0

                return {
                    "from_bearing_deg": from_bearing,
                    "next_exit_deg": next_bearing,
                    "tolerance_m": self.backtrack_tolerance_m,
                }

        return None  # 无路可退

    def is_dead_end(self, bearing_deg: float, ttl_s: float = None) -> bool:
        """
        检查某方向在 TTL 内是否被明确标记为死胡同。

        只检查 pattern=="dead_end" 的节点 (由 record_dead_end 写入)。
        不检查 fork 节点 — fork 的出口信息通过 unexplored 列表管理。
        """
        if ttl_s is None:
            ttl_s = self.dead_end_ttl_s

        sector = int(round(bearing_deg / 5.0)) % 72
        now = time.time()

        for node in reversed(self.nodes):
            if node.pattern == "dead_end":
                if node.entry_bearing == sector:
                    if now - node.timestamp < ttl_s:
                        return True
        return False

    # ── 持久化 ──

    def periodic_save(self):
        """每 save_interval_s 秒写一次快照。调用时机: 每帧或定时器。"""
        now = time.time()
        if now - self._last_save > self.save_interval_s:
            self._save_snapshot()
            self._last_save = now

    def _save_snapshot(self):
        """写 JSON 快照到磁盘。"""
        try:
            data = {
                "version": 1,
                "timestamp": time.time(),
                "nodes": [asdict(n) for n in self.nodes],
            }
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # 静默失败: 拓扑记忆是非关键的, 丢了也能继续飞

    def load_snapshot(self) -> bool:
        """从磁盘恢复快照。调用时机: 启动时。"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.nodes = [TopoNode(**n) for n in data.get("nodes", [])]
                return True
        except Exception:
            pass
        return False

    def clear_old_nodes(self, max_age_s: float = 300.0):
        """清理超旧节点 (默认 5 分钟)。控制内存。"""
        now = time.time()
        self.nodes = [n for n in self.nodes if now - n.timestamp < max_age_s]

    def __len__(self):
        return len(self.nodes)
