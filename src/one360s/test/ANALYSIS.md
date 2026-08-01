# one360s v1 潜在问题分析

> 逐模块审查。按严重度排列: ❌ bug / ⚠️ 设计缺陷 / 💡 优化建议。

---

## ❌ Bug — 必须修

### 1. `_select_best()` 用零数组计算当前方向夹点

**文件**: `roam_decision.py:262-264`

```python
current_pinch_val = pinch_point(
    np.zeros(72),  # placeholder ← BUG
    heading_deg, self.effective_R, self.max_range,
)
```

传入的是 `np.zeros(72)` 而非实际 distances。算出来的 pinch 永远是 0, 然后代码没有用这个结果——它用 `self.current_pinch` (上一帧的值), 但 `current_pinch_val` 算了个寂寞。

**后果**: 首帧 `self.current_pinch=40.0`, 即使飞机正前方 1m 有墙, 滞后逻辑也认为"当前方向 pinch=40", 不触发转向。首帧之后才纠正。

**修复**: 删除这段无用代码, `_select_best` 直接从 `scores` 或额外参数拿当前方向的夹点。

---

### 2. `_temporal_filter` 的 clear_counters 是所有扇区共用的

**文件**: `pipeline.py:360-380`

```python
def _temporal_filter(self, key, distances):
    for s in range(72):
        if d_new >= self.max_range_m * 0.5:  # 当前帧无障碍
            self._clear_counters[key] += 1   # ← 全局计数器!
```

只有一个计数器 `self._clear_counters[key]`, 72 个扇区共享。

**场景**: 扇区 0 连续 3 帧无障碍 → counter=3 → 清空扇区 0。但同时扇区 1 在第 2 帧出现障碍 → `d_new < 20m` → counter 重置为 0。扇区 0 的障碍确认过程被打断。

**后果**: 扇区之间互相干扰。密集障碍场景下, counter 频繁重置, 无障碍扇区永远无法确认清除。

**修复**: `clear_counters` 改为 `np.zeros(72)`, 每扇区独立计数。

---

### 3. `_inflate_angles` 内层逻辑存在冗余

**文件**: `pipeline.py`

```python
for offset in range(-inflate_bins, inflate_bins + 1):
    t = (s + offset) % num_sectors
    if distances[t] < self.max_range_m * 0.5:
        result[t] = min(result[t], d)
    else:
        result[t] = min(result[t], d)  # ← 两个分支一样!
```

if/else 两个分支做完全相同的事。实际意图应该是: 无障碍扇区也写入膨胀值(`d`), 有障碍扇区取更近的。但即使如此, `result[t] = min(result[t], d)` 对于原本 MAX 的扇区会错误地设为一个具体障碍距离, 导致虚假近障碍。

**后果**: 膨胀后, 原本无障碍的扇区被标上邻近扇区的障碍距离, 导致 pinch_point 在那些方向被虚假缩短。安全侧偏保守, 但可能导致过度保守(能过的通道被堵)。

**修复**: 删除冗余 if/else。如果意图是安全保守, 保留 `result[t] = min(result[t], d)` 但加注释说明。

---

### 4. `_handle_climb` 退出后不检查 L1 是否仍然全堵

**文件**: `roam_decision.py`

```python
def _handle_climb(self, ...):
    if current_alt_m >= self.climb_target_alt - 0.3:
        self.in_climb = False
        return {"action": "fly", ...}  # ← 直接切 fly
```

爬升到达目标高度后, 直接切回 `fly` 模式。但没有检查: 达到目标高度后, L1 层是否已经通畅了? 如果爬升高度不够(比如树冠比预期高), L1 仍然全堵, 下一帧会再次触发 `_handle_blocked` → 再次爬升, 形成循环。

**后果**: 爬升-水平-爬升-水平的振荡。

**修复**: 退出爬升前, 检查 L1 层是否真的通畅了。如果不通, 增加爬升目标继续爬。

---

## ⚠️ 设计缺陷 — 应该修

### 5. 伪压缩只在夹点距离处检查

**文件**: `pseudo_compress.py:expand_margin()`

宽度检查只在 `pinch` 这一个距离处做。通道可能在 pinch 距离处够宽, 但在更近处收窄。

**场景**:
```
飞机 → ░░░░ 宽 ░░░░ → ▓▓ 窄 ▓▓ → ░░░░ 宽 ░░░░
       0m          5m   pinch=8m          15m
```
pinch=8m 处的宽度够, 但 5m 处其实更窄。膨胀裕度虚高。

**后果**: 评分过高, 飞过去在 5m 处才发现窄 → 急刹车。

**缓解**: pinch_point 本身就是从近到远检查的——如果 5m 处真的窄到过不去, pinch 应该在 5m 处触发, 不会是 8m。所以"更近处更窄但夹点没触发"的情况只在漏检时发生。漏检才是根因。

**建议**: 问题不严重, 但可以在 `expand_margin` 里增加 1-2 个中间距离的抽查(比如 pinch/2 处)。

---

### 6. 回溯时看不到后方

**文件**: `roam_decision.py:_handle_blocked()`

```python
return {
    "action": "backtrack",
    "bearing_deg": bt["from_bearing_deg"],  # 可能是飞机后方
    ...
}
```

回溯方向可能是飞机的后方(>±90°)。但 `scan_pinch_map` 只扫描前方 ±90°。回溯指令发出后, 飞机需要先转过去, 转的过程中仍在向前飞。

**后果**: 回溯路径上可能有新障碍(比如后面来了另一架飞机), 系统看不到。

**建议**: v1 接受这个限制。回溯速度降到 1m/s, BendyRuler 作为最后防线。

---

### 7. `_score_all` 重复计算 expand_margin

**文件**: `roam_decision.py:_score_all()`

每帧对 36 个方向各调一次 `expand_margin`, 每次内部又遍历 72 扇区。加上 `scan_pinch_map` 已经遍历了 36×40×N 次。算力总计 ~0.5ms, 问题不大, 但 36 次 `expand_margin` 可以批量优化。

---

### 8. 四层 z-bin 的 L2 和 up 层可能缺少有效点

**场景**: 开阔场地上方, L2 和 up 层几乎无点 → `_extract_sectors` 返回全 MAX → `can_climb` 判定为"上方通畅" → 实际可能有电线/细枝。

**根因**: 点云稀疏 + min_points 阈值 → 上方障碍漏检。

**建议**: v1 接受。climb 前检查 min_points 实际命中数, 如果 L2 层的 `sectors_with_data` 太少(<10), 标记为"上方数据不足" → 禁止爬升。

---

### 9. `is_dead_end` 对 fork 节点的判断逻辑可疑

**文件**: `topo_memory.py`

```python
if node.pattern == "fork":
    if sector in node.unexplored or sector == node.chosen:
        return True
```

`sector == node.chosen` → 返回 True 意味着"你选过的方向就是死胡同"。但选过的方向不一定是死胡同——可能只是还没走完。这个逻辑应该在 `mark_dead_end` 被调用后才成立, 而不是只要 chosen 就成立。

**建议**: 把 `is_dead_end` 的检查改为只看 `node.pattern == "dead_end"`, 不处理 fork。fork 的出口信息通过 `unexplored` 列表管理已经够了。

---

## 💡 优化建议 — 可以修

### 10. PointCloud2 解析应该用 `sensor_msgs_py`

`processor_node._unpack_pointcloud` 手动解析二进制, 对不同 Mid-360 固件版本可能字段偏移不同。`sensor_msgs_py.point_cloud2.read_points` 自动处理。

### 11. 爬升时逐米试探效率低

`_handle_blocked` 里 `for test_dh in [1.0, 2.0, 3.0]` 逐米试。可以直接算: `需要的爬升量 = L1 最近障碍处对应的高度 + 余量`。

### 12. watchdog 用 `time.time()` 而非 ROS time

仿真回放 rosbag 时 watchdog 会误触发。应该用 `self.get_clock().now()`。

### 13. 类名残留 `AeroHaloPipeline`

`pipeline.py` 类名还叫 `AeroHaloPipeline`, 应改为 `One360sPipeline`。

---

## 汇总

| # | 问题 | 严重度 | 修法 |
|---|------|--------|------|
| 1 | `_select_best` 零数组计算当前夹点 | ❌ bug | 删无用代码, 从 scores 拿 |
| 2 | `clear_counters` 全局共享 | ❌ bug | 改为 np.zeros(72) |
| 3 | `_inflate_angles` if/else 冗余 | ❌ bug | 删冗余分支 |
| 4 | 爬升退出不检查 L1 通畅 | ❌ bug | 退出前验证 L1 |
| 5 | 伪压缩只查 pinch 处 | ⚠️ 设计 | 加中间距离抽查 |
| 6 | 回溯看不到后方 | ⚠️ 设计 | v1接受, BendyRuler兜底 |
| 7 | expand_margin 重复计算 | ⚠️ 设计 | 批量优化 |
| 8 | L2/up 层数据不足 | ⚠️ 设计 | 点数不足禁止爬升 |
| 9 | `is_dead_end` fork 逻辑 | ⚠️ 设计 | 简化判断逻辑 |
| 10 | PointCloud2 手动解析 | 💡 优化 | 用 sensor_msgs_py |
| 11 | 逐米试探爬升量 | 💡 优化 | 直接算 |
| 12 | watchdog 不用 ROS time | 💡 优化 | 改用 get_clock() |
| 13 | 类名残留 AeroHalo | 💡 优化 | 重命名 |
