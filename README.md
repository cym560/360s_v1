# one360s v1

**Single-LiDAR pinch-point planning for low-compute autonomous drone navigation.**

No mapping. No localization. Plan directly in sensor space.

## Hardware

- **Compute**: NVIDIA Jetson Nano
- **LiDAR**: Livox Mid-360 (360° FOV, ~40m range)
- **Flight Controller**: Pixhawk / PX4 with MAVLink

## Quick Start

```bash
# Build
cd ~/ws
colcon build --packages-select one360s
source install/setup.bash

# Launch (all 4 nodes)
ros2 launch one360s one360s.launch.py

# With custom config
ros2 launch one360s one360s.launch.py config_path:=/path/to/params.yaml
```

## Architecture

```
Livox Mid-360 LiDAR
       │  PointCloud2 @10Hz
       ▼
┌─────────────────┐
│ processor_node   │  pipeline.process() → 288 floats (4 layers × 72 sectors)
└────────┬────────┘
       │  /one360s/distances
       ▼
┌─────────────────┐
│ roam_node        │  RoamDecision.decide() → direction + speed + action
└────────┬────────┘
       │  /one360s/roam_target
       ▼
┌─────────────────┐
│ mavlink_node     │  MAVLink → OBSTACLE_DISTANCE + SET_POSITION_TARGET_LOCAL_NED
└────────┬────────┘
       │  serial (/dev/ttyTHS1)
       ▼
┌─────────────────┐
│ Flight Controller│  PX4 GUIDED mode
└─────────────────┘

watchdog_node: OK → DEGRADED → FAIL → emergency LAND
```

## Algorithm Stack

All core algorithms are **pure Python + NumPy, zero ROS dependency** (in `one360s/` package).

### 1. Perception Pipeline (`pipeline.py`)

7-stage processing, raw point cloud → 4-layer sector distance array:

| Stage | Description |
|-------|-------------|
| 1. Confidence filter | Drop low-confidence points (rain/fog/dust) via Mid-360 tag field |
| 2. Attitude compensation | Pitch/roll rotation matrix correction |
| 3. Spatial filter | Range clip + height clip + CropBox self-masking |
| 4. VoxelGrid downsample | 5cm leaf, one point per voxel |
| 5. RadiusOutlier denoise | Radius search, min 2 neighbors |
| 6. Sector binning | 72 sectors × 5°, 20th percentile per sector, adaptive min_points |
| 7. Temporal + inflation | Asymmetric smoothing + angle inflation + obstacle confidence EMA |

**4 z-bins:**
- **L0** [-3.0, -0.4m] — ground layer (descent / duck-under)
- **L1** [-0.4, 0.5m] — flight layer (primary horizontal planning)
- **L2** [0.5, 2.0m] — above-head space (climb clearance check)
- **up** [2.0, 6.0m] — canopy top (climb safety check)

### 2. Pinch Point Detection (`pinch_point.py`)

> How far can the drone fly along a bearing before its projected cone hits an obstacle?

- Projects drone geometry cone along a bearing
- Steps from 1m to 40m at 1m increments
- Returns the first distance where the cone intersects an obstacle sector
- `scan_pinch_map()` evaluates 37 directions (±90°, 5° step) in ~0.5ms

### 3. Pseudo-Compress (`pseudo_compress.py`)

> How wide is the passage at the pinch point?

- Progressively inflates obstacles (1.5×, 2.0×, 2.5×, 3.0×, 4.0×, 5.0×)
- Checks if the inflated cone still fits through at the pinch distance
- Higher margin = wider channel = safer direction
- Extra spot-checks at pinch/2 and pinch×0.75 for e≥3.0 to catch mid-channel narrowing

### 4. Direction Selection (`roam_decision.py`)

Fused scoring for each candidate direction:

```
score = depth_score × (0.5 + 0.5 × width_score) × goal_bias × dead_end_penalty
```

- **depth_score** = pinch / 30m (capped 1.0)
- **width_score** = expand_margin / 3.0 (capped 1.0)
- **goal_bias** = 1.0 + goal_weight × cos(Δbearing)
- **dead_end_penalty** = 0.3× if marked in topo memory

Features:
- **Hysteresis**: keep current direction unless new one is ≥1m better
- **Temporal confirmation**: requires 3 consecutive low-pinch frames before declaring "all blocked"
- **Declining trend detection**: pinch dropped to 30% within 3s + <8m → early backtrack

### 5. Blocked-Handling Priority

When all forward directions are blocked (`pinch < 1.5m`):

1. **Slope detection** → slope-follow climb along the incline
2. **Climb evaluation** → try climbing 1m / 2m / 3m (checks L2 + up layer clearance)
3. **Descend evaluation** → duck under via L0 layer (requires ≥1.5m ground clearance)
4. **Topological backtrack** → return to last fork's unexplored exit
5. **Hover alert** → manual intervention required

### 6. Topological Memory (`topo_memory.py`)

<2 KB memory footprint. Records key navigation events:

| Pattern | Trigger | Purpose |
|---------|---------|---------|
| `fork` | ≥2 exits with pinch >15m, separated by ≥30° | Record unexplored exits for backtracking |
| `dead_end` | All directions blocked | Mark sector to avoid on next pass |
| `narrowing` | Pinch halved within 3 seconds | Early backtrack signal |

- 60s periodic JSON persistence to `/tmp/one360s_topo.json`
- 30s dead-end TTL (scenes change, markings expire)
- Max 20 nodes (oldest evicted)
- Backtrack returns bearing of the unexplored exit + reverse bearing back to the fork

### 7. Climb & Slope (`climb_eval.py`)

- **`can_climb()`**: checks forward distance during climb time vs available horizontal space, L2/up layer clearance, up-laser priority, sparse-data safeguard (≥10 valid sectors required)
- **`detect_slope()`**: distinguishes sloped terrain (continuous pinch decrease across ≥5 sectors) from vertical walls (single-sector pinch drop)
- **`slope_follow_target()`**: computes target altitude to maintain ~2m above the slope surface

### 8. Speed from Physics

```
v = √(2 × max_decel × (pinch - 1.0)) × safety_factor
```

Brake-distance constrained. 1m safety margin. 0.5 safety factor for inertia + wind gust.

## Configuration

All parameters in `config/params.yaml`. To adapt to a different drone, **only change the `drone` section** — algorithm parameters (`effective_R`, `crop_box`, speeds) are auto-derived:

```yaml
drone:
  body_length_m: 0.65
  body_width_m:  0.80
  body_height_m: 0.55
  lidar_z_offset: 0.15
  safety_extra_m: 0.25
  max_speed_ms: 3.0
  max_decel_ms2: 3.0
  min_speed_ms: 0.5
```

Pre-arm validation checks all critical parameters at startup. Fails fast if anything is out of range.

## Mission Modes

| Mode | Behavior |
|------|----------|
| `roam` | Fixed-altitude wandering, no goal bias |
| `traverse` | Directional crossing with 0.3 goal weight |
| `inspect` | Inspection crossing with 0.6 goal weight |

Flight controller stays in GUIDED mode. Only the scoring strategy changes.

## Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/livox/lidar/pointcloud` | `PointCloud2` | sub | Mid-360 raw point cloud |
| `/mavros/imu/data` | `Imu` | sub | IMU for attitude compensation |
| `/one360s/distances` | `Float32MultiArray[288]` | pub | 4 layers × 72 sector distances (m) |
| `/one360s/roam_target` | `Float32MultiArray[5]` | pub | [bearing_deg, step_m, speed_ms, alt_m, action_code] |
| `/one360s/position` | `Float32MultiArray[5]` | pub | [heading, speed, alt, laser_alt, up_laser] |
| `/one360s/status` | `String` | pub | Processing stats (every 10 frames) |
| `/one360s/health` | `UInt8` | pub | 0=OK, 1=DEGRADED, 2=FAIL |

## Design Philosophy

- **Plan in sensor space** — no SLAM, no occupancy grid, no global coordinates
- **Conservative by default** — all thresholds err on the side of safety
- **Pure NumPy core** — algorithm library has zero ROS imports, fully testable offline
- **Thin ROS nodes** — nodes are ~30-line wrappers, all logic in the library
- **Pre-arm validation** — catch misconfiguration before takeoff, not in flight
- **Minimal compute** — designed for Jetson Nano, entire pipeline <15ms per frame

## License

MIT
