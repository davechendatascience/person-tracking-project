# follow_everything

A simulated leader-follower task: a follower robot must keep a moving leader in
view using only on-board sensors (LiDAR + camera detection + own odometry),
with no access to ground-truth maps or leader poses.

## What it does

A 2D world (configurable map) spawns a leader that patrols autonomously and a
follower that must track it. The follower uses a py_trees behavior tree (the
same family Nav2 uses) to switch between *Following*, *Chasing*, and several
*Recovery* behaviors when the leader leaves the camera's FOV.

Maps: `empty`, `corridor`, `cluttered`, `forest`, plus configurable Brownian
pedestrians.

## Architecture

```
┌─────────────────┐    /leader/*      ┌─────────────────┐
│  sim/world.py   │  (ground truth,   │  sim/leader.py  │
│   - physics     │  follower NEVER   │  (autonomous    │
│   - obstacles   │   subscribes)     │   patrol via A*)│
│   - pedestrians │                   └─────────────────┘
└────────┬────────┘
         │ /follower/odom              own pose
         │ /follower/scan              LiDAR
         │ /follower/camera/detections leader (body frame, only when visible)
         │ /follower/camera/pedestrians peds   (body frame, only when visible)
         ▼
┌──────────────────────────────────────────────┐
│  follower_pkg/python/                        │
│      follow_everything_follower.py           │
│                                              │
│  LearnedMap (log-odds occupancy from LiDAR)  │
│      └─ binary_dilation -> plan_grid         │
│      └─ EDT clearance -> cost_grid           │
│                                              │
│  Behavior tree (Selector, memory=False):     │
│    Retreating  | leader too close            │
│    Following   | leader visible, mid-range   │
│    Chasing     | leader visible, far         │
│    PlannedRec  | A* to last_seen             │
│    BackupRec   | direct goto last_seen       │
│    SpiralExpand| ring-search around predicted│
└────────────────┬─────────────────────────────┘
                 │ /follower/cmd_vel
                 ▼
              world.py
```

## Anti-cheat rules

- **No** subscription to `/leader/*` (only camera detections of the leader,
  in body frame).
- **No** loading of static maps. Obstacles are learned online from LiDAR via a
  log-odds occupancy grid.
- World *bounds* (e.g. 15 × 15 m) are deployment config, not cheating.
- `ROBOT_RADIUS` and other constants imported from `sim/` are config values,
  not map data.

## Running

```bash
# Build image (Linux/WSL with Docker)
docker compose build sim

# 8 episodes across mixed maps, headless
docker compose run --rm sim bash -c "
  source /opt/ros/humble/setup.bash &&
  python3 -m eval.run_episode --episodes 8 --maps mixed --episode-secs 60
"

# Render with X11 (WSLg or Linux desktop)
docker compose run --rm sim-viz bash -c "
  source /opt/ros/humble/setup.bash &&
  python3 -m eval.run_episode --episodes 1 --maps cluttered \
      --episode-secs 100 --render
"

# Add pedestrians
docker compose run --rm -e NUM_PEDESTRIANS=10 sim bash -c "
  source /opt/ros/humble/setup.bash &&
  python3 -m eval.run_episode --episodes 1 --maps cluttered --episode-secs 100
"
```

## Metrics

`eval/run_episode.py` prints and writes `results/last_run.json`:

- `mean_success` — fraction of ticks where leader is within 5 m **and** visible
- `mean_loss_ratio` — fraction of ticks leader not visible
- `collision_count` — follower-only collisions
- `mean_path_eff` — `follower_path_length / leader_path_length`
- `follower_stuck_ratio`, `leader_stuck_ratio` — pose diameter < 5 cm over 1 s

Current v34 baseline (8 eps, mixed maps, 60 s each):
`success 0.818, collisions 0, path_eff 0.93, follower_stuck 0.005`.

## Visualization

`sim/viz.py` opens two side-by-side panels:
- **Left**: ground truth (obstacles, leader, follower, pedestrians, FOV cone,
  LiDAR rays, planned A* path).
- **Right**: the follower's *learned* occupancy map. The two should agree once
  the follower has explored — divergence between them is a debugging signal.

Snapshot frames can be dumped via `SIM_SNAPSHOT_DIR=/path` and stitched with
`scripts/stitch_video.sh`.

## Layout

```
sim/             physics, leader, sensors, planner, viz
  world.py       physics + ROS pubs of /follower/odom and /scan
  leader.py      autonomous leader (random goals, A* planning)
  sensors.py     LiDAR + camera detector (FOV / occlusion gating)
  planner.py     A* + path smoothing
  geometry.py    AABB / circle / segment helpers
  maps/          *.txt grid maps
follower_pkg/python/
  follow_everything_follower.py   the follower (LearnedMap + BT)
eval/
  run_episode.py harness, metrics, subprocess spawn
resources/       design docs (incl. Chinese walkthrough of the BT)
results/         per-episode logs, JSON metrics, snapshots, MP4
```

## Key design notes

- **py_trees Selector with `memory=False`**: every tick re-evaluates the whole
  tree. No state machine, no transitions — priority order does the work.
- **Log-odds occupancy** with symmetric `±0.4` HIT/MISS and `±2.0` clamp.
  Pedestrians decay in ~0.25 s; static walls saturate.
- **Leader-cone mask** in LiDAR ingestion: the camera-detected leader's
  angular cone is excluded from log-odds updates so the leader doesn't ghost
  as a permanent obstacle.
- **Y-flip fix (v34)**: `merged_aabbs_from_grid` uses image-coords; `LearnedMap`
  uses `row j → y = j*MAP_RES`. Without `np.flipud`, every LiDAR-observed wall
  was mirrored across the world Y midline → A\* routed through phantom-free
  space. Critical to the planner ever working correctly.

See `resources/follow_everything_bt_zh.md` for a deeper walkthrough (in
Traditional Chinese) of the behavior tree and the planning grid pipeline.
