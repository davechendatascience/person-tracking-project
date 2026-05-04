# autoresearch — leader-following with Nav2 behavior trees

This is an experiment to have the LLM autonomously design and improve a leader-following robot policy. The follower must keep a moving leader within a safe distance using simulated **RGB camera + 2D LiDAR** sensing, both with limited range and occluded by obstacles. Control runs on a **Nav2 behavior tree**.

The leader moves around the environment on its own (random or scripted patrol). The follower must:
- detect the leader when in line of sight,
- choose the next pose (or recovery action) when detection is lost,
- avoid obstacles,
- never collide,
- minimize time the leader is out of view.

Inspired by *Follow Everything* ([arXiv:2504.19399](https://arxiv.org/html/2504.19399v1), `resources/follow_everything.md`) but **reimplemented on the Nav2 BT stack** — the agent's job is to design the BT, not to write a new navigation solver.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may3`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**. The relevant files for this task:
   - `README.md` — repository context.
   - `resources/follow_everything.md` — paper reference (read the abstract + method sections via WebFetch).
   - `sim/world.py` — fixed 2D physics + sensor simulation. **Do not modify.**
   - `sim/sensors.py` — fixed RGB-cone + LiDAR-ray sensor models. **Do not modify.**
   - `sim/leader.py` — fixed autonomous leader (random patrol with Nav2). **Do not modify.**
   - `eval/run_episode.py` — fixed evaluation harness; runs N episodes, computes metrics, writes results. **Do not modify.**
   - `follower_pkg/` — the ROS 2 package you modify. Behavior tree XML, BT plugin nodes, perception filtering, recovery actions.
4. **Build the Docker image**: `docker compose build` (uses ROS 2 Humble + Nav2 + simulator base). On Windows hosts, run inside WSL: `wsl -- bash -lc "cd <repo> && docker compose build sim"`.
5. **Smoke-test**: `docker compose run --rm sim eval/run_episode.py --episodes 1 --map empty` and confirm the harness completes without error.
6. **Initialize results.tsv**: create `results.tsv` with the header row. The baseline behavior tree will be recorded after the first run.
7. **Confirm and go**: tell the user the setup looks good.

Once you get confirmation, kick off the experimentation loop.

## The simulator

A lightweight 2D **continuous-space** physics simulation with ROS 2 message interfaces (so the real Nav2 stack runs unmodified on top).

- **World**: continuous 2D space. Obstacles are axis-aligned bounding boxes derived from ASCII map files (each `#` cell becomes a 0.5×0.5 m box, then merged greedily). Robot positions are continuous floats — *not* snapped to cells.
- **Collision**: analytical circle-vs-AABB check. A robot cannot move through an obstacle; if a commanded motion would penetrate, the position update is rejected (rotation in place is still allowed) and a collision is counted.
- **Robots**: leader + follower, both differential-drive disks (radius 0.25 m), max speed 1.5 m/s, max ang vel 1.5 rad/s.
- **Tick rate**: 20 Hz.
- **Visual mode**: matplotlib renderer activated via `SIM_RENDER=1` or `--render`. See *Inspecting visually* below.
- **Topics published per robot**:
  - `/<robot>/odom` — `nav_msgs/Odometry`, perfect odometry (no noise — sim limitation).
  - `/<robot>/scan` — `sensor_msgs/LaserScan`, 360° coverage, **8 m max range** (5° angular resolution).
  - `/<robot>/camera/detections` — `vision_msgs/Detection2DArray`, leader detection if visible.
  - `/<robot>/cmd_vel` — `geometry_msgs/Twist`, subscribed by sim.
- **Maps**: `empty`, `corridor`, `cluttered`, `forest` (provided in `sim/maps/`).

### Sensor models (the hard part of the task)

**RGB camera (simulated detector)** — `sim/sensors.py::CameraDetector`:
- Horizontal FOV: **90°**, pointing forward.
- Maximum range: **6 m**.
- Detection requires **clear line of sight** — analytical ray-vs-AABB intersection test against all obstacles between the camera origin and the leader.
- This is the **simulated SAM2** layer. We do NOT run any actual segmentation or detection model (no GPU required). The detector is an oracle that returns the leader's relative pose IFF the geometric conditions are met.
- When detected, publishes a `vision_msgs/Detection2DArray` containing the leader's `(x, y)` in the follower's body frame.
- When NOT detected (occluded, out of FOV, beyond range), publishes an empty array.

**2D LiDAR** — `sim/sensors.py::LidarSensor`:
- 360°, 72 rays (5° resolution), max range **8 m**.
- Each ray uses analytical ray-vs-AABB and ray-vs-circle (for the leader) intersection — continuous, no grid stepping.
- LaserScan does NOT label which return is the leader. The follower must infer.

**Result of these sensor models**: when an obstacle is between the follower and the leader, the follower **has no detection of the leader's position**. The agent's behavior tree must handle this with recovery / search behaviors.

## The task

Design a Nav2 behavior tree (XML) and accompanying plugin nodes that maximize the follow-success metric defined below. The BT should at minimum:

1. **Approach the leader** when detected (use `NavigateToPose` with the detected pose as a dynamic goal, OR a custom controller node).
2. **Maintain safe distance** (~2 m) — don't run into the leader when stopped.
3. **Recover when leader is lost** — search behavior (rotate in place, return to last known pose, explore) until detection re-acquires.
4. **Avoid obstacles** — Nav2's local costmap + DWB controller handles this if you wire it correctly.

You are free to add custom BT plugins for:
- Perception filtering (Kalman filter, particle filter, simple low-pass on detections)
- Goal selection logic (last known pose, predicted pose, exploration goal)
- Recovery strategies (spin, retrace, expand search radius)
- State memory (how long to remember a lost leader before giving up)

### What you CAN do

- Modify any file under `follower_pkg/`. This includes:
  - `bt_xml/follow.xml` — the behavior tree definition
  - `src/*.cpp` or `python/*.py` — custom BT plugin nodes
  - `config/*.yaml` — Nav2 parameter files (costmap, controller, planner) for the follower
  - `launch/follower.launch.py` — launch graph
- Add custom ROS nodes for perception, filtering, or planning helpers.
- Use any package available in the Docker image (Nav2, BT.CPP, OpenCV, NumPy, SciPy).

### What you CANNOT do

- Modify anything under `sim/`. The world, sensors, and leader behavior are fixed.
- Modify `eval/run_episode.py`. The evaluation metric is the ground truth.
- Cheat by reading the leader's true pose from any topic except `/<robot>/camera/detections`. The sim publishes a `/leader/pose_ground_truth` topic for the evaluator only — your follower must NOT subscribe to it. The evaluator will fail your run if it detects a subscription from the follower namespace.
- Modify the leader's controller or its random-patrol seed.
- Skip episodes or pre-compute the leader's path.

### The goal

Maximize **follow_success_rate** (defined below). Secondary goals: low collision rate, low average distance, low loss-time ratio.

## Metrics

Per episode (300 seconds wall-clock at 20 Hz = 6000 ticks):

- **follow_success_rate**: fraction of ticks where the leader is within 5 m AND visible to the camera.
- **loss_time_ratio**: fraction of ticks where the leader is NOT within camera FOV+range or is occluded.
- **collision_rate**: number of collisions per 100 m of follower travel.
- **avg_distance**: mean Euclidean distance between follower and leader during the episode.
- **path_efficiency**: `follower_path_length / leader_path_length` (lower is more efficient; 1.0 means perfectly tracking the leader's path).

Aggregated across N episodes (default N=10, seeds 0..9, mix of maps):
- **mean_success** = mean(follow_success_rate)  ← primary metric
- **collision_count** = total collisions across all episodes
- A run is a **crash** if any episode aborts (controller crash, BT exception, simulator hang).

## Output format

`eval/run_episode.py` prints a summary block:

```
---
mean_success:        0.781
mean_loss_ratio:     0.143
collision_count:     2
mean_distance:       2.34
mean_path_eff:       1.18
total_episodes:      10
crashes:             0
total_seconds:       312.4
strategy:            bt_v3_kalman_search
```

Extract the key metric:

```
grep "^mean_success:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated, **untracked by git**, leave it out of commits).

Header (5 columns):

```
commit	mean_success	collision_count	status	description
```

Example:

```
commit	mean_success	collision_count	status	description
a1b2c3d	0.000000	0	keep	baseline (BT does nothing — robot stays still)
b2c3d4e	0.412000	7	keep	naive: NavigateToPose to last detection, no recovery
c3d4e5f	0.658000	2	keep	+ spin-search recovery when leader lost > 2s
d4e5f6g	0.000000	0	crash	BT plugin segfault
```

## The experiment loop

LOOP FOREVER:

1. Inspect git state.
2. Edit a file in `follower_pkg/` with an experimental change.
3. `git commit`.
4. Run: `docker compose run --rm sim eval/run_episode.py --episodes 10 --maps mixed --seed 0 > run.log 2>&1`. Episodes are seeded so results are reproducible. **Do not pipe** — you don't want full per-tick logs flooding context. Just write to `run.log`.
5. Read results: `grep "^mean_success:\|^collision_count:\|^crashes:" run.log`. If empty, the run crashed — `tail -n 80 run.log` for traceback.
6. Record in `results.tsv`.
7. If `mean_success` improved AND `collision_count` did not regress meaningfully (≤ +2 over baseline), **keep** the commit.
8. Otherwise, `git reset` back.

**Timeout**: each evaluation run should complete in ≤ 15 minutes (10 episodes × ~90s plus startup). Kill and revert if exceeded.

**Crashes**: BT plugin crashes are usually fixable (compile errors, missing service registration, message-type mismatch). Read the traceback. If the IDEA is broken, log "crash" and move on.

**Common pitfalls — actively guard against:**

- **Subscribing to `/leader/pose_ground_truth`** — that's cheating. The evaluator detects it and zeros your score.
- **Costmap obstacle inflation too small** — robot grazes obstacles, collision count spikes.
- **No timeout in BT search behavior** — robot spins forever when leader is permanently lost.
- **Goal pose stale** — using a 30s-old detection as goal causes confused behavior. Add a freshness gate.
- **BT tick rate mismatch** — Nav2 default BT tick is 10 Hz; your perception might run at 20 Hz. Decide which one drives.
- **Reactive vs deliberative loop** — pure reactive (always re-plan to current detection) wastes CPU and is jittery; pure deliberative (long-running NavigateToPose) is slow to react. The BT should blend both.

**NEVER STOP**: once the loop has begun, do NOT ask the human if you should continue. The human expects you to keep iterating indefinitely until manually interrupted. If you run out of obvious changes, look at where episodes are failing (read per-episode logs), check whether failures cluster on specific maps, design a targeted fix.

## Idea space (non-exhaustive)

To get unstuck, consider:

- **Perception persistence**: keep last N detections, decay confidence over time, refuse to update goal if too old.
- **Predict where leader went**: small Kalman filter on detected (x, y, vx, vy); when occluded, keep extrapolating for ~1 s before recovery kicks in.
- **Search patterns**: spin in place / spiral outward / return to last known pose / explore the largest unobserved cell of the leader's habitable area.
- **Costmap tuning**: inflation radius, obstacle persistence, footprint padding — Nav2 default is conservative; tune for narrow corridors.
- **Controller choice**: DWB vs MPPI vs RPP — try each on the same BT.
- **Goal-pose offset**: don't navigate exactly to the leader's pose; navigate to a point 2 m behind on the leader's velocity vector (anticipate motion).
- **Memory / reactive blend**: BT condition `LeaderRecentlySeen(t<1s)` → reactive ApproachLeader; else `LeaderLost` → deliberative SearchPlan.
- **Sensor fusion**: use LiDAR returns to corroborate camera detections (if camera says leader at 4 m, LiDAR should show a ~0.5 m return at that bearing).
- **Re-detection bias**: when re-acquiring after a loss, accept detections only if they're geometrically consistent with the predicted location.

## Safety notes

- **Run on a dedicated branch.** The loop performs `git reset` when discarding experiments.
- **Docker isolation.** The simulator runs in a container. The agent should not attempt to connect to host network or mount host filesystems other than the workspace.
- **Resource limits.** The Docker compose file caps memory and CPU; the loop is allowed to be slow but not to OOM the host.
- **No real robot.** This is simulation only. The sim's perfect odometry, simple sensor model, and 2D world are not a substitute for hardware testing — a high-`mean_success` here does not guarantee real-world performance.

## Inspecting visually

The simulator has an optional matplotlib renderer for human inspection — useful when debugging a new behavior tree or watching how the leader navigates.

**Headless (default, fast, used by `eval/run_episode.py`):**
```
docker compose run --rm sim eval/run_episode.py --episodes 10 --maps mixed
```

**Visual mode** (separate compose service `sim-viz` forwards X11 to the host):
```
xhost +local:root            # one-time on the host (Linux)
docker compose run --rm sim-viz python -m sim.world &
docker compose run --rm sim-viz python -m sim.leader &
docker compose run --rm sim-viz ros2 launch follower_pkg/launch/follower.launch.py
```
Or with a single-episode evaluation harness:
```
docker compose run --rm sim-viz eval/run_episode.py --episodes 1 --maps cluttered --render
```

The render shows: obstacles (gray boxes), leader (blue disc + heading arrow), follower (green disc + heading arrow + yellow camera FOV cone), the LiDAR returns (orange dots), and a green/red line indicating whether the leader is currently visible to the follower.

On macOS or Windows hosts, X11 forwarding requires XQuartz or VcXsrv respectively; consult the project README.

## Reference resource

- `resources/follow_everything.md` → arXiv:2504.19399, "Follow Everything". Use as inspiration for failure modes, recovery state machine, and metrics design — but do NOT just copy their architecture. They use a custom planner; we use Nav2.
