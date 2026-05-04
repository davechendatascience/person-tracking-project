# follow_everything_nav2_3d

3D Gazebo port of [`follow_everything_nav2`](../follow_everything_nav2/), with [DAM4SAM](../DAM4SAM/) as the real perception backend (replacing the oracle camera detector). Built incrementally — each phase keeps a working demo.

The topic contract is frozen across phases (matches `follow_everything_nav2`):

| topic                            | type                          | direction        |
|----------------------------------|-------------------------------|------------------|
| `/follower/odom`                 | `nav_msgs/Odometry`           | sim → follower   |
| `/follower/scan`                 | `sensor_msgs/LaserScan`       | sim → follower   |
| `/follower/camera/detections`    | `vision_msgs/Detection2DArray`| sim → follower   |
| `/follower/cmd_vel`              | `geometry_msgs/Twist`         | follower → sim   |

## Phase status

- [x] Phase 1 — empty world, diff-drive follower, teleop.
- [x] Phase 2 — 360° lidar publishing `/follower/scan`.
- [x] Phase 3 — walking-human actor leader + oracle camera bridge publishing `/follower/camera/detections`.
- [x] Phase 4a — RGB-D camera + DAM4SAM tracker skeleton publishing `/follower/camera/detections_dam4sam` (no-op detector).
- [x] Phase 4b-i — CUDA torch + DAM4SAM/SAM2/YOLO Python deps in the image, GPU runtime + parent-repo mounts in compose.
- [x] Phase 4b-ii — real `SAM2Tracker` wired into `dam4sam_tracker.py`: YOLO bootstrap + frame-by-frame mask + depth back-projection → body-frame `(x, y)`.
- [x] **Phase 5** — DAM4SAM is primary on `/follower/camera/detections`; minimal P-controller `simple_follower.py` chases the leader end-to-end. *(this commit)*
- [ ] Phase 6 — odometry noise (EKF), latency, TF cleanup.

## Layout (mirrors [`follow_everything_nav2/`](../follow_everything_nav2/))

```
follow_everything_nav2_3d/
├── Dockerfile, docker-compose.yml
├── sim/                          # sim-side: world, URDF, oracle, sim launch
│   ├── urdf/follower.urdf.xacro
│   ├── worlds/empty.world
│   ├── python/oracle_camera.py
│   └── launch/empty_bringup.launch.py
└── follower_pkg/                 # follower-side: tracker, BT, follower launch
    ├── python/dam4sam_tracker.py
    ├── launch/follower.launch.py
    └── bt_xml/                   # (Phase 5+ behavior tree)
```

## Robot

Custom diff-drive disk mirroring the 2D sim ([`world.py`](../follow_everything_nav2/sim/world.py)):
- body radius 0.25 m
- max linear 1.5 m/s, max angular 1.5 rad/s
- nimble enough to chase the leader (which has identical kinematics)

URDF: [`sim/urdf/follower.urdf.xacro`](sim/urdf/follower.urdf.xacro).

## Stack

- ROS 2 Humble
- **Ignition Gazebo Fortress** (LTS, Tier 1 with Humble — used because `gazebo_ros` Classic-11 packages are not published for arm64/Jetson on Humble).
- `ros_gz_bridge` mirrors Gazebo Transport ↔ ROS topics.

## Display / X11 forwarding

Gazebo's GUI, RViz, and the DAM4SAM overlay all need an X server reachable
from inside the container. [`docker-compose.yml`](docker-compose.yml) mounts
`/tmp/.X11-unix` and exports `DISPLAY` + `XAUTHORITY` already; what differs
is how you make the host's X server (or your laptop's) accept connections.

### Local Linux host (sitting in front of the GB10)

```bash
xhost +local:root              # allow the container's root user to draw
touch /tmp/.docker.xauth       # XAUTHORITY mountpoint (compose binds it in)
```

That's it. `docker compose run --rm sim` then `ros2 launch sim/launch/empty_bringup.launch.py`
should pop a Gazebo window on the local display.

### SSH from a remote workstation (most common on GB10)

The GB10 is typically headless / accessed over SSH. Two viable paths:

**(a) X11 forwarding** — simplest, works for RViz and image streams,
sluggish for full Gazebo:

```bash
# from your workstation
ssh -X -C user@gb10           # -X enables forwarding, -C compresses
                              # (use -Y instead of -X if -X gives "X11 connection rejected")
```

On the GB10 (after sshing in):
```bash
echo "$DISPLAY"               # should be set to e.g. localhost:10.0
xauth list "$DISPLAY"         # should print one cookie line
xhost +local:root             # allow docker's root to use the X socket

# write the SSH cookie into the file the container will mount
touch /tmp/.docker.xauth
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

cd follow_everything_nav2_3d
docker compose run --rm sim
```

Inside the container, `xeyes` (if installed) or `glxgears` is a quick way
to confirm forwarding works before launching Gazebo.

**(b) VNC / NoMachine / RDP server on the GB10** — better for Gazebo's
3D rendering. Set up a VNC server outside this repo's scope; once you
have a desktop session, fall back to the **Local Linux host** flow above
(`xhost +local:root` from inside the VNC desktop's terminal).

### Verifying it works

After `docker compose run --rm sim`, before launching Gazebo:

```bash
# inside the container
echo "$DISPLAY"               # should print something
ls /tmp/.X11-unix             # should list X0 (or X<N>)
glxinfo -B 2>/dev/null | head -5 || echo "glxinfo not installed; that's fine"
```

If `DISPLAY` is empty or `/tmp/.X11-unix` is empty, fix that on the host
side first — the rest of the launch will fail with "failed to create
drawable" / "cannot connect to X" otherwise.

### Common failures

| symptom                                            | fix                                                                |
|----------------------------------------------------|--------------------------------------------------------------------|
| `Authorization required, but no authorization protocol specified` | Re-run `xhost +local:root` from a shell on the same display.      |
| `cannot open display:`                             | `DISPLAY` not exported or X11 socket not mounted. Re-check compose env + the bind on `/tmp/.X11-unix`. |
| Gazebo opens but window is black / "failed to create drawable" | OpenGL context unavailable in the container. With X11 forwarding, set `LIBGL_ALWAYS_INDIRECT=1` (Gazebo's GUI may still struggle — switch to VNC/local). |
| RViz / overlay works, Gazebo doesn't               | Gazebo wants direct GL; X11-forwarded GL is too weak. Use VNC.     |

## Phase 1 — running it

Build and enter the container (see *Display / X11 forwarding* above for the
one-time host setup):

```bash
docker compose build
docker compose run --rm sim
```

Inside the container:

```bash
# terminal 1 — Fortress + spawn the follower + ros_gz bridge
ros2 launch sim/launch/empty_bringup.launch.py

# terminal 2 — open another shell into the running container
docker exec -it follow_everything_nav2_3d bash
# then teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/follower/cmd_vel
```

Verify:

```bash
ros2 topic list | grep follower
ros2 topic hz /follower/odom
ros2 topic echo /follower/odom --once
```

## Phase 1 acceptance

- Ignition Gazebo opens with an empty ground plane and the green-disk follower at origin.
- `/follower/odom` publishes on the ROS side via `ros_gz_bridge` at ≥ 30 Hz.
- Teleop on `/follower/cmd_vel` moves the robot in Gazebo.
- `ros2 run tf2_tools view_frames` shows `follower/odom → base_footprint → base_link`.

## Phase 2 — 360° lidar

Adds a `gpu_lidar` sensor mounted on top of the follower body. Specs match
[`sim/sensors.py::LidarSensor`](../follow_everything_nav2/sim/sensors.py):

- 360° horizontal FOV
- 72 rays (5° resolution)
- 8 m max range
- 20 Hz update rate

Bridged onto ROS as `/follower/scan` (`sensor_msgs/LaserScan`).

### Phase 2 acceptance

After re-launching `empty_bringup.launch.py`:

```bash
ros2 topic hz   /follower/scan          # ~20 Hz
ros2 topic echo /follower/scan --once   # 72 ranges, angle_min ≈ -π, angle_max ≈ π
```

In RViz2 (`rviz2 --ros-args -p use_sim_time:=true`), set fixed frame to
`follower/lidar_link` and add a `LaserScan` display on `/follower/scan`. Empty
world means all returns are at the lidar's max range (`inf` / 8.0). Drop a
quick test obstacle in Gazebo and confirm the scan shows it.

## Phase 3 — walking-human leader + oracle camera bridge

The leader is now an **animated humanoid actor** (Mingfei Walking actor on
Fuel) walking a scripted rectangular patrol around the origin at ~0.7 m/s.
On first launch the actor mesh is fetched from Fuel and cached at
`/root/.ignition/fuel/`; subsequent launches are offline. The Dockerfile
also pre-caches the model at build time when network is available.

The actor is kinematic (no physics collision) — fine for an empty world. We
revisit when obstacles arrive.

### Oracle camera bridge

[`sim/python/oracle_camera.py`](sim/python/oracle_camera.py)
mirrors the 2D sim's [`CameraDetector`](../follow_everything_nav2/sim/sensors.py):
90° forward FOV, 6 m max range, body-frame `(x, y)`, `class_id="leader"`.
Ground-truth poses come from Gazebo's SceneBroadcaster, bridged onto a
**dedicated** `/gz_pose_truth` topic (kept off `/tf` to avoid clashing with
the diff-drive's own TF tree).

This is the safety net commit: the existing follower from
[`follow_everything_nav2`](../follow_everything_nav2/follower_pkg/python/follow_everything_follower.py)
should run **unmodified** on top, since the topic contract is identical.

### Phase 3 acceptance

```bash
# in container, after launch
ros2 topic hz   /follower/camera/detections        # ~20 Hz
ros2 topic echo /follower/camera/detections --once # detections[] when leader visible
ros2 topic echo /gz_pose_truth --once              # transforms[] include child_frame_id "follower" + "leader"
```

Drive the follower with teleop; when you point it at the actor and it's
within 6 m, you should see a `Detection2D` with `results[0].hypothesis.class_id="leader"`
and the body-frame `(x, y)` of the actor. Look away or back up >6 m and the
detections array empties.

To verify the full follower stack, copy
`../follow_everything_nav2/follower_pkg/python/follow_everything_follower.py`
into a second container shell and run it — it should chase the actor.

## Phase 4a — RGB-D camera + DAM4SAM tracker skeleton

Adds a forward-facing RGB-D camera to the follower URDF (90° H-FOV to match
the oracle, 640×480 @ 20 Hz, 0.1–10 m clip). Bridged onto ROS as:

| topic                              | type                       |
|------------------------------------|----------------------------|
| `/follower/camera/image`           | `sensor_msgs/Image`        |
| `/follower/camera/depth_image`     | `sensor_msgs/Image`        |
| `/follower/camera/camera_info`     | `sensor_msgs/CameraInfo`   |
| `/follower/camera/points`          | `sensor_msgs/PointCloud2`  |

[`follower_pkg/python/dam4sam_tracker.py`](follower_pkg/python/dam4sam_tracker.py)
subscribes to image / depth / camera_info and publishes a `Detection2DArray`
on **`/follower/camera/detections_dam4sam`** at 20 Hz. The detector is a
**no-op** for Phase 4a — it always emits an empty array. Phase 4b drops in
the real [`SAM2Tracker`](../follow_everything/perception/sam2_tracker.py)
and depth back-projection.

Running this in shadow mode means the oracle (`/follower/camera/detections`)
and the future DAM4SAM (`/follower/camera/detections_dam4sam`) coexist on
distinct topics, so we can compare them before cutting over in Phase 5.

### Phase 4a acceptance

```bash
# terminal 1 — sim + oracle
ros2 launch sim/launch/empty_bringup.launch.py

# terminal 2 — DAM4SAM tracker (skeleton)
docker exec -it follow_everything_nav2_3d bash
ros2 launch follower_pkg/launch/follower.launch.py

# terminal 3 — verify
ros2 topic hz   /follower/camera/image                # ~20 Hz
ros2 topic hz   /follower/camera/depth_image          # ~20 Hz
ros2 topic echo /follower/camera/camera_info --once   # K matrix populated
ros2 topic hz   /follower/camera/detections_dam4sam   # ~20 Hz (empty arrays)
```

In RViz2 with fixed frame `follower/camera_optical_frame`, add an `Image`
display on `/follower/camera/image` — you should see the actor walking
through the camera FOV when within 90°/10 m of the follower.

## Phase 4b-i — ML stack in Docker (CUDA + DAM4SAM/SAM2/YOLO deps)

Host: NVIDIA GB10 Grace Blackwell (sbsa aarch64), CUDA 13.0 driver. CUDA 13
is backward-compatible with cu12.x runtime, so the image installs the cu126
PyTorch wheels (which have aarch64 binaries on `download.pytorch.org`). If
those fail to install, the Dockerfile falls back to CPU-only torch.

Compose now mounts the parent repo's ML payloads read-only at the paths
`dam4sam_tracker.py` will look for in Phase 4b-ii:

| host path                              | container path                  |
|----------------------------------------|---------------------------------|
| `../DAM4SAM/`                          | `/opt/DAM4SAM`                  |
| `../follow_everything/`                | `/opt/follow_everything`        |
| `../sam2.1_hiera_large.pt`             | `/opt/sam2.1_hiera_large.pt`    |
| `../yolo11m.pt`                        | `/opt/yolo11m.pt`               |

(Mounted under `/opt/` rather than `/ws/` so Docker's bind-mount auto-creation
doesn't drop empty root-owned stubs into the host project directory.)

`PYTHONPATH` is set in the image so `from follow_everything.perception.sam2_tracker import SAM2Tracker` and `import sam2` (via DAM4SAM's modified package) both resolve.

### Phase 4b-i smoke test

```bash
docker compose build
docker compose run --rm sim
# inside container:
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python3 -c "from ultralytics import YOLO; print(YOLO('/ws/yolo11m.pt').names[0])"   # 'person'
python3 -c "import sys; sys.path.insert(0, '/ws/DAM4SAM'); import sam2; print('sam2 ok')"
ls -la /ws/sam2.1_hiera_large.pt /ws/yolo11m.pt /ws/DAM4SAM /ws/follow_everything
```

Acceptance:
- `torch.cuda.is_available()` returns **True** and prints the GB10 device name.
- All three import lines succeed without traceback.
- The four mounted paths exist with non-zero sizes.

If `cuda.is_available()` is False, the cu126 wheel didn't install (check the
build log) and we'll need a different strategy — most likely switch the base
to NGC `nvcr.io/nvidia/pytorch` and re-layer Humble + Fortress on top.

## Phase 4b-ii — real DAM4SAM tracker on `/follower/camera/detections_dam4sam`

[`follower_pkg/python/dam4sam_tracker.py`](follower_pkg/python/dam4sam_tracker.py)
wraps `follow_everything.perception.sam2_tracker.SAM2Tracker` end-to-end with
an **always-latest, drop-stale** streaming loader. SAM2 sees a contiguous
index sequence (required for `propagate_in_video`'s temporal memory) but
each slot is filled with whatever the freshest camera frame is at the
moment SAM2 asks for it — frames that arrived while SAM2 was busy on the
previous one are simply dropped.

Why this matters: sam2.1_hiera_large runs at ~5–8 Hz on the GB10 while the
camera publishes at 20 Hz. Naïvely writing every camera frame into the
loader backs up. Independence is preserved:

- `/follower/scan` (lidar) and `/follower/camera/detections` (oracle) keep
  their **20 Hz** rate.
- `/follower/camera/detections_dam4sam` ticks at **SAM2's actual rate** —
  the topic is published exactly when the tracker yields a result. The
  follower BT sees both rates and gates on detection freshness, the same
  way [`follow_everything_nav2`](../follow_everything_nav2/program.md) does.

Pipeline:

1. ROS RGB callback snapshots `(rgb, depth, K, stamp)` atomically as
   "latest". 20 Hz, fast.
2. **YOLO bootstrap** — on the first incoming frame containing a `person`
   with confidence ≥ 0.5, we capture that frame as **SAM2 frame 0** (so
   the YOLO bbox aligns with what SAM2 sees), then signal the worker.
3. Worker thread runs `tracker.track_sequence(LiveLoader, init_bbox)`.
   `LiveLoader` is a thin subclass of `StreamingFrameLoader` that, on
   `__getitem__(idx)`, asks the node to write the *current* latest RGB to
   that path before deferring to the parent loader's normal load.
4. Each yielded `(frame_idx, TrackResult)` is pushed onto a small queue;
   main-thread timer drains it (drops anything older than the most
   recent), projects the mask centroid `(u, v)` → camera optical frame
   via K → `follower/base_link` via tf2, and publishes
   `Detection2DArray` with body-frame `(x, y)` and `class_id="leader"`.
   Score = SAM2 mask confidence.

Shadow mode is preserved — the oracle still publishes on
`/follower/camera/detections`, DAM4SAM on `/follower/camera/detections_dam4sam`.
Phase 5 swaps the names.

### Phase 4b-ii acceptance

```bash
# terminal 1 — sim + oracle
ros2 launch sim/launch/empty_bringup.launch.py

# terminal 2 — DAM4SAM tracker
docker exec -it follow_everything_nav2_3d bash
ros2 launch follower_pkg/launch/follower.launch.py
# expect log lines:
#   "DAM4SAM tracker live. stream dir: /tmp/dam4sam_stream_..."
#   "YOLO bootstrap: init bbox [...] (conf 0.XX)"
#   "SAM2 device=cuda, building predictor (slow first time)..."

# terminal 3 — verify rates
ros2 topic hz /follower/camera/detections          # oracle, expect ~20 Hz
ros2 topic hz /follower/camera/detections_dam4sam  # DAM4SAM, expect 5–8 Hz on GB10
ros2 topic echo /follower/camera/detections_dam4sam --once
```

Drive the follower toward the actor. Both topics should report a leader
detection with `(x, y)` within ~0.3 m of each other when the actor is
clearly in view — the rate gap is expected, the agreement is what we
care about. DAM4SAM's score is the SAM2 mask confidence proxy (~0.7–0.95).

Known limitations:
- First-frame init only. If YOLO never sees a person on startup (e.g. the
  follower is facing away from the actor at boot), the tracker stays idle
  and waits. Phase 5 adds re-init logic.
- Streaming temp dir grows during the run (one JPEG per frame). On a long
  session, prune it manually or restart.
- Depth back-projection assumes the SAM2-internal 1024×1024 resize; that's
  scaled back to the depth resolution in `_project_to_base_link`.

## Phase 5 — DAM4SAM primary + minimal P-controller follower

DAM4SAM is now the primary detection source on `/follower/camera/detections`
(the topic the follower reads). The oracle is still bridged but moved to
`/follower/camera/detections_oracle` so it can run alongside for comparison.

Switching sources via launch arg:

| invocation                                                                | who drives `/follower/camera/detections`        |
|---------------------------------------------------------------------------|--------------------------------------------------|
| `ros2 launch sim/launch/empty_bringup.launch.py` *(default)*              | DAM4SAM (oracle remapped to `_oracle`)           |
| `ros2 launch sim/launch/empty_bringup.launch.py detection_source:=oracle` | oracle (DAM4SAM stays on `_dam4sam`)             |

The follower side mirrors the same arg:

| invocation                                                                                    | result                                         |
|-----------------------------------------------------------------------------------------------|------------------------------------------------|
| `ros2 launch follower_pkg/launch/follower.launch.py` *(default)*                              | tracker output remapped onto `/detections`     |
| `ros2 launch follower_pkg/launch/follower.launch.py detection_source:=oracle`                 | tracker stays on `_dam4sam`; follower follows the oracle |

### `simple_follower.py`

[`follower_pkg/python/simple_follower.py`](follower_pkg/python/simple_follower.py)
is a deliberately minimal proportional controller:

- subscribes `/follower/camera/detections` (Detection2DArray, body-frame `(x, y)`)
- publishes `/follower/cmd_vel` (Twist) at 20 Hz
- if `class_id == "leader"` is visible:
  - angular: `KW × atan2(y, x)`, clipped to `±MAX_W`
  - linear: `KV × (range − TARGET_DIST)`, scaled by `cos(bearing)` to avoid driving sideways
- if no detection in `LOSS_TIMEOUT` (0.5 s) → command zero

No behavior tree, no obstacle handling, no recovery search. Empty world
with one always-visible actor → P-control alone reproduces the demo. The
richer BT-based follower from [`follow_everything_nav2`](../follow_everything_nav2/follower_pkg/python/follow_everything_follower.py)
is what we'll wire in once we add obstacles + lost-leader recovery.

### Phase 5 acceptance

```bash
# t1 — sim + (DAM4SAM-driving) detections + oracle on _oracle
ros2 launch sim/launch/empty_bringup.launch.py

# t2 — DAM4SAM tracker + simple_follower
docker exec -it follow_everything_nav2_3d-sim-run-XXXXXX bash
ros2 launch follower_pkg/launch/follower.launch.py

# t3 — verify topics + watch the follower chase
ros2 topic hz /follower/camera/detections          # DAM4SAM rate (~5 Hz)
ros2 topic hz /follower/camera/detections_oracle   # oracle (~20 Hz)
ros2 topic echo /follower/cmd_vel --once
```

Visually (RViz Image display on `/follower/camera/dam4sam_overlay`): mask
follows the actor; the green follower disk in Gazebo trails the actor
around its rectangular patrol, holding ~1.5 m stand-off.

To regress: `detection_source:=oracle` on both launches and check the
follower behaves identically (same topic contract, just different source).
