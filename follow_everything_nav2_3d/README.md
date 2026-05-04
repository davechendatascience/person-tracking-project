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
- [x] **Phase 4a** — RGB-D camera + DAM4SAM tracker skeleton publishing `/follower/camera/detections_dam4sam` (no-op detector for now). *(this commit)*
- [ ] Phase 4b — wire real SAM2/DAM4SAM tracker (CUDA torch + sam2 + DAM4SAM), YOLO bootstrap, depth back-projection.
- [ ] Phase 5 — DAM4SAM becomes the primary detection source on `/follower/camera/detections`.
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

## Phase 1 — running it

One-time host setup (Linux, X11):

```bash
xhost +local:root
touch /tmp/.docker.xauth
```

Build and enter the container:

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
