"""Record a fixed-duration episode with separated per-process logs.

Mirrors follow_everything_nav2's results/logs/ep_<ts>_*/{world,leader,follower}.log
layout so we can debug the 3D port the same way as the 2D one.

Usage (inside the container, from /ws):
    python3 eval/record_episode.py [duration_sec] [detection_source]

  duration_sec      defaults to 30
  detection_source  defaults to oracle  (also accepts dam4sam)

Produces:
    results/logs/ep_<unix_ts>_empty_0/
        world.log     # gz Fortress + ros_gz_bridges
        leader.log    # oracle_camera (knows the leader's true pose)
        follower.log  # dam4sam_tracker + follow_everything_follower (BT)

Bypasses `ros2 launch` so each conceptual subsystem gets its own log file.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SRC = sys.argv[2] if len(sys.argv) > 2 else "oracle"
MAP = sys.argv[3] if len(sys.argv) > 3 else "empty"
WS  = os.environ.get("WS_ROOT", "/ws")
TS  = int(time.time())
DIR = Path(WS) / "results" / "logs" / f"ep_{TS}_{MAP}_0"
SNAPS = DIR / "snapshots"
DIR.mkdir(parents=True, exist_ok=True)
SNAPS.mkdir(parents=True, exist_ok=True)

print(f"Recording {DUR}s map={MAP} detection_source={SRC} -> {DIR}")

# For non-empty maps, regenerate the world from the 2D map first.
if MAP == "empty":
    WORLD_PATH = f"{WS}/sim/worlds/empty.world"
else:
    import subprocess as _sp
    _b = _sp.run(
        ["python3", f"{WS}/sim/python/build_world.py", MAP],
        capture_output=True, text=True, check=True)
    WORLD_PATH = _b.stdout.strip().splitlines()[-1]
    print(f"Built world: {WORLD_PATH}")

# Per-log file handles, opened once and reused so multiple subprocesses
# can write into the same log (line-buffered).
log_handles: dict[str, "io.IOBase"] = {}
def log(name: str):
    if name not in log_handles:
        log_handles[name] = open(DIR / f"{name}.log", "w", buffering=1)
    return log_handles[name]


procs: list[tuple[str, subprocess.Popen]] = []
def spawn(log_name: str, cmd: list[str], env: dict | None = None) -> None:
    """Launch `cmd` in its own process group, output appended to <log_name>.log."""
    f = log(log_name)
    f.write(f"\n=== spawning: {' '.join(cmd)}\n")
    f.flush()
    p = subprocess.Popen(
        cmd,
        stdout=f, stderr=subprocess.STDOUT,
        env=env or os.environ.copy(),
        preexec_fn=os.setsid,
    )
    procs.append((log_name, p))


# ---------------------------------------------------------------------------
# 1) WORLD: gz Fortress + ros_gz_bridges. All sim infrastructure pooled.
# ---------------------------------------------------------------------------
spawn("world", [
    "ign", "gazebo", "-r", "-v", "3", WORLD_PATH,
])
time.sleep(3)  # gz needs a moment before bridges can connect

spawn("world", [
    "ros2", "run", "ros_gz_bridge", "parameter_bridge",
    "/follower/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
    # /follower/odom comes from world_odom_publisher.py (world frame), not gz.
    "/follower/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
    "/follower/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
    # gz publishes the raw lidar to _raw; lidar_leader_filter.py strips the
    # leader's own body hits and republishes the cleaned scan on
    # /follower/scan (which the BT subscribes to).
    "/follower/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
    "/follower/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
    "/follower/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
    "/follower/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
    "/leader/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
    "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
    "--ros-args",
    "-r", "/follower/scan:=/follower/scan_raw",
])
spawn("world", [
    "ros2", "run", "ros_gz_bridge", "parameter_bridge",
    "/world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
    "--ros-args", "-r",
    "/world/empty/dynamic_pose/info:=/gz_pose_truth",
])
spawn("world", [
    "ros2", "run", "ros_gz_bridge", "parameter_bridge",
    "/world/empty/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
    "--ros-args", "-r",
    "/world/empty/pose/info:=/gz_pose_truth",
])

# ---------------------------------------------------------------------------
# 2) LEADER: oracle_camera (its job is publishing the leader's body-frame
#    detection — closest analog to the 2D project's leader process).
# ---------------------------------------------------------------------------
oracle_cmd = ["python3", "-u", f"{WS}/sim/python/oracle_camera.py"]
if SRC == "dam4sam":
    oracle_cmd += [
        "--ros-args", "-r",
        "/follower/camera/detections:=/follower/camera/detections_oracle",
    ]
spawn("leader", oracle_cmd)

# Patrol controller for the leader model — drives /leader/cmd_vel.
leader_env = dict(os.environ)
leader_env["EP_MAP"] = MAP
spawn("leader", [
    "python3", "-u", f"{WS}/sim/python/leader_controller.py",
], env=leader_env)

# World-frame odom for the BT — replaces gz's local-frame odom.
# EP_MAP gates WORLD_ORIGIN_OFFSET inside world_odom_publisher.py so the
# bot's published pose lands inside the BT's planning grid (empty needs
# +7.5,+7.5 to escape gz's centered origin; map-file worlds don't).
odom_env = dict(os.environ)
odom_env["EP_MAP"] = MAP
spawn("world", [
    "python3", "-u", f"{WS}/sim/python/world_odom_publisher.py",
], env=odom_env)

# Lidar leader-body filter: subscribes /follower/scan_raw (from gz bridge),
# strips beams that hit the leader's mesh, republishes /follower/scan.
spawn("world", [
    "python3", "-u", f"{WS}/sim/python/lidar_leader_filter.py",
])

# Snapshot recorder — saves a top-down PNG every second to <DIR>/snapshots/.
snap_env = dict(os.environ)
snap_env["SNAP_DIR"] = str(SNAPS)
snap_env["EP_MAP"] = MAP
snap_env["SNAP_PERIOD_SEC"] = "1.0"
f = log("snapshots")
f.write(f"\n=== spawning: snapshot_recorder.py (dir={SNAPS})\n"); f.flush()
p = subprocess.Popen(
    ["python3", "-u", f"{WS}/sim/python/snapshot_recorder.py"],
    stdout=f, stderr=subprocess.STDOUT, env=snap_env,
    preexec_fn=os.setsid)
procs.append(("snapshots", p))

# ---------------------------------------------------------------------------
# 3) FOLLOWER: DAM4SAM tracker + the BT-based follow_everything_follower.
# ---------------------------------------------------------------------------
tracker_cmd = ["python3", "-u", f"{WS}/follower_pkg/python/dam4sam_tracker.py"]
if SRC == "dam4sam":
    tracker_cmd += [
        "--ros-args", "-r",
        "/follower/camera/detections_dam4sam:=/follower/camera/detections",
    ]
spawn("follower", tracker_cmd)

time.sleep(2)  # let the tracker register before the BT subscribes

fenv = dict(os.environ)
fenv["PYTHONPATH"] = (
    "/opt/follow_everything_nav2:" + fenv.get("PYTHONPATH", "")).rstrip(":")
# BT reads SIM_MAP for its world dimensions (W*0.5, H*0.5). Empty has no
# map file → /dev/null forces the BT's 15×15 fallback. Non-empty maps must
# pass the actual file so the BT's planning grid covers the whole world
# (otherwise the bot, spawned per the F cell, lives outside A*'s domain).
if MAP == "empty":
    fenv.setdefault("SIM_MAP", "/dev/null")
else:
    fenv["SIM_MAP"] = f"/opt/follow_everything_nav2/sim/maps/{MAP}.txt"
spawn("follower", [
    "python3", "-u",
    "/opt/follow_everything_nav2/follower_pkg/python/follow_everything_follower.py",
], env=fenv)

# ---------------------------------------------------------------------------
# 4) Run for DUR seconds, then shut down cleanly.
# ---------------------------------------------------------------------------
print(f"Running... (Ctrl-C to stop early)")
try:
    time.sleep(DUR)
except KeyboardInterrupt:
    print("Interrupted")

print("Stopping subprocesses...")
for _, p in procs:
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except ProcessLookupError:
        pass
for _, p in procs:
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
for f in log_handles.values():
    f.close()

print(f"\nLogs:")
for fn in sorted(DIR.iterdir()):
    sz = fn.stat().st_size
    print(f"  {fn.name:14s} {sz/1024:8.1f} KB")
print(f"\nTo inspect:  ls {DIR}")
