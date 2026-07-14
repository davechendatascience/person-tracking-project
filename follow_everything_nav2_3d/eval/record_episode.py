"""Record a fixed-duration episode with separated per-process logs.

Mirrors follow_everything_nav2's results/logs/ep_<ts>_*/{world,leader,follower}.log
layout so we can debug the 3D port the same way as the 2D one.

Usage (inside the container, from /ws):
    PERCEPTION=<oracle|aot|sam2_aot_memory> \
        python3 eval/record_episode.py [duration_sec] [map]

  duration_sec  defaults to 30
  map           defaults to empty
  PERCEPTION    (env) the single perception-backend selector, default oracle:
                  oracle          - oracle_camera drives the contract topic;
                                    NO tracker is spawned (ground truth).
                  aot             - aot_tracker.py drives it (DeAOT/AOT).
                  sam2_aot_memory - edgetam_tracker.py drives it, routed to
                                    sam2_aot_memory.SAM2AOTMemoryStreamingTracker
                                    (EdgeTAM + AOT long/short-term memory).

Produces:
    results/logs/ep_<unix_ts>_empty_0/
        world.log     # gz Fortress + ros_gz_bridges
        leader.log    # oracle_camera (knows the leader's true pose)
        follower.log  # edgetam_tracker + follow_everything_follower (BT)

Bypasses `ros2 launch` so each conceptual subsystem gets its own log file.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MAP = sys.argv[2] if len(sys.argv) > 2 else "empty"

# Single perception-backend selector (replaces the old detection_source
# positional + TRACKER_KIND env). oracle | aot | sam2_aot_memory.
PERCEPTION = os.environ.get("PERCEPTION", "oracle").strip()
_VALID_PERCEPTION = ("oracle", "aot", "sam2_aot_memory")
# Catch the legacy CLI shape `record_episode.py <dur> <detection_source> <map>`:
# argv[2] used to be oracle/edgetam, now it's the map. Fail loud with a hint.
if MAP in ("oracle", "edgetam", "aot", "sam2_aotmem", "sam2_aot_memory"):
    sys.exit(
        f"'{MAP}' is not a map. The CLI changed: perception is now the "
        f"PERCEPTION env var and argv[2] is the map.\n"
        f"  e.g.  PERCEPTION=aot python3 eval/record_episode.py {DUR} empty")
if PERCEPTION not in _VALID_PERCEPTION:
    sys.exit(f"PERCEPTION={PERCEPTION!r} invalid; "
             f"choose from {_VALID_PERCEPTION}")
# Whether a tracker process is spawned at all (oracle drives detections itself).
USE_TRACKER = PERCEPTION in ("aot", "sam2_aot_memory")
WS  = os.environ.get("WS_ROOT", "/ws")
TS  = int(time.time())
DIR = Path(WS) / "results" / "logs" / f"ep_{TS}_{MAP}_0"
SNAPS = DIR / "snapshots"
DIR.mkdir(parents=True, exist_ok=True)
SNAPS.mkdir(parents=True, exist_ok=True)

print(f"Recording {DUR}s map={MAP} perception={PERCEPTION} -> {DIR}")

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
# 2) LEADER: oracle_camera (publishes the leader's body-frame detection).
#    We hold off on spawning leader_controller until the tracker has
#    finished building its predictor + run its first init pass — otherwise
#    the leader walks away during the ~30 s EdgeTAM build and the
#    perception system races a moving target it hasn't locked onto yet.
# ---------------------------------------------------------------------------
oracle_cmd = ["python3", "-u", f"{WS}/sim/python/oracle_camera.py"]
if USE_TRACKER:
    # A tracker drives the contract topic, so keep the oracle on a side
    # topic (ground-truth logging / projection lookup only).
    oracle_cmd += [
        "--ros-args", "-r",
        "/follower/camera/detections:=/follower/camera/detections_oracle",
    ]
spawn("leader", oracle_cmd)

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
# 3) FOLLOWER: perception tracker (unless oracle) + BT follow_everything_follower.
# ---------------------------------------------------------------------------
# PERCEPTION picks the backend that drives /follower/camera/detections:
#   oracle          -> oracle_camera drives it; NO tracker is spawned.
#   aot             -> aot_tracker.py.
#   sam2_aot_memory -> edgetam_tracker.py with EDGETAM_TRACKER=sam2_aot_memory,
#                      routed to sam2_aot_memory.SAM2AOTMemoryStreamingTracker
#                      (appearance-gated LT promotion, distractor rejection,
#                      self-consistency audit).
# BT's taskset cores are read regardless of whether a tracker runs.
follower_cores = os.environ.get("FOLLOWER_TASKSET_CORES", "").strip()

if USE_TRACKER:
    if PERCEPTION == "aot":
        tracker_script    = f"{WS}/follower_pkg/python/aot_tracker.py"
        tracker_topic     = "/follower/camera/detections_aot"
        INIT_READY_MARKER = "AOT init: mask shape="
        edgetam_variant   = None
    else:  # sam2_aot_memory
        tracker_script    = f"{WS}/follower_pkg/python/edgetam_tracker.py"
        tracker_topic     = "/follower/camera/detections_edgetam"
        INIT_READY_MARKER = "EdgeTAM init: mask shape="
        edgetam_variant   = "sam2_aot_memory"

    # The selected tracker always drives the contract topic (remap).
    tracker_cmd = [
        "python3", "-u", tracker_script,
        "--ros-args", "-r",
        f"{tracker_topic}:=/follower/camera/detections",
    ]
    # CPU pinning. The AOT pure-PyTorch fallback can saturate a CPU core and
    # starve the BT's 20 Hz tick — choppy follow in heavy maps (forest,
    # cluttered). Pin the tracker via TRACKER_TASKSET_CORES (e.g. "0,1"); the
    # BT picks up the complement via FOLLOWER_TASKSET_CORES. Default unpinned.
    # Needs `privileged: true` on compose (already set) for the BT's nice case.
    tracker_cores = os.environ.get("TRACKER_TASKSET_CORES", "").strip()
    if tracker_cores:
        tracker_cmd = ["taskset", "-c", tracker_cores] + tracker_cmd
        print(f"Pinning tracker to cores {tracker_cores}")
    # Forward the episode log dir so the tracker can dump init RGB + the first
    # few propagated frames for offline inspection.
    tracker_env = dict(os.environ)
    tracker_env["EP_LOG_DIR"] = str(DIR)
    if edgetam_variant:
        # Name the EdgeTAM-hosted variant explicitly (no on/off boolean).
        tracker_env["EDGETAM_TRACKER"] = edgetam_variant
        print(f"PERCEPTION=sam2_aot_memory -> EDGETAM_TRACKER={edgetam_variant} "
              "(EdgeTAM + AOT long/short-term memory)")
    spawn("follower", tracker_cmd, env=tracker_env)

    # Block until the tracker has both (a) finished building the predictor
    # (~30 s cold for EdgeTAM, ~5 s for AOT; sam2_aot_memory adds a one-time
    # torch.compile of the image encoder on the first frame) AND (b) run its
    # first init pass on the stationary leader. The init line only appears once
    # the tracker got a camera frame + oracle bbox AND processed it. No timeout:
    # a longer compile just makes this wait longer, which is fine.
    print(f"Waiting for tracker init ({INIT_READY_MARKER!r})...")
    _t0 = time.time()
    _follower_log = DIR / "follower.log"
    while True:
        if _follower_log.exists():
            with open(_follower_log) as _fh:
                if INIT_READY_MARKER in _fh.read():
                    break
        time.sleep(0.5)
    print(f"Tracker ready after {time.time() - _t0:.1f}s. "
          "Spawning leader_controller + BT.")
else:
    # oracle: oracle_camera drives detections from ground truth — no tracker to
    # build, so no lock-on window to protect. Give oracle a moment to come up,
    # then start the leader.
    print("PERCEPTION=oracle -> no tracker; oracle_camera drives detections.")
    time.sleep(3)

# Patrol controller for the leader — for tracker modes this is only reached
# AFTER tracker init, so the leader doesn't walk away during the build window.
leader_env = dict(os.environ)
leader_env["EP_MAP"] = MAP
spawn("leader", [
    "python3", "-u", f"{WS}/sim/python/leader_controller.py",
], env=leader_env)

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
follower_cmd = [
    "python3", "-u",
    "/opt/follow_everything_nav2/follower_pkg/python/follow_everything_follower.py",
]
if follower_cores:
    follower_cmd = ["taskset", "-c", follower_cores] + follower_cmd
    print(f"Pinning BT/Nav2 follower to cores {follower_cores}")
spawn("follower", follower_cmd, env=fenv)

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
