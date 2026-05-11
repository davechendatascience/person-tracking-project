"""Follower-side launch: EdgeTAM tracker + BT-based follower.

The follower is the BT-based `follow_everything_follower.py` from the 2D
project at ../follow_everything_nav2/. Its topic contract (odom, scan,
camera/detections, cmd_vel) is identical in 2D and 3D, so we mount the
2D project read-only at /opt/follow_everything_nav2 (see docker-compose),
add /opt/follow_everything_nav2 to PYTHONPATH (so its `from sim.world …`
imports resolve), and just exec it.

Toggle:
  detection_source:=edgetam (default) — EdgeTAM output remapped onto the
                                        contract topic.
  detection_source:=oracle            — oracle drives the contract topic.

Toggle:
  follower_kind:=bt (default) — runs the BT-based follow_everything_follower.
  follower_kind:=simple       — runs our P-controller simple_follower.py
                                 (kept around for regression / smoke tests).
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


# Where the 2D project's BT follower lives inside the container — see
# docker-compose.yml's read-only mount.
BT_FOLLOWER_PATH = "/opt/follow_everything_nav2/follower_pkg/python/follow_everything_follower.py"


def _bringup(context, *args, **kwargs):
    repo = os.environ.get("WS_ROOT", "/ws")
    simple_follower = os.path.join(
        repo, "follower_pkg", "python", "simple_follower.py")

    detection_source = LaunchConfiguration("detection_source").perform(context)
    follower_kind    = LaunchConfiguration("follower_kind").perform(context)
    tracker_kind     = LaunchConfiguration("tracker_kind").perform(context)

    # Each tracker variant publishes on its own pre-remap topic; the
    # `detection_source:=edgetam` arg below points that topic at the
    # BT's contract topic regardless of which tracker is active.
    if tracker_kind == "aot":
        tracker_script = "aot_tracker.py"
        tracker_topic  = "/follower/camera/detections_aot"
    else:
        tracker_script = "edgetam_tracker.py"
        tracker_topic  = "/follower/camera/detections_edgetam"
    tracker = os.path.join(repo, "follower_pkg", "python", tracker_script)

    tracker_cmd = [sys.executable, "-u", tracker]
    if detection_source == "edgetam":
        tracker_cmd += [
            "--ros-args", "-r",
            f"{tracker_topic}:=/follower/camera/detections",
        ]

    if follower_kind == "simple":
        follower_cmd = [sys.executable, "-u", simple_follower]
    else:
        # follow_everything_follower.py reads SIM_MAP for its A* grid size.
        # We point it at a non-existent path so its built-in fallback
        # returns world_size=(15, 15) m — fine for our empty world.
        follower_cmd = [sys.executable, "-u", BT_FOLLOWER_PATH]

    follower_env = dict(os.environ)
    follower_env.setdefault("SIM_MAP", "/dev/null")  # forces 15x15 fallback
    # /ws/sim/ and /opt/follow_everything_nav2/sim/ are *both* `sim`
    # packages (we kept the same layout name). The BT follower wants the
    # 2D project's `sim.world / sim.geometry / sim.planner`, so put that
    # path *first* in the BT follower's PYTHONPATH. Only affects this
    # subprocess — our oracle / sim launch aren't impacted.
    fenv_pp = follower_env.get("PYTHONPATH", "")
    follower_env["PYTHONPATH"] = (
        "/opt/follow_everything_nav2:" + fenv_pp).rstrip(":")

    return [
        ExecuteProcess(
            cmd=tracker_cmd,
            output="both", cwd=repo, emulate_tty=True),
        ExecuteProcess(
            cmd=follower_cmd,
            env=follower_env,
            output="both", cwd=repo, emulate_tty=True),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "detection_source",
            default_value="edgetam",
            description="edgetam (tracker drives the contract topic) | "
                        "oracle (oracle drives it directly)"),
        DeclareLaunchArgument(
            "tracker_kind",
            default_value="edgetam",
            description="edgetam (SAM2 fork, default) | "
                        "aot (AOT/DeAOT family, occlusion-robust memory)"),
        DeclareLaunchArgument(
            "follower_kind",
            default_value="bt",
            description="bt (follow_everything_follower with BT) | "
                        "simple (proportional simple_follower for regression)"),
        OpaqueFunction(function=_bringup),
    ])
