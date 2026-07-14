"""Follower-side launch: EdgeTAM tracker + BT-based follower.

The follower is the BT-based `follow_everything_follower.py` from the 2D
project at ../follow_everything_nav2/. Its topic contract (odom, scan,
camera/detections, cmd_vel) is identical in 2D and 3D, so we mount the
2D project read-only at /opt/follow_everything_nav2 (see docker-compose),
add /opt/follow_everything_nav2 to PYTHONPATH (so its `from sim.world …`
imports resolve), and just exec it.

Toggle:
  perception:=oracle (default)     — oracle drives the contract topic; no
                                     tracker is spawned.
  perception:=aot                  — aot_tracker.py drives it.
  perception:=sam2_aot_memory      — edgetam_tracker.py drives it, routed to
                                     sam2_aot_memory.SAM2AOTMemoryStreamingTracker
                                     (EdgeTAM + AOT long/short-term memory).

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

    perception    = LaunchConfiguration("perception").perform(context)
    follower_kind = LaunchConfiguration("follower_kind").perform(context)
    rviz          = LaunchConfiguration("rviz").perform(context)

    # `perception` is the single backend selector for who drives the BT's
    # contract topic /follower/camera/detections:
    #   oracle          -> oracle drives it (external node); NO tracker spawned.
    #   aot             -> aot_tracker.py.
    #   sam2_aot_memory -> edgetam_tracker.py (the ROS node) with
    #                      EDGETAM_TRACKER=sam2_aot_memory, which routes its
    #                      streaming tracker to
    #                      sam2_aot_memory.SAM2AOTMemoryStreamingTracker (the
    #                      real AOT long/short-term memory: appearance-gated LT
    #                      promotion, distractor rejection, self-consistency
    #                      audit). sam2_aot_memory.py is a library module, not a
    #                      runnable node, so it is driven through this node —
    #                      never spawned directly. Bounded LT -> flat per-frame
    #                      cost, so faster on average than AOT on long runs.
    tracker_env = dict(os.environ)
    tracker_cmd = None
    if perception == "aot":
        tracker_script = "aot_tracker.py"
        tracker_topic  = "/follower/camera/detections_aot"
    elif perception == "sam2_aot_memory":
        tracker_script = "edgetam_tracker.py"
        tracker_topic  = "/follower/camera/detections_edgetam"
        tracker_env["EDGETAM_TRACKER"] = "sam2_aot_memory"
    elif perception == "oracle":
        tracker_script = None   # oracle drives the contract topic directly
    else:
        raise RuntimeError(
            f"perception={perception!r} invalid; "
            "choose oracle | aot | sam2_aot_memory")

    if tracker_script is not None:
        tracker = os.path.join(repo, "follower_pkg", "python", tracker_script)
        # The selected tracker always drives the contract topic (remap).
        tracker_cmd = [
            sys.executable, "-u", tracker,
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

    procs = []
    if tracker_cmd is not None:  # oracle mode spawns no tracker
        procs.append(ExecuteProcess(
            cmd=tracker_cmd, env=tracker_env,
            output="both", cwd=repo, emulate_tty=True))
    procs.append(ExecuteProcess(
        cmd=follower_cmd,
        env=follower_env,
        output="both", cwd=repo, emulate_tty=True))

    # Optional RViz with the EdgeTAM/SAM2-AOTmem overlay preconfigured.
    # The shipped config has an Image display on the overlay topic with
    # Reliable QoS (matching the publisher) — the overlay shows on open,
    # no manual display setup needed. Needs an X server in the container
    # (see README's RViz/X11 section).
    if rviz.lower() in ("1", "true", "yes"):
        rviz_cfg = os.path.join(repo, "follower_pkg", "rviz", "follower.rviz")
        procs.append(ExecuteProcess(
            cmd=["rviz2", "-d", rviz_cfg],
            output="both", cwd=repo, emulate_tty=True))

    return procs


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "perception",
            default_value="oracle",
            description="perception backend that drives the contract topic: "
                        "oracle (ground truth, no tracker) | "
                        "aot (AOT/DeAOT family, occlusion-robust memory) | "
                        "sam2_aot_memory (EdgeTAM + AOT long/short-term memory)"),
        DeclareLaunchArgument(
            "follower_kind",
            default_value="bt",
            description="bt (follow_everything_follower with BT) | "
                        "simple (proportional simple_follower for regression)"),
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="true -> open RViz with follower_pkg/rviz/follower.rviz "
                        "(EdgeTAM overlay preconfigured). Needs X in container."),
        OpaqueFunction(function=_bringup),
    ])
