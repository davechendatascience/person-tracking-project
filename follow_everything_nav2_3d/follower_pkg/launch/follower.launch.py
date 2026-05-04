"""Follower-side launch: DAM4SAM tracker + minimal P-controller follower.

  detection_source:=dam4sam  (default) — tracker output remapped to
                              /follower/camera/detections so the follower
                              consumes DAM4SAM. Phase 5 primary mode.
  detection_source:=oracle   — tracker output stays on _dam4sam; the
                              follower (which subscribes to /detections)
                              ends up driven by the oracle published from
                              sim/launch/empty_bringup.launch.py. Useful as
                              a regression baseline.
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _bringup(context, *args, **kwargs):
    repo = os.environ.get("WS_ROOT", "/ws")
    tracker  = os.path.join(repo, "follower_pkg", "python", "dam4sam_tracker.py")
    follower = os.path.join(repo, "follower_pkg", "python", "simple_follower.py")
    detection_source = LaunchConfiguration("detection_source").perform(context)

    tracker_cmd = [sys.executable, "-u", tracker]
    if detection_source == "dam4sam":
        # DAM4SAM output drives the follower contract topic.
        tracker_cmd += [
            "--ros-args", "-r",
            "/follower/camera/detections_dam4sam:=/follower/camera/detections",
        ]

    return [
        ExecuteProcess(
            cmd=tracker_cmd,
            output="both", cwd=repo, emulate_tty=True),
        ExecuteProcess(
            cmd=[sys.executable, "-u", follower],
            output="both", cwd=repo, emulate_tty=True),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "detection_source",
            default_value="dam4sam",
            description="dam4sam | oracle — see module docstring."),
        OpaqueFunction(function=_bringup),
    ])
