"""Follower-side launch: starts the DAM4SAM tracker (and later, the BT/follower
script). Kept separate from sim/launch/empty_bringup.launch.py so the sim and
the follower can be brought up in different shells, mirroring the layout of
../follow_everything_nav2/follower_pkg/launch/follower.launch.py.
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    repo = os.environ.get("WS_ROOT", "/ws")
    tracker = os.path.join(repo, "follower_pkg", "python", "dam4sam_tracker.py")

    return LaunchDescription([
        ExecuteProcess(
            cmd=[sys.executable, "-u", tracker],
            output="both",
            cwd=repo,
            emulate_tty=True,
        ),
    ])
