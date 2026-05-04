"""Minimal launch for the baseline follower. Spawns the python script directly so
the package does not need to be colcon-built. The autoresearch agent should extend
this to include nav2_bringup, bt_navigator, costmap_2d, etc."""
import os
import sys

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    repo = os.environ.get("WS_ROOT", "/ws")
    # Pick the implementation via FOLLOWER env var: "baseline" or
    # "follow_everything" (default). Both publish /follower/cmd_vel.
    impl = os.environ.get("FOLLOWER", "follow_everything")
    fname = {
        "baseline": "baseline_follower.py",
        "follow_everything": "follow_everything_follower.py",
    }.get(impl, "follow_everything_follower.py")
    script = os.path.join(repo, "follower_pkg", "python", fname)
    return LaunchDescription([
        ExecuteProcess(
            cmd=[sys.executable, "-u", script],   # -u: unbuffered stdout
            output="both",                         # write to screen AND log files
            cwd=repo,
            emulate_tty=True,                       # forces line buffering
        ),
    ])
