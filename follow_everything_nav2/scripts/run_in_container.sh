#!/usr/bin/env bash
# Sources ROS 2 + sets PYTHONPATH, then runs whatever args you pass.
# Designed to be the entrypoint for `docker compose run sim-viz scripts/run_in_container.sh ...`.
set -eo pipefail
source /opt/ros/humble/setup.bash
export PYTHONPATH="/ws:${PYTHONPATH:-}"
exec "$@"
