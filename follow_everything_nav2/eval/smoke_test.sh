#!/usr/bin/env bash
# Quick smoke test: build image, run a 30s episode without follower, then with baseline.
# Run from repo root inside WSL or any Linux shell with docker:
#   bash eval/smoke_test.sh
set -e

cd "$(dirname "$0")/.."

echo "=== Building Docker image ==="
docker compose build sim

echo
echo "=== Smoke 1: world + leader only, no follower (expect zero success) ==="
docker compose run --rm sim python -m eval.run_episode \
    --episodes 1 --maps empty --episode-secs 20 --no-follower

echo
echo "=== Smoke 2: world + leader + baseline follower (expect some movement) ==="
docker compose run --rm sim bash -c "
    source /opt/ros/humble/setup.bash &&
    python -m eval.run_episode --episodes 1 --maps cluttered --episode-secs 30
"

echo
echo "Smoke tests complete. Check eval/results/last_run.json for full per-episode metrics."
