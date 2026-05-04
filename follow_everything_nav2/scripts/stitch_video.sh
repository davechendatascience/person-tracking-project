#!/usr/bin/env bash
# Install ffmpeg if missing, then stitch /ws/results/snapshots/*.png into MP4.
set -e
if ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq ffmpeg >/dev/null 2>&1
fi
ffmpeg -y -r 10 -f concat -safe 0 -i /ws/results/frames.txt \
    -c:v libx264 -pix_fmt yuv420p \
    -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=10' \
    /ws/results/leader_follower_v33_100s.mp4 2>&1 | tail -3
