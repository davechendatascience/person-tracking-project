#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
: > results/frames.txt
for f in $(ls results/snapshots/ | sort); do
    echo "file '/ws/results/snapshots/$f'" >> results/frames.txt
done
echo "wrote $(wc -l < results/frames.txt) frame entries"
