#!/bin/bash
# DeepStream environment setup + tracker runner
# Run this INSIDE the nvcr.io/nvidia/deepstream:7.1-triton-multiarch container

set -e

DS_LIB="/opt/nvidia/deepstream/deepstream/lib"

echo "[1/4] Setting up DeepStream library path..."
export LD_LIBRARY_PATH="${DS_LIB}:$LD_LIBRARY_PATH"

echo "[2/4] Creating versioned symlinks for missing .so files..."
for lib in "${DS_LIB}"/libnv*.so; do
    base=$(basename "$lib" .so)
    versioned="${DS_LIB}/${base}.so.1.0.0"
    if [ ! -e "$versioned" ]; then
        echo "  Symlinking: $versioned -> $lib"
        ln -sf "$lib" "$versioned"
    fi
done

echo "[3/4] Registering path with ldconfig..."
echo "${DS_LIB}" > /etc/ld.so.conf.d/deepstream.conf
ldconfig

echo "[4/4] Verifying pyds..."
python3 -c 'import pyds; print("pyds import: OK")'

echo ""
echo "Starting mot_tracker_deepstream.py..."
python3 mot_tracker_deepstream.py "$@"
