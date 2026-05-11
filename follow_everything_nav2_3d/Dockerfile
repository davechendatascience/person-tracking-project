# Phase 1 image: ROS 2 Humble + Ignition Gazebo Fortress + ros_gz bridge.
#
# arm64 hosts (e.g. Jetson) do not get `ros-humble-gazebo-ros-pkgs` from the
# ROS 2 apt repo, so we use Ignition Fortress (Tier 1 with Humble) and bridge
# Gazebo Transport <-> ROS via ros_gz_bridge. Topic contract on the ROS side
# is unchanged: /follower/{cmd_vel, odom, scan, camera/detections}.
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg lsb-release ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# OSRF Gazebo apt source for Ignition Fortress on jammy arm64/amd64.
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/gazebo-stable.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        ignition-fortress \
        ros-humble-ros-gz \
        ros-humble-ros-gz-bridge \
        ros-humble-ros-gz-sim \
        ros-humble-ros-gz-interfaces \
        ros-humble-xacro \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-teleop-twist-keyboard \
        ros-humble-tf2-tools \
        ros-humble-tf2-ros \
        ros-humble-tf-transformations \
        ros-humble-cv-bridge \
        ros-humble-rviz2 \
        ros-humble-vision-msgs \
        python3-colcon-common-extensions \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Pre-cache the walking-human actor mesh from Fuel so the first launch does
# not require internet. Falls through silently if Fuel is unreachable at build
# time — Fortress will then fetch on first run if the container has network.
RUN ign fuel download \
        -u "https://fuel.gazebosim.org/1.0/Mingfei/models/actor" \
        -v 4 || echo "[warn] Fuel pre-cache failed; will retry at runtime."

# --- Phase 4b-i: ML stack for EdgeTAM / SAM2 ---------------------------------
# Host is Grace Blackwell (sbsa aarch64) with CUDA 13.0 driver. CUDA 13 is
# backward-compatible with cu12.x runtime, so we use the cu126 PyTorch wheels —
# they have aarch64 binaries on download.pytorch.org and are stable. (No
# cu130 wheels exist for SBSA at time of writing.)
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.5.1 torchvision==0.20.1 \
 || pip install --no-cache-dir torch torchvision  # CPU fallback if cu126 wheel unavailable

# EdgeTAM / SAM2 runtime deps. hydra-core + iopath are SAM2's loader
# dependencies; timm is needed by EdgeTAM's RepViT backbone; ultralytics
# provides YOLO for the first-frame bootstrap. (vot-toolkit / vot-trax
# were DAM4SAM-specific and are no longer needed since we stripped the
# DAM4SAM dependency.)
RUN pip install --no-cache-dir \
        "numpy<2" \
        opencv-python-headless \
        Pillow \
        hydra-core \
        iopath \
        einops \
        ultralytics \
        timm \
        tqdm \
        scipy \
        py_trees

WORKDIR /ws
# PYTHONPATH:
#   /opt              parent of `follow_everything/` and `follow_everything_nav2/`
#                     so `from follow_everything.perception import …` resolves.
#   /opt/EdgeTAM      so `import sam2` resolves to EdgeTAM's `sam2` fork
#                     (we use it via build_sam2_video_predictor).
#   /opt/follow_everything_nav2
#                     so `from sim.world import ROBOT_RADIUS` resolves
#                     (the 2D project's BT follower expects `sim` as a
#                     top-level package — its sim/ folder lives at
#                     /opt/follow_everything_nav2/sim/).
ENV WS_ROOT=/ws \
    PYTHONPATH=/ws:/opt:/opt/EdgeTAM:/opt/follow_everything_nav2:${PYTHONPATH}
