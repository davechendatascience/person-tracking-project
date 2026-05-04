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

WORKDIR /ws
ENV WS_ROOT=/ws
