# ROS 2 Humble + Nav2 + simulator for autoresearch leader-following.
# Headless / CPU-only. The 2D simulator publishes ROS topics; Nav2 runs unmodified.
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    PYTHONUNBUFFERED=1

# Nav2 + BT.CPP + commonly used messages + Python deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-nav2-bringup \
        ros-humble-nav2-bt-navigator \
        ros-humble-nav2-behavior-tree \
        ros-humble-nav2-behaviors \
        ros-humble-nav2-collision-monitor \
        ros-humble-nav2-controller \
        ros-humble-nav2-planner \
        ros-humble-nav2-smoother \
        ros-humble-nav2-velocity-smoother \
        ros-humble-nav2-msgs \
        ros-humble-vision-msgs \
        ros-humble-tf2-tools \
        ros-humble-tf-transformations \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-behaviortree-cpp-v3 \
        python3-colcon-common-extensions \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-matplotlib \
        python3-yaml \
        git build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

WORKDIR /ws
COPY . /ws

# Build Python wheels needed by sim (none beyond apt for now). Add transforms3d for quaternion math.
RUN pip3 install --no-cache-dir transforms3d py_trees==2.2.3

# Source ROS on every shell
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'if [ -f /ws/install/setup.bash ]; then source /ws/install/setup.bash; fi' >> /root/.bashrc

CMD ["bash"]
