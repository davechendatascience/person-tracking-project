"""Phase 3 bringup: empty Fortress world + diff-drive follower + walking-human
actor leader + oracle camera bridge.

Topology:
  - Ignition Fortress (gz sim) with empty.world (contains the actor leader)
  - robot_state_publisher under /follower namespace (xacro-rendered URDF)
  - ros_gz_sim spawn for the follower
  - parameter_bridge: ROS <-> gz for the contract topics
  - second parameter_bridge: SceneBroadcaster /world/empty/dynamic_pose/info
    -> tf2_msgs/TFMessage on /gz_pose_truth (kept off /tf to avoid clash with
    the diff-drive's own TF tree)
  - oracle_camera: subscribes /gz_pose_truth, publishes
    /follower/camera/detections (vision_msgs/Detection2DArray)

ROS-side topic contract after Phase 3:
    /follower/cmd_vel             (geometry_msgs/Twist)
    /follower/odom                (nav_msgs/Odometry)
    /follower/scan                (sensor_msgs/LaserScan)
    /follower/camera/detections   (vision_msgs/Detection2DArray)
"""
import os

import xacro
from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction
from launch_ros.actions import Node


def _bringup(context, *args, **kwargs):
    repo = os.environ.get("WS_ROOT", "/ws")
    xacro_path = os.path.join(repo, "sim", "urdf", "follower.urdf.xacro")
    world_path = os.path.join(repo, "sim", "worlds", "empty.world")
    oracle_path = os.path.join(repo, "sim", "python", "oracle_camera.py")

    robot_description = xacro.process_file(xacro_path).toxml()

    gz_sim = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "3", world_path],
        output="screen",
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="follower",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "follower",
            "-topic", "/follower/robot_description",
            "-x", "0", "-y", "0", "-z", "0.05",
        ],
        output="screen",
    )

    contract_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/follower/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
            "/follower/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            "/follower/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
            "/follower/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            "/follower/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
            "/follower/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/follower/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/follower/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            # PointCloud2 bridge dropped: ignition.msgs.PointCloudPacked -> ROS
            # serialization is fragile in Fortress and has been seen to throw
            # "sequence size exceeds remaining buffer". DAM4SAM only needs RGB
            # + depth + intrinsics, so we don't need the cloud anyway.
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
        output="screen",
    )

    # Ground-truth poses for the oracle camera. Remap to /gz_pose_truth so we
    # don't pollute /tf.
    pose_truth_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
        ],
        remappings=[
            ("/world/empty/dynamic_pose/info", "/gz_pose_truth"),
        ],
        output="screen",
    )

    oracle = ExecuteProcess(
        cmd=["python3", "-u", oracle_path],
        output="screen",
    )

    return [gz_sim, rsp, spawn, contract_bridge, pose_truth_bridge, oracle]


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=_bringup)])
