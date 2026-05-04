"""Empty-world bringup: gz Fortress + bridges + oracle camera.

The follower is now a native SDF model declared inline in empty.world (no
URDF / xacro / robot_state_publisher / spawn-from-topic), so launch is
just:
  - ign gazebo -r empty.world          (model already in the world)
  - parameter_bridge for the contract topics + /clock
  - parameter_bridge for /world/empty/dynamic_pose/info -> /gz_pose_truth
  - oracle_camera

ROS-side topic contract:
    /follower/cmd_vel             (geometry_msgs/Twist)
    /follower/odom                (nav_msgs/Odometry)
    /follower/scan                (sensor_msgs/LaserScan)
    /follower/camera/image        (sensor_msgs/Image)
    /follower/camera/depth_image  (sensor_msgs/Image)
    /follower/camera/camera_info  (sensor_msgs/CameraInfo)
    /follower/camera/detections   (vision_msgs/Detection2DArray)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bringup(context, *args, **kwargs):
    repo = os.environ.get("WS_ROOT", "/ws")
    world_path = os.path.join(repo, "sim", "worlds", "empty.world")
    oracle_path = os.path.join(repo, "sim", "python", "oracle_camera.py")
    detection_source = LaunchConfiguration("detection_source").perform(context)

    gz_sim = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "3", world_path],
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
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
        ],
        output="screen",
    )

    # dynamic_pose/info only emits updates for *moving* entities, so a
    # stationary follower wouldn't appear there at startup. pose/info is the
    # periodic full snapshot — both feed /gz_pose_truth.
    pose_truth_bridge_dynamic = Node(
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
    pose_truth_bridge_static = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/empty/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
        ],
        remappings=[
            ("/world/empty/pose/info", "/gz_pose_truth"),
        ],
        output="screen",
    )

    oracle_cmd = ["python3", "-u", oracle_path]
    if detection_source == "dam4sam":
        oracle_cmd += [
            "--ros-args", "-r",
            "/follower/camera/detections:=/follower/camera/detections_oracle",
        ]
    oracle = ExecuteProcess(cmd=oracle_cmd, output="screen")

    return [gz_sim, contract_bridge,
            pose_truth_bridge_dynamic, pose_truth_bridge_static, oracle]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "detection_source",
            default_value="dam4sam",
            description="Which source drives /follower/camera/detections: "
                        "'dam4sam' (primary, oracle moves to _oracle) or "
                        "'oracle' (oracle on contract topic, DAM4SAM stays "
                        "on _dam4sam).",
        ),
        OpaqueFunction(function=_bringup),
    ])
