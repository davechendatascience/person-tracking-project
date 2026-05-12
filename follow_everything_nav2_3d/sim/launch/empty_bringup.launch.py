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
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bringup(context, *args, **kwargs):
    repo = os.environ.get("WS_ROOT", "/ws")
    oracle_path = os.path.join(repo, "sim", "python", "oracle_camera.py")
    detection_source = LaunchConfiguration("detection_source").perform(context)
    map_name = LaunchConfiguration("map").perform(context)

    # Map -> world file. For "empty" we use the hand-authored template;
    # for any 2D-project map (cluttered, corridor, forest, ...) we
    # generate a /tmp/<map>_generated.world by parsing
    # /opt/follow_everything_nav2/sim/maps/<map>.txt.
    if map_name == "empty":
        world_path = os.path.join(repo, "sim", "worlds", "empty.world")
    else:
        builder = os.path.join(repo, "sim", "python", "build_world.py")
        proc = subprocess.run(
            ["python3", builder, map_name],
            capture_output=True, text=True)
        if proc.stderr:
            print(proc.stderr, end="")
        if proc.returncode != 0:
            raise RuntimeError(
                f"build_world.py {map_name} failed (exit {proc.returncode}). "
                f"stderr above.")
        world_path = proc.stdout.strip().splitlines()[-1]
        print(f"[empty_bringup] using generated world: {world_path}")

    gz_sim = ExecuteProcess(
        cmd=["ign", "gazebo", "-r", "-v", "3", world_path],
        output="screen",
    )

    contract_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/follower/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
            # /follower/odom is published by world_odom_publisher.py in WORLD
            # frame (gz's diff-drive odom is parked on /follower/odom_local).
            "/follower/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
            "/follower/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            "/follower/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
            "/follower/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/follower/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/follower/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            # Leader's velocity command — Twist sent by leader_controller.py
            # is forwarded into gz's VelocityControl plugin on the leader
            # model.
            "/leader/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
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
    if detection_source == "edgetam":
        oracle_cmd += [
            "--ros-args", "-r",
            "/follower/camera/detections:=/follower/camera/detections_oracle",
        ]
    oracle = ExecuteProcess(cmd=oracle_cmd, output="screen")

    leader_ctrl_path = os.path.join(repo, "sim", "python", "leader_controller.py")
    leader_ctrl = ExecuteProcess(
        cmd=["python3", "-u", leader_ctrl_path],
        additional_env={"EP_MAP": map_name},
        output="screen")

    world_odom_path = os.path.join(repo, "sim", "python", "world_odom_publisher.py")
    world_odom = ExecuteProcess(
        cmd=["python3", "-u", world_odom_path], output="screen")

    return [gz_sim, contract_bridge,
            pose_truth_bridge_dynamic, pose_truth_bridge_static,
            oracle, leader_ctrl, world_odom]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "detection_source",
            default_value="edgetam",
            description="Which source drives /follower/camera/detections: "
                        "'edgetam' (primary, oracle moves to _oracle) or "
                        "'oracle' (oracle on contract topic, EdgeTAM stays "
                        "on _edgetam).",
        ),
        DeclareLaunchArgument(
            "map",
            default_value="empty",
            description="Map name: empty | corridor | cluttered | forest. "
                        "Non-empty maps are generated from the 2D project's "
                        "ASCII grids at /opt/follow_everything_nav2/sim/maps/.",
        ),
        OpaqueFunction(function=_bringup),
    ])
