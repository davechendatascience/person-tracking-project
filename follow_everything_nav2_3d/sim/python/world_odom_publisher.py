"""Publish /follower/odom in the WORLD frame, not gz's local odom frame.

By default, gz-sim's diff-drive plugin publishes Odometry with origin at
the follower's spawn pose: /follower/odom reports `(0, 0)` at startup,
even though the follower is at world `(Fx, Fy)`. The BT consumes this
odom as if it were world coordinates, which mis-places its A* grid by
the spawn offset and breaks the recovery / planning logic.

This node bypasses that. It subscribes to /gz_pose_truth (the
SceneBroadcaster ground-truth poses), takes the entry where
`child_frame_id == "follower"`, and republishes that pose as a
nav_msgs/Odometry on /follower/odom — already in WORLD frame.

Net effect:
  - bb.x, bb.y, bb.yaw on the BT side == world coordinates.
  - learned_map's `(0, 0)` cell == world `(0, 0)`.
  - A* plans on a grid that actually covers the world.

We must also stop gz's diff-drive from publishing on /follower/odom; the
SDF redirects it to /follower/odom_local, and the launch's bridge no
longer carries /follower/odom from gz.
"""
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage


PUB_RATE_HZ = 50.0
FOLLOWER_FRAME = "follower"

# The 2D project's BT (mounted at /opt/follow_everything_nav2) indexes its
# log-odds map and planning grid with `int(wx / MAP_RES)` and rejects cells
# where `cj < 0` or `ci < 0`. So the BT's world is strictly first-quadrant,
# (0,0) → (Wm, Hm) = (15,15) in the SIM_MAP=/dev/null fallback. Gazebo's
# world is centered at the origin, so any time the follower drifted into
# negative-y / negative-x the BT's lidar mapper dropped the hits and A*
# refused goals there. Shift gz coords into the positive quadrant before
# publishing /follower/odom so the BT's first-quadrant assumption holds.
WORLD_ORIGIN_OFFSET = (7.5, 7.5)  # half of the 15×15 fallback world


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class WorldOdomPublisher(Node):
    def __init__(self) -> None:
        super().__init__("world_odom_publisher")
        self._latest_pose = None  # (px, py, pz, qx, qy, qz, qw)
        self.create_subscription(
            TFMessage, "/gz_pose_truth", self._on_poses, 50)
        self.pub = self.create_publisher(Odometry, "/follower/odom", 10)
        self.create_timer(1.0 / PUB_RATE_HZ, self._tick)
        self.get_logger().info(
            "world_odom_publisher live; mirroring gz_pose_truth follower "
            "into /follower/odom (world frame)")

    def _on_poses(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            if tr.child_frame_id == FOLLOWER_FRAME:
                t = tr.transform.translation
                r = tr.transform.rotation
                self._latest_pose = (t.x, t.y, t.z, r.x, r.y, r.z, r.w)
                return

    def _tick(self) -> None:
        if self._latest_pose is None:
            return
        px, py, pz, qx, qy, qz, qw = self._latest_pose
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "follower/odom"
        msg.child_frame_id  = "chassis"
        msg.pose.pose.position.x = px + WORLD_ORIGIN_OFFSET[0]
        msg.pose.pose.position.y = py + WORLD_ORIGIN_OFFSET[1]
        msg.pose.pose.position.z = pz
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        # Twist left zero — the BT we're feeding doesn't read it.
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = WorldOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
