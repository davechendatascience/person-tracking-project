"""Oracle camera bridge — Phase 3 safety net before DAM4SAM.

Reads ground-truth follower + actor leader poses from Gazebo's SceneBroadcaster
(bridged onto /gz_pose_truth as tf2_msgs/TFMessage) and publishes a
vision_msgs/Detection2DArray that exactly mirrors the 2D sim's CameraDetector:

    - 90° forward FOV cone   (sim/sensors.py CAMERA_FOV_DEG)
    - 6 m max range          (sim/sensors.py CAMERA_MAX_RANGE)
    - line-of-sight check    (skipped: empty world has no obstacles yet)
    - body-frame (x, y) of the leader, class_id="leader"

The follower stack consumes /follower/camera/detections without caring whether
the source is this oracle or the real DAM4SAM tracker — the topic contract is
the same, which is what makes Phase 3 work as a regression baseline.
"""
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Quaternion
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


CAMERA_FOV_DEG = 90.0
CAMERA_MAX_RANGE = 6.0
PUBLISH_RATE_HZ = 20.0

FOLLOWER_FRAME = "follower"
LEADER_FRAME = "leader"


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OracleCamera(Node):
    def __init__(self) -> None:
        super().__init__("oracle_camera")
        self.follower_xyy = None  # (x, y, yaw)
        self.leader_xy = None     # (x, y)

        self.create_subscription(
            TFMessage, "/gz_pose_truth", self._on_poses, 50)
        self.pub = self.create_publisher(
            Detection2DArray, "/follower/camera/detections", 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._tick)

    def _on_poses(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            child = tr.child_frame_id
            t = tr.transform.translation
            if child == FOLLOWER_FRAME:
                self.follower_xyy = (t.x, t.y, yaw_from_quat(tr.transform.rotation))
            elif child == LEADER_FRAME:
                self.leader_xy = (t.x, t.y)

    def _tick(self) -> None:
        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "follower/base_link"

        if self.follower_xyy is None or self.leader_xy is None:
            self.pub.publish(out)
            return

        fx, fy, fyaw = self.follower_xyy
        lx, ly = self.leader_xy
        dx, dy = lx - fx, ly - fy
        d = math.hypot(dx, dy)
        if d > CAMERA_MAX_RANGE:
            self.pub.publish(out)
            return

        bearing_world = math.atan2(dy, dx)
        rel = (bearing_world - fyaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(math.degrees(rel)) > CAMERA_FOV_DEG / 2.0:
            self.pub.publish(out)
            return

        # Body-frame transform: rotate world delta by -follower yaw.
        c, s = math.cos(-fyaw), math.sin(-fyaw)
        body_x = c * dx - s * dy
        body_y = s * dx + c * dy

        det = Detection2D()
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "leader"
        hyp.hypothesis.score = 1.0
        hyp.pose.pose.position.x = body_x
        hyp.pose.pose.position.y = body_y
        det.results.append(hyp)
        out.detections.append(det)
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = OracleCamera()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
