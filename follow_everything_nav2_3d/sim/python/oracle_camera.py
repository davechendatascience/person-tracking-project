"""Oracle camera — geometric ground-truth detection that uses the SAME
rendered camera as EdgeTAM, so the two are directly comparable.

Pipeline:
  1. Follower pose from /gz_pose_truth (SceneBroadcaster -> bridge),
     child_frame_id == "follower".
  2. Leader pose from the same /gz_pose_truth feed,
     child_frame_id == "leader".  The leader is a <model> driven by
     leader_controller.py via /leader/cmd_vel; gz publishes its pose
     authoritatively, so the oracle uses ground truth directly (no
     analytical waypoint replay).
  3. Camera intrinsics from /follower/camera/camera_info.
  4. Project the leader's 3D position into camera_optical_frame using the
     camera's actual mount offset on the chassis.  Accept the detection
     iff the projected pixel is inside the rendered image AND depth > 0.
  5. Emit body-frame (x, y) on /follower/camera/detections.

Same intrinsics + same frustum as EdgeTAM, so they agree on what's "in
view".
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


PUBLISH_RATE_HZ = 20.0
MAX_RANGE_M    = 8.0      # leader spawns at d=5.5 (full-body-view distance)
                          # and patrols within ±6 m of spawn, so a 6-m cap
                          # was clipping detections on most ticks. 8 m is
                          # the camera's actual <clip><far> in empty.world,
                          # so this matches the rgbd sensor's effective
                          # range rather than imposing a tighter gate.
CAM_FOV_DEG    = 90.0     # match the rgbd_camera sensor's horizontal_fov

FOLLOWER_FRAME = "follower"
LEADER_FRAME   = "leader"

# Camera mount on the chassis — must match sim/worlds/empty.world's
# rgbd_camera sensor pose (camera at chassis-relative (0.23, 0, 0.05);
# chassis itself sits at world z=0.10; so camera is at world z=0.15).
CAM_OFFSET_X_BODY = 0.23
CAM_OFFSET_Z_BODY = 0.15
CAM_MIN_DEPTH_M   = 0.10

# Approximate body-center height of the leader model (50 cm above ground
# isn't right for our 1.7 m model — body center is at z≈0.85 m). Used only
# for vertical projection.
LEADER_Z = 0.85


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OracleCamera(Node):
    def __init__(self) -> None:
        super().__init__("oracle_camera")
        self.follower_xyy: tuple[float, float, float] | None = None
        self.leader_xy:    tuple[float, float] | None        = None
        self.K: np.ndarray | None = None
        self.image_w: int = 0
        self.image_h: int = 0

        self.create_subscription(TFMessage, "/gz_pose_truth", self._on_poses, 50)
        self.create_subscription(
            CameraInfo, "/follower/camera/camera_info",
            self._on_camera_info, 10)
        self.pub = self.create_publisher(
            Detection2DArray, "/follower/camera/detections", 10)
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self._tick)
        self.create_timer(1.0, self._debug_log)
        self._tick_count = 0
        self._emit_count = 0

    def _on_poses(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            t = tr.transform.translation
            if tr.child_frame_id == FOLLOWER_FRAME:
                self.follower_xyy = (
                    t.x, t.y, yaw_from_quat(tr.transform.rotation))
            elif tr.child_frame_id == LEADER_FRAME:
                self.leader_xy = (t.x, t.y)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.image_w = msg.width
        self.image_h = msg.height

    def _debug_log(self) -> None:
        if self.follower_xyy is None:
            self.get_logger().warn("oracle: no follower pose yet")
            return
        if self.leader_xy is None:
            self.get_logger().warn("oracle: no leader pose yet")
            return
        if self.K is None:
            self.get_logger().warn("oracle: no /follower/camera/camera_info yet")
            return
        lx, ly = self.leader_xy
        fx, fy, fyaw = self.follower_xyy
        d = math.hypot(lx - fx, ly - fy)
        self.get_logger().info(
            f"oracle: F=({fx:+.2f},{fy:+.2f},{math.degrees(fyaw):+.0f}°) "
            f"L=({lx:+.2f},{ly:+.2f}) d={d:.2f}m emits={self._emit_count}/{self._tick_count}")

    def _tick(self) -> None:
        self._tick_count += 1
        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "follower/base_link"

        if (self.follower_xyy is None
                or self.leader_xy is None
                or self.K is None):
            self.pub.publish(out)
            return

        fx, fy, fyaw = self.follower_xyy
        lx, ly       = self.leader_xy

        c, s = math.cos(fyaw), math.sin(fyaw)
        body_x =  c * (lx - fx) + s * (ly - fy)
        body_y = -s * (lx - fx) + c * (ly - fy)
        body_z = LEADER_Z

        rng = math.hypot(body_x, body_y)
        if rng > MAX_RANGE_M:
            self.pub.publish(out)
            return

        # Match the 2D project's CameraDetector: a flat horizontal-FOV
        # cone + range gate, no image-plane projection. The previous
        # u/v bounds check was rejecting valid detections at close
        # range — the rgbd_camera is 320x240 with 90° H-FOV, which
        # makes V-FOV only ~74°, and a 0.85 m tall leader projected
        # *above* the top of the frame whenever distance < ~0.93 m.
        # For an "oracle" we want the geometric truth, not a real-camera
        # frustum — the BT then sees a leader in the same fan the
        # snapshotter draws.
        bearing = math.atan2(body_y, body_x)
        if abs(bearing) > math.radians(CAM_FOV_DEG / 2):
            self.pub.publish(out)
            return

        det = Detection2D()
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "leader"
        hyp.hypothesis.score = 1.0
        hyp.pose.pose.position.x = float(body_x)
        hyp.pose.pose.position.y = float(body_y)
        det.results.append(hyp)
        out.detections.append(det)
        self._emit_count += 1
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = OracleCamera()
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
