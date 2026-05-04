"""Oracle camera — geometric ground-truth detection that uses the SAME
rendered camera as DAM4SAM, so the two are directly comparable.

Pipeline:
  1. Follower pose from /gz_pose_truth (SceneBroadcaster -> bridge).
  2. Leader pose: replayed from /clock + the trajectory waypoints below
     (Fortress doesn't expose <actor> entity poses on any topic; the
     trajectory is deterministic and lives in sim/worlds/empty.world).
  3. Camera intrinsics from /follower/camera/camera_info.
  4. Project the leader's 3D position into the camera_optical_frame using
     the camera's actual pose on the chassis. Accept the detection iff
     the projected pixel (u, v) is inside the image AND depth > min_clip.
  5. Emit body-frame (x, y) on /follower/camera/detections.

This way the oracle's FOV is *exactly* what the gz camera renders, and
DAM4SAM (which runs on the rendered RGB) and the oracle agree on what
"in view" means.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Quaternion
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


PUBLISH_RATE_HZ = 20.0
MAX_RANGE_M    = 6.0      # match the 2D sim's CameraDetector range gate

FOLLOWER_FRAME = "follower"

# Camera mount on the chassis — must match sim/worlds/empty.world's
# rgbd_camera sensor pose (camera at chassis-relative (0.23, 0, 0.05);
# chassis itself sits at world z=0.10; so camera is at world z=0.15).
CAM_OFFSET_X_BODY = 0.23   # camera_link x in follower body frame
CAM_OFFSET_Z_BODY = 0.15   # camera_link z in world frame (≈ body frame z)
CAM_MIN_DEPTH_M   = 0.10   # near clip from the rgbd_camera in the SDF

# Leader trajectory (mirrors empty.world::<actor name="leader">).
# (t_seconds, x, y, z). z = mesh body-center height ≈ 0.50 m for the
# Mingfei actor, used for the projection's vertical component.
LEADER_Z = 0.50
LEADER_WAYPOINTS = [
    ( 0.0,  3.0,  0.0),
    ( 6.0,  3.0,  4.0),
    ( 8.0,  3.0,  4.0),
    (17.0, -3.0,  4.0),
    (19.0, -3.0,  4.0),
    (28.0, -3.0, -2.0),
    (30.0, -3.0, -2.0),
    (39.0,  3.0, -2.0),
    (41.0,  3.0, -2.0),
    (44.0,  3.0,  0.0),
]
LEADER_LOOP_PERIOD = LEADER_WAYPOINTS[-1][0]


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def leader_xy_at(t_sec: float) -> tuple[float, float]:
    t = t_sec % LEADER_LOOP_PERIOD
    for i in range(len(LEADER_WAYPOINTS) - 1):
        t0, x0, y0 = LEADER_WAYPOINTS[i]
        t1, x1, y1 = LEADER_WAYPOINTS[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return x0, y0
            a = (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return LEADER_WAYPOINTS[-1][1], LEADER_WAYPOINTS[-1][2]


class OracleCamera(Node):
    def __init__(self) -> None:
        super().__init__("oracle_camera")
        self.follower_xyy: tuple[float, float, float] | None = None
        self.sim_time_sec: float | None = None
        self.K: np.ndarray | None = None
        self.image_w: int = 0
        self.image_h: int = 0

        self.create_subscription(TFMessage, "/gz_pose_truth", self._on_poses, 50)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
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
            if tr.child_frame_id == FOLLOWER_FRAME:
                t = tr.transform.translation
                self.follower_xyy = (t.x, t.y, yaw_from_quat(tr.transform.rotation))
                return

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time_sec = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.image_w = msg.width
        self.image_h = msg.height

    def _debug_log(self) -> None:
        if self.follower_xyy is None:
            self.get_logger().warn("oracle: no follower pose yet")
            return
        if self.sim_time_sec is None:
            self.get_logger().warn("oracle: no /clock yet")
            return
        if self.K is None:
            self.get_logger().warn("oracle: no /follower/camera/camera_info yet")
            return
        lx, ly = leader_xy_at(self.sim_time_sec)
        fx, fy, fyaw = self.follower_xyy
        d = math.hypot(lx - fx, ly - fy)
        self.get_logger().info(
            f"oracle: t={self.sim_time_sec:.1f}s "
            f"F=({fx:+.2f},{fy:+.2f},{math.degrees(fyaw):+.0f}°) "
            f"L=({lx:+.2f},{ly:+.2f}) d={d:.2f}m emits={self._emit_count}/{self._tick_count}")

    def _tick(self) -> None:
        self._tick_count += 1
        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "follower/base_link"

        if (self.follower_xyy is None or self.sim_time_sec is None
                or self.K is None):
            self.pub.publish(out)
            return

        # 1. Leader pose (world).
        lx, ly = leader_xy_at(self.sim_time_sec)
        # 2. Follower pose (world).
        fx, fy, fyaw = self.follower_xyy
        # 3. Leader in follower body frame (x-forward, y-left, z-up).
        c, s = math.cos(fyaw), math.sin(fyaw)
        body_x =  c * (lx - fx) + s * (ly - fy)
        body_y = -s * (lx - fx) + c * (ly - fy)
        body_z = LEADER_Z

        # 4. Range gate first (cheap), matches the 2D sim's 6 m guard.
        rng = math.hypot(body_x, body_y)
        if rng > MAX_RANGE_M:
            self.pub.publish(out)
            return

        # 5. Transform body -> camera_link (translate forward + up).
        cl_x = body_x - CAM_OFFSET_X_BODY
        cl_y = body_y
        cl_z = body_z - CAM_OFFSET_Z_BODY

        # 6. camera_link (REP-103, x-fwd, y-left, z-up) -> camera_optical
        #    (z-fwd, x-right, y-down):
        opt_x = -cl_y
        opt_y = -cl_z
        opt_z =  cl_x
        if opt_z <= CAM_MIN_DEPTH_M:
            self.pub.publish(out)
            return

        # 7. Project through K. Accept iff inside image bounds.
        u = self.K[0, 0] * opt_x / opt_z + self.K[0, 2]
        v = self.K[1, 1] * opt_y / opt_z + self.K[1, 2]
        if not (0.0 <= u < self.image_w and 0.0 <= v < self.image_h):
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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
