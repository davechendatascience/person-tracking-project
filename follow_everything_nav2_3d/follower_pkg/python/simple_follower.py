"""Minimal proportional leader-follower for Phase 5.

No behavior tree, no pedestrians, no obstacles, no Nav2 — just consumes the
body-frame leader detection on /follower/camera/detections (whichever source
the launch args wire there: oracle in dev, EdgeTAM in primary mode) and
drives /follower/cmd_vel to keep ~TARGET_DIST metres behind the leader.

The follow_everything_nav2 project has a much richer BT-based follower; we'll
swap that in when we actually need recovery + obstacle avoidance. For empty
world with a single visible actor, this is enough to verify the contract
(detection -> motion) end-to-end.
"""
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray


# Tuned for our 0.25 m radius diff-drive (1.5 m/s, 1.5 rad/s) chasing the
# Mingfei walking actor at ~0.7 m/s.
TARGET_DIST = 1.5      # metres — stand-off behind the leader
DIST_DEADBAND = 0.1    # metres — hysteresis around TARGET_DIST
KV   = 1.0             # m/s per metre of (range - TARGET_DIST)
KW   = 1.6             # rad/s per radian of bearing error
MAX_V = 1.5
MAX_W = 1.5
LOSS_TIMEOUT = 0.5     # seconds without a detection -> stop


class SimpleFollower(Node):
    def __init__(self) -> None:
        super().__init__("simple_follower")
        self._last_detect_t: float | None = None
        self._last_xy: tuple[float, float] | None = None
        self._msgs_seen = 0
        self._leader_seen = 0
        self._last_v = 0.0
        self._last_w = 0.0

        self.create_subscription(
            Detection2DArray, "/follower/camera/detections",
            self._on_detect, 10)
        self.pub = self.create_publisher(Twist, "/follower/cmd_vel", 10)
        self.create_timer(0.05, self._tick)  # 20 Hz
        self.create_timer(1.0, self._debug_log)

        self.get_logger().info(
            f"simple_follower live — chasing /follower/camera/detections "
            f"(target {TARGET_DIST} m, max v={MAX_V} m/s w={MAX_W} rad/s).")

    def _debug_log(self) -> None:
        age = (time.time() - self._last_detect_t) if self._last_detect_t else float("inf")
        xy = self._last_xy if self._last_xy else (None, None)
        self.get_logger().info(
            f"sf: msgs={self._msgs_seen} leader_msgs={self._leader_seen} "
            f"last_xy={xy} age={age:.2f}s last_cmd=v={self._last_v:+.2f} w={self._last_w:+.2f}")

    def _on_detect(self, msg: Detection2DArray) -> None:
        self._msgs_seen += 1
        # An empty detections array == "leader not visible right now".
        # Treat this as immediate loss so the robot stops as soon as the
        # leader leaves FOV, instead of coasting on stale data for
        # LOSS_TIMEOUT seconds.
        leader_xy = None
        for det in msg.detections:
            for hyp in det.results:
                if hyp.hypothesis.class_id == "leader":
                    p = hyp.pose.pose.position
                    leader_xy = (float(p.x), float(p.y))
                    break
            if leader_xy is not None:
                break
        if leader_xy is not None:
            self._last_xy = leader_xy
            self._last_detect_t = time.time()
            self._leader_seen += 1
        else:
            # Latest oracle/EdgeTAM frame doesn't see the leader.
            self._last_xy = None
            self._last_detect_t = None

    def _tick(self) -> None:
        cmd = Twist()
        if (self._last_xy is None
                or self._last_detect_t is None
                or time.time() - self._last_detect_t > LOSS_TIMEOUT):
            self.pub.publish(cmd)  # zero
            return

        x, y = self._last_xy
        rng = math.hypot(x, y)
        bearing = math.atan2(y, x)

        # Angular: align with the bearing.
        w = max(-MAX_W, min(MAX_W, KW * bearing))

        # Linear: drive to TARGET_DIST. Don't push forward if we'd be heading
        # away from the leader (|bearing| near pi).
        err = rng - TARGET_DIST
        if abs(err) < DIST_DEADBAND:
            v = 0.0
        else:
            v = max(-MAX_V, min(MAX_V, KV * err))
        # Slow forward when we're not pointed at the leader yet.
        v *= max(0.0, math.cos(bearing))

        cmd.linear.x  = v
        cmd.angular.z = w
        self._last_v, self._last_w = v, w
        self.pub.publish(cmd)


def main() -> None:
    rclpy.init()
    node = SimpleFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
