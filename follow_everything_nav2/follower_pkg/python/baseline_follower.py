"""
Baseline follower: drives toward the visible leader and uses a thin LiDAR
safety layer to avoid head-on collisions. Still intentionally dumb — no
memory of where the leader went, no path planning, no recovery search.

The autoresearch agent should REPLACE this with a Nav2 BT-based controller.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray


# LiDAR safety geometry. Front cone = ±FRONT_HALFCONE_DEG; if anything in that
# cone is closer than DANGER_M, we hard-stop forward motion and rotate toward
# the side with more room.
FRONT_HALFCONE_DEG = 35.0
DANGER_M = 0.55
SLOW_M = 1.20


class BaselineFollower(Node):
    def __init__(self):
        super().__init__("baseline_follower")
        self.create_subscription(Detection2DArray, "/follower/camera/detections",
                                 self._on_detect, 10)
        self.create_subscription(LaserScan, "/follower/scan", self._on_scan, 10)
        self.pub = self.create_publisher(Twist, "/follower/cmd_vel", 10)
        self.last_target = None  # body-frame (x, y) of leader
        self.last_scan = None
        self.create_timer(0.05, self._tick)
        self.get_logger().info("Baseline follower up.")

    def _on_detect(self, msg: Detection2DArray):
        if not msg.detections:
            self.last_target = None
            return
        p = msg.detections[0].results[0].pose.pose.position
        self.last_target = (p.x, p.y)

    def _on_scan(self, msg: LaserScan):
        self.last_scan = msg

    def _front_clearance(self):
        """Returns (min range in front cone, side bias).
        side bias > 0 => more room on left, < 0 => right."""
        if self.last_scan is None or not self.last_scan.ranges:
            return float("inf"), 0.0
        s = self.last_scan
        n = len(s.ranges)
        cone = math.radians(FRONT_HALFCONE_DEG)
        max_r = s.range_max if s.range_max > 0 else 8.0
        front_min = max_r
        left_min = max_r
        right_min = max_r
        for i, r in enumerate(s.ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            ang = s.angle_min + i * s.angle_increment
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if -cone <= ang <= cone:
                front_min = min(front_min, r)
                if ang > 0:
                    left_min = min(left_min, r)
                else:
                    right_min = min(right_min, r)
        return front_min, left_min - right_min

    def _tick(self):
        cmd = Twist()
        if self.last_target is None:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.4  # spin to search
            self.pub.publish(cmd)
            return

        x, y = self.last_target
        d = math.hypot(x, y)
        bearing = math.atan2(y, x)
        cmd.angular.z = max(-1.0, min(1.0, 1.5 * bearing))
        stand_off = max(0.0, d - 2.0)
        cmd.linear.x = max(0.0, min(1.0, stand_off))

        # LiDAR safety: brake near obstacles, swing to the freer side.
        front_min, side_bias = self._front_clearance()
        if front_min < DANGER_M:
            cmd.linear.x = 0.0
            # Steer hard toward the side with more clearance. side_bias>0 => left.
            cmd.angular.z = 1.2 if side_bias >= 0 else -1.2
        elif front_min < SLOW_M:
            scale = (front_min - DANGER_M) / (SLOW_M - DANGER_M)
            cmd.linear.x *= max(0.1, scale)
            # Nudge toward freer side proportional to crowding.
            nudge = (1.0 - scale) * 0.6 * (1.0 if side_bias >= 0 else -1.0)
            cmd.angular.z = max(-1.0, min(1.0, cmd.angular.z + nudge))

        self.pub.publish(cmd)


def main():
    rclpy.init()
    n = BaselineFollower()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
