"""Strip the leader's own body from /follower/scan before the BT sees it.

The 360° lidar on the follower hits the leader actor mesh as the actor
walks past. Each hit lands on a cell in the BT's learned occupancy map
and tips its log-odds toward "blocked", so the leader leaves a moving
ghost-wall trail. That ghost causes two real problems:

  1) `Following.update()` calls `line_segment_clear(follower → orbit_goal)`
     against the learned-map-derived obstacles. The leader's own ghost
     fails the check, so Following hands off to PlannedRecovery (A*),
     even though the leader is right there in front of us.
  2) PlannedRecovery's `_approach_target` projects the goal *past* the
     blocked cell — so the planned star sits past where the leader was.

We can't touch the BT (its first-quadrant grid + ghost-handling are
upstream concerns). Instead, intercept the scan on the sim side:

  gz bridge  → /follower/scan_raw
  this node  → /follower/scan (filtered, leader cells stripped)

For each beam, compute its world-frame endpoint from the *unshifted*
follower pose, then null out (set to range_max) any range whose
endpoint is within LEADER_RADIUS_M of the leader's world-frame xy.
We keep the leader ground-truth pose from /gz_pose_truth — it's the
same SceneBroadcaster feed the oracle uses, so this filter is no
more "cheating" than the oracle camera already is.
"""
import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


LEADER_RADIUS_M = 0.45  # generous; actor mesh + lidar quantization


class LidarLeaderFilter(Node):
    def __init__(self) -> None:
        super().__init__("lidar_leader_filter")
        self.follower_xyy: tuple[float, float, float] | None = None
        self.leader_xy:    tuple[float, float] | None        = None
        self.create_subscription(TFMessage, "/gz_pose_truth", self._on_poses, 50)
        self.create_subscription(LaserScan, "/follower/scan_raw", self._on_scan, 10)
        self.pub = self.create_publisher(LaserScan, "/follower/scan", 10)
        self._dropped = 0
        self._kept = 0
        self.create_timer(2.0, self._log_stats)
        self.get_logger().info(
            f"lidar_leader_filter live; mask radius {LEADER_RADIUS_M:.2f} m around leader. "
            "in: /follower/scan_raw, out: /follower/scan")

    def _on_poses(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            t = tr.transform.translation
            if tr.child_frame_id == "leader":
                self.leader_xy = (t.x, t.y)
            elif tr.child_frame_id == "follower":
                # yaw from quat
                q = tr.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                self.follower_xyy = (t.x, t.y, math.atan2(siny_cosp, cosy_cosp))

    def _on_scan(self, msg: LaserScan) -> None:
        # No leader pose yet → pass through unchanged so the BT still gets
        # a steady stream of scans during the first few hundred ms.
        if self.follower_xyy is None or self.leader_xy is None:
            self.pub.publish(msg)
            return

        fx, fy, fyaw = self.follower_xyy
        lx, ly       = self.leader_xy
        r2 = LEADER_RADIUS_M * LEADER_RADIUS_M

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment  = msg.time_increment
        out.scan_time       = msg.scan_time
        out.range_min       = msg.range_min
        out.range_max       = msg.range_max
        out.intensities     = list(msg.intensities)

        ranges = list(msg.ranges)
        n = len(ranges)
        a = msg.angle_min
        for i in range(n):
            r = ranges[i]
            if math.isfinite(r) and msg.range_min < r < msg.range_max:
                # Endpoint in world frame — lidar is mounted on the
                # follower, scans rotate with the chassis, so beam world
                # angle is fyaw + a.
                ex = fx + r * math.cos(fyaw + a)
                ey = fy + r * math.sin(fyaw + a)
                dx, dy = ex - lx, ey - ly
                if dx * dx + dy * dy < r2:
                    # Beam endpoint is on the leader's body. Treat as
                    # max-range (no return) — same as how the oracle map
                    # would represent unoccupied space.
                    ranges[i] = float("inf")
                    self._dropped += 1
                else:
                    self._kept += 1
            a += msg.angle_increment
        out.ranges = ranges
        self.pub.publish(out)

    def _log_stats(self) -> None:
        total = self._dropped + self._kept
        if total == 0:
            return
        pct = 100.0 * self._dropped / total
        self.get_logger().info(
            f"lidar_leader_filter: stripped {self._dropped}/{total} beams "
            f"({pct:.1f}%) so far")


def main() -> None:
    rclpy.init()
    node = LidarLeaderFilter()
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
