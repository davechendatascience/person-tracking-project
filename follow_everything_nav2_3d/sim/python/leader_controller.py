"""Leader patrol controller — drives /leader/cmd_vel.

Single mode: random-goal patrol, mirroring the 2D project's leader.py.
Pick a random reachable point in a 4 m box around the spawn, plan a
path to it (A* + smooth_path when obstacles exist; straight line when
not), drive the path with a proportional controller, repeat on arrival.

Why not the old rectangular patrol for empty: it produced 90° corners
at every waypoint transition, which forced the leader to pivot in
place. The follower's trailing-orbit goal jumped 90° around the leader
each pivot, the leader exited the FOV during the follower's catch-up
turn, and SpiralExpand (which always rotates CCW) sometimes spun the
wrong way to reacquire. The 2D sim doesn't hit this because its leader
already uses A*-smoothed random goals — so we now match it.

Closed-loop on /gz_pose_truth (child_frame_id=="leader") for current
pose, /clock for sim time, publishes Twist on /leader/cmd_vel. The gz
VelocityControl plugin on the leader model applies it kinematically.
"""
import math
import os
import random
import sys
from pathlib import Path

# Bring in the 2D project's planner — its build_planning_grid + astar +
# smooth_path are exactly what we want, and they only depend on numpy.
sys.path.insert(0, "/opt/follow_everything_nav2")

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

from geometry_msgs.msg import Twist, Quaternion  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from sim.geometry import merged_aabbs_from_grid, circle_collides_aabbs  # noqa: E402
from sim.planner import build_planning_grid, astar, smooth_path  # noqa: E402


PUB_RATE_HZ   = 20.0
WALK_SPEED    = 0.7
KW            = 2.0
MAX_W         = 1.5
MAX_V         = 0.9
WP_REACHED_M  = 0.30  # advance to next waypoint when within this distance
GOAL_REACHED_M = 0.40 # current goal reached -> pick a new one
LEADER_RADIUS = 0.30  # used to inflate obstacles for the planner
SAMPLE_RADIUS_M = 6.0 # random-goal box half-extent around spawn
MIN_GOAL_DIST_M = 3.0 # reject goals closer than this to current pose
MAX_TURN_DEG    = 60.0 # reject goals that would require sharper than this turn

LEADER_FRAME = "leader"
EP_MAP       = os.environ.get("EP_MAP", "empty")
MAPS_DIR     = Path("/opt/follow_everything_nav2/sim/maps")
CELL_M       = 0.5


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def load_map_obstacles(map_name: str):
    """Returns (obstacles, world_size_xy) — obstacles is a list[AABB]."""
    path = MAPS_DIR / f"{map_name}.txt"
    if map_name == "empty" or not path.exists():
        return [], (15.0, 15.0)
    rows = []
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("//"):
                continue
            rows.append(s)
    H, W = len(rows), max(len(r) for r in rows)
    grid = np.zeros((H, W), dtype=bool)
    for j, row in enumerate(rows):
        for i, ch in enumerate(row.ljust(W)):
            if ch == "#":
                grid[j, i] = True
    return merged_aabbs_from_grid(grid, CELL_M), (W * CELL_M, H * CELL_M)


class LeaderController(Node):
    def __init__(self) -> None:
        super().__init__("leader_controller")
        self.cur_xyy: tuple[float, float, float] | None = None
        self.sim_time_sec: float | None = None
        self.patrol_origin: tuple[float, float] | None = None

        # Planner state (only used when map != 'empty').
        self.obstacles, self.world_size = load_map_obstacles(EP_MAP)
        self.use_planner = bool(self.obstacles)
        if self.use_planner:
            self.plan_grid = build_planning_grid(
                self.obstacles, self.world_size,
                radius_m=LEADER_RADIUS, resolution=CELL_M / 2)
            self.get_logger().info(
                f"planner mode (A*): map={EP_MAP} world={self.world_size} "
                f"obstacles={len(self.obstacles)} "
                f"grid={self.plan_grid.shape} blocked={int(self.plan_grid.sum())}")
        else:
            self.plan_grid = None
            self.get_logger().info(
                f"random-goal mode (no obstacles): map={EP_MAP} "
                f"sampling within ±{SAMPLE_RADIUS_M:.1f}m of spawn")

        self.path: list[tuple[float, float]] = []  # current waypoint sequence
        self.path_idx = 0
        self.rng = random.Random(0)

        self.create_subscription(
            TFMessage, "/gz_pose_truth", self._on_poses, 50)
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.pub = self.create_publisher(Twist, "/leader/cmd_vel", 10)
        self.create_timer(1.0 / PUB_RATE_HZ, self._tick)

    # ------------------------------------------------------------------
    def _on_poses(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            if tr.child_frame_id == LEADER_FRAME:
                t = tr.transform.translation
                self.cur_xyy = (
                    t.x, t.y, yaw_from_quat(tr.transform.rotation))
                if self.patrol_origin is None:
                    self.patrol_origin = (t.x, t.y)
                    self.get_logger().info(
                        f"patrol origin locked at ({t.x:.2f}, {t.y:.2f})")
                return

    def _on_clock(self, msg: Clock) -> None:
        self.sim_time_sec = msg.clock.sec + msg.clock.nanosec * 1e-9

    # ------------------------------------------------------------------
    def _pick_random_goal(self) -> tuple[float, float] | None:
        """Sample a random goal in a SAMPLE_RADIUS_M box around the spawn,
        clipped to the map interior, with progressive constraint relaxation.

        Three passes:
          1. strict — require min distance + max turn angle + obstacle clear.
          2. drop the turn constraint (sharper turns OK if needed).
          3. drop the distance constraint (any free spot will do).

        All three clip the sample to the (margin, Wm-margin) interior so we
        never hand astar a goal outside its grid (which would silently fail
        and freeze the leader)."""
        if self.patrol_origin is None or self.cur_xyy is None:
            return None
        ox, oy = self.patrol_origin
        cx, cy, cyaw = self.cur_xyy
        max_turn_rad = math.radians(MAX_TURN_DEG)
        Wm, Hm = self.world_size
        margin = LEADER_RADIUS + 0.1

        def _sample_one():
            gx = ox + self.rng.uniform(-SAMPLE_RADIUS_M, SAMPLE_RADIUS_M)
            gy = oy + self.rng.uniform(-SAMPLE_RADIUS_M, SAMPLE_RADIUS_M)
            gx = max(margin, min(Wm - margin, gx))
            gy = max(margin, min(Hm - margin, gy))
            return gx, gy

        for _ in range(100):
            gx, gy = _sample_one()
            dx, dy = gx - cx, gy - cy
            if math.hypot(dx, dy) < MIN_GOAL_DIST_M:
                continue
            target_yaw = math.atan2(dy, dx)
            yaw_err = abs(
                (target_yaw - cyaw + math.pi) % (2 * math.pi) - math.pi)
            if yaw_err > max_turn_rad:
                continue
            if self.use_planner and circle_collides_aabbs(
                    gx, gy, LEADER_RADIUS, self.obstacles):
                continue
            return (gx, gy)
        for _ in range(50):
            gx, gy = _sample_one()
            if math.hypot(gx - cx, gy - cy) < MIN_GOAL_DIST_M:
                continue
            if self.use_planner and circle_collides_aabbs(
                    gx, gy, LEADER_RADIUS, self.obstacles):
                continue
            return (gx, gy)
        for _ in range(50):
            gx, gy = _sample_one()
            if self.use_planner and circle_collides_aabbs(
                    gx, gy, LEADER_RADIUS, self.obstacles):
                continue
            return (gx, gy)
        return None

    def _replan_to(self, goal_xy: tuple[float, float]) -> None:
        if self.cur_xyy is None:
            return
        cx, cy, _ = self.cur_xyy
        if self.use_planner:
            path = astar(self.plan_grid, (cx, cy), goal_xy,
                         resolution=self.world_size[0] / self.plan_grid.shape[1])
            if not path:
                self.path = []
                return
            path = smooth_path(path, self.obstacles, clearance=LEADER_RADIUS)
        else:
            # No obstacles → straight line is the path.
            path = [(cx, cy), goal_xy]
        self.path = path
        self.path_idx = 1 if len(path) > 1 else 0  # skip the current pose

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        cmd = Twist()
        if (self.cur_xyy is None or self.sim_time_sec is None
                or self.patrol_origin is None):
            self.pub.publish(cmd)
            return

        cx, cy, cyaw = self.cur_xyy

        # Pick a fresh random goal if we have no path or finished it.
        if not self.path or self.path_idx >= len(self.path):
            goal = self._pick_random_goal()
            if goal is None:
                self.pub.publish(cmd)
                return
            self._replan_to(goal)
            if not self.path:
                self.pub.publish(cmd)
                return
        tx, ty = self.path[self.path_idx]
        # Advance through path waypoints when close.
        if math.hypot(tx - cx, ty - cy) < WP_REACHED_M:
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                return  # next tick will pick a new goal
            tx, ty = self.path[self.path_idx]

        # Proportional drive toward (tx, ty).
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            self.pub.publish(cmd)
            return
        target_yaw = math.atan2(dy, dx)
        yaw_err = (target_yaw - cyaw + math.pi) % (2 * math.pi) - math.pi
        w = max(-MAX_W, min(MAX_W, KW * yaw_err))
        v_target = min(MAX_V, max(WALK_SPEED, WALK_SPEED * dist))
        v = v_target * max(0.0, math.cos(yaw_err))
        cmd.linear.x  = v
        cmd.angular.z = w
        self.pub.publish(cmd)


def main() -> None:
    rclpy.init()
    node = LeaderController()
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
