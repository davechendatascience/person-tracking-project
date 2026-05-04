"""
Autonomous leader: picks random reachable goals in the CONTINUOUS world and
follows A*-planned paths (planner uses fine grid internally only). Speed is
capped at 1.0 m/s so the follower can keep up. DO NOT MODIFY.
"""
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry

from sim.world import load_map, MAX_LIN_VEL, MAX_ANG_VEL, ROBOT_RADIUS
from sim.geometry import circle_collides_aabbs
from sim.planner import build_planning_grid, astar, smooth_path

PEDESTRIAN_RADIUS = 0.25  # mirror sim/world.py value


GOAL_TOLERANCE = 0.6
WAYPOINT_TOLERANCE = 0.5
GOAL_TIMEOUT_S = 45.0
ROTATE_KP = 1.6
LIN_KP = 0.9
SAFE_LIN_VEL = 1.0
# Plan with extra inflation so the path keeps the robot well clear of obstacle
# corners — important on cluttered/forest maps where 0.15m clearance proved
# tight enough for the controller to drift into a snag.
INFLATE_RADIUS = 0.6
PLAN_RES = 0.2
BACKUP_TICKS = 4   # ~0.2s; just enough to clear the wedge, not retreat


def _clearance_to_obstacles(x, y, obstacles):
    """Distance from (x,y) to the closest point on the closest AABB."""
    best = float("inf")
    for b in obstacles:
        cx, cy = b.closest_point(x, y)
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if d2 < best:
            best = d2
    return math.sqrt(best)


def random_free_pose(rng, obstacles, world_size):
    """Sample uniformly until we hit a free pose with enough clearance to
    not be jammed against a wall corner. STOCHASTIC: each call returns a
    different goal (so _replan's 40-try loop actually explores 40
    candidates). The clearance threshold scales down if the map is too
    tight to find a 0.8m-clear pose, so corridor-style maps still work."""
    Wm, Hm = world_size
    for clear_min in (0.8, 0.5, 0.0):
        for _ in range(400):
            x = rng.uniform(2.0, Wm - 2.0)
            y = rng.uniform(2.0, Hm - 2.0)
            if circle_collides_aabbs(x, y, INFLATE_RADIUS, obstacles):
                continue
            clear = _clearance_to_obstacles(x, y, obstacles)
            if clear >= clear_min:
                return (x, y)
    raise RuntimeError("Could not sample a free pose for the leader.")


class LeaderPatrol(Node):
    def __init__(self, map_path: str, seed: int = 0):
        super().__init__("leader_patrol")
        self.np_rng = np.random.default_rng(seed + 1000)
        _, self.obstacles, _, _, self.world_size = load_map(map_path)
        self.plan_grid = build_planning_grid(self.obstacles, self.world_size,
                                              radius_m=INFLATE_RADIUS,
                                              resolution=PLAN_RES)
        self.cur = None
        self.prev_xy = None
        self.path = []
        self.goal_t = 0.0
        self.last_replan_t = 0.0
        self.blocked_ticks = 0
        self.stall_ticks = 0
        self.backup_ticks_left = 0
        self.pedestrians = []  # list of (x, y) — ground-truth from sim
        self.create_subscription(Odometry, "/leader/odom", self._on_odom, 10)
        self.create_subscription(PoseArray, "/pedestrians/poses",
                                 self._on_peds, 10)
        self.pub = self.create_publisher(Twist, "/leader/cmd_vel", 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"Leader patrol up. plan grid {self.plan_grid.shape}, "
            f"{int(self.plan_grid.sum())} blocked cells.")

    def _on_odom(self, msg):
        ori = msg.pose.pose.orientation
        yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y),
                         1 - 2 * (ori.y * ori.y + ori.z * ori.z))
        self.cur = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def _on_peds(self, msg):
        self.pedestrians = [(p.position.x, p.position.y) for p in msg.poses]

    def _ped_aware_grid(self, inflate_m):
        """Static plan grid + pedestrian positions inflated to `inflate_m`."""
        if not self.pedestrians:
            return self.plan_grid
        grid = self.plan_grid.copy()
        H, W = grid.shape
        r_cells = int(math.ceil(inflate_m / PLAN_RES))
        r2 = r_cells * r_cells
        for px, py in self.pedestrians:
            ci = int(px / PLAN_RES)
            cj = int(py / PLAN_RES)
            for dj in range(-r_cells, r_cells + 1):
                jj = cj + dj
                if jj < 0 or jj >= H:
                    continue
                for di in range(-r_cells, r_cells + 1):
                    ii = ci + di
                    if ii < 0 or ii >= W:
                        continue
                    if dj * dj + di * di <= r2:
                        grid[jj, ii] = True
        return grid

    def _replan(self):
        """Pick a fresh random goal and route to it on the STATIC map only.
        The leader has full ground-truth knowledge of static obstacles, so
        A* always finds a path between any two free cells. Pedestrians are
        intentionally NOT in the planning grid — they're Brownian and will
        wander on; planning around their current cells would just produce
        long detours around what amounts to noise. World physics (bounce
        on collision) handles any actual ped contact during execution."""
        if self.cur is None:
            return
        for _ in range(40):
            gx, gy = random_free_pose(self.np_rng, self.obstacles,
                                      self.world_size)
            if math.hypot(gx - self.cur[0], gy - self.cur[1]) < 6.0:
                continue
            path = astar(self.plan_grid, (self.cur[0], self.cur[1]), (gx, gy),
                          resolution=PLAN_RES)
            if path and len(path) > 1:
                self.path = smooth_path(path, self.obstacles,
                                        clearance=ROBOT_RADIUS + 0.05)[1:]
                self.goal_t = time.time()
                self.last_replan_t = time.time()
                return
        self.get_logger().warn("Failed to find a reachable random goal.")

    def _tick(self):
        if self.cur is None:
            return
        cx, cy, cyaw = self.cur

        # Replan ONLY once after stuck or goal-reached; if the prior replan
        # already ran in the last second, just wait — re-running A* every
        # tick when no path exists wastes CPU and floods the log.
        if not self.path:
            if (time.time() - self.last_replan_t) > 1.0:
                self._replan()
                self.last_replan_t = time.time()
            return
        while self.path and math.hypot(self.path[0][0] - cx, self.path[0][1] - cy) < WAYPOINT_TOLERANCE:
            self.path.pop(0)
        if not self.path:
            self._replan()
            return
        if (time.time() - self.goal_t) > GOAL_TIMEOUT_S:
            self._replan()
            return

        gx, gy = self.path[0]
        dx, dy = gx - cx, gy - cy
        bearing = math.atan2(dy, dx)
        err = (bearing - cyaw + math.pi) % (2 * math.pi) - math.pi
        w = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, ROTATE_KP * err))
        v_desired = min(SAFE_LIN_VEL, LIN_KP * math.hypot(dx, dy)) if abs(err) < 0.7 else 0.0
        v = v_desired

        # Predicted-block check considers ONLY static obstacles. Peds are
        # transient — if one's in our path, we walk through (the world's
        # bounce-on-collision physics keeps us out of it; the actual-stall
        # detector below catches a true wedge). Treating peds as plan
        # blockers caused chronic stuck-loops in dense crowds.
        nx = cx + v * math.cos(cyaw) * 0.15
        ny = cy + v * math.sin(cyaw) * 0.15
        predicted_block = v_desired > 0.05 and circle_collides_aabbs(
            nx, ny, ROBOT_RADIUS, self.obstacles)
        actual_stall = False
        if self.prev_xy is not None and v_desired > 0.05:
            moved = math.hypot(cx - self.prev_xy[0], cy - self.prev_xy[1])
            if moved < 0.01:
                self.stall_ticks += 1
                actual_stall = self.stall_ticks > 6  # ~0.3s of no progress
            else:
                self.stall_ticks = 0
        self.prev_xy = (cx, cy)

        if predicted_block or actual_stall:
            v = 0.0
            self.blocked_ticks += 1
            # Replan ONCE after being stuck >0.2s, then wait 1s before the
            # next replan attempt regardless of stuck state. The cooldown
            # gates the next attempt via last_replan_t in the no-path branch.
            if (self.blocked_ticks > 4
                    and (time.time() - self.last_replan_t) > 1.0):
                self.path = []
                self.blocked_ticks = 0
                self.stall_ticks = 0
                return
        else:
            self.blocked_ticks = 0

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub.publish(msg)


def main():
    rclpy.init()
    map_path = os.environ.get("SIM_MAP", "sim/maps/empty.txt")
    seed = int(os.environ.get("SIM_SEED", "0"))
    node = LeaderPatrol(map_path, seed)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
