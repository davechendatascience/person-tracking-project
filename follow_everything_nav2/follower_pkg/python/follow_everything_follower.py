"""
Follow-Everything follower built as a py_trees behavior tree (v2).

Implements the paper's 5-state machine (Following / Chasing / Retreating /
Planning(=Lost) / Switching) over our simulated geometric perception.

DATA SOURCES (anti-cheat audit):
  * /follower/odom              — own pose (allowed)
  * /follower/scan              — own LiDAR (allowed)
  * /follower/camera/detections — leader's body-frame pose IFF visible (allowed)
  * sim/maps/<map>.txt          — static a-priori map for the planner. This is
                                  the moral equivalent of a Nav2 static-map
                                  layer loaded from a .yaml that came out of a
                                  prior SLAM run — NOT live ground-truth.
  * NEVER subscribes to /leader/pose_ground_truth, /leader/odom, or any other
    leader topic. The follower has no live information about the leader other
    than what its camera saw.

v2 improvements over v1:
  * Following: orbit point sits on the leader's trail (opposite velocity)
    instead of robot-bearing — follower retraces leader's path through corners.
  * Adaptive D_t: 0.6 + 0.4*|v_leader|, clipped [0.8, 2.0].
  * Planning: predictive recovery — target = last_seen + v_leader * dt with
    a 1 s horizon cap (rejects when the predicted point lands in an obstacle).
  * Sticky visibility: require 3 consecutive missed detections before
    declaring lost; removes single-frame occlusion thrash.
  * LiDAR brake tightened (0.45 / 1.0) so the follower can thread closer
    obstacles while chasing through clutter.
"""
import math
import os
import time
from collections import deque

import numpy as np
import py_trees
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray

from sim.world import ROBOT_RADIUS  # only the robot's own radius — config, not map
from sim.geometry import (
    AABB, circle_collides_aabbs, line_segment_clear, ray_aabb_distance,
    segment_hits_circle, merged_aabbs_from_grid,
)
from sim.planner import build_planning_grid, astar, smooth_path
from scipy.ndimage import binary_dilation, distance_transform_edt
import heapq


def astar_weighted(cost_grid, start_xy, goal_xy, resolution):
    """A* with per-cell costs. Cells with cost >= 1e6 are impassable.
    Step cost = (1 or sqrt(2)) * cell_cost. Heuristic = octile distance
    (admissible since min cell cost is 1.0). The follower uses this to
    prefer confirmed-free paths over paths through unknown territory."""
    H, W = cost_grid.shape
    si = int(start_xy[0] / resolution)
    sj = int(start_xy[1] / resolution)
    gi = int(goal_xy[0] / resolution)
    gj = int(goal_xy[1] / resolution)
    if not (0 <= sj < H and 0 <= si < W):
        return None
    if not (0 <= gj < H and 0 <= gi < W):
        return None
    if cost_grid[gj, gi] >= 1e6:
        return None

    def heur(a, b):
        dy = abs(a[0] - b[0])
        dx = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    s_cell = (sj, si)
    g_cell = (gj, gi)
    open_q = [(heur(s_cell, g_cell), 0.0, s_cell, None)]
    came = {}
    closed = set()
    while open_q:
        f, g, cur, parent = heapq.heappop(open_q)
        if cur in closed:
            continue
        closed.add(cur)
        came[cur] = parent
        if cur == g_cell:
            cells = []
            n = cur
            while n is not None:
                cells.append(n)
                n = came[n]
            cells.reverse()
            return [((c[1] + 0.5) * resolution, (c[0] + 0.5) * resolution)
                    for c in cells]
        cj, ci = cur
        for dj, di in nbrs:
            nj, ni = cj + dj, ci + di
            if not (0 <= nj < H and 0 <= ni < W):
                continue
            cell_cost = float(cost_grid[nj, ni])
            if cell_cost >= 1e6:
                continue
            base_step = math.sqrt(2) if (dj != 0 and di != 0) else 1.0
            step = base_step * cell_cost
            ng = g + step
            heapq.heappush(open_q, (
                ng + heur((nj, ni), g_cell), ng, (nj, ni), cur))
    return None


FOLLOW_RANGE = 3.5          # inside this -> Following; outside -> Chasing
D_T_MIN = 0.8               # tightest standoff when leader is slow
D_T_MAX = 2.0               # loosest standoff when leader is fast
D_T_VEL_GAIN = 0.4
D_T_BASE = 0.6
RETREAT_TRIGGER_DIST = 0.9
V_MAX_HARD = 1.4
V_MIN_CHASE = 0.3
ALPHA_V_VEL = 0.5           # paper Eq.4 alpha_1
ALPHA_V_DIST = 0.7          # paper Eq.4 alpha_2
LIDAR_FRONT_HALFCONE_DEG = 30.0
LIDAR_DANGER_M = 0.45
LIDAR_SLOW_M = 1.00
PLAN_RES = 0.2
INFLATE_RADIUS = 0.4
LOST_REPLAN_S = 0.8         # replan more aggressively while lost
WAYPOINT_TOL = 0.4
PRED_HORIZON_S = 1.0        # cap on predictive extrapolation
MISS_TICKS_FOR_LOST = 3     # debounce single-frame occlusions
LOS_REPLAN_S = 0.6          # ClearLOS replan interval (when no live plan)
LOS_PLAN_COMMIT_S = 4.0     # safety cap on commit-to-plan
CORNER_MARGIN = 0.55        # how far past obstacle corner to put a via-point
HEADING_PENALTY_W = 0.6     # weight on heading-consistency vs path length
PEDESTRIAN_TTL_S = 1.5      # forget pedestrian sightings after this
PEDESTRIAN_RADIUS = 0.3     # treated as a circle for occlusion / via-points


# ---------------------------------------------------------------------------
# Online occupancy mapping from LiDAR. Replaces the static-map cheat:
# the follower no longer loads sim/maps/X.txt — it builds its grid from
# accumulated LiDAR scans, like a real Nav2 robot using slam_toolbox in
# pure-localization mode (or the local-costmap obstacle layer).
#
# We do keep WORLD BOUNDS as known config (the work-area dimensions), so
# we can pre-allocate the log-odds grid. Bounds-but-not-contents is the
# realistic deployment assumption — the follower has been told "you operate
# in this 15x15m area" without being told what's in it.
# ---------------------------------------------------------------------------
# Symmetric log-odds with low clamp: a cell becomes "blocked" only after
# many consecutive hits, and decays just as fast on free-space passes.
# Effect: a 1s-stationary pedestrian creates a cell that ghosts for ~0.25s
# after the ped leaves, then fades. Static walls stay maxed out because
# the LiDAR sees them every scan.
LOG_ODDS_HIT = 0.4
LOG_ODDS_MISS = -0.4
LOG_ODDS_CLAMP = 2.0
LOG_ODDS_THRESH_BLOCKED = 0.3   # any hit (+0.4) confirms wall on next refresh
LOG_ODDS_THRESH_FREE = -0.3     # any miss (-0.4) confirms free
LOG_ODDS_THRESH = LOG_ODDS_THRESH_BLOCKED   # backwards-compat alias
MAP_RES = PLAN_RES


class LearnedMap:
    def __init__(self, world_size):
        self.world_size = world_size
        self.W_cells = int(math.ceil(world_size[0] / MAP_RES))
        self.H_cells = int(math.ceil(world_size[1] / MAP_RES))
        self.log_odds = np.zeros((self.H_cells, self.W_cells), dtype=np.float32)
        # Cached AABB list and plan grid; refreshed from log-odds on demand.
        self._cache_t = -1.0
        self._cached_aabbs = []
        self._cached_plan_grid = None

    def update_from_scan(self, scan, robot_x, robot_y, robot_yaw,
                         leader_world=None):
        """Update log-odds from one LiDAR scan.

        Pedestrians: filtered purely by temporal decay (symmetric log-odds
        with low clamp — moving peds fade within ~0.25s of leaving a cell;
        static walls stay maxed because they're hit every scan).

        Leader: special. The follower stays close to the leader for long
        stretches, so the leader's LiDAR cell would saturate and ghost as
        a permanent obstacle. We mask the angular cone around the camera-
        detected leader. Sensor fusion is fair for the LEADER specifically
        because the camera is the leader's dedicated sensor — this is the
        same as 'I see a friend, don't write them into my static map'."""
        if scan is None or not scan.ranges:
            return
        max_r = scan.range_max if scan.range_max > 0 else 8.0
        step_m = MAP_RES * 0.5
        # Pre-compute leader cone if visible.
        leader_cone = None
        if leader_world is not None:
            ldx = leader_world[0] - robot_x
            ldy = leader_world[1] - robot_y
            ld = math.hypot(ldx, ldy)
            if ld > 1e-3:
                leader_bearing = math.atan2(ldy, ldx)
                leader_half_ext = math.atan2(0.30, ld) + 0.05  # leader radius+margin
                leader_cone = (leader_bearing, leader_half_ext)
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            ang = scan.angle_min + i * scan.angle_increment
            world_th = robot_yaw + ang
            world_th_n = (world_th + math.pi) % (2 * math.pi) - math.pi
            if leader_cone is not None:
                d_ang = abs((world_th_n - leader_cone[0] + math.pi) %
                            (2 * math.pi) - math.pi)
                if d_ang < leader_cone[1]:
                    continue
            cos_t, sin_t = math.cos(world_th), math.sin(world_th)
            stop = min(r, max_r) - step_m
            t = step_m
            while t < stop:
                wx = robot_x + t * cos_t
                wy = robot_y + t * sin_t
                ci = int(wx / MAP_RES)
                cj = int(wy / MAP_RES)
                if 0 <= cj < self.H_cells and 0 <= ci < self.W_cells:
                    self.log_odds[cj, ci] = max(
                        -LOG_ODDS_CLAMP,
                        self.log_odds[cj, ci] + LOG_ODDS_MISS)
                t += step_m
            if r < max_r - step_m:
                ex = robot_x + r * cos_t
                ey = robot_y + r * sin_t
                ci = int(ex / MAP_RES)
                cj = int(ey / MAP_RES)
                if 0 <= cj < self.H_cells and 0 <= ci < self.W_cells:
                    self.log_odds[cj, ci] = min(
                        LOG_ODDS_CLAMP,
                        self.log_odds[cj, ci] + LOG_ODDS_HIT)

    def _refresh_cache(self):
        """Recompute AABB list, binary plan grid, and weighted COST GRID
        from log-odds. The cost grid is used by the follower's weighted
        A* so it prefers paths through CONFIRMED-FREE cells over paths
        through UNKNOWN cells. Without this, A* takes long detours
        through unmapped territory because it optimistically treats
        unknown == free, then commits to a path that runs into actual
        unmapped walls."""
        binary = self.log_odds > LOG_ODDS_THRESH_BLOCKED
        # merged_aabbs_from_grid uses image-coords (row 0 = TOP of world,
        # ymax = (H-j)*cell). Our log_odds uses row j -> y = j*MAP_RES
        # (NO flip). Without flipud here, every observed wall comes out
        # mirrored across the world Y midline — A* then routes through
        # phantom-free space where the real wall is. This was the root
        # cause of "A* happily passes through walls". Flip in, no flip out.
        self._cached_aabbs = merged_aabbs_from_grid(
            np.flipud(binary), MAP_RES)
        # Build planning grid directly from the binary log_odds via
        # morphological dilation. This (a) avoids any chance of further
        # coord-frame drift through the AABB round-trip, and (b) fattens
        # each LiDAR endpoint hit into a disk so 1-cell-thin walls
        # cannot be slipped between by diagonal A* moves.
        r_cells = max(1, int(math.ceil(INFLATE_RADIUS / PLAN_RES)))
        yy, xx = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
        struct = (xx * xx + yy * yy) <= r_cells * r_cells
        plan_grid = binary_dilation(binary, structure=struct)
        # World-bounds margin (robot can't stand at world edge).
        Wm, Hm = self.world_size
        nx = int(math.ceil(Wm / PLAN_RES))
        ny = int(math.ceil(Hm / PLAN_RES))
        if plan_grid.shape != (ny, nx):
            pg = np.zeros((ny, nx), dtype=bool)
            h_copy = min(plan_grid.shape[0], ny)
            w_copy = min(plan_grid.shape[1], nx)
            pg[:h_copy, :w_copy] = plan_grid[:h_copy, :w_copy]
            plan_grid = pg
        margin = max(1, int(math.ceil(INFLATE_RADIUS / PLAN_RES)))
        plan_grid[:margin, :] = True
        plan_grid[-margin:, :] = True
        plan_grid[:, :margin] = True
        plan_grid[:, -margin:] = True
        self._cached_plan_grid = plan_grid
        # Cost grid with CLEARANCE PENALTY. For each free cell, its cost
        # rises as it gets closer to the nearest blocked cell — A* will
        # naturally pick wide-open corridors over narrow ones when both
        # reach the goal. Done via Euclidean distance transform.
        # Beyond PREFERRED_CLEARANCE_CELLS (~1m) cells cost 1.0; closer
        # to walls they cost up to PREFERRED_CLEARANCE_CELLS+1.
        PREFERRED_CLEARANCE_CELLS = 5    # ~1m at 0.2m/cell
        dist = distance_transform_edt(~self._cached_plan_grid)
        clearance_penalty = np.maximum(
            0.0, PREFERRED_CLEARANCE_CELLS - dist)
        cost = (1.0 + clearance_penalty).astype(np.float32)
        cost[self._cached_plan_grid] = 1e6   # inflated obstacles impassable
        self._cached_cost_grid = cost
        self._cache_t = time.time()

    def aabbs(self):
        if (time.time() - self._cache_t) > 0.25:
            self._refresh_cache()
        return self._cached_aabbs

    def plan_grid(self):
        if (time.time() - self._cache_t) > 0.25:
            self._refresh_cache()
        return self._cached_plan_grid

    def cost_grid(self):
        if (time.time() - self._cache_t) > 0.25:
            self._refresh_cache()
        return self._cached_cost_grid


class Blackboard:
    def __init__(self, world_size):
        self.world_size = world_size
        self.learned_map = LearnedMap(world_size)
        # Pose (world frame) from odom
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # Latest detection (body-frame) and its derived world-frame leader pose
        self.leader_visible = False
        self.leader_world = None
        self.last_seen = None
        self.last_seen_t = 0.0
        self._last_obs = None
        self.leader_vel = (0.0, 0.0)
        self.miss_count = 0
        # Latest LiDAR scan
        self.scan = None
        # Last A* recovery plan (world-frame waypoints)
        self.recovery_path = []
        self.recovery_t = 0.0
        # Spiral-search bookkeeping
        self.spiral_radius = 0.0
        self.spiral_angle = 0.0
        # Pedestrian sightings (TTL-aged); world frame
        self.pedestrians = []

    @property
    def obstacles(self):
        return self.learned_map.aabbs()

    def update_leader_from_detection(self, body_x, body_y):
        """Convert body-frame detection to world frame and update vel EWMA."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wx = self.x + body_x * c - body_y * s
        wy = self.y + body_x * s + body_y * c
        now = time.time()
        # EWMA velocity update (alpha=0.4). Reject huge jumps as detection
        # noise — leader cannot exceed 1.5 m/s, so drop samples implying >3.
        if self._last_obs is not None:
            t0, x0, y0 = self._last_obs
            dt = max(1e-3, now - t0)
            inst_vx = (wx - x0) / dt
            inst_vy = (wy - y0) / dt
            if math.hypot(inst_vx, inst_vy) <= 3.0:
                a = 0.4
                self.leader_vel = (
                    a * inst_vx + (1 - a) * self.leader_vel[0],
                    a * inst_vy + (1 - a) * self.leader_vel[1],
                )
        self._last_obs = (now, wx, wy)
        self.leader_world = (wx, wy)
        self.leader_visible = True
        self.miss_count = 0
        self.last_seen = (wx, wy)
        self.last_seen_t = now
        # Re-detection: reset spiral search.
        self.spiral_radius = 0.0
        self.spiral_angle = 0.0

    def mark_lost(self):
        # Debounce: a single missed frame doesn't mean lost. Only flip the
        # visibility flag after several consecutive misses so a momentary
        # occlusion-corner glitch doesn't kick the BT into Planning.
        self.miss_count += 1
        if self.miss_count >= MISS_TICKS_FOR_LOST:
            self.leader_visible = False
            self.leader_world = None

    def dist_to_leader(self):
        if self.leader_world is None:
            return float("inf")
        lx, ly = self.leader_world
        return math.hypot(lx - self.x, ly - self.y)

    def dist_to_last_seen(self):
        if self.last_seen is None:
            return float("inf")
        lx, ly = self.last_seen
        return math.hypot(lx - self.x, ly - self.y)

    def update_pedestrians_from_detection(self, body_xys):
        """Replace fresh pedestrian sightings with the current ones (in world
        frame); keep older ones until they age out."""
        now = time.time()
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        fresh = []
        for bx, by in body_xys:
            wx = self.x + bx * c - by * s
            wy = self.x  # placeholder; corrected below
            wy = self.y + bx * s + by * c
            fresh.append((now, wx, wy))
        # Drop old entries near a fresh one (simple proximity dedupe).
        kept = []
        for t, x, y in self.pedestrians:
            if (now - t) > PEDESTRIAN_TTL_S:
                continue
            if any((x - fx) ** 2 + (y - fy) ** 2 < 0.5 ** 2
                   for _, fx, fy in fresh):
                continue
            kept.append((t, x, y))
        self.pedestrians = kept + fresh

    def known_pedestrian_xys(self):
        now = time.time()
        return [(x, y) for t, x, y in self.pedestrians
                if (now - t) <= PEDESTRIAN_TTL_S]


# ---------------------------------------------------------------------------
# LiDAR safety wrap and geometry helpers.
# ---------------------------------------------------------------------------
def lidar_front_clearance(scan: LaserScan):
    if scan is None or not scan.ranges:
        return float("inf"), 0.0
    cone = math.radians(LIDAR_FRONT_HALFCONE_DEG)
    max_r = scan.range_max if scan.range_max > 0 else 8.0
    front_min = max_r
    left_min = max_r
    right_min = max_r
    for i, r in enumerate(scan.ranges):
        if not math.isfinite(r) or r <= 0.0:
            continue
        ang = scan.angle_min + i * scan.angle_increment
        ang = (ang + math.pi) % (2 * math.pi) - math.pi
        if -cone <= ang <= cone:
            front_min = min(front_min, r)
            if ang > 0:
                left_min = min(left_min, r)
            else:
                right_min = min(right_min, r)
    return front_min, left_min - right_min


def apply_lidar_safety(v_lin, w_ang, scan):
    front_min, side_bias = lidar_front_clearance(scan)
    if v_lin > 0 and front_min < LIDAR_DANGER_M:
        return 0.0, (1.2 if side_bias >= 0 else -1.2)
    if v_lin > 0 and front_min < LIDAR_SLOW_M:
        scale = (front_min - LIDAR_DANGER_M) / (LIDAR_SLOW_M - LIDAR_DANGER_M)
        v_lin *= max(0.1, scale)
        nudge = (1.0 - scale) * 0.6 * (1.0 if side_bias >= 0 else -1.0)
        w_ang = max(-1.0, min(1.0, w_ang + nudge))
    return v_lin, w_ang


def adaptive_v_max(bb: Blackboard) -> float:
    """Paper Eq. 4: speed scales with leader velocity AND robot-leader distance."""
    speed = math.hypot(*bb.leader_vel)
    dist = bb.dist_to_leader()
    return float(np.clip(ALPHA_V_VEL * speed + ALPHA_V_DIST * dist,
                         V_MIN_CHASE, V_MAX_HARD))


def adaptive_d_t(bb: Blackboard) -> float:
    """Paper Eq. 5 substitute (we have no Kalman NIS — use leader speed)."""
    speed = math.hypot(*bb.leader_vel)
    return float(np.clip(D_T_BASE + D_T_VEL_GAIN * speed, D_T_MIN, D_T_MAX))


def goto_command(bb: Blackboard, target_xy, v_cap):
    """Diff-drive controller. Pure-rotation when target is behind us
    (no translation = no bounce-induced yaw flips), proportional forward
    when aligned. NO reverse — the robot must always make progress
    TOWARD the target, never away from it. This is what the user
    repeatedly clarified: 'just go to last_seen'.

    For the 180°-rotation tie-break (target exactly behind), rotation
    continues in whichever direction err points to. Once translating,
    the 0.6-rad threshold keeps us on a tight cone toward target."""
    tx, ty = target_xy
    dx, dy = tx - bb.x, ty - bb.y
    bearing = math.atan2(dy, dx)
    err = (bearing - bb.yaw + math.pi) % (2 * math.pi) - math.pi
    dist = math.hypot(dx, dy)
    w = max(-1.5, min(1.5, 1.6 * err))
    if abs(err) > 0.6:
        return 0.0, w
    v = max(0.0, min(v_cap, 0.9 * dist))
    return v, w


# ---------------------------------------------------------------------------
# Behavior tree leaves. Each ticks once per BT pass and writes (v, w) into
# bb.cmd. The root is a Selector that picks the highest-priority active leaf.
# ---------------------------------------------------------------------------
class _Leaf(py_trees.behaviour.Behaviour):
    def __init__(self, name, bb):
        super().__init__(name)
        self.bb = bb


class Retreating(_Leaf):
    def update(self):
        if not self.bb.leader_visible:
            return py_trees.common.Status.FAILURE
        d = self.bb.dist_to_leader()
        if d >= RETREAT_TRIGGER_DIST:
            return py_trees.common.Status.FAILURE
        # Approach test: leader velocity points roughly at robot.
        if self.bb.leader_world is None:
            return py_trees.common.Status.FAILURE
        lx, ly = self.bb.leader_world
        vx, vy = self.bb.leader_vel
        to_robot = (self.bb.x - lx, self.bb.y - ly)
        norm = math.hypot(*to_robot) + 1e-6
        dot = (vx * to_robot[0] + vy * to_robot[1]) / norm
        if dot < 0.05:
            return py_trees.common.Status.FAILURE
        # Back up while keeping orientation toward leader
        bearing = math.atan2(ly - self.bb.y, lx - self.bb.x)
        err = (bearing - self.bb.yaw + math.pi) % (2 * math.pi) - math.pi
        self.bb.cmd = (-0.4, max(-1.0, min(1.0, 1.2 * err)))
        return py_trees.common.Status.RUNNING


class Following(_Leaf):
    def update(self):
        if not self.bb.leader_visible:
            return py_trees.common.Status.FAILURE
        d = self.bb.dist_to_leader()
        if d > FOLLOW_RANGE:
            return py_trees.common.Status.FAILURE
        lx, ly = self.bb.leader_world
        vx, vy = self.bb.leader_vel
        speed = math.hypot(vx, vy)
        d_t = adaptive_d_t(self.bb)
        if speed > 0.15:
            # Trailing orbit: stand D_t metres behind the leader along its
            # velocity vector. As the leader rounds an obstacle the follower
            # naturally retraces the same arc and keeps line of sight.
            ux, uy = vx / speed, vy / speed
            goal = (lx - d_t * ux, ly - d_t * uy)
        else:
            # Stationary leader: just sit on the line robot->leader at D_t.
            br = math.atan2(self.bb.y - ly, self.bb.x - lx)
            goal = (lx + d_t * math.cos(br), ly + d_t * math.sin(br))
        # If a wall sits between us and the orbit goal, straight-line
        # steering would smack the wall. Hand off to PlannedRecovery (which
        # runs A* with peds inflated). Visibility of the leader does not
        # imply traversability of the line to it.
        if not line_segment_clear(self.bb.x, self.bb.y, goal[0], goal[1],
                                  self.bb.obstacles):
            return py_trees.common.Status.FAILURE
        v_cap = adaptive_v_max(self.bb)
        v, w = goto_command(self.bb, goal, v_cap)
        self.bb.cmd = (v, w)
        return py_trees.common.Status.RUNNING


class Chasing(_Leaf):
    def update(self):
        if not self.bb.leader_visible:
            return py_trees.common.Status.FAILURE
        d = self.bb.dist_to_leader()
        if d <= FOLLOW_RANGE:
            return py_trees.common.Status.FAILURE
        # Same wall-aware check as Following — straight chase only when LOS
        # to the leader is wall-clear; otherwise fall through to A*.
        if not line_segment_clear(self.bb.x, self.bb.y,
                                   self.bb.leader_world[0],
                                   self.bb.leader_world[1],
                                   self.bb.obstacles):
            return py_trees.common.Status.FAILURE
        v, w = goto_command(self.bb, self.bb.leader_world, V_MAX_HARD)
        self.bb.cmd = (v, w)
        return py_trees.common.Status.RUNNING


class Planning(_Leaf):
    """Drive to last-seen pose using an A* recovery plan, refreshed on demand.
    (Legacy — superseded by PlannedRecovery; kept for completeness.)"""
    def __init__(self, name, bb, plan_grid=None):
        super().__init__(name, bb)

    def _need_replan(self):
        if not self.bb.recovery_path:
            return True
        if (time.time() - self.bb.recovery_t) > LOST_REPLAN_S:
            return True
        return False

    def _predicted_target(self):
        """last_seen + leader_vel * dt with a horizon cap. Falls back to
        last_seen if the prediction lands inside an obstacle (i.e. the leader
        almost certainly didn't go through a wall)."""
        if self.bb.last_seen is None:
            return None
        dt = min(time.time() - self.bb.last_seen_t, PRED_HORIZON_S)
        vx, vy = self.bb.leader_vel
        Wm, Hm = self.bb.world_size
        px = self.bb.last_seen[0] + vx * dt
        py = self.bb.last_seen[1] + vy * dt
        px = max(0.5, min(Wm - 0.5, px))
        py = max(0.5, min(Hm - 0.5, py))
        if circle_collides_aabbs(px, py, ROBOT_RADIUS + 0.1, self.bb.obstacles):
            return self.bb.last_seen
        return (px, py)

    def _replan(self):
        target = self._predicted_target()
        if target is None:
            self.bb.recovery_path = []
            return
        plan_grid = self.bb.learned_map.plan_grid()
        path = astar(plan_grid, (self.bb.x, self.bb.y),
                     target, resolution=PLAN_RES)
        if not path or len(path) < 2:
            if target != self.bb.last_seen and self.bb.last_seen is not None:
                path = astar(plan_grid, (self.bb.x, self.bb.y),
                             self.bb.last_seen, resolution=PLAN_RES)
        if not path or len(path) < 2:
            self.bb.recovery_path = []
            return
        self.bb.recovery_path = smooth_path(
            path, self.bb.obstacles,
            clearance=ROBOT_RADIUS + 0.05)[1:]
        self.bb.recovery_t = time.time()

    def update(self):
        if self.bb.leader_visible:
            return py_trees.common.Status.FAILURE
        if self.bb.last_seen is None:
            return py_trees.common.Status.FAILURE
        # Did we arrive?
        if self.bb.dist_to_last_seen() < 0.6:
            self.bb.last_seen = None
            self.bb.recovery_path = []
            return py_trees.common.Status.FAILURE
        if self._need_replan():
            self._replan()
        # Pop reached waypoints
        while self.bb.recovery_path and \
                math.hypot(self.bb.recovery_path[0][0] - self.bb.x,
                           self.bb.recovery_path[0][1] - self.bb.y) < WAYPOINT_TOL:
            self.bb.recovery_path.pop(0)
        if not self.bb.recovery_path:
            return py_trees.common.Status.FAILURE
        v, w = goto_command(self.bb, self.bb.recovery_path[0], V_MAX_HARD)
        self.bb.cmd = (v, w)
        return py_trees.common.Status.RUNNING


def first_blocker(robot_xy, target_xy, obstacles, pedestrian_xys=()):
    """First occluder intersected by the segment robot->target. Returns either
    an AABB (static obstacle) or a tuple ('ped', cx, cy, r) for a pedestrian,
    or None if the segment is clear."""
    rx, ry = robot_xy
    tx, ty = target_xy
    dx, dy = tx - rx, ty - ry
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-3:
        return None
    ux, uy = dx / seg_len, dy / seg_len
    best_t = float("inf")
    best_obj = None
    for box in obstacles:
        if box.xmax - box.xmin > 8.0 or box.ymax - box.ymin > 8.0:
            continue
        d = ray_aabb_distance(rx, ry, ux, uy, box, max_t=seg_len)
        if d is not None and d < best_t:
            best_t = d
            best_obj = box
    for px, py in pedestrian_xys:
        if segment_hits_circle(rx, ry, tx, ty, px, py, PEDESTRIAN_RADIUS):
            # Approximate distance to pedestrian along ray (closest point).
            t = ((px - rx) * ux + (py - ry) * uy)
            t = max(0.0, min(seg_len, t))
            if t < best_t:
                best_t = t
                best_obj = ("ped", px, py, PEDESTRIAN_RADIUS)
    return best_obj


PRED_TRUST_S = 0.3   # prediction is only trusted for very recent losses

def predicted_leader(bb):
    """Where the leader probably is right now.

    For VERY recent detections (< PRED_TRUST_S), extrapolate from velocity.
    For longer losses, prediction drift dominates — just use last_seen
    directly. Otherwise the follower commits to a stale extrapolation
    pointing far from where the leader actually was, and walks the wrong
    way (observed in ep_1777817759: follower drove NW for 10s while
    last_seen was 3m east)."""
    if bb.last_seen is None:
        return None
    dt = time.time() - bb.last_seen_t
    if dt > PRED_TRUST_S:
        return bb.last_seen
    return (bb.last_seen[0] + bb.leader_vel[0] * dt,
            bb.last_seen[1] + bb.leader_vel[1] * dt)


def build_ped_aware_grid(static_grid, ped_xys, plan_res, inflate_m):
    """Return a copy of `static_grid` with all cells within `inflate_m` of any
    pedestrian marked blocked. Cheap: ~r_cells^2 marks per pedestrian."""
    if not ped_xys:
        return static_grid
    grid = static_grid.copy()
    H, W = grid.shape
    r_cells = int(math.ceil(inflate_m / plan_res))
    r2 = r_cells * r_cells
    for px, py in ped_xys:
        ci = int(px / plan_res)
        cj = int(py / plan_res)
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


class ClearLOS(_Leaf):
    """When LOS to the predicted leader is blocked by an obstacle, drive to a
    vantage point that re-establishes LOS. Generates 4 corner via-points
    around the blocker, picks the one that:

      1. Is reachable (A* finds a path),
      2. Has clear LOS to the predicted leader from there,
      3. Minimises (path_length * (1 + W * heading_misalignment)) so we prefer
         the side the leader actually went.

    If a NEW obstacle becomes the blocker after we round the first one, the
    leaf naturally recomputes (replans every LOS_REPLAN_S). This is the
    iterative 'always clear the next occluder' behavior.
    """
    def __init__(self, name, bb, plan_grid=None):
        super().__init__(name, bb)
        self._plan = []
        self._plan_t = 0.0

    def _candidate_via_points(self, blocker):
        if isinstance(blocker, tuple) and blocker[0] == "ped":
            # Pedestrian: 6 points on a ring around the centre.
            _, cx, cy, r = blocker
            ring_r = r + CORNER_MARGIN
            return [(cx + ring_r * math.cos(a), cy + ring_r * math.sin(a))
                    for a in [0, math.pi/3, 2*math.pi/3, math.pi,
                              4*math.pi/3, 5*math.pi/3]]
        m = CORNER_MARGIN
        return [
            (blocker.xmin - m, blocker.ymin - m),
            (blocker.xmax + m, blocker.ymin - m),
            (blocker.xmax + m, blocker.ymax + m),
            (blocker.xmin - m, blocker.ymax + m),
            (blocker.xmin - m, (blocker.ymin + blocker.ymax) / 2),
            (blocker.xmax + m, (blocker.ymin + blocker.ymax) / 2),
            ((blocker.xmin + blocker.xmax) / 2, blocker.ymin - m),
            ((blocker.xmin + blocker.xmax) / 2, blocker.ymax + m),
        ]

    def _score_via(self, via, blocker, pred, dyn_grid):
        Wm, Hm = self.bb.world_size
        vx, vy = via
        if not (0.5 < vx < Wm - 0.5 and 0.5 < vy < Hm - 0.5):
            return None
        if circle_collides_aabbs(vx, vy, ROBOT_RADIUS + 0.1, self.bb.obstacles):
            return None
        if not line_segment_clear(vx, vy, pred[0], pred[1], self.bb.obstacles):
            return None
        # Pedestrians can also occlude from the via-point.
        peds = self.bb.known_pedestrian_xys()
        for px, py in peds:
            if segment_hits_circle(vx, vy, pred[0], pred[1], px, py,
                                   PEDESTRIAN_RADIUS):
                return None
        path = astar(dyn_grid, (self.bb.x, self.bb.y), (vx, vy),
                     resolution=PLAN_RES)
        if not path or len(path) < 2:
            return None
        length = sum(math.hypot(path[i + 1][0] - path[i][0],
                                path[i + 1][1] - path[i][1])
                     for i in range(len(path) - 1))
        # Heading consistency: prefer corners on the side the leader went.
        speed = math.hypot(*self.bb.leader_vel)
        if speed > 0.1:
            if isinstance(blocker, tuple) and blocker[0] == "ped":
                _, bcx, bcy, _ = blocker
            else:
                bcx = (blocker.xmin + blocker.xmax) / 2
                bcy = (blocker.ymin + blocker.ymax) / 2
            corner_b = math.atan2(vy - bcy, vx - bcx)
            vel_b = math.atan2(self.bb.leader_vel[1], self.bb.leader_vel[0])
            d_b = abs((corner_b - vel_b + math.pi) % (2 * math.pi) - math.pi)
            penalty = d_b / math.pi  # 0..1
        else:
            penalty = 0.0
        return path, length * (1.0 + HEADING_PENALTY_W * penalty)

    def update(self):
        if self.bb.leader_visible:
            return py_trees.common.Status.FAILURE
        pred = predicted_leader(self.bb)
        if pred is None:
            return py_trees.common.Status.FAILURE
        blocker = first_blocker((self.bb.x, self.bb.y), pred,
                                 self.bb.obstacles,
                                 pedestrian_xys=self.bb.known_pedestrian_xys())
        if blocker is None:
            return py_trees.common.Status.FAILURE  # let DriveToPrediction handle

        # Pop reached waypoints first so the commit check sees an accurate
        # "plan still has work" signal.
        while self._plan and math.hypot(self._plan[0][0] - self.bb.x,
                                        self._plan[0][1] - self.bb.y) < WAYPOINT_TOL:
            self._plan.pop(0)

        # COMMIT-TO-PLAN: keep the existing plan as long as it has work and
        # hasn't aged out. Transient pedestrians crossing LOS mid-route no
        # longer reset the plan — finish what you started.
        plan_alive = self._plan and (time.time() - self._plan_t) < LOS_PLAN_COMMIT_S
        if not plan_alive:
            # Build a dynamic plan grid: learned static map + current
            # pedestrian positions inflated. The learned static layer is
            # whatever LiDAR has revealed so far; peds add a temporary
            # exclusion zone for the A* call.
            static_grid = self.bb.learned_map.plan_grid()
            dyn_grid = build_ped_aware_grid(
                static_grid, self.bb.known_pedestrian_xys(),
                PLAN_RES, PEDESTRIAN_RADIUS + ROBOT_RADIUS + 0.05)
            best_path = None
            best_score = float("inf")
            for via in self._candidate_via_points(blocker):
                scored = self._score_via(via, blocker, pred, dyn_grid)
                if scored is None:
                    continue
                path, score = scored
                if score < best_score:
                    best_score = score
                    best_path = path
            # If the dynamic grid blocks every via-point (very dense crowd),
            # retry with the static grid only — better a path through some
            # peds than no path at all.
            if best_path is None:
                for via in self._candidate_via_points(blocker):
                    scored = self._score_via(via, blocker, pred, static_grid)
                    if scored is None:
                        continue
                    path, score = scored
                    if score < best_score:
                        best_score = score
                        best_path = path
            self._plan = (smooth_path(best_path, self.bb.obstacles,
                                      clearance=ROBOT_RADIUS + 0.05)[1:]
                          if best_path else [])
            self._plan_t = time.time()
        if not self._plan:
            return py_trees.common.Status.FAILURE
        v, w = goto_command(self.bb, self._plan[0], V_MAX_HARD)
        self.bb.cmd = (v, w)
        return py_trees.common.Status.RUNNING


class DriveToPrediction(_Leaf):
    """LOS to predicted leader is clear but we still don't see them — drive
    straight to the prediction at full speed. Sits between ClearLOS (handles
    occluded case) and SpiralExpand (handles 'prediction is wrong' case)."""
    def update(self):
        if self.bb.leader_visible:
            return py_trees.common.Status.FAILURE
        pred = predicted_leader(self.bb)
        if pred is None:
            return py_trees.common.Status.FAILURE
        # If we're already close to the prediction and still no detection,
        # the prediction is stale → let SpiralExpand take over.
        if math.hypot(pred[0] - self.bb.x, pred[1] - self.bb.y) < 0.6:
            return py_trees.common.Status.FAILURE
        v, w = goto_command(self.bb, pred, V_MAX_HARD)
        self.bb.cmd = (v, w)
        return py_trees.common.Status.RUNNING


class SpiralExpand(_Leaf):
    """Two-phase recovery when PlannedRecovery couldn't find a path.
      Phase 1: drive STRAIGHT to the predicted-leader position (the
               spiral center). The leader was last there, so check
               that location first.
      Phase 2: only once we've ARRIVED at the center and still have no
               detection, start the concentric-ring search to look in
               the surrounding area.
    Goes back to phase 1 whenever last_seen updates (new detection)."""
    def __init__(self, name, bb, plan_grid=None):
        super().__init__(name, bb)
        self._target_t = 0.0
        self._target = None
        self._anchored_t = 0.0  # last_seen_t when this leaf last anchored
        self._reached_center = False

    def update(self):
        if self.bb.leader_visible:
            self._reached_center = False
            self._anchored_t = 0.0
            return py_trees.common.Status.FAILURE
        if self.bb.last_seen is None:
            self.bb.cmd = (0.0, 0.4)
            return py_trees.common.Status.RUNNING

        dt = time.time() - self.bb.last_seen_t
        horizon = min(dt, PRED_HORIZON_S)
        cx = self.bb.last_seen[0] + self.bb.leader_vel[0] * horizon
        cy = self.bb.last_seen[1] + self.bb.leader_vel[1] * horizon

        # Reset to phase 1 whenever we get a fresh sighting (last_seen_t
        # advanced since we anchored).
        if self.bb.last_seen_t != self._anchored_t:
            self._reached_center = False
            self._anchored_t = self.bb.last_seen_t
            self.bb.spiral_radius = 0.0
            self.bb.spiral_angle = 0.0

        # Phase 1: head directly to the predicted leader pose.
        if not self._reached_center:
            d_to_center = math.hypot(cx - self.bb.x, cy - self.bb.y)
            if d_to_center < 0.6:
                self._reached_center = True
                self._target = None  # force fresh ring target on next tick
            else:
                v, w = goto_command(self.bb, (cx, cy), V_MAX_HARD)
                self.bb.cmd = (v, w)
                return py_trees.common.Status.RUNNING

        # Phase 2: grow concentric rings around the spiral center.
        need_new = (
            self._target is None
            or math.hypot(self._target[0] - self.bb.x,
                          self._target[1] - self.bb.y) < 0.6
            or (time.time() - self._target_t) > 1.5
        )
        if need_new:
            self.bb.spiral_radius = min(4.0, self.bb.spiral_radius + 0.6)
            self.bb.spiral_angle = (self.bb.spiral_angle + math.radians(70)) % (2 * math.pi)
            tx = cx + self.bb.spiral_radius * math.cos(self.bb.spiral_angle)
            ty = cy + self.bb.spiral_radius * math.sin(self.bb.spiral_angle)
            Wm, Hm = self.bb.world_size
            tx = max(0.5, min(Wm - 0.5, tx))
            ty = max(0.5, min(Hm - 0.5, ty))
            if circle_collides_aabbs(tx, ty, ROBOT_RADIUS + 0.1, self.bb.obstacles):
                tx, ty = cx, cy
                tx = max(0.5, min(Wm - 0.5, tx))
                ty = max(0.5, min(Hm - 0.5, ty))
            self._target = (tx, ty)
            self._target_t = time.time()

        v, w = goto_command(self.bb, self._target, V_MAX_HARD)
        self.bb.cmd = (v, w)
        return py_trees.common.Status.RUNNING


class PlannedRecovery(_Leaf):
    """When the leader is lost, plan an A* path from current pose to the
    predicted leader position, with all known peds inflated as obstacles.
    Walk the waypoints. If A* finds NO path (i.e. peds form a wall and there
    is no safe route), return FAILURE rather than pushing forward into a
    stuck-prone direction — let SpiralExpand try a different region.

    This replaces the older ClearLOS/DriveToPrediction split: a single A*
    call captures both 'navigate around the blocker' and 'walk the clear
    line' in one mechanism, identical in spirit to the leader's planner."""
    def __init__(self, name, bb, plan_grid=None):
        super().__init__(name, bb)
        self._plan = []
        self._plan_t = 0.0
        self._target = None

    def _approach_target(self, grid, target):
        """If `target` is in a blocked cell (LiDAR-mapped obstacle on top
        of where we last saw the leader), project the goal slightly BEYOND
        the obstacle along the ray from us to target. The map doesn't
        know what's past the obstacle (occlusion = unknown, treated as
        free), so A* can plan to a beyond point and naturally route around
        the obstacle."""
        ti = int(target[0] / PLAN_RES)
        tj = int(target[1] / PLAN_RES)
        if not (0 <= tj < grid.shape[0] and 0 <= ti < grid.shape[1]):
            return target
        if not grid[tj, ti]:
            return target
        rx, ry = self.bb.x, self.bb.y
        dx, dy = target[0] - rx, target[1] - ry
        L = math.hypot(dx, dy)
        if L < 1e-3:
            return target
        ux, uy = dx / L, dy / L
        step = PLAN_RES * 0.5
        # Extend up to 2m past the target looking for the first free cell
        # past the obstacle. That's our "beyond the wall" approximate target.
        max_extend = 2.0
        t = L
        while t < L + max_extend:
            x = rx + ux * t
            y = ry + uy * t
            ci = int(x / PLAN_RES)
            cj = int(y / PLAN_RES)
            if not (0 <= cj < grid.shape[0] and 0 <= ci < grid.shape[1]):
                break
            if not grid[cj, ci]:
                return (x, y)
            t += step
        # Couldn't find a beyond-point. Fall back to the original target;
        # let A* refuse and the BT will fall through to SpiralExpand.
        return target

    def _ped_aware_grid(self, target_xy=None):
        """A* uses the LiDAR-learned occupancy grid. No explicit ped layer
        (camera fusion would be cheating) — the LiDAR map already shows
        peds as transient ghosts that fade in ~0.25s.
        Only the visible-LEADER cell gets unblocked here, because the
        leader's position is camera-confirmed and we're trying to reach
        it. For other targets (last_seen / predicted), if the cell is
        blocked we use _approach_target instead of unblocking."""
        grid = self.bb.learned_map.plan_grid()
        if self.bb.leader_world is not None:
            grid = grid.copy()
            lx, ly = self.bb.leader_world
            li = int(lx / PLAN_RES)
            lj = int(ly / PLAN_RES)
            r_cells = int(math.ceil(0.5 / PLAN_RES))
            for dj in range(-r_cells, r_cells + 1):
                for di in range(-r_cells, r_cells + 1):
                    jj, ii = lj + dj, li + di
                    if (0 <= jj < grid.shape[0]
                            and 0 <= ii < grid.shape[1]
                            and dj * dj + di * di <= r_cells * r_cells):
                        grid[jj, ii] = False
        return grid

    def _try_plan(self, target):
        """Weighted A* on a clearance-aware cost grid: blocked cells are
        impassable, and free cells cost more the closer they are to a wall.
        Result: A* prefers wide-open corridors over narrow ones when both
        reach the goal. Reactive replan in PlannedRecovery.update
        invalidates the plan when a new wall lands on it.

        Fallback chain still tries last_seen if predicted target fails."""
        cost = self.bb.learned_map.cost_grid().copy()
        grid = self.bb.learned_map.plan_grid()
        # Unblock the leader's known cell so A* doesn't refuse the goal.
        if self.bb.leader_world is not None:
            lx, ly = self.bb.leader_world
            li = int(lx / PLAN_RES)
            lj = int(ly / PLAN_RES)
            r_cells = int(math.ceil(0.5 / PLAN_RES))
            for dj in range(-r_cells, r_cells + 1):
                for di in range(-r_cells, r_cells + 1):
                    jj, ii = lj + dj, li + di
                    if (0 <= jj < cost.shape[0]
                            and 0 <= ii < cost.shape[1]
                            and dj * dj + di * di <= r_cells * r_cells):
                        cost[jj, ii] = 1.0
        candidates = [target]
        if (self.bb.last_seen is not None
                and (math.hypot(target[0] - self.bb.last_seen[0],
                                target[1] - self.bb.last_seen[1]) > 0.4)):
            candidates.append(self.bb.last_seen)
        for cand in candidates:
            cand = self._approach_target(grid, cand)
            path = astar_weighted(cost, (self.bb.x, self.bb.y), cand,
                                   resolution=PLAN_RES)
            if path and len(path) > 1:
                return smooth_path(path, self.bb.obstacles,
                                    clearance=ROBOT_RADIUS + 0.05)[1:]
        return []

    def update(self):
        # NOTE: no longer gated on leader_visible. Following/Chasing run
        # FIRST in the BT (higher priority); they only fall through to here
        # when their straight-line LOS check fails — i.e. visible-but-wall-
        # blocked. In that case we still want to A* toward the leader.
        if self.bb.leader_visible and self.bb.leader_world is not None:
            target = self.bb.leader_world
        else:
            target = predicted_leader(self.bb)
        if target is None:
            return py_trees.common.Status.FAILURE
        # ARRIVED-AT-LAST-SEEN GUARD: if we're already at the recovery
        # target and still don't see the leader, hand off to SpiralExpand
        # (which will start searching outward from here). Without this we
        # keep replanning a trivial 1-waypoint path and never give the
        # search leaf a turn — observed in ep_1777825358 from t=2280
        # onward (follower oscillated at last_seen for >40s).
        if not self.bb.leader_visible:
            d_to_target = math.hypot(target[0] - self.bb.x,
                                      target[1] - self.bb.y)
            if d_to_target < 0.6:
                return py_trees.common.Status.FAILURE
        # Pop reached waypoints
        while self._plan and math.hypot(self._plan[0][0] - self.bb.x,
                                        self._plan[0][1] - self.bb.y) < WAYPOINT_TOL:
            self._plan.pop(0)
        target_moved = (
            self._target is None
            or math.hypot(target[0] - self._target[0],
                          target[1] - self._target[1]) > 0.8
        )
        # Reactive replan: if the learned map now shows a waypoint of the
        # current plan inside a blocked cell, the plan is stale (LiDAR
        # discovered an obstacle on our route). Drop it immediately and
        # let A* find a fresh path. This is the standard Nav2 obstacle-
        # update -> replan pattern.
        plan_blocked = False
        if self._plan:
            grid = self.bb.learned_map.plan_grid()
            for wx, wy in self._plan:
                ci = int(wx / PLAN_RES)
                cj = int(wy / PLAN_RES)
                if (0 <= cj < grid.shape[0] and 0 <= ci < grid.shape[1]
                        and grid[cj, ci]):
                    plan_blocked = True
                    break
        plan_alive = (self._plan
                      and (time.time() - self._plan_t) < LOS_PLAN_COMMIT_S
                      and not target_moved
                      and not plan_blocked)
        if not plan_alive:
            self._plan = self._try_plan(target)
            self._plan_t = time.time()
            self._target = target
        if not self._plan:
            return py_trees.common.Status.FAILURE
        v, w = goto_command(self.bb, self._plan[0], V_MAX_HARD)
        self.bb.cmd = (v, w)
        return py_trees.common.Status.RUNNING


class BackupRecovery(_Leaf):
    """Wall-stuck recovery: when the follower commands forward motion
    for 1s+ but doesn't actually translate (wedged on a wall), back up
    briefly to break free, then yield. Crucial gates so we DON'T fire
    on 'arrived at last_seen' (which would loop with PlannedRecovery's
    arrived-handoff to SpiralExpand):
      * leader_visible -> not our problem, let Following handle.
      * dist to last_seen < 0.8m -> we're at the goal, let SpiralExpand
        take over, NOT us.
      * pose-bbox over 1s window < 5cm -> truly wedged.
    Place AFTER PlannedRecovery so PlannedRecovery's plan-execution
    gets first crack. Only fires when execution is actually stuck."""
    STUCK_WINDOW_TICKS = 20
    STUCK_TRANSLATION_M = 0.05
    AT_TARGET_DIST_M = 0.8
    BACKUP_DURATION_TICKS = 10
    COOLDOWN_TICKS = 30

    def __init__(self, name, bb):
        super().__init__(name, bb)
        from collections import deque
        self._pose_window = deque(maxlen=self.STUCK_WINDOW_TICKS)
        self._backup_left = 0
        self._cooldown_left = 0

    def update(self):
        self._pose_window.append((self.bb.x, self.bb.y))
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
        # Already executing a backup? Continue it.
        if self._backup_left > 0:
            self._backup_left -= 1
            self.bb.cmd = (-0.4, 0.0)
            return py_trees.common.Status.RUNNING
        # Don't fire if leader is visible — Following/Chasing handle it.
        if self.bb.leader_visible:
            return py_trees.common.Status.FAILURE
        # Don't fire if we've reached last_seen (let SpiralExpand search).
        if self.bb.last_seen is not None:
            d = math.hypot(self.bb.last_seen[0] - self.bb.x,
                           self.bb.last_seen[1] - self.bb.y)
            if d < self.AT_TARGET_DIST_M:
                return py_trees.common.Status.FAILURE
        # Need a full window to assess stuckness.
        if len(self._pose_window) < self.STUCK_WINDOW_TICKS:
            return py_trees.common.Status.FAILURE
        if self._cooldown_left > 0:
            return py_trees.common.Status.FAILURE
        xs = [p[0] for p in self._pose_window]
        ys = [p[1] for p in self._pose_window]
        bbox = max(max(xs) - min(xs), max(ys) - min(ys))
        if bbox < self.STUCK_TRANSLATION_M:
            self._backup_left = self.BACKUP_DURATION_TICKS
            self._cooldown_left = self.COOLDOWN_TICKS
            self.bb.cmd = (-0.4, 0.0)
            return py_trees.common.Status.RUNNING
        return py_trees.common.Status.FAILURE


def build_tree(bb: Blackboard, plan_grid) -> py_trees.trees.BehaviourTree:
    """BT (v34):
      Retreating         - too close + leader closing
      Following          - in range, trail-orbit at adaptive D_t
      Chasing            - visible but out of follow range
      PlannedRecovery    - lost: A* to last_seen. Yields once arrived so
                           SpiralExpand can search.
      BackupRecovery     - wedged on wall while trying to make progress;
                           gated by 'not at last_seen yet' so it doesn't
                           steal SpiralExpand's turn.
      SpiralExpand       - search outward in rings from last_seen

    Note placement: BackupRecovery is BEFORE SpiralExpand. If we're
    wedged en route to last_seen, we back up to escape; once arrived
    (within 0.8m), BackupRecovery's gate forces FAILURE and SpiralExpand
    takes over.
    """
    root = py_trees.composites.Selector(name="follow_everything_v34", memory=False)
    root.add_children([
        Retreating("Retreating", bb),
        Following("Following", bb),
        Chasing("Chasing", bb),
        PlannedRecovery("PlannedRecovery", bb, plan_grid),
        BackupRecovery("BackupRecovery", bb),
        SpiralExpand("SpiralExpand", bb, plan_grid),
    ])
    return py_trees.trees.BehaviourTree(root)


# ---------------------------------------------------------------------------
# ROS node glue.
# ---------------------------------------------------------------------------
class FollowEverythingNode(Node):
    def __init__(self, map_path: str):
        super().__init__("follow_everything_follower")
        # NO map cheat. We accept work-area BOUNDS as deployment config (the
        # robot is told "you operate in this 15x15m area") but learn the
        # obstacle CONTENTS online from LiDAR. last_seen starts None — the
        # follower has no idea where the leader is until the camera detects
        # one. This is the strict-no-leakage configuration.
        world_size = self._world_size_from_env(map_path)
        self.bb = Blackboard(world_size)
        self.bb.cmd = (0.0, 0.0)
        self.tree = build_tree(self.bb, None)
        self.tree.setup(timeout=1.0)

        self.create_subscription(Detection2DArray,
                                 "/follower/camera/detections",
                                 self._on_detect, 10)
        self.create_subscription(Detection2DArray,
                                 "/follower/camera/pedestrians",
                                 self._on_ped_detect, 10)
        self.create_subscription(LaserScan, "/follower/scan",
                                 self._on_scan, 10)
        self.create_subscription(Odometry, "/follower/odom",
                                 self._on_odom, 10)
        self.pub = self.create_publisher(Twist, "/follower/cmd_vel", 10)
        self.pub_map = self.create_publisher(
            OccupancyGrid, "/follower/learned_map", 10)
        self.pub_path = self.create_publisher(
            Path, "/follower/planned_path", 10)
        self.create_timer(0.05, self._tick)
        self.create_timer(0.25, self._publish_learned_map)
        self.create_timer(0.10, self._publish_planned_path)

    def _world_size_from_env(self, map_path):
        """Read just the dimensions of the map file — NOT the obstacles or
        spawn cells. The follower is told 'you work in this many metres
        square'; obstacles are unknown until LiDAR sees them."""
        try:
            with open(map_path) as f:
                rows = []
                for line in f:
                    s = line.rstrip("\n")
                    if not s.strip() or s.lstrip().startswith("//"):
                        continue
                    rows.append(s)
            H = len(rows)
            W = max(len(r) for r in rows)
            return (W * 0.5, H * 0.5)
        except Exception:
            return (15.0, 15.0)
        self.get_logger().info(
            f"Follow-Everything follower up. World bounds {world_size}; "
            f"obstacles unknown — will be learned from LiDAR.")

    def _on_odom(self, msg):
        ori = msg.pose.pose.orientation
        yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y),
                         1 - 2 * (ori.y * ori.y + ori.z * ori.z))
        self.bb.x = msg.pose.pose.position.x
        self.bb.y = msg.pose.pose.position.y
        self.bb.yaw = yaw

    def _on_scan(self, msg):
        self._scan_count = getattr(self, "_scan_count", 0) + 1
        if self._scan_count <= 3 or self._scan_count % 40 == 0:
            print(f"[FOLLOWER] _on_scan fired #{self._scan_count} ranges={len(msg.ranges)}", flush=True)
        self.bb.scan = msg
        leader = self.bb.leader_world if self.bb.leader_visible else None
        try:
            self.bb.learned_map.update_from_scan(
                msg, self.bb.x, self.bb.y, self.bb.yaw, leader_world=leader)
        except Exception as e:
            import traceback
            print(f"[FOLLOWER] _on_scan CRASH: {e!r}", flush=True)
            traceback.print_exc()
        self._scan_count = getattr(self, "_scan_count", 0) + 1
        if self._scan_count % 40 == 0:
            log_odds = self.bb.learned_map.log_odds
            n_occ = int(np.sum(log_odds > LOG_ODDS_THRESH))
            n_free = int(np.sum(log_odds < -LOG_ODDS_THRESH))
            print(f"[FOLLOWER] scan #{self._scan_count}: ranges={len(msg.ranges)}, "
                  f"pose=({self.bb.x:.1f},{self.bb.y:.1f},{math.degrees(self.bb.yaw):.0f}), "
                  f"map occ={n_occ} free={n_free} max_lo={float(log_odds.max()):.2f}",
                  flush=True)

    def _publish_planned_path(self):
        # Walk the BT and find the PlannedRecovery leaf's current plan.
        plan = []
        for child in self.tree.root.children:
            if isinstance(child, PlannedRecovery):
                plan = list(child._plan or [])
                break
        msg = Path()
        msg.header.frame_id = "world"
        msg.header.stamp = self.get_clock().now().to_msg()
        # Prepend current pose so the line starts AT the follower.
        all_pts = [(self.bb.x, self.bb.y)] + plan
        for x, y in all_pts:
            ps = PoseStamped()
            ps.header.frame_id = "world"
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def _publish_learned_map(self):
        try:
            self._publish_learned_map_impl()
        except Exception as e:
            self.get_logger().error(f"publish_learned_map crash: {e!r}")

    def _publish_learned_map_impl(self):
        lm = self.bb.learned_map
        msg = OccupancyGrid()
        msg.info.resolution = float(MAP_RES)
        msg.info.width = lm.W_cells
        msg.info.height = lm.H_cells
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        # Encode as standard OccupancyGrid: -1 unknown, 0 free, 100 occupied.
        cells = []
        flat = lm.log_odds
        for j in range(lm.H_cells):
            for i in range(lm.W_cells):
                v = flat[j, i]
                if v > LOG_ODDS_THRESH:
                    cells.append(100)
                elif v < -LOG_ODDS_THRESH:
                    cells.append(0)
                else:
                    cells.append(-1)
        msg.data = cells
        self.pub_map.publish(msg)

    def _on_detect(self, msg: Detection2DArray):
        if not msg.detections:
            self.bb.mark_lost()
            return
        p = msg.detections[0].results[0].pose.pose.position
        self.bb.update_leader_from_detection(p.x, p.y)

    def _on_ped_detect(self, msg: Detection2DArray):
        body_xys = []
        for det in msg.detections:
            if not det.results:
                continue
            p = det.results[0].pose.pose.position
            body_xys.append((p.x, p.y))
        self.bb.update_pedestrians_from_detection(body_xys)

    def _tick(self):
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count <= 3 or self._tick_count % 40 == 0:
            print(f"[FOLLOWER] _tick fired #{self._tick_count}", flush=True)
        self.bb.cmd = (0.0, 0.0)
        try:
            self.tree.tick()
        except Exception as e:
            import traceback
            print(f"[FOLLOWER] BT tick CRASH: {e!r}", flush=True)
            traceback.print_exc()
        v, w = self.bb.cmd
        try:
            v, w = apply_lidar_safety(v, w, self.bb.scan)
        except Exception as e:
            print(f"[FOLLOWER] lidar_safety CRASH: {e!r}", flush=True)
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count % 40 == 0:
            active = []
            for child in self.tree.root.children:
                if child.status == py_trees.common.Status.RUNNING:
                    active.append(child.name)
            print(f"[FOLLOWER] tick #{self._tick_count}: pose=({self.bb.x:.1f},{self.bb.y:.1f}) "
                  f"vis={self.bb.leader_visible} last_seen={self.bb.last_seen} "
                  f"v={v:.2f} w={w:.2f} active={active}", flush=True)
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)


def main():
    print("[FOLLOWER] main() entered", flush=True)
    rclpy.init()
    print("[FOLLOWER] rclpy.init done", flush=True)
    map_path = os.environ.get("SIM_MAP", "sim/maps/cluttered.txt")
    print(f"[FOLLOWER] map_path={map_path}", flush=True)
    try:
        node = FollowEverythingNode(map_path)
        print("[FOLLOWER] node constructed; spinning", flush=True)
    except Exception as e:
        import traceback
        print(f"[FOLLOWER] node init CRASH: {e!r}", flush=True)
        traceback.print_exc()
        return
    try:
        rclpy.spin(node)
    except Exception as e:
        import traceback
        print(f"[FOLLOWER] spin CRASH: {e!r}", flush=True)
        traceback.print_exc()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
