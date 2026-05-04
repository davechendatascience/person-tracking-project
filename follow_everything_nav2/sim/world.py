"""
Continuous-space 2D simulator. Robots are continuous (x, y, yaw); obstacles are
axis-aligned boxes derived from the ASCII map at load time. Sensor rays use
analytical ray-vs-box intersection. NO grid stepping.

DO NOT MODIFY for autoresearch experiments — this file is part of the fixed
evaluator. The follower interacts only via ROS topics.
"""
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, PoseArray, Pose, Quaternion, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from tf2_ros import TransformBroadcaster

from sim.geometry import (
    AABB, circle_collides_aabbs, line_segment_clear, merged_aabbs_from_grid,
)
from sim.planner import build_planning_grid, astar, smooth_path
from sim.sensors import LidarSensor, CameraDetector


PED_PLAN_RES = 0.2
PED_PLAN_INFLATE = 0.4   # > radius so paths keep clear of corners

CELL_M = 0.5
TICK_HZ = 20.0
TICK_DT = 1.0 / TICK_HZ
ROBOT_RADIUS = 0.25
MAX_LIN_VEL = 1.5
MAX_ANG_VEL = 1.5
PEDESTRIAN_RADIUS = 0.25
PEDESTRIAN_SPEED = 0.5
NUM_PEDESTRIANS = int(os.environ.get("NUM_PEDESTRIANS", "0"))

DEFAULT_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


def yaw_to_quat(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    wz: float = 0.0


@dataclass
class PedestrianState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_t: float = 0.0
    stuck_ticks: int = 0
    boost_ticks: int = 0   # time left in escape-mode (low repulsion)
    prev_x: float = 0.0
    prev_y: float = 0.0
    path: list = None       # A* waypoints to target

    def __post_init__(self):
        if self.path is None:
            self.path = []


def load_map(path: str):
    """Returns (occupancy[H,W] bool, obstacles List[AABB], leader_xy, follower_xy,
    world_size_xy). Map syntax: '#' = obstacle, '.' = free, 'L' = leader spawn,
    'F' = follower spawn. Lines starting with '//' are comments. ALL other non-empty
    lines are treated as map rows (so '#' is unambiguous as an obstacle char)."""
    with open(path) as f:
        rows = []
        for line in f:
            s = line.rstrip("\n")
            if not s.strip():
                continue
            if s.lstrip().startswith("//"):
                continue
            rows.append(s)
    H = len(rows)
    W = max(len(r) for r in rows)
    grid = np.zeros((H, W), dtype=bool)
    leader_xy = None
    follower_xy = None
    for j, row in enumerate(rows):
        for i, ch in enumerate(row.ljust(W)):
            if ch == "#":
                grid[j, i] = True
            elif ch == "L":
                leader_xy = ((i + 0.5) * CELL_M, (H - j - 0.5) * CELL_M)
            elif ch == "F":
                follower_xy = ((i + 0.5) * CELL_M, (H - j - 0.5) * CELL_M)
    if leader_xy is None or follower_xy is None:
        raise ValueError(f"Map {path} must include 'L' and 'F' spawn cells.")
    obstacles = merged_aabbs_from_grid(grid, CELL_M)
    world_size = (W * CELL_M, H * CELL_M)
    return grid, obstacles, leader_xy, follower_xy, world_size


class World(Node):
    def __init__(self, map_path: str, seed: int = 0, render: bool = False):
        super().__init__("sim_world")
        self.rng = np.random.default_rng(seed)
        (self.grid, self.obstacles, leader_xy, follower_xy,
         self.world_size) = load_map(map_path)
        self.leader = RobotState(x=leader_xy[0], y=leader_xy[1], yaw=0.0)
        # Initialize follower facing the leader so the chase can begin without
        # a long blind-spin search.
        f_yaw = math.atan2(leader_xy[1] - follower_xy[1],
                           leader_xy[0] - follower_xy[0])
        self.follower = RobotState(x=follower_xy[0], y=follower_xy[1], yaw=f_yaw)
        self.collision_count_follower = 0
        self.collision_count_leader = 0

        self.lidar = LidarSensor()
        self.camera = CameraDetector()

        self.pedestrians = self._spawn_pedestrians(
            NUM_PEDESTRIANS, leader_xy, follower_xy)
        # Planning grid shared by all pedestrians (static obstacles, inflated
        # by ped radius + margin so A* paths actually fit a ped's body).
        self.ped_plan_grid = build_planning_grid(
            self.obstacles, self.world_size,
            radius_m=PED_PLAN_INFLATE, resolution=PED_PLAN_RES)

        self._viz = None
        if render:
            from sim.viz import Viz
            self._viz = Viz(render_every=2)
        self._last_detection_visible = False
        self._last_lidar_ranges = None
        # Follower's learned map (subscribed only for rendering — the world
        # itself doesn't use it).
        self._learned_map_msg = None
        self._planned_path_msg = None
        if render:
            self.create_subscription(OccupancyGrid,
                                     "/follower/learned_map",
                                     self._on_learned_map, DEFAULT_QOS)
            self.create_subscription(Path,
                                     "/follower/planned_path",
                                     self._on_planned_path, DEFAULT_QOS)

        self.create_subscription(Twist, "/leader/cmd_vel",
                                 lambda m: self._on_cmd("leader", m), DEFAULT_QOS)
        self.create_subscription(Twist, "/follower/cmd_vel",
                                 lambda m: self._on_cmd("follower", m), DEFAULT_QOS)

        self.pub_leader_odom = self.create_publisher(Odometry, "/leader/odom", DEFAULT_QOS)
        self.pub_follower_odom = self.create_publisher(Odometry, "/follower/odom", DEFAULT_QOS)
        self.pub_follower_scan = self.create_publisher(LaserScan, "/follower/scan", DEFAULT_QOS)
        self.pub_follower_detect = self.create_publisher(Detection2DArray,
                                                         "/follower/camera/detections", DEFAULT_QOS)
        self.pub_leader_truth = self.create_publisher(PoseStamped,
                                                     "/leader/pose_ground_truth", DEFAULT_QOS)
        self.pub_follower_ped_detect = self.create_publisher(
            Detection2DArray, "/follower/camera/pedestrians", DEFAULT_QOS)
        # Global ground-truth pedestrian poses for the autonomous LEADER's
        # planner (the leader is part of the sim, not the autoresearch agent —
        # it's allowed to "see everyone" the way an aware human would).
        # The follower MUST NOT subscribe to this.
        self.pub_pedestrian_poses = self.create_publisher(
            PoseArray, "/pedestrians/poses", DEFAULT_QOS)

        self.tfb = TransformBroadcaster(self)

        self.tick = 0
        self.create_timer(TICK_DT, self._step)
        self.get_logger().info(
            f"World up: map={map_path}, {len(self.obstacles)} obstacle boxes, "
            f"world={self.world_size[0]:.1f}x{self.world_size[1]:.1f}m")

    # ---- helpers ---- #
    def _world_collides(self, x: float, y: float, radius: float) -> bool:
        # World boundary
        if (x < radius or y < radius or
                x > self.world_size[0] - radius or y > self.world_size[1] - radius):
            return True
        return circle_collides_aabbs(x, y, radius, self.obstacles)

    def _hits_pedestrian(self, x, y, radius, exclude=None):
        for p in self.pedestrians:
            if p is exclude:
                continue
            if (x - p.x) ** 2 + (y - p.y) ** 2 < (radius + PEDESTRIAN_RADIUS) ** 2:
                return True
        return False

    def _spawn_pedestrians(self, n, leader_xy, follower_xy):
        peds = []
        Wm, Hm = self.world_size
        # Adapt spacing so we can pack ~100 peds in a 15x15 map. Reserve a
        # 1.5m bubble around leader/follower so the chase can begin cleanly,
        # and require pedestrians to be at least one diameter apart.
        leader_bubble = 1.5
        ped_spacing = 2 * PEDESTRIAN_RADIUS + 0.05
        attempts_per = max(200, 20 * n)
        for _ in range(n):
            placed = False
            for _ in range(attempts_per):
                px = float(self.rng.uniform(0.6, Wm - 0.6))
                py = float(self.rng.uniform(0.6, Hm - 0.6))
                if self._world_collides(px, py, PEDESTRIAN_RADIUS + 0.05):
                    continue
                if math.hypot(px - leader_xy[0], py - leader_xy[1]) < leader_bubble:
                    continue
                if math.hypot(px - follower_xy[0], py - follower_xy[1]) < leader_bubble:
                    continue
                if any((px - q.x) ** 2 + (py - q.y) ** 2 < ped_spacing ** 2
                       for q in peds):
                    continue
                peds.append(PedestrianState(
                    x=px, y=py,
                    yaw=float(self.rng.uniform(-math.pi, math.pi)),
                    target_x=px, target_y=py, target_t=0.0))
                placed = True
                break
            if not placed:
                self.get_logger().warn(
                    f"Could only place {len(peds)} of {n} pedestrians.")
                break
        return peds

    # Pedestrian motion intentionally kept simple: random walk with
    # reflection off walls and other peds. No targets, no planning. Cheap and
    # impossible to deadlock — peds just wander like billiard balls with a
    # heading-noise term.

    def _step_pedestrians(self):
        """Brownian-motion pedestrians: walk forward + small heading noise +
        reflect on collision. WALL hits take priority over ped hits when both
        fire — guarantees a ped against a wall always bounces away from the
        wall, not toward it. Plus a hard-escape: if a ped hasn't actually
        moved for ~1s, randomize its heading to break out of any cycle."""
        HEADING_NOISE = 0.10
        for p in self.pedestrians:
            p.yaw += float(self.rng.normal(0.0, HEADING_NOISE))
            nx = p.x + PEDESTRIAN_SPEED * math.cos(p.yaw) * TICK_DT
            ny = p.y + PEDESTRIAN_SPEED * math.sin(p.yaw) * TICK_DT
            hit_box = circle_collides_aabbs(nx, ny, PEDESTRIAN_RADIUS,
                                             self.obstacles)
            hit_ped = self._hits_pedestrian(nx, ny, PEDESTRIAN_RADIUS,
                                             exclude=p)
            hit_robot = (
                (nx - self.leader.x) ** 2 + (ny - self.leader.y) ** 2 <
                (PEDESTRIAN_RADIUS + ROBOT_RADIUS) ** 2 or
                (nx - self.follower.x) ** 2 + (ny - self.follower.y) ** 2 <
                (PEDESTRIAN_RADIUS + ROBOT_RADIUS) ** 2)
            blocked = hit_box or hit_ped or hit_robot
            if not blocked:
                p.x, p.y = nx, ny
            else:
                # Wall normal first (priority): if any obstacle is within
                # radius, use its outward normal regardless of agent hits.
                # Otherwise use the closest agent's normal.
                if hit_box:
                    best = None
                    best_d2 = float("inf")
                    for box in self.obstacles:
                        ccx, ccy = box.closest_point(p.x, p.y)
                        d2 = (p.x - ccx) ** 2 + (p.y - ccy) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best = (ccx, ccy)
                    nxn, nyn = p.x - best[0], p.y - best[1]
                else:
                    others = [(q.x, q.y) for q in self.pedestrians if q is not p]
                    others.append((self.leader.x, self.leader.y))
                    others.append((self.follower.x, self.follower.y))
                    cx, cy = min(others, key=lambda c:
                                  (c[0] - p.x) ** 2 + (c[1] - p.y) ** 2)
                    nxn, nyn = p.x - cx, p.y - cy
                nn = math.hypot(nxn, nyn) + 1e-6
                nxn, nyn = nxn / nn, nyn / nn
                vx, vy = math.cos(p.yaw), math.sin(p.yaw)
                dot = vx * nxn + vy * nyn
                rvx, rvy = vx - 2 * dot * nxn, vy - 2 * dot * nyn
                p.yaw = math.atan2(rvy, rvx) + float(self.rng.uniform(-0.1, 0.1))

            # Hard-escape: if no actual translation in ~1s of trying, snap to
            # a random heading. Catches any reflection-cycle the geometry
            # might trap us in.
            actual_move = math.hypot(p.x - p.prev_x, p.y - p.prev_y)
            if actual_move < 0.005:
                p.stuck_ticks += 1
                if p.stuck_ticks > 20:
                    p.yaw = float(self.rng.uniform(-math.pi, math.pi))
                    p.stuck_ticks = 0
            else:
                p.stuck_ticks = 0
            p.prev_x, p.prev_y = p.x, p.y

    def _on_learned_map(self, msg: OccupancyGrid):
        self._learned_map_msg = msg

    def _on_planned_path(self, msg: Path):
        self._planned_path_msg = msg

    def _on_cmd(self, who: str, msg: Twist):
        v = max(-MAX_LIN_VEL, min(MAX_LIN_VEL, msg.linear.x))
        w = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, msg.angular.z))
        getattr(self, who).vx = v
        getattr(self, who).wz = w

    def _collision_normal(self, x, y, radius, exclude_agent=None):
        """If a disk at (x,y,radius) collides with anything, return (nx,ny)
        outward unit normal pointing away from the colliding surface, else
        None. Walls take priority over agent contacts."""
        # Walls (closest point on closest AABB)
        best_box = None
        best_d2 = float("inf")
        for box in self.obstacles:
            ccx, ccy = box.closest_point(x, y)
            d2 = (x - ccx) ** 2 + (y - ccy) ** 2
            if d2 < radius * radius and d2 < best_d2:
                best_d2 = d2
                best_box = (ccx, ccy)
        if best_box is not None:
            nxn, nyn = x - best_box[0], y - best_box[1]
            n = math.hypot(nxn, nyn) + 1e-6
            return (nxn / n, nyn / n)
        # Other agents: closest center
        candidates = [(p.x, p.y, PEDESTRIAN_RADIUS) for p in self.pedestrians
                      if p is not exclude_agent]
        if exclude_agent is not self.leader:
            candidates.append((self.leader.x, self.leader.y, ROBOT_RADIUS))
        if exclude_agent is not self.follower:
            candidates.append((self.follower.x, self.follower.y, ROBOT_RADIUS))
        best = None
        best_d2 = float("inf")
        for ax, ay, ar in candidates:
            d2 = (x - ax) ** 2 + (y - ay) ** 2
            if d2 < (radius + ar) ** 2 and d2 < best_d2:
                best_d2 = d2
                best = (ax, ay)
        if best is not None:
            nxn, nyn = x - best[0], y - best[1]
            n = math.hypot(nxn, nyn) + 1e-6
            return (nxn / n, nyn / n)
        return None

    def _integrate(self, s: RobotState, dt: float, can_push: bool = False) -> bool:
        """Continuous motion integration with elastic bounce on collision.
        On any collision (wall / ped / other robot), the agent's heading is
        REFLECTED across the surface normal — same physics the Brownian peds
        use. The agent's own controller sees the redirected yaw next tick
        and reorients toward its goal. This means hitting a wall doesn't
        freeze the agent — it slides/bounces away naturally.

        If `can_push` is set (leader only), the leader also displaces any
        pedestrian OR the follower a small amount in its heading direction
        on contact, so a determined leader can shove past blockers it
        couldn't otherwise route around."""
        new_yaw = s.yaw + s.wz * dt
        nx = s.x + s.vx * math.cos(new_yaw) * dt
        ny = s.y + s.vx * math.sin(new_yaw) * dt
        # Robots (leader/follower) DON'T bounce. Bounce physics was useful
        # for Brownian peds (random motion) but actively hurts a controlled
        # robot: when the controller carefully rotates toward last_seen,
        # the world reflecting yaw across a nearby wall normal destroys
        # the controller's progress. So for robots, collision = motion
        # rejected, yaw preserved (controller decides what to do next).
        # Pure rotation in place is always allowed.
        if not self._world_collides(nx, ny, ROBOT_RADIUS):
            other = self.follower if s is self.leader else self.leader
            hits_other = ((nx - other.x) ** 2 + (ny - other.y) ** 2 <
                          (2 * ROBOT_RADIUS) ** 2)
            hits_ped = self._hits_pedestrian(nx, ny, ROBOT_RADIUS)
            if not (hits_other or hits_ped):
                s.x, s.y, s.yaw = nx, ny, new_yaw
                return True
            # Push if allowed
            if can_push and s.vx > 0.05:
                push_x = math.cos(new_yaw) * 0.08
                push_y = math.sin(new_yaw) * 0.08
                if hits_ped:
                    for p in self.pedestrians:
                        if ((nx - p.x) ** 2 + (ny - p.y) ** 2 <
                                (ROBOT_RADIUS + PEDESTRIAN_RADIUS) ** 2):
                            cand_x, cand_y = p.x + push_x, p.y + push_y
                            if (not self._world_collides(
                                    cand_x, cand_y, PEDESTRIAN_RADIUS) and
                                    not self._hits_pedestrian(
                                        cand_x, cand_y, PEDESTRIAN_RADIUS, exclude=p)):
                                p.x, p.y = cand_x, cand_y
                if hits_other:
                    cand_x, cand_y = other.x + push_x, other.y + push_y
                    if (not self._world_collides(cand_x, cand_y, ROBOT_RADIUS)
                            and not self._hits_pedestrian(
                                cand_x, cand_y, ROBOT_RADIUS)):
                        other.x, other.y = cand_x, cand_y
        # Blocked: motion rejected, yaw preserved. Controller will see we
        # didn't move and replan. No reflection.
        s.yaw = new_yaw
        return False

    def _step(self):
        if not self._integrate(self.leader, TICK_DT, can_push=True):
            self.collision_count_leader += 1
        if not self._integrate(self.follower, TICK_DT):
            self.collision_count_follower += 1
        self._step_pedestrians()

        self.tick += 1
        now = self.get_clock().now().to_msg()

        for who, st in (("leader", self.leader), ("follower", self.follower)):
            od = Odometry()
            od.header.stamp = now
            od.header.frame_id = f"{who}/odom"
            od.child_frame_id = f"{who}/base_link"
            od.pose.pose.position.x = st.x
            od.pose.pose.position.y = st.y
            od.pose.pose.orientation = yaw_to_quat(st.yaw)
            od.twist.twist.linear.x = st.vx
            od.twist.twist.angular.z = st.wz
            getattr(self, f"pub_{who}_odom").publish(od)

            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = f"{who}/odom"
            tf.child_frame_id = f"{who}/base_link"
            tf.transform.translation.x = st.x
            tf.transform.translation.y = st.y
            tf.transform.rotation = yaw_to_quat(st.yaw)
            self.tfb.sendTransform(tf)

        # LiDAR ray-vs-circle hits BOTH the leader and any pedestrians, so the
        # follower's brake/avoidance treats pedestrians as moving obstacles.
        scan = self.lidar.scan(self.obstacles, self.follower,
                               others=[self.leader] + list(self.pedestrians))
        scan.header.stamp = now
        scan.header.frame_id = "follower/base_link"
        self.pub_follower_scan.publish(scan)
        self._last_lidar_ranges = list(scan.ranges)

        # Camera occludes the leader detection through any pedestrian standing
        # between follower and leader.
        detect = self.camera.detect(self.obstacles, self.follower,
                                    leader=self.leader,
                                    pedestrians=self.pedestrians)
        detect.header.stamp = now
        detect.header.frame_id = "follower/base_link"
        self.pub_follower_detect.publish(detect)
        self._last_detection_visible = bool(detect.detections)

        # Separate topic for pedestrian detections — labelled "pedestrian", so
        # the follower distinguishes them from the leader.
        ped_detect = self.camera.detect_pedestrians(
            self.obstacles, self.follower, pedestrians=self.pedestrians)
        ped_detect.header.stamp = now
        ped_detect.header.frame_id = "follower/base_link"
        self.pub_follower_ped_detect.publish(ped_detect)

        gt = PoseStamped()
        gt.header.stamp = now
        gt.header.frame_id = "world"
        gt.pose.position.x = self.leader.x
        gt.pose.position.y = self.leader.y
        gt.pose.orientation = yaw_to_quat(self.leader.yaw)
        self.pub_leader_truth.publish(gt)

        # Pedestrian ground-truth poses for the leader's planner.
        pa = PoseArray()
        pa.header.stamp = now
        pa.header.frame_id = "world"
        for p in self.pedestrians:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.orientation = yaw_to_quat(p.yaw)
            pa.poses.append(pose)
        self.pub_pedestrian_poses.publish(pa)

        if self._viz is not None:
            self._viz.update(self,
                             last_detection=self._last_detection_visible,
                             last_lidar_ranges=self._last_lidar_ranges,
                             learned_map=self._learned_map_msg,
                             planned_path=self._planned_path_msg)


def main():
    rclpy.init()
    map_path = os.environ.get("SIM_MAP", "sim/maps/empty.txt")
    seed = int(os.environ.get("SIM_SEED", "0"))
    render = os.environ.get("SIM_RENDER", "0") not in ("0", "", "false", "False")
    node = World(map_path, seed, render=render)
    try:
        rclpy.spin(node)
    finally:
        if node._viz is not None:
            node._viz.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
