"""
Continuous-space sensor models. DO NOT MODIFY.

CameraDetector: 90° FOV cone, 6 m max range. Returns the leader's pose in the
follower's body frame IFF the leader is in cone, in range, and line-of-sight is
not blocked by any obstacle (analytical ray-vs-AABB intersection — no grid
stepping). This is the "simulated SAM2" — no real inference, just a geometry oracle.

LidarSensor: 360°, 72 rays at 5° spacing, 8 m max. Each ray is closed-form
intersected with all AABB obstacles AND with the leader (modeled as a circle)
and reports the nearest hit distance.
"""
import math
import os
from typing import List

import numpy as np
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

from sim.geometry import (
    AABB, ray_min_distance, ray_circle_distance,
    line_segment_clear, segment_hits_circle,
)


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class C:
    # Default sensor params per program.md. Overridable by env for stress
    # testing only — the autoresearch agent must NOT change these.
    CAMERA_FOV_DEG = _env_float("CAMERA_FOV_DEG", 90.0)
    CAMERA_MAX_RANGE = _env_float("CAMERA_MAX_RANGE", 6.0)

    LIDAR_FOV_DEG = 360.0
    LIDAR_NUM_RAYS = 72
    LIDAR_MAX_RANGE = _env_float("LIDAR_MAX_RANGE", 8.0)

    LEADER_RADIUS = 0.25  # match world.ROBOT_RADIUS


class LidarSensor:
    """360° 2D LiDAR (continuous-space). Hits both AABB obstacles and the leader."""

    def __init__(self):
        self.angles = np.deg2rad(np.linspace(
            0.0, C.LIDAR_FOV_DEG, C.LIDAR_NUM_RAYS, endpoint=False))

    def scan(self, obstacles: List[AABB], follower, others=()) -> LaserScan:
        msg = LaserScan()
        msg.angle_min = 0.0
        msg.angle_max = math.radians(C.LIDAR_FOV_DEG)
        msg.angle_increment = math.radians(C.LIDAR_FOV_DEG) / C.LIDAR_NUM_RAYS
        msg.range_min = 0.05
        msg.range_max = float(C.LIDAR_MAX_RANGE)

        ranges = []
        for theta in self.angles:
            world_th = follower.yaw + theta
            dx, dy = math.cos(world_th), math.sin(world_th)
            d = ray_min_distance(follower.x, follower.y, dx, dy,
                                 obstacles, max_t=C.LIDAR_MAX_RANGE)
            for o in others:
                d_o = ray_circle_distance(follower.x, follower.y, dx, dy,
                                          o.x, o.y, C.LEADER_RADIUS)
                if d_o is not None and 0 < d_o < d:
                    d = d_o
            ranges.append(float(d))
        msg.ranges = ranges
        return msg


class CameraDetector:
    """Forward-facing camera detector. Returns the leader's pose in the follower's
    body frame iff: (1) within cone, (2) within range, (3) clear line of sight
    against AABB obstacles AND any pedestrian circles passed in."""

    def detect(self, obstacles: List[AABB], follower, leader,
               pedestrians=()) -> Detection2DArray:
        msg = Detection2DArray()
        dx = leader.x - follower.x
        dy = leader.y - follower.y
        d = math.hypot(dx, dy)
        if d > C.CAMERA_MAX_RANGE:
            return msg
        bearing_world = math.atan2(dy, dx)
        rel = (bearing_world - follower.yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(math.degrees(rel)) > C.CAMERA_FOV_DEG / 2.0:
            return msg
        # Line-of-sight: segment from camera to (just inside) the leader's body.
        target_x = leader.x - (dx / d) * (C.LEADER_RADIUS - 0.01)
        target_y = leader.y - (dy / d) * (C.LEADER_RADIUS - 0.01)
        if not line_segment_clear(follower.x, follower.y, target_x, target_y, obstacles):
            return msg
        # Pedestrians block the LOS too.
        for p in pedestrians:
            if segment_hits_circle(follower.x, follower.y, target_x, target_y,
                                   p.x, p.y, C.LEADER_RADIUS):
                return msg

        det = Detection2D()
        c, s = math.cos(-follower.yaw), math.sin(-follower.yaw)
        body_x = c * dx - s * dy
        body_y = s * dx + c * dy
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "leader"
        hyp.hypothesis.score = 1.0
        hyp.pose.pose.position.x = body_x
        hyp.pose.pose.position.y = body_y
        det.results.append(hyp)
        msg.detections.append(det)
        return msg

    def detect_pedestrians(self, obstacles: List[AABB], follower,
                           pedestrians=()) -> Detection2DArray:
        """Separate detection topic for pedestrians — the follower sees them
        labelled distinctly from the leader (analogous to SAM2 instance ID).
        Each visible pedestrian is reported in body frame."""
        msg = Detection2DArray()
        for p in pedestrians:
            dx = p.x - follower.x
            dy = p.y - follower.y
            d = math.hypot(dx, dy)
            if d > C.CAMERA_MAX_RANGE:
                continue
            bearing_world = math.atan2(dy, dx)
            rel = (bearing_world - follower.yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(math.degrees(rel)) > C.CAMERA_FOV_DEG / 2.0:
                continue
            tx = p.x - (dx / d) * (C.LEADER_RADIUS - 0.01)
            ty = p.y - (dy / d) * (C.LEADER_RADIUS - 0.01)
            if not line_segment_clear(follower.x, follower.y, tx, ty, obstacles):
                continue
            # Other pedestrians can occlude this one too.
            blocked = False
            for q in pedestrians:
                if q is p:
                    continue
                if segment_hits_circle(follower.x, follower.y, tx, ty,
                                       q.x, q.y, C.LEADER_RADIUS):
                    blocked = True
                    break
            if blocked:
                continue
            det = Detection2D()
            c, s = math.cos(-follower.yaw), math.sin(-follower.yaw)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "pedestrian"
            hyp.hypothesis.score = 1.0
            hyp.pose.pose.position.x = c * dx - s * dy
            hyp.pose.pose.position.y = s * dx + c * dy
            det.results.append(hyp)
            msg.detections.append(det)
        return msg
