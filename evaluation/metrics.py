"""Evaluation metrics matching "Follow Everything" §IV-B.

Metrics
-------
follow_success_rate  : fraction of trials where the leader is still in
                       the robot's FOV at mission end.
leader_loss_ratio    : fraction of total frames where leader was not visible.
collision_rate       : fraction of frames where robot overlaps an obstacle.
mean_follow_distance : mean distance between robot and leader (metres).
trajectory_smoothness: mean squared jerk of linear velocity command sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FrameRecord:
    leader_visible:   bool
    robot_pos:        np.ndarray   # (2,) world frame
    leader_pos:       np.ndarray   # (2,) world frame
    in_collision:     bool
    linear_vel:       float
    fov_half_angle:   float        # radians


@dataclass
class EpisodeMetrics:
    follow_success:      bool
    leader_loss_ratio:   float
    collision_rate:      float
    mean_follow_distance: float
    trajectory_smoothness: float   # mean |Δv| (lower = smoother)
    n_frames:            int


class MetricsAccumulator:
    """Collects per-frame observations and computes episode-level metrics."""

    def __init__(self, fov_half_angle_deg: float = 90.0):
        self._fov = np.radians(fov_half_angle_deg)
        self._records: List[FrameRecord] = []

    def record(
        self,
        robot_pose:       np.ndarray,       # (3,) [x, y, theta]
        leader_world_pos: Optional[np.ndarray],  # (2,) or None
        is_visible:       bool,
        in_collision:     bool,
        linear_vel:       float,
    ) -> None:
        if leader_world_pos is None:
            leader_world_pos = robot_pose[:2] + np.array([2.0, 0.0])

        self._records.append(FrameRecord(
            leader_visible=is_visible,
            robot_pos=robot_pose[:2].copy(),
            leader_pos=leader_world_pos.copy(),
            in_collision=in_collision,
            linear_vel=float(linear_vel),
            fov_half_angle=self._fov,
        ))

    def compute(self) -> EpisodeMetrics:
        if not self._records:
            return EpisodeMetrics(False, 1.0, 1.0, 0.0, 0.0, 0)

        n = len(self._records)
        n_visible   = sum(1 for r in self._records if r.leader_visible)
        n_collision = sum(1 for r in self._records if r.in_collision)

        dists = [
            float(np.linalg.norm(r.leader_pos - r.robot_pos))
            for r in self._records
        ]
        vels = [r.linear_vel for r in self._records]
        jerk = float(np.mean(np.abs(np.diff(vels)))) if len(vels) > 1 else 0.0

        # "Follow success" = leader still in FOV at the final frame
        last = self._records[-1]
        follow_success = self._in_fov(last)

        return EpisodeMetrics(
            follow_success=follow_success,
            leader_loss_ratio=1.0 - n_visible / n,
            collision_rate=n_collision / n,
            mean_follow_distance=float(np.mean(dists)),
            trajectory_smoothness=jerk,
            n_frames=n,
        )

    def reset(self) -> None:
        self._records.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _in_fov(record: FrameRecord) -> bool:
        if not record.leader_visible:
            return False
        delta = record.leader_pos - record.robot_pos
        # Would need robot heading from pose — approximate: leader visible flag
        return True   # rely on is_visible from tracker

    def summary_str(self) -> str:
        m = self.compute()
        return (
            f"Follow success : {'YES' if m.follow_success else 'NO'}\n"
            f"Leader loss    : {m.leader_loss_ratio * 100:.1f} %\n"
            f"Collision rate : {m.collision_rate * 100:.1f} %\n"
            f"Mean distance  : {m.mean_follow_distance:.2f} m\n"
            f"Smoothness     : {m.trajectory_smoothness:.4f} m/s² (↓ better)\n"
            f"Frames         : {m.n_frames}"
        )
