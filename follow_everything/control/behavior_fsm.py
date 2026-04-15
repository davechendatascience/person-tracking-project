"""Five-state Behaviour Finite State Machine.

Paper reference: "Follow Everything" §III-B / Fig. 3

States
------
FOLLOWING  — leader visible, within [D_min, D_max].
             Goal: circle of radius D_safe centred on leader.
             D_safe = clip(nis_alpha * NIS, D_min, D_max)

CHASING    — leader visible but farther than D_max.
             Goal: point toward leader at D_max distance.
             V_max adapted to leader speed + distance.

RETREATING — leader closer than D_min (approaching too fast).
             Goal: point on D_min circle, away from leader.

SEARCHING  — leader not visible; navigate to last known position.

STOPPED    — leader has been missing for > search_timeout_s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from follow_everything.estimation.leader_ekf import LeaderState
from follow_everything.utils.geometry import (
    angle_to_point,
    normalize_angle,
)


class State(Enum):
    FOLLOWING  = auto()
    CHASING    = auto()
    RETREATING = auto()
    SEARCHING  = auto()
    STOPPED    = auto()


@dataclass
class ControlOutput:
    linear_vel:  float   # m/s, positive = forward
    angular_vel: float   # rad/s, positive = counter-clockwise
    state:       State
    goal:        Optional[np.ndarray]   # 2-D world frame, for visualisation


class BehaviorFSM:
    def __init__(
        self,
        d_min: float = 0.8,
        d_max: float = 3.0,
        nis_alpha: float = 0.4,
        v_alpha1: float = 0.6,      # leader speed weight for chase V_max
        v_alpha2: float = 0.4,      # leader distance weight for chase V_max
        max_linear_vel: float = 1.5,
        max_angular_vel: float = 1.2,
        k_angular: float = 2.0,     # proportional heading gain
        lookahead: float = 0.8,     # pure-pursuit lookahead distance (m)
        search_timeout_s: float = 5.0,
    ):
        self.d_min = d_min
        self.d_max = d_max
        self.nis_alpha = nis_alpha
        self.v_alpha1 = v_alpha1
        self.v_alpha2 = v_alpha2
        self.max_v   = max_linear_vel
        self.max_w   = max_angular_vel
        self.k_w     = k_angular
        self.lookahead = lookahead
        self.search_timeout_s = search_timeout_s

        self._state: State = State.SEARCHING
        self._last_seen_time: float = time.monotonic()
        self._last_known_pos: Optional[np.ndarray] = None  # world frame (2,)

    # ------------------------------------------------------------------
    def step(
        self,
        robot_pose:    np.ndarray,         # (3,) [x, y, theta] world frame
        leader_state:  Optional[LeaderState],
        is_visible:    bool,
        nis:           float,
        path:          List[np.ndarray],   # waypoints from planner (world frame)
    ) -> ControlOutput:
        """Advance the FSM one time-step and return velocity commands."""
        now = time.monotonic()

        # Update last-seen bookkeeping
        if is_visible and leader_state is not None:
            self._last_seen_time = now
            self._last_known_pos = leader_state.position  # already in world frame

        # Distance: robot to leader (both in world frame)
        dist = (
            float(np.linalg.norm(leader_state.position - robot_pose[:2]))
            if leader_state is not None else np.inf
        )

        # ---- State transitions ----------------------------------------
        elapsed_lost = now - self._last_seen_time

        if not is_visible:
            self._state = (
                State.STOPPED if elapsed_lost > self.search_timeout_s
                else State.SEARCHING
            )
        elif dist < self.d_min:
            self._state = State.RETREATING
        elif dist > self.d_max:
            self._state = State.CHASING
        else:
            self._state = State.FOLLOWING

        # ---- Goal computation ----------------------------------------
        goal = self._compute_goal(robot_pose, leader_state, dist, nis)

        # ---- Velocity controller ------------------------------------
        if self._state == State.STOPPED or goal is None:
            return ControlOutput(0.0, 0.0, self._state, goal)

        # Use first waypoint from planner if available, else direct goal
        waypoint = path[0] if path else goal
        v, w = self._pursue(robot_pose, waypoint, dist, leader_state)

        # State-specific speed scaling
        if self._state == State.CHASING:
            v_scale = self._chase_speed_scale(leader_state, dist)
            v = min(v * v_scale, self.max_v)
        elif self._state == State.RETREATING:
            v = -abs(v) * 0.5   # slow reverse

        return ControlOutput(
            linear_vel=np.clip(v, -self.max_v, self.max_v),
            angular_vel=np.clip(w, -self.max_w, self.max_w),
            state=self._state,
            goal=goal,
        )

    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    def _compute_goal(
        self,
        robot_pose:   np.ndarray,
        leader_state: Optional[LeaderState],
        dist:         float,
        nis:          float,
    ) -> Optional[np.ndarray]:
        """Return the 2-D world-frame goal point for the current state."""
        if self._state == State.STOPPED:
            return None

        if self._state == State.SEARCHING:
            return self._last_known_pos

        assert leader_state is not None
        leader_world = leader_state.position   # EKF tracks in world frame

        if self._state == State.FOLLOWING:
            d_safe = float(np.clip(
                self.nis_alpha * nis, self.d_min, self.d_max
            ))
            # Point on the safe-distance circle closest to the robot
            direction = robot_pose[:2] - leader_world
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0])
            else:
                direction /= norm
            return leader_world + direction * d_safe

        if self._state == State.CHASING:
            direction = leader_world - robot_pose[:2]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return leader_world
            return robot_pose[:2] + direction / norm * self.d_max

        if self._state == State.RETREATING:
            # Back up to D_min circle
            direction = robot_pose[:2] - leader_world
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0])
            else:
                direction /= norm
            return leader_world + direction * self.d_min

        return None

    def _pursue(
        self,
        robot_pose:   np.ndarray,
        waypoint:     np.ndarray,
        dist:         float,
        leader_state: Optional[LeaderState],
    ) -> Tuple[float, float]:
        """Proportional heading controller toward waypoint."""
        angle_err = angle_to_point(robot_pose, waypoint)
        w = float(np.clip(self.k_w * angle_err, -self.max_w, self.max_w))

        dist_to_wp = float(np.linalg.norm(waypoint - robot_pose[:2]))
        # Reduce speed when turning sharply or near waypoint
        heading_factor = max(0.0, 1.0 - abs(angle_err) / np.pi)
        dist_factor = min(dist_to_wp / max(self.lookahead, 1e-3), 1.0)
        v = self.max_v * heading_factor * dist_factor

        return v, w

    def _chase_speed_scale(
        self, leader_state: LeaderState, dist: float
    ) -> float:
        """V_max adaptation: eq. (4) of "Follow Everything".

        V_max = clip(α1 * |v_leader| + α2 * |d_leader|, 0, 1.5)
        """
        v_leader = float(np.linalg.norm(leader_state.velocity))
        raw = self.v_alpha1 * v_leader + self.v_alpha2 * dist
        return float(np.clip(raw / self.max_v, 0.5, 1.5))
