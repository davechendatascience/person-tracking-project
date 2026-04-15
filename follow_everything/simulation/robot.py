"""Differential-drive kinematic simulator.

Used for offline evaluation: we feed velocity commands from the FSM
into this model to simulate where the robot would have moved, then
compare against the ground-truth sequences in CrowdBot_v2.
"""

from __future__ import annotations

import numpy as np

from follow_everything.utils.geometry import normalize_angle


class DiffDriveRobot:
    """Unicycle (point-mass differential drive) kinematic model.

    State:  [x, y, theta]  — world frame; x-east, y-north, theta from x.
    Input:  [v, omega]     — linear (m/s) and angular (rad/s) velocity.
    """

    def __init__(
        self,
        dt: float = 0.067,
        max_linear_vel: float = 1.5,
        max_angular_vel: float = 1.2,
        initial_pose: np.ndarray | None = None,
    ):
        self.dt = dt
        self.max_v = max_linear_vel
        self.max_w = max_angular_vel
        self.pose = (
            np.zeros(3) if initial_pose is None else np.array(initial_pose, dtype=float)
        )
        self._trajectory: list[np.ndarray] = [self.pose.copy()]

    # ------------------------------------------------------------------
    def step(self, v: float, omega: float) -> np.ndarray:
        """Euler-integrate one timestep; returns new pose."""
        v     = float(np.clip(v,     -self.max_v, self.max_v))
        omega = float(np.clip(omega, -self.max_w, self.max_w))

        x, y, theta = self.pose
        x     += v * np.cos(theta) * self.dt
        y     += v * np.sin(theta) * self.dt
        theta  = normalize_angle(theta + omega * self.dt)

        self.pose = np.array([x, y, theta])
        self._trajectory.append(self.pose.copy())
        return self.pose

    def reset(self, pose: np.ndarray | None = None) -> None:
        self.pose = np.zeros(3) if pose is None else np.array(pose, dtype=float)
        self._trajectory = [self.pose.copy()]

    @property
    def position(self) -> np.ndarray:
        return self.pose[:2].copy()

    @property
    def trajectory(self) -> np.ndarray:
        """(T, 3) array of all visited poses."""
        return np.array(self._trajectory)
