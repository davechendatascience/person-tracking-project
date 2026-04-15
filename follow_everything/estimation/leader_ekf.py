"""Kalman filter for 2-D leader state estimation (position + velocity).

Paper reference: "Follow Everything" §III-B — the NIS value from this filter
drives the adaptive safe-distance in the FOLLOWING state.
"""

import numpy as np
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter


@dataclass
class LeaderState:
    position: np.ndarray   # [x, y] in robot frame (metres)
    velocity: np.ndarray   # [vx, vy] in robot frame (m/s)
    covariance: np.ndarray  # 4×4 state covariance


class LeaderEKF:
    """Constant-velocity Kalman filter tracking leader in 2-D robot frame.

    State  x = [px, py, vx, vy]
    Obs    z = [px, py]  (from depth-projected mask centroid)
    """

    def __init__(
        self,
        dt: float = 0.067,
        process_noise_std: float = 0.3,
        measurement_noise_std: float = 0.4,
        initial_uncertainty: float = 5.0,
    ):
        self.dt = dt
        self._kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition — constant-velocity model
        self._kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Observation model — measure position only
        self._kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise (continuous white-noise acceleration model)
        q = process_noise_std ** 2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        self._kf.Q = q * np.array([
            [dt4 / 4, 0,       dt3 / 2, 0      ],
            [0,       dt4 / 4, 0,       dt3 / 2],
            [dt3 / 2, 0,       dt2,     0      ],
            [0,       dt3 / 2, 0,       dt2    ],
        ])

        # Measurement noise
        r = measurement_noise_std ** 2
        self._kf.R = r * np.eye(2)

        # Initial covariance
        self._kf.P = initial_uncertainty * np.eye(4)

        # Initialise with leader straight ahead at 2 m
        self._kf.x = np.array([2.0, 0.0, 0.0, 0.0])

        self._initialized = False
        self._last_innovation = np.zeros(2)
        self._last_S = np.eye(2)

    # ------------------------------------------------------------------
    def initialize(self, position: np.ndarray) -> None:
        """Seed the filter from a known first observation."""
        self._kf.x = np.array([position[0], position[1], 0.0, 0.0])
        self._initialized = True

    def predict(self) -> LeaderState:
        """Advance the filter one time-step (call even when no measurement)."""
        self._kf.predict()
        return self._state_snapshot()

    def update(self, measurement: np.ndarray) -> LeaderState:
        """Fuse a new [x, y] observation. Calls predict() internally."""
        if not self._initialized:
            self.initialize(measurement)
            return self._state_snapshot()

        self._kf.predict()

        # Cache pre-update innovation for NIS computation
        y = measurement - (self._kf.H @ self._kf.x_prior)
        S = self._kf.H @ self._kf.P_prior @ self._kf.H.T + self._kf.R
        self._last_innovation = y
        self._last_S = S

        self._kf.update(measurement)
        return self._state_snapshot()

    # ------------------------------------------------------------------
    @property
    def nis(self) -> float:
        """Normalized Innovation Squared — used to scale safe following distance.

        High NIS → filter is surprised → increase following distance.
        """
        try:
            S_inv = np.linalg.inv(self._last_S)
            return float(self._last_innovation @ S_inv @ self._last_innovation)
        except np.linalg.LinAlgError:
            return 1.0

    @property
    def state(self) -> LeaderState:
        return self._state_snapshot()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    def _state_snapshot(self) -> LeaderState:
        x = self._kf.x
        return LeaderState(
            position=x[:2].copy(),
            velocity=x[2:].copy(),
            covariance=self._kf.P.copy(),
        )
