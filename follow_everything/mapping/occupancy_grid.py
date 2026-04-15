"""2-D occupancy grid built from LiDAR point clouds.

The grid is centred on the robot (robot frame: x-forward, y-left, z-up).
After raw cell marking, an inflation pass expands each obstacle by the
robot radius so that path planning can treat the robot as a point.
"""

import numpy as np
from scipy.ndimage import binary_dilation, label as nd_label
from typing import List, Tuple


class OccupancyGrid:
    def __init__(
        self,
        resolution: float = 0.1,       # metres per cell
        map_size: float = 12.0,         # side length of square map (metres)
        inflation_radius: float = 0.45, # robot radius + safety clearance
        min_height: float = 0.05,       # ignore floor returns below this
        max_height: float = 2.0,        # ignore returns above this
    ):
        self.resolution = resolution
        self.map_size = map_size
        self.inflation_radius = inflation_radius
        self.min_height = min_height
        self.max_height = max_height

        self.n = int(map_size / resolution)
        # Robot sits at grid centre
        self._origin = np.array([map_size / 2, map_size / 2])

        self._raw: np.ndarray = np.zeros((self.n, self.n), dtype=bool)
        self._inflated: np.ndarray = np.zeros((self.n, self.n), dtype=bool)
        self._disk = self._make_disk(int(np.ceil(inflation_radius / resolution)))

    # ------------------------------------------------------------------
    def update(self, lidar_points: np.ndarray) -> None:
        """Rebuild grid from a new LiDAR sweep.

        Args:
            lidar_points: (N, 3+) array; columns 0-2 are XYZ in robot frame.
        """
        self._raw[:] = False

        pts = lidar_points[:, :3]
        mask = (pts[:, 2] >= self.min_height) & (pts[:, 2] <= self.max_height)
        pts = pts[mask]

        if len(pts) == 0:
            self._inflated[:] = False
            return

        cols = np.floor((pts[:, 0] + self._origin[0]) / self.resolution).astype(int)
        rows = np.floor((pts[:, 1] + self._origin[1]) / self.resolution).astype(int)
        valid = (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
        self._raw[rows[valid], cols[valid]] = True

        self._inflated = binary_dilation(self._raw, structure=self._disk)

    # ------------------------------------------------------------------
    def is_occupied(self, x: float, y: float) -> bool:
        """Check inflated grid at world-frame (robot-centric) position."""
        row, col = self.world_to_cell(x, y)
        if 0 <= row < self.n and 0 <= col < self.n:
            return bool(self._inflated[row, col])
        return False  # out-of-map: unknown, treat as free for collision metric

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x + self._origin[0]) / self.resolution)
        row = int((y + self._origin[1]) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = col * self.resolution - self._origin[0]
        y = row * self.resolution - self._origin[1]
        return x, y

    # ------------------------------------------------------------------
    def obstacle_clusters(self) -> List[np.ndarray]:
        """Return list of (N, 2) world-coord arrays, one per obstacle cluster."""
        labelled, n_clusters = nd_label(self._inflated)
        clusters = []
        for i in range(1, n_clusters + 1):
            cells = np.argwhere(labelled == i)  # (N, 2) row, col
            world = np.array([self.cell_to_world(r, c) for r, c in cells])
            clusters.append(world)
        return clusters

    # ------------------------------------------------------------------
    @property
    def raw_grid(self) -> np.ndarray:
        return self._raw

    @property
    def inflated_grid(self) -> np.ndarray:
        return self._inflated

    @property
    def origin(self) -> np.ndarray:
        return self._origin.copy()

    # ------------------------------------------------------------------
    @staticmethod
    def _make_disk(radius: int) -> np.ndarray:
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        return (x ** 2 + y ** 2) <= radius ** 2
