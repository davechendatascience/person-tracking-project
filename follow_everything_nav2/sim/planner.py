"""
A* planner used by the leader's autonomous patrol. Operates on a fine
discretization of the continuous obstacle list, so the path is collision-free
for a robot of given radius. Smoothing then converts back to continuous
waypoints. Available for the agent's follower stack to import if useful.
"""
import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

from sim.geometry import AABB, circle_collides_aabbs, line_segment_clear


def build_planning_grid(obstacles: List[AABB], world_size: Tuple[float, float],
                         radius_m: float, resolution: float = 0.25) -> np.ndarray:
    """Discretise the continuous world for grid-based planning. A cell is marked
    'blocked' if a disk of `radius_m` placed at the cell center would collide with
    any obstacle (or wall)."""
    Wm, Hm = world_size
    nx = int(math.ceil(Wm / resolution))
    ny = int(math.ceil(Hm / resolution))
    grid = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5) * resolution
            y = (j + 0.5) * resolution
            if (x < radius_m or y < radius_m or
                    x > Wm - radius_m or y > Hm - radius_m):
                grid[j, i] = True
                continue
            if circle_collides_aabbs(x, y, radius_m, obstacles):
                grid[j, i] = True
    return grid


def astar(grid: np.ndarray,
          start_xy: Tuple[float, float],
          goal_xy: Tuple[float, float],
          resolution: float = 0.25) -> Optional[List[Tuple[float, float]]]:
    H, W = grid.shape

    def world_to_cell(p):
        i = int(p[0] / resolution)
        j = int(p[1] / resolution)
        return (j, i)

    def cell_to_world(c):
        j, i = c
        return ((i + 0.5) * resolution, (j + 0.5) * resolution)

    s = world_to_cell(start_xy)
    g = world_to_cell(goal_xy)
    if not (0 <= s[0] < H and 0 <= s[1] < W):
        return None
    if not (0 <= g[0] < H and 0 <= g[1] < W):
        return None
    if grid[g]:
        return None

    def heur(a, b):
        dy = abs(a[0] - b[0])
        dx = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_q = [(heur(s, g), 0.0, s, None)]
    came = {}
    closed = set()
    while open_q:
        f, gscore, cur, parent = heapq.heappop(open_q)
        if cur in closed:
            continue
        closed.add(cur)
        came[cur] = parent
        if cur == g:
            cells = []
            n = cur
            while n is not None:
                cells.append(n)
                n = came[n]
            cells.reverse()
            return [cell_to_world(c) for c in cells]
        cj, ci = cur
        for dj, di in nbrs:
            nj, ni = cj + dj, ci + di
            if not (0 <= nj < H and 0 <= ni < W):
                continue
            if grid[nj, ni]:
                continue
            step = math.sqrt(2) if (dj != 0 and di != 0) else 1.0
            ng = gscore + step
            heapq.heappush(open_q, (ng + heur((nj, ni), g), ng, (nj, ni), cur))
    return None


def smooth_path(path: List[Tuple[float, float]],
                obstacles: List[AABB],
                clearance: float = 0.0) -> List[Tuple[float, float]]:
    """Greedy line-of-sight smoothing in CONTINUOUS space.

    `clearance` inflates each obstacle by this many metres before the LOS check
    so that a robot of `clearance` radius can travel the smoothed segment
    without grazing corners. Pass clearance=ROBOT_RADIUS (or larger) when the
    path will be executed by a finite-radius vehicle.
    """
    if not path:
        return path
    if clearance > 0:
        infl = [AABB(b.xmin - clearance, b.ymin - clearance,
                     b.xmax + clearance, b.ymax + clearance) for b in obstacles]
    else:
        infl = obstacles
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = i + 1
        for k in range(i + 2, len(path)):
            if line_segment_clear(path[i][0], path[i][1], path[k][0], path[k][1], infl):
                j = k
            else:
                break
        out.append(path[j])
        i = j
    return out
