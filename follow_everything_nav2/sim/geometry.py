"""
Continuous-space geometry: axis-aligned bounding boxes (AABBs), circle-vs-AABB
collision, and ray-vs-AABB intersection (slab method). Used by world.py for
collision and by sensors.py for line-of-sight checks. NO grid-stepping.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AABB:
    """Axis-aligned box in continuous world coords. xmin <= xmax, ymin <= ymax."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def closest_point(self, x: float, y: float) -> Tuple[float, float]:
        cx = max(self.xmin, min(self.xmax, x))
        cy = max(self.ymin, min(self.ymax, y))
        return cx, cy


def circle_collides_aabbs(x: float, y: float, radius: float,
                          boxes: List[AABB]) -> bool:
    """True if a disk at (x, y) of given radius overlaps any AABB."""
    for b in boxes:
        cx, cy = b.closest_point(x, y)
        dx = x - cx
        dy = y - cy
        if dx * dx + dy * dy < radius * radius:
            return True
    return False


def ray_aabb_distance(x0: float, y0: float, dx: float, dy: float,
                      box: AABB, max_t: float = float("inf")) -> Optional[float]:
    """Distance along ray (x0+t*dx, y0+t*dy) to box surface (t >= 0).
    Returns None if no hit ahead. (dx, dy) must be normalised."""
    inv_dx = 1.0 / dx if abs(dx) > 1e-9 else float("inf")
    inv_dy = 1.0 / dy if abs(dy) > 1e-9 else float("inf")

    if dx > 0:
        tx1, tx2 = (box.xmin - x0) * inv_dx, (box.xmax - x0) * inv_dx
    elif dx < 0:
        tx1, tx2 = (box.xmax - x0) * inv_dx, (box.xmin - x0) * inv_dx
    else:
        if x0 < box.xmin or x0 > box.xmax:
            return None
        tx1, tx2 = -float("inf"), float("inf")

    if dy > 0:
        ty1, ty2 = (box.ymin - y0) * inv_dy, (box.ymax - y0) * inv_dy
    elif dy < 0:
        ty1, ty2 = (box.ymax - y0) * inv_dy, (box.ymin - y0) * inv_dy
    else:
        if y0 < box.ymin or y0 > box.ymax:
            return None
        ty1, ty2 = -float("inf"), float("inf")

    t_enter = max(tx1, ty1)
    t_exit = min(tx2, ty2)
    if t_exit < t_enter or t_exit < 0:
        return None
    if t_enter < 0:  # ray origin inside the box
        return 0.0
    if t_enter > max_t:
        return None
    return t_enter


def ray_min_distance(x0: float, y0: float, dx: float, dy: float,
                     boxes: List[AABB], max_t: float) -> float:
    """Minimum hit distance over a list of AABBs (returns max_t if none hit)."""
    best = max_t
    for b in boxes:
        d = ray_aabb_distance(x0, y0, dx, dy, b, max_t=best)
        if d is not None and d < best:
            best = d
    return best


def ray_circle_distance(x0: float, y0: float, dx: float, dy: float,
                        cx: float, cy: float, radius: float) -> Optional[float]:
    """Smallest positive ray distance to circle, or None."""
    fx = x0 - cx
    fy = y0 - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    t1 = (-b - s) / (2 * a)
    t2 = (-b + s) / (2 * a)
    for t in (t1, t2):
        if t > 1e-3:
            return t
    return None


def line_segment_clear(x0: float, y0: float, x1: float, y1: float,
                       boxes: List[AABB]) -> bool:
    """True if straight segment (x0,y0)->(x1,y1) intersects no AABB."""
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return True
    ux, uy = dx / L, dy / L
    for b in boxes:
        d = ray_aabb_distance(x0, y0, ux, uy, b, max_t=L)
        if d is not None and d < L:
            return False
    return True


def segment_hits_circle(x0: float, y0: float, x1: float, y1: float,
                        cx: float, cy: float, r: float) -> bool:
    """True if segment (x0,y0)->(x1,y1) passes within `r` of center (cx,cy)."""
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-9:
        return (x0 - cx) ** 2 + (y0 - cy) ** 2 < r * r
    t = ((cx - x0) * dx + (cy - y0) * dy) / L2
    t = max(0.0, min(1.0, t))
    px = x0 + t * dx
    py = y0 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 < r * r


def aabbs_from_grid(grid: np.ndarray, cell_m: float) -> List[AABB]:
    """Convert a binary occupancy grid (True = obstacle cell) into a list of
    AABBs in continuous coords. Each obstacle cell becomes one box.

    NOTE: this is N boxes per occupied cell — for big maps consider merging
    contiguous cells into bigger rectangles. For our 30x30 maps it's fine."""
    H, W = grid.shape
    boxes: List[AABB] = []
    obs_j, obs_i = np.where(grid)
    for j, i in zip(obs_j, obs_i):
        # World convention used elsewhere: x = i*cell_m, y = (H-j-1)*cell_m
        xmin = i * cell_m
        ymin = (H - j - 1) * cell_m
        boxes.append(AABB(xmin=xmin, ymin=ymin,
                          xmax=xmin + cell_m, ymax=ymin + cell_m))
    return boxes


def merged_aabbs_from_grid(grid: np.ndarray, cell_m: float) -> List[AABB]:
    """Greedy row-then-column merge of occupied cells into larger AABBs.
    Reduces collision/ray queries by ~10x on typical maps."""
    H, W = grid.shape
    used = np.zeros_like(grid, dtype=bool)
    boxes: List[AABB] = []
    for j in range(H):
        i = 0
        while i < W:
            if grid[j, i] and not used[j, i]:
                # Extend right
                k = i
                while k + 1 < W and grid[j, k + 1] and not used[j, k + 1]:
                    k += 1
                # Extend down: rows where ALL of [i..k] are obstacle and unused
                jb = j
                while jb + 1 < H and all(grid[jb + 1, ii] and not used[jb + 1, ii]
                                         for ii in range(i, k + 1)):
                    jb += 1
                used[j:jb + 1, i:k + 1] = True
                xmin = i * cell_m
                ymin = (H - jb - 1) * cell_m
                xmax = (k + 1) * cell_m
                ymax = (H - j) * cell_m
                boxes.append(AABB(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
                i = k + 1
            else:
                i += 1
    return boxes
