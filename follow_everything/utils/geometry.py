"""Shared 2-D geometry helpers used across all modules."""

import numpy as np
from typing import Tuple, Optional


def rotation_matrix_2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def robot_to_world(point_robot: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """Transform a 2-D point from robot frame to world frame.

    Args:
        point_robot: (2,) array [x, y] in robot frame.
        robot_pose:  (3,) array [x, y, theta] world pose of robot.
    """
    x, y, theta = robot_pose
    return rotation_matrix_2d(theta) @ point_robot + np.array([x, y])


def world_to_robot(point_world: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """Transform a 2-D point from world frame to robot frame."""
    x, y, theta = robot_pose
    return rotation_matrix_2d(-theta) @ (point_world - np.array([x, y]))


def normalize_angle(angle: float) -> float:
    """Wrap angle to (−π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_to_point(robot_pose: np.ndarray, target: np.ndarray) -> float:
    """Heading error (rad) from robot pose to a 2-D target in world frame."""
    dx = target[0] - robot_pose[0]
    dy = target[1] - robot_pose[1]
    return normalize_angle(np.arctan2(dy, dx) - robot_pose[2])


def euclidean_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    """Binary mask → [x1, y1, x2, y2] bounding box, or None if mask is empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Return (u, v) centroid of a binary mask."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return mask.shape[1] / 2.0, mask.shape[0] / 2.0
    cy, cx = coords.mean(axis=0)
    return float(cx), float(cy)


def mask_area_ratio(mask: np.ndarray) -> float:
    return mask.sum() / max(mask.size, 1)


def project_3d_to_image(
    point_3d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float
) -> Tuple[float, float]:
    """Project a 3-D point (camera frame, z-forward) to pixel (u, v)."""
    x, y, z = point_3d
    if z <= 0:
        return cx, cy
    u = fx * x / z + cx
    v = fy * y / z + cy
    return float(u), float(v)


def bbox_from_3d_in_image(
    cx_3d: float, cy_3d: float, cz_3d: float,
    length: float, width: float, height: float,
    fx: float, fy: float, cx_img: float, cy_img: float,
    img_w: int, img_h: int,
    padding: float = 0.15,
) -> Optional[np.ndarray]:
    """Estimate a 2-D image bounding box from a 3-D bounding box in camera frame.

    Returns [x1, y1, x2, y2] clipped to image bounds, or None if behind camera.
    """
    if cz_3d <= 0.1:
        return None
    half_l = (length / 2 + padding)
    half_w = (width  / 2 + padding)
    half_h = (height / 2 + padding)
    # 8 corners of the 3-D box in camera frame
    corners = np.array([
        [cx_3d + dx, cy_3d + dy, cz_3d + dz]
        for dx in (-half_l, half_l)
        for dy in (-half_w, half_w)
        for dz in (-half_h, half_h)
    ])
    us, vs = [], []
    for pt in corners:
        if pt[2] > 0:
            u, v = project_3d_to_image(pt, fx, fy, cx_img, cy_img)
            us.append(u)
            vs.append(v)
    if not us:
        return None
    x1 = max(0, int(min(us)))
    y1 = max(0, int(min(vs)))
    x2 = min(img_w - 1, int(max(us)))
    y2 = min(img_h - 1, int(max(vs)))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def segment_intersects_grid_cell(
    p1: np.ndarray,
    p2: np.ndarray,
    grid: np.ndarray,
    origin: np.ndarray,
    resolution: float,
) -> bool:
    """Bresenham line-of-sight check on a boolean occupancy grid.

    Returns True if the segment p1→p2 (world coords) passes through an occupied cell.
    """
    def to_cell(pt):
        col = int((pt[0] + origin[0]) / resolution)
        row = int((pt[1] + origin[1]) / resolution)
        return row, col

    n = int(np.linalg.norm(p2 - p1) / resolution) + 1
    for i in range(n + 1):
        t = i / max(n, 1)
        pt = p1 + t * (p2 - p1)
        row, col = to_cell(pt)
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
            if grid[row, col]:
                return True
    return False
