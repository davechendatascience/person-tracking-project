"""
Standalone unit tests for sim/geometry.py — no ROS required. Run with:
    uv run python -m sim.test_geometry
"""
import math
import sys

import numpy as np

from sim.geometry import (
    AABB,
    circle_collides_aabbs,
    ray_aabb_distance,
    ray_min_distance,
    ray_circle_distance,
    line_segment_clear,
    aabbs_from_grid,
    merged_aabbs_from_grid,
)


def almost(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_aabb_contains():
    b = AABB(0.0, 0.0, 1.0, 1.0)
    assert b.contains_point(0.5, 0.5)
    assert not b.contains_point(1.5, 0.5)
    assert b.contains_point(0.0, 0.0)
    print("test_aabb_contains: PASS")


def test_circle_collides():
    boxes = [AABB(2.0, 2.0, 3.0, 3.0)]
    # Far away
    assert not circle_collides_aabbs(0.0, 0.0, 0.5, boxes)
    # Tangent (just outside)
    assert not circle_collides_aabbs(2.0 - 0.5 - 0.01, 2.5, 0.5, boxes)
    # Just inside
    assert circle_collides_aabbs(2.0 - 0.5 + 0.01, 2.5, 0.5, boxes)
    # Center inside box
    assert circle_collides_aabbs(2.5, 2.5, 0.1, boxes)
    print("test_circle_collides: PASS")


def test_ray_aabb_distance():
    b = AABB(2.0, -1.0, 3.0, 1.0)
    # Ray hitting the front face from the left
    d = ray_aabb_distance(0.0, 0.0, 1.0, 0.0, b)
    assert d is not None and almost(d, 2.0)
    # Ray missing (going up)
    d = ray_aabb_distance(0.0, 0.0, 0.0, 1.0, b)
    assert d is None
    # Ray pointing away (still hits if pointing toward and we're outside)
    d = ray_aabb_distance(5.0, 0.0, -1.0, 0.0, b)
    assert d is not None and almost(d, 2.0)
    # Ray ORIGIN inside the box
    d = ray_aabb_distance(2.5, 0.0, 1.0, 0.0, b)
    assert d == 0.0
    # max_t cuts off
    d = ray_aabb_distance(0.0, 0.0, 1.0, 0.0, b, max_t=1.5)
    assert d is None
    print("test_ray_aabb_distance: PASS")


def test_ray_circle_distance():
    # Circle at (5, 0), r=1. Ray from origin along +x hits at d=4.
    d = ray_circle_distance(0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0)
    assert d is not None and almost(d, 4.0)
    # Ray going +y misses the circle
    d = ray_circle_distance(0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 1.0)
    assert d is None
    # Ray facing away — circle "behind" — should not return positive distance
    d = ray_circle_distance(10.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0)
    assert d is None
    print("test_ray_circle_distance: PASS")


def test_line_segment_clear():
    boxes = [AABB(2.0, -1.0, 3.0, 1.0)]
    assert line_segment_clear(0.0, 0.0, 1.5, 0.0, boxes)        # short, ends before box
    assert not line_segment_clear(0.0, 0.0, 5.0, 0.0, boxes)    # crosses through
    assert line_segment_clear(0.0, 5.0, 5.0, 5.0, boxes)        # above box
    print("test_line_segment_clear: PASS")


def test_grid_to_aabbs_merge():
    # 3x3 with two separate single cells
    g = np.array([
        [True, False, False],
        [False, False, True],
        [False, False, False],
    ])
    cell = 0.5
    raw = aabbs_from_grid(g, cell)
    assert len(raw) == 2
    merged = merged_aabbs_from_grid(g, cell)
    assert len(merged) == 2  # no merging possible
    # 2x2 block should merge into one
    g2 = np.array([
        [True, True],
        [True, True],
    ])
    raw2 = aabbs_from_grid(g2, cell)
    merged2 = merged_aabbs_from_grid(g2, cell)
    assert len(raw2) == 4 and len(merged2) == 1
    b = merged2[0]
    assert almost(b.xmax - b.xmin, 1.0) and almost(b.ymax - b.ymin, 1.0)
    print("test_grid_to_aabbs_merge: PASS")


def test_lidar_geometry_consistency():
    """Sanity: a 360° sweep against a single box should produce a contiguous arc of
    short ranges roughly subtending the angle the box covers from the sensor."""
    boxes = [AABB(2.0, -0.5, 3.0, 0.5)]  # 1m wide block 2m ahead, 1m tall
    n_rays = 360
    angles = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
    ranges = []
    for th in angles:
        ranges.append(ray_min_distance(0.0, 0.0, math.cos(th), math.sin(th),
                                        boxes, max_t=10.0))
    ranges = np.array(ranges)
    # Block subtends roughly atan2(0.5, 2.0)*2 = ~28 deg around angle 0
    short = ranges < 5.0
    n_short = short.sum()
    assert 20 < n_short < 50, f"Expected ~28 short rays, got {n_short}"
    # The shortest range should be ~2.0 (closest face)
    assert almost(ranges.min(), 2.0, tol=0.05)
    print(f"test_lidar_geometry_consistency: PASS ({n_short} short rays, min={ranges.min():.3f})")


def main():
    fns = [
        test_aabb_contains,
        test_circle_collides,
        test_ray_aabb_distance,
        test_ray_circle_distance,
        test_line_segment_clear,
        test_grid_to_aabbs_merge,
        test_lidar_geometry_consistency,
    ]
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            print(f"{fn.__name__}: FAIL - {e}", file=sys.stderr)
            sys.exit(1)
    print(f"\nAll {len(fns)} geometry tests passed.")


if __name__ == "__main__":
    main()
