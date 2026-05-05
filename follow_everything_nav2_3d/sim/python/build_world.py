"""Generate a 3D Gazebo world from a 2D ASCII map.

Reuses the 2D project's `merged_aabbs_from_grid` to merge adjacent '#'
cells into rectangular obstacles, then emits one SDF static `<model>` per
AABB. Also reads 'L' / 'F' spawn cells and rewrites the follower/leader
model `<pose>` in the world template so they spawn where the map says.

Output: /tmp/<map>_generated.world  (gz reads this directly).

Usage (called from sim/launch/empty_bringup.launch.py):
    python3 sim/python/build_world.py <map_name>

Map name is one of: empty, corridor, cluttered, forest. Files come from
/opt/follow_everything_nav2/sim/maps/<name>.txt (mounted in compose).
"""
import os
import re
import sys
from pathlib import Path

# /opt/follow_everything_nav2 is mounted by docker-compose; the 2D project's
# `sim` package lives there with the geometry helpers we want to reuse.
# Insert at front so Python finds 2D-sim's `sim/` instead of /ws/sim/.
sys.path.insert(0, "/opt/follow_everything_nav2")

import numpy as np  # noqa: E402  (sys.path tweak first)
from sim.geometry import merged_aabbs_from_grid  # noqa: E402


WS = os.environ.get("WS_ROOT", "/ws")
TEMPLATE_PATH = Path(WS) / "sim" / "worlds" / "empty.world"
MAPS_DIR = Path("/opt/follow_everything_nav2/sim/maps")
CELL_M = 0.5
OBSTACLE_HEIGHT = 1.5  # tall enough to occlude the actor + lidar
GROUND_PLANE_PADDING = 5.0


def parse_map(map_name: str):
    """Read map file -> (grid bool[H, W], leader_xy, follower_xy, world_size)."""
    path = MAPS_DIR / f"{map_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"map not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("//"):
                continue
            rows.append(s)
    H = len(rows)
    W = max(len(r) for r in rows)
    grid = np.zeros((H, W), dtype=bool)
    leader_xy = follower_xy = None
    for j, row in enumerate(rows):
        for i, ch in enumerate(row.ljust(W)):
            wx = (i + 0.5) * CELL_M
            wy = (H - j - 0.5) * CELL_M
            if ch == "#":
                grid[j, i] = True
            elif ch == "L":
                leader_xy = (wx, wy)
            elif ch == "F":
                follower_xy = (wx, wy)
    if leader_xy is None or follower_xy is None:
        raise ValueError(f"{map_name}: map must include 'L' and 'F' cells")
    return grid, leader_xy, follower_xy, (W * CELL_M, H * CELL_M)


def aabb_sdf(idx: int, aabb) -> str:
    """One static box model per merged AABB."""
    cx = (aabb.xmin + aabb.xmax) / 2.0
    cy = (aabb.ymin + aabb.ymax) / 2.0
    sx = aabb.xmax - aabb.xmin
    sy = aabb.ymax - aabb.ymin
    sz = OBSTACLE_HEIGHT
    return f"""    <model name="obstacle_{idx}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} {sz/2:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material>
            <ambient>0.4 0.3 0.2 1</ambient>
            <diffuse>0.5 0.4 0.3 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""


def shift_leader_waypoints(template: str, lx: float, ly: float) -> str:
    """Move the leader model's <pose> so it spawns at (lx, ly)."""
    # The template's leader <model name="leader"><pose>3 0 0 0 0 1.5708</pose>...
    return re.sub(
        r'(<model name="leader">\s*<pose>)([^<]+)(</pose>)',
        rf'\g<1>{lx:.3f} {ly:.3f} 0 0 0 1.5708\g<3>',
        template,
        count=1,
    )


def shift_follower_pose(
        template: str, fx: float, fy: float, fyaw: float) -> str:
    """Move the follower model's <pose> so it spawns at (fx, fy) with yaw."""
    return re.sub(
        r'(<model name="follower">\s*<pose>)([^<]+)(</pose>)',
        rf'\g<1>{fx:.3f} {fy:.3f} 0 0 0 {fyaw:.4f}\g<3>',
        template,
        count=1,
    )


def insert_obstacles(template: str, obstacle_xml: str) -> str:
    """Splice obstacle <model> blocks just before </world>."""
    return template.replace("</world>", obstacle_xml + "  </world>", 1)


def main():
    if len(sys.argv) < 2:
        print("usage: build_world.py <map_name>", file=sys.stderr)
        sys.exit(2)
    map_name = sys.argv[1]

    if map_name == "empty":
        # The hand-authored empty.world already has the leader patrol setup we want.
        print(TEMPLATE_PATH)
        return

    grid, leader_xy, follower_xy, (Wm, Hm) = parse_map(map_name)
    aabbs = merged_aabbs_from_grid(grid, CELL_M)
    print(f"[build_world] map={map_name} size={Wm}x{Hm}m "
          f"leader={leader_xy} follower={follower_xy} "
          f"obstacles={len(aabbs)} (merged from {int(grid.sum())} cells)",
          file=sys.stderr)

    # Orient the follower toward the leader so the leader is in the camera
    # FOV from frame 0 (the 2D maps put F/L in line-of-sight, but we'd
    # otherwise spawn the follower facing +x which may point it away from L).
    import math as _m
    f_to_l_yaw = _m.atan2(
        leader_xy[1] - follower_xy[1],
        leader_xy[0] - follower_xy[0])

    template = TEMPLATE_PATH.read_text()
    template = shift_follower_pose(
        template, follower_xy[0], follower_xy[1], f_to_l_yaw)
    template = shift_leader_waypoints(template, leader_xy[0], leader_xy[1])
    obstacle_xml = "".join(aabb_sdf(i, a) for i, a in enumerate(aabbs))
    template = insert_obstacles(template, obstacle_xml)

    out = Path(f"/tmp/{map_name}_generated.world")
    out.write_text(template)
    print(out)  # path on stdout for the launch to pick up


if __name__ == "__main__":
    main()
