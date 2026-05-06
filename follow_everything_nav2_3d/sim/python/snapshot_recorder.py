"""Periodic top-down PNG snapshots of the running episode.

Mirrors the 2D autoresearch sim's matplotlib renderer so we get the same
"see what the BT is doing" debug surface in 3D. One PNG per second by
default, dumped to $SNAP_DIR (set by record_episode.py) showing:

  * obstacle AABBs from the chosen map
  * follower position + heading + path trail
  * leader position + path trail
  * current lidar fan, in world frame
  * camera FOV cone (the same 90° / 6 m the oracle uses)

Headless matplotlib (Agg backend) — no display required.
"""
import math
import os
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Wedge  # noqa: E402
import time  # noqa: E402

# Reuse the 2D project's geometry helpers / map parser.
sys.path.insert(0, "/opt/follow_everything_nav2")
import numpy as np  # noqa: E402
from sim.geometry import merged_aabbs_from_grid  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

from geometry_msgs.msg import Quaternion  # noqa: E402
from nav_msgs.msg import Odometry, OccupancyGrid  # noqa: E402
from nav_msgs.msg import Path as PathMsg  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from vision_msgs.msg import Detection2DArray  # noqa: E402


SNAP_DIR = Path(os.environ.get("SNAP_DIR", "/tmp/snapshots"))
SNAP_DIR.mkdir(parents=True, exist_ok=True)
SNAP_PERIOD = float(os.environ.get("SNAP_PERIOD_SEC", "1.0"))
MAP_NAME = os.environ.get("EP_MAP", "empty")
MAPS_DIR = Path("/opt/follow_everything_nav2/sim/maps")
TRAIL_MAX = 600  # ~30s of 20Hz odom samples
CAM_FOV_DEG = 90.0
CAM_RANGE_M = 6.0

# Must match world_odom_publisher.py — the BT lives in a first-quadrant
# (0..Wm, 0..Hm) world, gz lives in a centered world. Shift everything we
# read directly from /gz_pose_truth so it overlays correctly on the BT's
# OccupancyGrid (which is published at origin (0, 0)).
# Mirror world_odom_publisher's per-map convention: empty needs +7.5 to
# move the bot off gz's centered origin into the BT's first-quadrant grid;
# map-file worlds already have gz origin at the bottom-left corner so no
# offset is needed (otherwise the bot is drawn well outside the grid).
WORLD_ORIGIN_OFFSET = (7.5, 7.5) if MAP_NAME == "empty" else (0.0, 0.0)


def yaw_from_quat(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def load_map_obstacles(map_name: str):
    """Return (list of AABB, world_size_xy) — empty list for 'empty'."""
    path = MAPS_DIR / f"{map_name}.txt"
    if map_name == "empty" or not path.exists():
        return [], (15.0, 15.0)
    rows = []
    with open(path) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("//"):
                continue
            rows.append(s)
    H, W = len(rows), max(len(r) for r in rows)
    grid = np.zeros((H, W), dtype=bool)
    for j, row in enumerate(rows):
        for i, ch in enumerate(row.ljust(W)):
            if ch == "#":
                grid[j, i] = True
    return merged_aabbs_from_grid(grid, 0.5), (W * 0.5, H * 0.5)


class SnapshotRecorder(Node):
    def __init__(self) -> None:
        super().__init__("snapshot_recorder")
        self.obstacles, self.world_size = load_map_obstacles(MAP_NAME)

        self.f_xyy: tuple[float, float, float] | None = None
        self.l_xy:  tuple[float, float] | None        = None
        self.scan: LaserScan | None = None
        self.learned_map: OccupancyGrid | None = None
        self.planned_path: list[tuple[float, float]] = []
        self.last_seen: tuple[float, float] | None = None
        self.last_seen_t: float = 0.0
        self.f_trail: deque = deque(maxlen=TRAIL_MAX)
        self.l_trail: deque = deque(maxlen=TRAIL_MAX)

        self.create_subscription(Odometry, "/follower/odom", self._on_odom, 10)
        self.create_subscription(TFMessage, "/gz_pose_truth", self._on_poses, 50)
        self.create_subscription(LaserScan, "/follower/scan", self._on_scan, 10)
        self.create_subscription(
            OccupancyGrid, "/follower/learned_map", self._on_map, 1)
        self.create_subscription(
            PathMsg, "/follower/planned_path", self._on_path, 1)
        self.create_subscription(
            Detection2DArray, "/follower/camera/detections",
            self._on_detect, 10)
        self.create_timer(SNAP_PERIOD, self._snap)
        self._idx = 0
        self.get_logger().info(
            f"snapshot_recorder live; map={MAP_NAME}, dir={SNAP_DIR}, "
            f"period={SNAP_PERIOD}s, obstacles={len(self.obstacles)}")

    def _on_odom(self, msg: Odometry) -> None:
        # Diff-drive odom is in the follower's local "follower/odom" frame
        # (origin at the follower's spawn pose), NOT the world frame, so it
        # would mis-place the follower relative to obstacles + leader.
        # We keep the subscriber for completeness but DON'T use the odom
        # position for plotting — use /gz_pose_truth (world frame) below.
        return

    def _on_poses(self, msg: TFMessage) -> None:
        ox, oy = WORLD_ORIGIN_OFFSET
        for tr in msg.transforms:
            t = tr.transform.translation
            if tr.child_frame_id == "leader":
                self.l_xy = (t.x + ox, t.y + oy)
                self.l_trail.append(self.l_xy)
            elif tr.child_frame_id == "follower":
                self.f_xyy = (
                    t.x + ox, t.y + oy,
                    yaw_from_quat(tr.transform.rotation))
                self.f_trail.append((self.f_xyy[0], self.f_xyy[1]))

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan = msg

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.learned_map = msg

    def _on_path(self, msg: PathMsg) -> None:
        self.planned_path = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def _on_detect(self, msg: Detection2DArray) -> None:
        if not msg.detections:
            return
        det = msg.detections[0]
        if not det.results:
            return
        if det.results[0].hypothesis.class_id != "leader":
            return
        px = det.results[0].pose.pose.position.x
        py = det.results[0].pose.pose.position.y
        if msg.header.frame_id == "follower/odom":
            # Already world-frame (publisher did the body→world lift at the
            # frame's own timestamp — no further pose math here).
            self.last_seen = (px, py)
        else:
            if self.f_xyy is None:
                return
            fx, fy, fyaw = self.f_xyy
            c, s = math.cos(fyaw), math.sin(fyaw)
            self.last_seen = (fx + c * px - s * py, fy + s * px + c * py)
        self.last_seen_t = time.time()

    def _draw_gt(self, ax) -> None:
        """Left subplot: ground-truth obstacle layout + scan + entities."""
        # Obstacles (true)
        for a in self.obstacles:
            ax.add_patch(Rectangle(
                (a.xmin, a.ymin), a.xmax - a.xmin, a.ymax - a.ymin,
                facecolor="0.4", edgecolor="0.2", linewidth=0.5))
        # Trails
        if self.f_trail:
            xs, ys = zip(*self.f_trail)
            ax.plot(xs, ys, "g-", linewidth=1.0, alpha=0.6)
        if self.l_trail:
            xs, ys = zip(*self.l_trail)
            ax.plot(xs, ys, "b-", linewidth=1.0, alpha=0.6)
        # Lidar fan in world frame
        if self.scan is not None and self.f_xyy is not None:
            fx, fy, fyaw = self.f_xyy
            angles = np.linspace(
                self.scan.angle_min, self.scan.angle_max,
                len(self.scan.ranges))
            for r, a in zip(self.scan.ranges, angles):
                if not (0.1 < r < 7.5):
                    continue
                hx = fx + r * math.cos(fyaw + a)
                hy = fy + r * math.sin(fyaw + a)
                ax.plot([fx, hx], [fy, hy],
                        color="orange", linewidth=0.3, alpha=0.4)
        # Follower marker + heading + camera FOV
        if self.f_xyy is not None:
            fx, fy, fyaw = self.f_xyy
            ax.add_patch(plt.Circle((fx, fy), 0.20, color="green", alpha=0.7))
            ax.add_patch(Wedge(
                (fx, fy), CAM_RANGE_M,
                math.degrees(fyaw) - CAM_FOV_DEG / 2,
                math.degrees(fyaw) + CAM_FOV_DEG / 2,
                facecolor="yellow", alpha=0.15,
                edgecolor="0.6", linewidth=0.5))
            ax.arrow(
                fx, fy,
                0.5 * math.cos(fyaw), 0.5 * math.sin(fyaw),
                head_width=0.15, head_length=0.15, fc="black")
        if self.l_xy is not None:
            lx, ly = self.l_xy
            ax.add_patch(plt.Circle((lx, ly), 0.20, color="blue", alpha=0.7))
        # last_seen marker — hollow cyan circle. Where the BT thinks the
        # leader was; A* recovery aims here (or here + velocity prediction).
        if self.last_seen is not None:
            lsx, lsy = self.last_seen
            ax.add_patch(plt.Circle(
                (lsx, lsy), 0.30, fill=False, edgecolor="cyan",
                linewidth=2.0, zorder=4))
        # A* / planned path (waypoints + connecting lines + start/end markers)
        if self.planned_path:
            xs = [p[0] for p in self.planned_path]
            ys = [p[1] for p in self.planned_path]
            ax.plot(xs, ys, color="magenta", linewidth=1.6, alpha=0.9,
                    zorder=5, label=f"A* path ({len(xs)} wpts)")
            ax.scatter(xs, ys, s=18, color="magenta", alpha=0.9, zorder=6)
            ax.scatter([xs[-1]], [ys[-1]], s=70, marker="*",
                       color="magenta", edgecolor="black", zorder=7)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title("GT map (obstacles + scan + A* path)")

    def _draw_lidar_map(self, ax) -> None:
        """Right subplot: BT's lidar-built occupancy grid (/follower/learned_map)."""
        if self.learned_map is None:
            ax.text(0.5, 0.5, "(awaiting /follower/learned_map)",
                    transform=ax.transAxes, ha="center")
            ax.set_title("Lidar-built map")
            return
        m = self.learned_map
        W, H = m.info.width, m.info.height
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        res = m.info.resolution
        data = np.array(m.data, dtype=np.int8).reshape(H, W)
        # OccupancyGrid: -1 = unknown, 0 = free, 100 = occ.
        # Visualise: grey for unknown, white for free, dark for occupied.
        img = np.full((H, W, 3), 0.85, dtype=float)            # unknown ~ grey
        img[data == 0]   = (1.00, 1.00, 1.00)                  # free = white
        img[data >= 50]  = (0.10, 0.10, 0.10)                  # occupied = black
        ax.imshow(
            img, origin="lower",
            extent=[ox, ox + W * res, oy, oy + H * res],
            interpolation="nearest")
        # Overlay follower position + heading (lives in the same world frame).
        if self.f_xyy is not None:
            fx, fy, fyaw = self.f_xyy
            ax.add_patch(plt.Circle((fx, fy), 0.20, color="green", alpha=0.7))
            ax.arrow(
                fx, fy,
                0.5 * math.cos(fyaw), 0.5 * math.sin(fyaw),
                head_width=0.15, head_length=0.15, fc="black")
        if self.l_xy is not None:
            lx, ly = self.l_xy
            ax.add_patch(plt.Circle((lx, ly), 0.20, color="blue", alpha=0.7))
        if self.planned_path:
            xs = [p[0] for p in self.planned_path]
            ys = [p[1] for p in self.planned_path]
            ax.plot(xs, ys, color="magenta", linewidth=1.6, alpha=0.9,
                    zorder=5)
            ax.scatter(xs, ys, s=18, color="magenta", alpha=0.9, zorder=6)
            ax.scatter([xs[-1]], [ys[-1]], s=70, marker="*",
                       color="magenta", edgecolor="black", zorder=7)
        ax.set_title(f"Lidar-built map ({W}x{H} @ {res:.2f}m)")

    def _set_view(self, ax) -> None:
        Wm, Hm = self.world_size
        if self.f_xyy is not None:
            cx, cy = self.f_xyy[0], self.f_xyy[1]
            half = max(8.0, Wm / 2)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
        else:
            ax.set_xlim(0, Wm)
            ax.set_ylim(0, Hm)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    def _snap(self) -> None:
        if self.f_xyy is None and self.l_xy is None:
            return  # nothing to draw yet

        fig, (ax_gt, ax_lm) = plt.subplots(1, 2, figsize=(14, 7), dpi=100)
        self._draw_gt(ax_gt)
        self._draw_lidar_map(ax_lm)
        for ax in (ax_gt, ax_lm):
            self._set_view(ax)

        d_str = ""
        if self.f_xyy and self.l_xy:
            d = math.hypot(self.l_xy[0] - self.f_xyy[0],
                           self.l_xy[1] - self.f_xyy[1])
            d_str = f" | d={d:.2f}m"
        if self.last_seen is None:
            ls_str = " | last_seen=None"
        else:
            age = time.time() - self.last_seen_t
            ls_str = (f" | last_seen=({self.last_seen[0]:+.1f},"
                      f"{self.last_seen[1]:+.1f}) age={age:.1f}s")
        fig.suptitle(
            f"map={MAP_NAME} | snap #{self._idx:04d}{d_str}{ls_str}")

        out = SNAP_DIR / f"snap_{self._idx:04d}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        self._idx += 1


def main() -> None:
    rclpy.init()
    node = SnapshotRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
