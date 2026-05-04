"""
Optional matplotlib visualizer for the leader-follower simulation.

Activate by setting SIM_RENDER=1 (or passing --render to eval/run_episode.py).
For Docker visual mode use the docker-compose `sim-viz` profile which forwards X11.

Renders at ~10 Hz showing:
  * Occupancy grid (gray = obstacle)
  * Leader (blue disc + heading)
  * Follower (green disc + heading + camera FOV cone)
  * Detection state (green line follower→leader if visible, dashed red if blocked)
  * LiDAR rays (faint)
"""
import math
import os

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))  # falls back to Agg in headless mode
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge

from sim.sensors import C as SC


TICK_HZ = 20.0


class Viz:
    """Stateful renderer; call .update(world) each sim tick. Cheap if render skip > 1.
    If SIM_SNAPSHOT_DIR is set in env, also saves a PNG every snapshot_every ticks
    so an offline reviewer (or LLM) can scrub the episode visually."""

    def __init__(self, render_every: int = 2, snapshot_every: int = 60):
        self.render_every = render_every
        # Allow env override so we can capture every-N-tick frames for video.
        self.snapshot_every = int(os.environ.get("SIM_SNAPSHOT_EVERY",
                                                  snapshot_every))
        self.snapshot_dir = os.environ.get("SIM_SNAPSHOT_DIR")
        self.tick = 0
        self.fig = None
        self.ax = None
        self.background = None
        self._artists = {}
        if self.snapshot_dir:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def _init(self, world):
        plt.ion()
        # Two panels: ground-truth world on the left, follower's learned
        # occupancy map on the right (same scale, side-by-side).
        self.fig, (self.ax, self.ax_map) = plt.subplots(
            1, 2, figsize=(14, 8))
        Wm, Hm = world.world_size
        self.ax.set_xlim(0, Wm)
        self.ax.set_ylim(0, Hm)
        self.ax.set_aspect("equal")
        self.ax.set_title("Ground truth")
        self.ax_map.set_xlim(0, Wm)
        self.ax_map.set_ylim(0, Hm)
        self.ax_map.set_aspect("equal")
        self.ax_map.set_title("Follower's learned map (LiDAR-only)")
        # Learned-map image artist; image origin matches world coords.
        self._artists["learned_img"] = self.ax_map.imshow(
            np.zeros((1, 1)),
            extent=(0, Wm, 0, Hm),
            origin="lower", cmap="gray_r", vmin=-1, vmax=100,
            interpolation="nearest", zorder=1)
        # Mirror robot positions on the learned-map view too.
        self._artists["lm_leader"] = Circle((0, 0), 0.25,
                                             color="#3b82f6", zorder=4)
        self._artists["lm_follower"] = Circle((0, 0), 0.25,
                                               color="#22c55e", zorder=4)
        self.ax_map.add_patch(self._artists["lm_leader"])
        self.ax_map.add_patch(self._artists["lm_follower"])

        # Continuous-space obstacles drawn as filled rectangles
        for b in world.obstacles:
            self.ax.add_patch(Rectangle(
                (b.xmin, b.ymin), b.xmax - b.xmin, b.ymax - b.ymin,
                facecolor="#4b5563", edgecolor="#1f2937", linewidth=0.5, zorder=1))

        # Robots
        self._artists["leader_body"] = Circle((0, 0), 0.25, color="#3b82f6", zorder=4)
        self._artists["follower_body"] = Circle((0, 0), 0.25, color="#22c55e", zorder=4)
        for k in ("leader_body", "follower_body"):
            self.ax.add_patch(self._artists[k])
        # Follower's planned A* path. Drawn on both panels so the user can
        # see the planned waypoints relative to ground truth and to what
        # the follower THINKS the world looks like.
        self._artists["planned_path_gt"], = self.ax.plot(
            [], [], color="#0ea5e9", lw=2.0, alpha=0.85, zorder=3,
            marker="o", markersize=3, markerfacecolor="#0284c7")
        self._artists["planned_path_map"], = self.ax_map.plot(
            [], [], color="#0ea5e9", lw=2.0, alpha=0.85, zorder=3,
            marker="o", markersize=3, markerfacecolor="#0284c7")
        # Pedestrians (variable count). Magenta to distinguish from leader/follower.
        self._artists["pedestrian_bodies"] = []
        self._artists["pedestrian_path_lines"] = []
        for p in getattr(world, "pedestrians", []):
            c = Circle((p.x, p.y), 0.25, color="#c026d3", zorder=4)
            self.ax.add_patch(c)
            self._artists["pedestrian_bodies"].append(c)
            line, = self.ax.plot([], [], color="#c026d3", lw=0.8, alpha=0.5,
                                  zorder=2)
            self._artists["pedestrian_path_lines"].append(line)

        # Heading arrows
        self._artists["leader_arrow"] = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.6))
        self._artists["follower_arrow"] = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.6))

        # Camera FOV wedge
        self._artists["cam_fov"] = Wedge(
            (0, 0), SC.CAMERA_MAX_RANGE, 0, 0,
            facecolor="#facc15", alpha=0.15, edgecolor="#a16207", lw=0.6, zorder=2)
        self.ax.add_patch(self._artists["cam_fov"])

        # Detection line
        self._artists["det_line"], = self.ax.plot([0, 0], [0, 0],
                                                   color="lime", lw=2, alpha=0.0, zorder=5)

        # LiDAR scatter
        self._artists["lidar_scatter"] = self.ax.scatter(
            [], [], s=4, c="orange", alpha=0.55, zorder=3)

        # HUD text
        self._artists["hud"] = self.ax.text(
            0.01, 0.99, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        plt.tight_layout()
        plt.show(block=False)

    def update(self, world, last_detection=None, last_lidar_ranges=None,
               learned_map=None, planned_path=None):
        self.tick += 1
        if self.fig is None:
            self._init(world)
        if self.tick % self.render_every != 0:
            return

        L = world.leader
        F = world.follower

        self._artists["leader_body"].center = (L.x, L.y)
        self._artists["follower_body"].center = (F.x, F.y)
        for body, line, p in zip(self._artists["pedestrian_bodies"],
                                  self._artists["pedestrian_path_lines"],
                                  getattr(world, "pedestrians", [])):
            body.center = (p.x, p.y)
            xs = [p.x] + [w[0] for w in p.path]
            ys = [p.y] + [w[1] for w in p.path]
            line.set_data(xs, ys)

        ar_len = 0.6
        a = self._artists["leader_arrow"]
        a.xy = (L.x + ar_len * math.cos(L.yaw), L.y + ar_len * math.sin(L.yaw))
        a.xyann = (L.x, L.y)
        a = self._artists["follower_arrow"]
        a.xy = (F.x + ar_len * math.cos(F.yaw), F.y + ar_len * math.sin(F.yaw))
        a.xyann = (F.x, F.y)

        # FOV wedge: matplotlib Wedge angles are absolute (0 = +x, CCW degrees)
        f_yaw_deg = math.degrees(F.yaw)
        half_fov = SC.CAMERA_FOV_DEG / 2
        wedge = self._artists["cam_fov"]
        wedge.set_center((F.x, F.y))
        wedge.set_theta1(f_yaw_deg - half_fov)
        wedge.set_theta2(f_yaw_deg + half_fov)

        # Detection line
        det_line = self._artists["det_line"]
        if last_detection is not None and last_detection:
            det_line.set_data([F.x, L.x], [F.y, L.y])
            det_line.set_alpha(0.9)
        else:
            det_line.set_alpha(0.0)

        # LiDAR scatter (in world frame)
        if last_lidar_ranges is not None:
            from sim.sensors import LidarSensor
            angles = np.deg2rad(np.linspace(0, SC.LIDAR_FOV_DEG, SC.LIDAR_NUM_RAYS, endpoint=False))
            xs, ys = [], []
            for r, th in zip(last_lidar_ranges, angles):
                if r >= SC.LIDAR_MAX_RANGE - 1e-3:
                    continue
                world_th = F.yaw + th
                xs.append(F.x + r * math.cos(world_th))
                ys.append(F.y + r * math.sin(world_th))
            self._artists["lidar_scatter"].set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))

        # Learned-map panel: render OccupancyGrid as image.
        if learned_map is not None and "learned_img" in self._artists:
            w = learned_map.info.width
            h = learned_map.info.height
            if w > 0 and h > 0 and len(learned_map.data) == w * h:
                arr = np.array(learned_map.data, dtype=np.int8).reshape(h, w)
                self._artists["learned_img"].set_data(arr)
            self._artists["lm_leader"].center = (L.x, L.y)
            self._artists["lm_follower"].center = (F.x, F.y)

        # Follower's planned A* path. Drawn on both panels.
        pxs, pys = [], []
        if planned_path is not None:
            for ps in planned_path.poses:
                pxs.append(ps.pose.position.x)
                pys.append(ps.pose.position.y)
        self._artists["planned_path_gt"].set_data(pxs, pys)
        self._artists["planned_path_map"].set_data(pxs, pys)

        # HUD
        d_lf = math.hypot(L.x - F.x, L.y - F.y)
        det_str = "VISIBLE " if last_detection else "BLOCKED "
        self._artists["hud"].set_text(
            f"tick {world.tick:>5d}  d={d_lf:5.2f}m  {det_str}\n"
            f"L=({L.x:5.1f},{L.y:5.1f}) yaw={math.degrees(L.yaw):+6.1f}°\n"
            f"F=({F.x:5.1f},{F.y:5.1f}) yaw={math.degrees(F.yaw):+6.1f}°\n"
            f"collisions L={world.collision_count_leader} F={world.collision_count_follower}")

        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass

        if self.snapshot_dir and (self.tick % self.snapshot_every) == 0:
            try:
                path = os.path.join(self.snapshot_dir, f"t{self.tick:05d}.png")
                self.fig.savefig(path, dpi=80, bbox_inches="tight")
            except Exception:
                pass

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
