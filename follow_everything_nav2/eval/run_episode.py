#!/usr/bin/env python3
"""
Evaluation harness for the autoresearch leader-following task. DO NOT MODIFY.

Spawns the simulator + leader patrol + (optionally) the follower stack,
runs N episodes for a fixed wall-clock budget each, computes metrics, and
prints the canonical summary block + writes results.json.

Episode metrics:
  follow_success_rate: fraction of ticks where leader is within 5m AND visible
  loss_time_ratio: fraction of ticks where leader is NOT visible
  collision_count: number of follower collisions during the episode
  avg_distance: mean Euclidean distance follower<->leader
  path_efficiency: follower_path_length / max(leader_path_length, eps)

Aggregates across episodes:
  mean_success, mean_loss_ratio, collision_count, mean_distance, mean_path_eff,
  total_episodes, crashes

CLI:
  python eval/run_episode.py --episodes 10 --maps mixed --seed 0 --episode-secs 90
  python eval/run_episode.py --episodes 1  --maps empty  --seed 0 --render
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray


STUCK_WINDOW_TICKS = 20        # 1s at 20Hz
STUCK_WINDOW_DIST_M = 0.05     # bounding-box diameter under 5cm over 1s = stuck


def _window_diameter(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys))


REPO = Path(__file__).resolve().parent.parent
MAPS_DIR = REPO / "sim" / "maps"
QOS = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
SUCCESS_DISTANCE = 5.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--maps", type=str, default="mixed",
                   help="One of: empty, corridor, cluttered, forest, mixed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-secs", type=int, default=90,
                   help="Wall-clock budget per episode")
    p.add_argument("--render", action="store_true",
                   help="Open matplotlib visualizer (single-episode use)")
    p.add_argument("--no-follower", action="store_true",
                   help="Don't launch follower stack (eval as if BT outputs zero cmd_vel)")
    p.add_argument("--follower-launch", type=str,
                   default="follower_pkg/launch/follower.launch.py",
                   help="Path to the follower's ROS launch file (relative to repo)")
    p.add_argument("--out", type=str, default="results/last_run.json")
    return p.parse_args()


def select_maps(name: str, n: int):
    all_maps = ["empty", "corridor", "cluttered", "forest"]
    if name in all_maps:
        return [name] * n
    if name == "mixed":
        return [all_maps[i % len(all_maps)] for i in range(n)]
    raise ValueError(f"Unknown map name: {name}")


class EpisodeRecorder(Node):
    """Subscribes to odom + detections + ground-truth pose. Records per-tick metrics."""

    def __init__(self):
        super().__init__("episode_recorder")
        self.last_leader_xy = None
        self.last_follower_xy = None
        self.last_ped_xys = []
        self.last_detect_visible = False
        self.tick = 0
        self.ticks_visible_close = 0
        self.ticks_visible = 0
        self.distances = []
        self.follower_path_len = 0.0
        self.leader_path_len = 0.0
        self.collision_count = 0
        self._prev_follower_xy = None
        self._prev_leader_xy = None
        # Window-based stuck tracking. We keep a deque of (x,y) for each agent
        # over the last STUCK_WINDOW_TICKS ticks; "stuck now" iff the diameter
        # of those positions is under STUCK_WINDOW_DIST_M.
        from collections import deque
        self._leader_window = deque(maxlen=STUCK_WINDOW_TICKS)
        self._follower_window = deque(maxlen=STUCK_WINDOW_TICKS)
        self._ped_windows = []  # list of deques
        self.leader_stuck_ticks = 0
        self.follower_stuck_ticks = 0
        self.ped_stuck_ticks = []  # one accumulator per pedestrian

        self.create_subscription(PoseStamped, "/leader/pose_ground_truth",
                                 self._on_leader_truth, QOS)
        self.create_subscription(Odometry, "/follower/odom", self._on_follower_odom, QOS)
        self.create_subscription(Detection2DArray, "/follower/camera/detections",
                                 self._on_detect, QOS)
        self.create_subscription(PoseArray, "/pedestrians/poses",
                                 self._on_peds, QOS)
        # Tick at 20 Hz, matching world
        self.create_timer(0.05, self._tick)

    def _on_leader_truth(self, msg):
        self.last_leader_xy = (msg.pose.position.x, msg.pose.position.y)

    def _on_follower_odom(self, msg):
        self.last_follower_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_detect(self, msg):
        self.last_detect_visible = bool(msg.detections)

    def _on_peds(self, msg):
        self.last_ped_xys = [(p.position.x, p.position.y) for p in msg.poses]

    def _tick(self):
        if self.last_leader_xy is None or self.last_follower_xy is None:
            return
        self.tick += 1
        lx, ly = self.last_leader_xy
        fx, fy = self.last_follower_xy
        d = math.hypot(lx - fx, ly - fy)
        self.distances.append(d)
        if self.last_detect_visible:
            self.ticks_visible += 1
            if d <= SUCCESS_DISTANCE:
                self.ticks_visible_close += 1

        # Path length accumulators
        if self._prev_leader_xy is not None:
            self.leader_path_len += math.hypot(
                lx - self._prev_leader_xy[0], ly - self._prev_leader_xy[1])
        if self._prev_follower_xy is not None:
            self.follower_path_len += math.hypot(
                fx - self._prev_follower_xy[0], fy - self._prev_follower_xy[1])
        self._prev_follower_xy = (fx, fy)
        self._prev_leader_xy = (lx, ly)

        # Window-based stuck tracking. An agent is "stuck this tick" iff the
        # bounding box of its last STUCK_WINDOW_TICKS positions is < 5cm —
        # i.e. it hasn't actually translated anywhere over the past second.
        # Pure rotation-in-place no longer counts.
        self._leader_window.append((lx, ly))
        self._follower_window.append((fx, fy))
        if len(self._leader_window) == STUCK_WINDOW_TICKS:
            if _window_diameter(self._leader_window) < STUCK_WINDOW_DIST_M:
                self.leader_stuck_ticks += 1
        if len(self._follower_window) == STUCK_WINDOW_TICKS:
            if _window_diameter(self._follower_window) < STUCK_WINDOW_DIST_M:
                self.follower_stuck_ticks += 1

        if self.last_ped_xys:
            if len(self._ped_windows) != len(self.last_ped_xys):
                from collections import deque
                self._ped_windows = [deque(maxlen=STUCK_WINDOW_TICKS)
                                      for _ in self.last_ped_xys]
                self.ped_stuck_ticks = [0] * len(self.last_ped_xys)
            for i, (px, py) in enumerate(self.last_ped_xys):
                self._ped_windows[i].append((px, py))
                if (len(self._ped_windows[i]) == STUCK_WINDOW_TICKS and
                        _window_diameter(self._ped_windows[i]) < STUCK_WINDOW_DIST_M):
                    self.ped_stuck_ticks[i] += 1

    def reset_metrics(self):
        """Discard whatever was recorded during the startup grace period."""
        self.tick = 0
        self.ticks_visible = 0
        self.ticks_visible_close = 0
        self.distances = []
        self.follower_path_len = 0.0
        self.leader_path_len = 0.0
        self._prev_follower_xy = None
        self._prev_leader_xy = None
        self.leader_stuck_ticks = 0
        self.follower_stuck_ticks = 0
        self.ped_stuck_ticks = []
        from collections import deque
        self._leader_window = deque(maxlen=STUCK_WINDOW_TICKS)
        self._follower_window = deque(maxlen=STUCK_WINDOW_TICKS)
        self._ped_windows = []

    def metrics(self):
        n = max(1, self.tick)
        ped_stuck_ratios = ([t / n for t in self.ped_stuck_ticks]
                             if self.ped_stuck_ticks else [])
        return dict(
            follow_success_rate=self.ticks_visible_close / n,
            loss_time_ratio=1.0 - self.ticks_visible / n,
            avg_distance=float(np.mean(self.distances)) if self.distances else float("nan"),
            path_efficiency=self.follower_path_len / max(self.leader_path_len, 1e-6),
            collision_count=self.collision_count,
            leader_stuck_ratio=self.leader_stuck_ticks / n,
            follower_stuck_ratio=self.follower_stuck_ticks / n,
            ped_stuck_ratio_mean=(float(np.mean(ped_stuck_ratios))
                                   if ped_stuck_ratios else 0.0),
            ped_stuck_ratio_max=(float(np.max(ped_stuck_ratios))
                                  if ped_stuck_ratios else 0.0),
            num_pedestrians=len(self.ped_stuck_ticks),
            ticks=self.tick,
        )


@contextmanager
def spawn(cmd: list, env: dict, name: str, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"[eval] spawning {name}: {' '.join(cmd)}  -> {log_path}")
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    try:
        yield proc
    finally:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            else:
                proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            log_f.close()
        except Exception:
            pass


def run_episode(map_name: str, seed: int, episode_secs: int, render: bool,
                no_follower: bool, follower_launch: str) -> dict:
    map_path = str(MAPS_DIR / f"{map_name}.txt")
    env = os.environ.copy()
    env["SIM_MAP"] = map_path
    env["SIM_SEED"] = str(seed)
    # Spawned follower runs as a script (file path), so /ws isn't on its
    # sys.path automatically. Prepend cwd so `from sim.world import ...` works.
    cwd = str(REPO)
    env["PYTHONPATH"] = (cwd + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else cwd)
    if render:
        env["SIM_RENDER"] = "1"

    py = sys.executable
    sim_cmd = [py, "-m", "sim.world"]
    leader_cmd = [py, "-m", "sim.leader"]
    # Spawn follower directly (bypass ros2 launch) so stdout reaches the
    # captured follower.log without launch's buffering / log-file tricks.
    impl = os.environ.get("FOLLOWER", "follow_everything")
    fname = {
        "baseline": "baseline_follower.py",
        "follow_everything": "follow_everything_follower.py",
    }.get(impl, "follow_everything_follower.py")
    follower_cmd = ([py, "-u", f"follower_pkg/python/{fname}"]
                    if not no_follower else None)

    log_dir = Path("results") / "logs" / f"ep_{int(time.time())}_{map_name}_{seed}"
    procs = []
    rclpy.init()
    recorder = EpisodeRecorder()
    try:
        with spawn(sim_cmd, env, "world", log_dir) as p_world, \
             spawn(leader_cmd, env, "leader", log_dir) as p_leader:
            procs += [p_world, p_leader]
            p_follower_ctx = None
            if follower_cmd:
                # Use the spawn helper for the follower too — captures stderr cleanly
                follower_log = open(log_dir / "follower.log", "w")
                p_follower = subprocess.Popen(
                    follower_cmd, env=env,
                    stdout=follower_log, stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )
                procs.append(p_follower)

            # Brief startup grace so subscribers come up before metrics begin
            for _ in range(20):
                rclpy.spin_once(recorder, timeout_sec=0.05)

            recorder.reset_metrics()
            t0 = time.time()
            while time.time() - t0 < episode_secs:
                rclpy.spin_once(recorder, timeout_sec=0.05)
                if any(p.poll() is not None for p in procs):
                    print(f"[eval] a child process exited unexpectedly (logs in {log_dir})",
                          file=sys.stderr)
                    break

            metrics = recorder.metrics()
            metrics["map"] = map_name
            metrics["seed"] = seed
            metrics["episode_secs"] = episode_secs
            metrics["log_dir"] = str(log_dir)
            return metrics
    finally:
        try:
            recorder.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


def main():
    args = parse_args()
    maps = select_maps(args.maps, args.episodes)
    print(f"[eval] {args.episodes} episodes, maps {maps}, seed base {args.seed}, "
          f"{args.episode_secs}s each")

    per_episode = []
    crashes = 0
    t_start = time.time()
    for k in range(args.episodes):
        seed_k = args.seed + k
        try:
            m = run_episode(maps[k], seed_k, args.episode_secs, args.render,
                            args.no_follower, args.follower_launch)
            per_episode.append(m)
            print(f"  ep{k:02d} {m['map']:>9s} seed={seed_k} "
                  f"success={m['follow_success_rate']:.3f} "
                  f"loss={m['loss_time_ratio']:.3f} "
                  f"d={m['avg_distance']:.2f} eff={m['path_efficiency']:.2f}")
        except Exception as e:
            print(f"  ep{k:02d} CRASH: {e}", file=sys.stderr)
            crashes += 1

    if per_episode:
        mean_success = float(np.mean([m["follow_success_rate"] for m in per_episode]))
        mean_loss = float(np.mean([m["loss_time_ratio"] for m in per_episode]))
        coll = int(sum(m["collision_count"] for m in per_episode))
        mean_d = float(np.mean([m["avg_distance"] for m in per_episode]))
        mean_eff = float(np.mean([m["path_efficiency"] for m in per_episode]))
        mean_l_stuck = float(np.mean([m["leader_stuck_ratio"] for m in per_episode]))
        mean_f_stuck = float(np.mean([m["follower_stuck_ratio"] for m in per_episode]))
        mean_p_stuck = float(np.mean([m["ped_stuck_ratio_mean"] for m in per_episode]))
        max_p_stuck = float(np.max([m["ped_stuck_ratio_max"] for m in per_episode]))
    else:
        mean_success = mean_loss = mean_d = mean_eff = 0.0
        mean_l_stuck = mean_f_stuck = mean_p_stuck = max_p_stuck = 0.0
        coll = 0

    total_t = time.time() - t_start
    summary = dict(
        mean_success=mean_success,
        mean_loss_ratio=mean_loss,
        collision_count=coll,
        mean_distance=mean_d,
        mean_path_eff=mean_eff,
        total_episodes=len(per_episode),
        crashes=crashes,
        total_seconds=total_t,
        episodes=per_episode,
    )

    # Canonical machine-readable block
    print("\n---")
    print(f"mean_success:        {mean_success:.3f}")
    print(f"mean_loss_ratio:     {mean_loss:.3f}")
    print(f"collision_count:     {coll}")
    print(f"mean_distance:       {mean_d:.2f}")
    print(f"mean_path_eff:       {mean_eff:.2f}")
    print(f"leader_stuck_ratio:  {mean_l_stuck:.3f}")
    print(f"follower_stuck_ratio:{mean_f_stuck:.3f}")
    print(f"ped_stuck_ratio_mean:{mean_p_stuck:.3f}")
    print(f"ped_stuck_ratio_max: {max_p_stuck:.3f}")
    print(f"total_episodes:      {len(per_episode)}")
    print(f"crashes:             {crashes}")
    print(f"total_seconds:       {total_t:.1f}")
    print(f"strategy:            {os.environ.get('STRATEGY', 'unspecified')}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
