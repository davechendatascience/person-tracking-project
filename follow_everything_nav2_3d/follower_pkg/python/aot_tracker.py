"""AOT/DeAOT streaming tracker bridge.

Parallel ROS node to edgetam_tracker.py / cutie_tracker.py that runs
the AOT family (yoxu515/aot-benchmark) — DeAOT, R50-DeAOTL, etc. —
as the streaming tracker. Picks model variant via the AOT_MODEL env
var; default is R50-DeAOTL (VOT-grade memory + matching architecture).

Per-frame flow mirrors `tools/demo.py` from the AOT repo:
  frame 0 → engine.add_reference_frame(img, mask, ...)
  frame N → engine.match_propogate_one_frame(img)
            + engine.decode_current_logits((H, W))

Two extras over vanilla:
  * Spatial-correlation-sampler is NOT required — the AOT source has
    a pure-PyTorch fallback (networks/layers/attention.py:213-219 et
    al.) we lean on, so no CUDA toolkit / extension compile.
  * DMAOT-style FIFO cap on the long-term memory bank. Vanilla
    DeAOT grows memory unboundedly (concat-along-dim-0 every Nth
    frame) → OOM on long videos (~355 frames @ 1040px on edge).
    The wrapper monkey-patches update_long_term_memory to keep
    only the most-recent AOT_LT_MAX entries.

The camera is 20 Hz; we publish on /follower/camera/detections_aot
at the tracker's actual rate. On every track() call we snapshot the
*latest* camera RGB; frames that arrived while the tracker was busy
are dropped.

Bootstrap: oracle leader pose gives us a bbox. We hand AOT a filled-
rectangle mask from that bbox at frame 0 — AOT's memory attention
refines the silhouette over the next few frames. Until init, the
node publishes empty Detection2DArrays.
"""
import os
# Must be set before torch is imported anywhere. Allocator tweaks for
# long-running tracking (validated via the AOT demo on a 1001-frame
# clip): max_split_size_mb=128 keeps big blocks intact for the
# long-term memory bank's concat operations;
# garbage_collection_threshold=0.7 proactively releases free segments;
# expandable_segments returns fragmented reserved memory to the OS
# instead of hoarding it. The demo showed frag dropping from 4.2% to
# 2.1% over a 1001-frame run with this config.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True",
)

import gc
import math
import sys
import threading
import queue
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2  # used in _publish_overlay
import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


# ---------------------------------------------------------------------------
# sys.path injection so the follow_everything + EdgeTAM checkouts mounted
# at /opt/* (see docker-compose.yml) are importable. /opt/EdgeTAM goes on
# the path directly because we import its `sam2` subdir as a top-level
# module via build_sam2_video_predictor.
for _p in ("/ws", "/opt", "/opt/EdgeTAM"):
    if _p not in sys.path and Path(_p).is_dir():
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
SAM2_CFG = {
    # EdgeTAM (facebookresearch/EdgeTAM) — a SAM2-compatible fork
    # optimised for edge devices. Mounted at /opt/EdgeTAM with its own
    # `sam2` package and checkpoint. ~10× faster propagation than
    # sam2_hiera_large; we replaced the DAM4SAM (DRM) wrapper with an
    # inline streaming wrapper that drives EdgeTAM directly — no
    # DAM4SAM dependency. See _build_aot_streaming_tracker.
    "model_cfg":  "configs/edgetam.yaml",
    "checkpoint": "/opt/EdgeTAM/checkpoints/edgetam.pt",
    "device":     "cuda",
    "image_size": 1024,
}
# Heuristic gates for our TrackResult — EdgeTAM gives us the mask, we
# decide whether the mask is "real enough" to publish a body-frame xy.
PERCEPTION_CFG = {
    "min_mask_area_ratio":  0.0005,
}

YOLO_WEIGHTS = "/opt/yolo11m.pt"
YOLO_CONF    = 0.5
PERSON_CLS   = 0  # COCO

# Camera mount on the chassis (must match sim/worlds/empty.world's
# rgbd_camera sensor pose AND sim/python/oracle_camera.py). Used by the
# hardcoded optical→base_link transform: the gz bridge does NOT publish
# follower/camera_optical_frame in /tf, so TF lookup fails. Same trick
# the oracle uses — bypass TF, rely on URDF-fixed offsets.
CAM_OFFSET_X_BODY = 0.10
CAM_OFFSET_Z_BODY = 1.30  # camera on mast at chassis z=1.20 + chassis link z=0.10
                          # below the 1.5 m obstacle height so the bot
                          # can't peek over walls

# Hard cap on total frames the worker will process before exiting. At
# ~5 Hz that's ~5.5 hours; plenty for a long demo.
MAX_FRAMES = 100_000
SIDE_DATA_KEEP = 64  # only need the most recent few for projection lookup

# Leader's body extents used to project a ground-truth bbox + build a
# depth-filtered init mask at SAM2 init. The bbox itself is generous (full
# body); the *mask* we hand SAM2 only includes pixels inside the bbox whose
# depth is within ±LEADER_DEPTH_TOL of the expected leader distance — that
# rejects wall/floor pixels that share the bbox area at close range and
# was the cause of the "last_seen lands on the wall" failure mode.
LEADER_HALF_W = 0.30
LEADER_HALF_H = 0.85
LEADER_Z_CTR  = 0.85
LEADER_DEPTH_TOL = 0.5
# Wait this long for the leader to walk into a non-clipped pose before
# falling back to a head-clipped seed. The leader patrol moves at ~0.7 m/s
# so within a few seconds it usually drifts past the 4 m full-view distance.
FULL_VIEW_TIMEOUT_SEC = 5.0


# Result emitted by the worker thread for each tracker step.
@dataclass
class TrackResult:
    mask: Optional[np.ndarray]
    confidence: float
    centroid_uv: Optional[Tuple[float, float]]
    is_visible: bool


# ---------------------------------------------------------------------------
def _msg_to_rgb(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding.lower().startswith("bgr"):
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def _msg_to_depth(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


def _stamp_to_ns(t) -> int:
    """builtin_interfaces/Time → integer nanoseconds since epoch."""
    return int(t.sec) * 1_000_000_000 + int(t.nanosec)


# ---------------------------------------------------------------------------
class AOTTracker(Node):
    def __init__(self) -> None:
        super().__init__("aot_tracker")

        # ---- Atomic snapshot of the latest ROS frame --------------------
        self._snap_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_K: Optional[np.ndarray] = None
        self._latest_stamp = None
        self._latest_stamp_seen_by_worker = None
        self._got_first_rgb = threading.Event()

        # Captured at YOLO bootstrap so SAM2 frame 0 == YOLO frame.
        self._init_bbox: Optional[np.ndarray] = None
        self._init_mask: Optional[np.ndarray] = None  # set by oracle path
        self._init_rgb: Optional[np.ndarray] = None
        self._init_depth: Optional[np.ndarray] = None
        self._init_K: Optional[np.ndarray] = None
        self._init_stamp = None
        self._init_event = threading.Event()

        # Side data keyed by SAM2 contiguous frame index — populated inside
        # the loader's request_cb at the moment we hand SAM2 a new frame.
        # Tuple is (rgb, depth, K, stamp); rgb is kept so the overlay topic
        # can render the mask on the exact RGB SAM2 saw.
        self._side_data: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, object]]" = OrderedDict()
        self._side_lock = threading.Lock()

        self._yolo = None
        self._results_q: "queue.Queue[Tuple[int, object]]" = queue.Queue(maxsize=4)
        self._stop = threading.Event()

        # Projection-path diagnostics. Logged once a second so we can see at
        # a glance whether lidar is reaching us and whether the cross-check
        # is dropping frames. Hot-path counters; no lock needed for these
        # int increments under CPython's GIL.
        self._proj_lidar_only = 0
        self._proj_depth_only = 0
        self._proj_both_agree = 0
        self._proj_disagree = 0
        self._proj_neither = 0
        self._centroid_log_n = 0
        self._last_scan_skew_ms = 0.0  # set in _publish_results

        # Time-aligned lidar buffer. The tracker runs ~7 Hz so by the time we
        # project a centroid to body-frame xy, the bot may have rotated ~17°
        # since the camera frame was captured. Sampling lidar at "now" would
        # use a beam pointing in a different direction. Instead we keep ~2 s
        # of recent scans and pick the one whose stamp is closest to the
        # camera frame's stamp.
        self._scan_buf: "deque[Tuple[int, LaserScan]]" = deque(maxlen=40)
        self._scan_lock = threading.Lock()

        # Odom buffer with the same time-alignment idea. Lets us:
        #   1. Convert body-frame xy at frame time → world frame (for log).
        #   2. Convert world frame → body frame at "now" before publishing,
        #      so the BT (which does body→world with its own current pose)
        #      lands the leader at the right world xy. Otherwise the bot's
        #      rotation between frame capture and BT consumption (up to
        #      ~25° at 1.5 rad/s × 200 ms) systematically biases the
        #      world-frame leader position.
        self._odom_buf: "deque[Tuple[int, float, float, float]]" = deque(maxlen=80)
        self._odom_lock = threading.Lock()

        self.create_subscription(Image, "/follower/camera/image", self._on_rgb, 10)
        self.create_subscription(Image, "/follower/camera/depth_image", self._on_depth, 10)
        self.create_subscription(CameraInfo, "/follower/camera/camera_info",
                                 self._on_camera_info, 10)
        # /gz_pose_truth — used ONLY as a fallback init seed for SAM2 when
        # YOLO can't find the leader (low-poly actor + low-res camera fail
        # the conf gate at distance). Once SAM2 is initialized we never
        # consult these poses; tracking is purely visual after frame 0.
        self._gz_follower_xyy: Optional[Tuple[float, float, float]] = None
        self._gz_leader_xy:    Optional[Tuple[float, float]]        = None
        self._gz_lock = threading.Lock()
        self._first_rgb_t: Optional[float] = None
        self.create_subscription(
            TFMessage, "/gz_pose_truth", self._on_gz_poses, 50)
        # /follower/scan_raw — the un-filtered lidar (lidar_leader_filter strips
        # leader hits from /follower/scan; we want the raw one so beams hitting
        # the leader are still there to give us range).
        self.create_subscription(LaserScan, "/follower/scan_raw", self._on_scan, 10)
        # /follower/odom — already shifted to first-quadrant world by
        # world_odom_publisher.py (matches the frame the BT uses).
        self.create_subscription(Odometry, "/follower/odom", self._on_odom, 20)
        self.pub = self.create_publisher(
            Detection2DArray, "/follower/camera/detections_aot", 10)
        # Debug overlay — RGB frame SAM2 actually processed, with mask
        # tinted red, centroid drawn, and frame_idx in the corner. View in
        # RViz with fixed_frame=follower/camera_optical_frame.
        self.overlay_pub = self.create_publisher(
            Image, "/follower/camera/aot_overlay", 10)

        # 50 Hz drain — non-blocking. Whenever SAM2 has yielded, we publish.
        self.create_timer(0.02, self._publish_results)
        self.create_timer(2.0, self._log_proj_stats)
        self._worker = threading.Thread(target=self._tracker_worker, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"AOT tracker live. "
            f"awaiting first 'person' YOLO detection (conf ≥ {YOLO_CONF}); "
            f"detection topic ticks at the tracker's own rate "
            f"(slower than the camera).")

    # ------------------------------------------------------------------
    # ROS callbacks (main thread)
    # ------------------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with self._snap_lock:
            self._latest_K = K

    def _on_depth(self, msg: Image) -> None:
        d = _msg_to_depth(msg)
        with self._snap_lock:
            self._latest_depth = d

    def _on_rgb(self, msg: Image) -> None:
        rgb = _msg_to_rgb(msg)
        with self._snap_lock:
            self._latest_rgb = rgb
            self._latest_stamp = msg.header
        self._got_first_rgb.set()
        if self._first_rgb_t is None:
            self._first_rgb_t = time.time()

        if self._init_bbox is None:
            # Oracle bootstrap is preferred — build_world spawns the bot
            # facing the leader so the projected bbox is on the actual
            # leader pixels at frame 0. YOLO on the low-poly actor at low
            # camera resolution often fails the conf gate, and a delayed
            # bootstrap (after the bot has rotated) seeded SAM2 onto walls.
            self._try_oracle_bootstrap(rgb, msg.header)
            if self._init_bbox is None:
                self._try_yolo_bootstrap(rgb, msg.header)

    def _on_gz_poses(self, msg: TFMessage) -> None:
        for tr in msg.transforms:
            t = tr.transform.translation
            if tr.child_frame_id == "follower":
                yaw = math.atan2(
                    2.0 * (tr.transform.rotation.w * tr.transform.rotation.z
                           + tr.transform.rotation.x * tr.transform.rotation.y),
                    1.0 - 2.0 * (tr.transform.rotation.y ** 2
                                  + tr.transform.rotation.z ** 2))
                with self._gz_lock:
                    self._gz_follower_xyy = (t.x, t.y, yaw)
            elif tr.child_frame_id == "leader":
                with self._gz_lock:
                    self._gz_leader_xy = (t.x, t.y)

    def _try_oracle_bootstrap(self, rgb: np.ndarray, header) -> None:
        """Build a tight, depth-filtered SAM2 init mask from ground-truth
        leader pose. The projected full-body bbox alone covered > 60% of
        the image at our 1–2 m spawn distance — SAM2 latched onto walls
        sharing the bbox and last_seen drifted onto the wall. Filtering the
        bbox region by depth ≈ expected leader distance keeps only leader
        pixels.

        Also waits for the leader to be in *full body view* (no clipping at
        the image edges) before seeding — a head-clipped seed produces a
        DRM template that doesn't match the leader's actual silhouette
        when they later move further away. Falls back to the clipped seed
        after FULL_VIEW_TIMEOUT_SEC so maps where the leader can't walk
        far enough still bootstrap eventually."""
        with self._snap_lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            K = None if self._latest_K is None else self._latest_K.copy()
        with self._gz_lock:
            f_xyy = self._gz_follower_xyy
            l_xy  = self._gz_leader_xy
        if depth is None or K is None or f_xyy is None or l_xy is None:
            return
        bbox_full = self._project_leader_bbox(rgb.shape, K, f_xyy, l_xy)
        if bbox_full is None:
            return
        bbox, fully_in_view = bbox_full
        if not fully_in_view:
            elapsed = (time.time() - self._first_rgb_t
                       if self._first_rgb_t is not None else 0.0)
            if elapsed < FULL_VIEW_TIMEOUT_SEC:
                # Leader is too close (head clipped) — wait for it to walk
                # farther so SAM2 sees the whole body.
                return
            self.get_logger().warn(
                f"oracle bootstrap: full-view timeout after {elapsed:.1f}s — "
                f"seeding with clipped bbox")
        x1, y1, x2, y2 = bbox
        bbox_xywh = np.array(
            [x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

        # Expected horizontal distance from camera mount to leader's body.
        fxw, fyw, _ = f_xyy
        lx, ly = l_xy
        d_expected = math.hypot(lx - fxw, ly - fyw) - CAM_OFFSET_X_BODY
        # Pixel mask: inside bbox AND depth within tolerance of expected.
        h, w = depth.shape
        u = np.arange(w)[None, :].repeat(h, axis=0)
        v = np.arange(h)[:, None].repeat(w, axis=1)
        in_bbox = (
            (u >= int(x1)) & (u <= int(x2)) &
            (v >= int(y1)) & (v <= int(y2)))
        depth_ok = (
            np.isfinite(depth)
            & (depth > max(0.1, d_expected - LEADER_DEPTH_TOL))
            & (depth < d_expected + LEADER_DEPTH_TOL))
        mask = (in_bbox & depth_ok).astype(np.uint8)
        n_on = int(mask.sum())
        if n_on < 50:
            self.get_logger().warn(
                f"oracle bootstrap: depth filter found only {n_on} px in bbox "
                f"d_expected={d_expected:.2f} m — falling back to bbox-only seed")
            self._init_mask = None
        else:
            self._init_mask = mask

        self._init_rgb   = rgb.copy()
        self._init_depth = depth
        self._init_K     = K
        self._init_stamp = header
        self._init_bbox  = bbox_xywh
        self._init_event.set()
        self.get_logger().info(
            f"oracle bootstrap: bbox xyxy=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
            f"d_expected={d_expected:.2f}m mask_px={n_on}")

    @staticmethod
    def _project_leader_bbox(
        img_shape, K: np.ndarray,
        f_xyy: Tuple[float, float, float],
        l_xy:  Tuple[float, float],
    ) -> Optional[Tuple[Tuple[float, float, float, float], bool]]:
        """Project an axis-aligned 3D box around the leader's body to image
        pixel coords. Returns ((x1, y1, x2, y2), fully_in_view) where the
        bbox is clipped to the image, and fully_in_view is True iff the
        unclipped projection fit entirely within the image (no head/foot
        cropping). None if leader is entirely outside the FOV."""
        h, w = img_shape[:2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        fxw, fyw, fyaw = f_xyy
        lx, ly = l_xy
        cosY, sinY = math.cos(fyaw), math.sin(fyaw)

        u_min = float("inf"); v_min = float("inf")
        u_max = -float("inf"); v_max = -float("inf")
        any_in_front = False
        for dx in (-LEADER_HALF_W, LEADER_HALF_W):
            for dy in (-LEADER_HALF_W, LEADER_HALF_W):
                for dz in (-LEADER_HALF_H, LEADER_HALF_H):
                    # World-frame corner.
                    wx = lx + dx
                    wy = ly + dy
                    wz = LEADER_Z_CTR + dz
                    # World → body frame (inverse rotation by bot yaw,
                    # then subtract bot xy).
                    rx = wx - fxw
                    ry = wy - fyw
                    body_x =  cosY * rx + sinY * ry
                    body_y = -sinY * rx + cosY * ry
                    body_z = wz
                    # Body → optical frame (the URDF camera mount):
                    # opt is at (CAM_OFFSET_X_BODY, 0, CAM_OFFSET_Z_BODY)
                    # in body frame, pointing along body +x.
                    opt_z = body_x - CAM_OFFSET_X_BODY
                    if opt_z <= 0.05:
                        continue  # behind / inside camera; skip this corner
                    any_in_front = True
                    opt_x = -body_y
                    opt_y = -(body_z - CAM_OFFSET_Z_BODY)
                    u = fx * opt_x / opt_z + cx
                    v = fy * opt_y / opt_z + cy
                    u_min = min(u_min, u); v_min = min(v_min, v)
                    u_max = max(u_max, u); v_max = max(v_max, v)
        if not any_in_front:
            return None
        fully_in_view = (
            u_min >= 0.0 and u_max <= w - 1.0
            and v_min >= 0.0 and v_max <= h - 1.0)
        x1 = max(0.0, min(w - 1.0, u_min))
        y1 = max(0.0, min(h - 1.0, v_min))
        x2 = max(0.0, min(w - 1.0, u_max))
        y2 = max(0.0, min(h - 1.0, v_max))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return ((x1, y1, x2, y2), fully_in_view)

    def _on_scan(self, msg: LaserScan) -> None:
        ns = _stamp_to_ns(msg.header.stamp)
        with self._scan_lock:
            empty_before = not self._scan_buf
            self._scan_buf.append((ns, msg))
        if empty_before:
            n = len(msg.ranges)
            finite = [r for r in msg.ranges if math.isfinite(r)]
            min_str = f"{min(finite):.2f}" if finite else "nan"
            max_str = f"{max(finite):.2f}" if finite else "nan"
            self.get_logger().info(
                f"first scan: n={n} angle_min={msg.angle_min:.3f} "
                f"angle_max={msg.angle_max:.3f} inc={msg.angle_increment:.4f} "
                f"range_min={msg.range_min:.2f} range_max={msg.range_max:.2f} "
                f"finite_beams={len(finite)} "
                f"min_finite={min_str} max_finite={max_str}")

    def _log_proj_stats(self) -> None:
        with self._scan_lock:
            buf_len = len(self._scan_buf)
        total = (self._proj_lidar_only + self._proj_depth_only
                 + self._proj_both_agree + self._proj_disagree
                 + self._proj_neither)
        if total == 0 and buf_len == 0:
            return
        self.get_logger().info(
            f"proj: scan_buf={buf_len} "
            f"lidar+depth_agree={self._proj_both_agree} "
            f"disagree={self._proj_disagree} "
            f"lidar_only={self._proj_lidar_only} "
            f"depth_only={self._proj_depth_only} "
            f"neither={self._proj_neither}")

    def _on_odom(self, msg: Odometry) -> None:
        ns = _stamp_to_ns(msg.header.stamp)
        p = msg.pose.pose
        q = p.orientation
        # yaw from quaternion
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._odom_lock:
            self._odom_buf.append((ns, p.position.x, p.position.y, yaw))

    def _pose_at(
        self, target_ns: int,
    ) -> Tuple[Optional[Tuple[float, float, float]], float]:
        """Return ((x_w, y_w, yaw), skew_ms) for the buffered odom pose
        whose stamp is closest to target_ns. Returns (None, skew) if no
        pose within 80 ms or if the buffer is empty.

        80 ms cap chosen vs 200 ms: at the bot's max 1.5 rad/s a 200 ms
        skew would bake 17° of yaw error into the world projection
        (≈ 1.5 m offset at a 5 m leader distance). 80 ms keeps that under
        7° / 0.6 m. Odom publishes at 30 Hz so the buffer is dense enough
        to satisfy the tighter window in steady state."""
        with self._odom_lock:
            if not self._odom_buf:
                return None, 0.0
            ns, x, y, yaw = min(
                self._odom_buf, key=lambda kv: abs(kv[0] - target_ns))
        skew_ns = ns - target_ns
        if abs(skew_ns) > 80_000_000:
            return None, skew_ns / 1e6
        return (x, y, yaw), skew_ns / 1e6

    def _latest_pose(self) -> Optional[Tuple[float, float, float]]:
        with self._odom_lock:
            if not self._odom_buf:
                return None
            _, x, y, yaw = self._odom_buf[-1]
            return (x, y, yaw)

    def _scan_at(self, target_ns: int) -> Tuple[Optional[LaserScan], float]:
        """Return (scan, skew_ms) where scan is the buffered LaserScan whose
        stamp is closest to target_ns, and skew_ms = scan_stamp - frame_stamp
        in milliseconds. Returns (None, 0.0) if buffer is empty or nothing
        within 200 ms (a safety bound — a half-second-old scan would have
        the bot pointing 40°+ away from where it was at frame capture)."""
        with self._scan_lock:
            if not self._scan_buf:
                return None, 0.0
            ns, scan = min(self._scan_buf, key=lambda kv: abs(kv[0] - target_ns))
        skew_ns = ns - target_ns
        if abs(skew_ns) > 200_000_000:
            return None, skew_ns / 1e6
        return scan, skew_ns / 1e6

    def _try_yolo_bootstrap(self, rgb: np.ndarray, header) -> None:
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(YOLO_WEIGHTS)
        results = self._yolo(rgb[:, :, ::-1], verbose=False, conf=YOLO_CONF)
        if not results:
            return
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        person_mask = (cls == PERSON_CLS) & (conf >= YOLO_CONF)
        if not person_mask.any():
            return
        i = int(np.argmax(np.where(person_mask, conf, -1.0)))
        x1, y1, x2, y2 = xyxy[i].astype(np.float32)
        # DAM4SAM/dam4sam_tracker.py:278 expects [x, y, w, h] and computes
        # [x, y, x+w, y+h]. Passing YOLO's xyxy directly produced a box
        # ~10× the actual person area, contaminating the DRM template with
        # background — visible as occasional mask drift onto walls/shadow.
        bbox_xywh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

        with self._snap_lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            K = None if self._latest_K is None else self._latest_K.copy()
        if depth is None or K is None:
            return  # need depth + K to do anything useful with a hit

        self._init_rgb   = rgb.copy()
        self._init_depth = depth
        self._init_K     = K
        self._init_stamp = header
        self._init_bbox  = bbox_xywh
        self._init_event.set()
        self.get_logger().info(
            f"YOLO bootstrap: init bbox xyxy=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
            f"→ xywh={bbox_xywh.tolist()} (conf {conf[i]:.2f})")

    # ------------------------------------------------------------------
    # Worker thread — full EdgeTAM tracker, one track() per ROS frame.
    # ------------------------------------------------------------------
    def _tracker_worker(self) -> None:
        # Wait for YOLO bootstrap.
        while not self._stop.is_set():
            if self._init_event.wait(timeout=0.5):
                break
        if self._stop.is_set():
            return

        log = self.get_logger()
        try:
            tracker = self._build_aot_streaming_tracker(log)
        except Exception as e:
            import traceback
            log.error(f"AOT build failed: {e!r}\n{traceback.format_exc()}")
            return

        from PIL import Image as PILImage
        import torch

        # ---- Frame 0: bootstrap with the YOLO snapshot --------------
        with self._side_lock:
            self._side_data[0] = (
                self._init_rgb, self._init_depth,
                self._init_K, self._init_stamp)

        init_pil = PILImage.fromarray(self._init_rgb)
        try:
            # AOT (unlike SAM2/EdgeTAM) is trained on dense per-object
            # masks from YT-VOS / DAVIS — it expects a real silhouette,
            # not a bbox blob. The oracle bootstrap path builds a
            # depth-filtered mask at `self._init_mask`; we pass that
            # straight through when available. Filled-rect-from-bbox
            # confuses AOT's first-frame conditioning because the
            # rect covers visible background (sky, ground, trees in
            # forest map), which AOT then treats as part of the
            # target → drift the moment the bot moves.
            #
            # Fallback: if the depth-filtered mask is too sparse
            # (< MIN_INIT_MASK_PX), fall back to bbox-only and let
            # the wrapper convert to filled rect — better than nothing.
            init_bbox = self._init_bbox.astype(np.float32).tolist()
            init_mask = self._init_mask
            MIN_INIT_MASK_PX = 100
            if init_mask is not None and int(init_mask.sum()) >= MIN_INIT_MASK_PX:
                log.info(
                    f"AOT init source: depth-filtered mask "
                    f"(px={int(init_mask.sum())}, bbox={init_bbox})")
                out_dict = tracker.initialize(
                    init_pil, init_mask=init_mask, bbox=None)
            else:
                log.warn(
                    f"AOT init source: bbox→filled-rect fallback "
                    f"(depth mask px={0 if init_mask is None else int(init_mask.sum())} "
                    f"< {MIN_INIT_MASK_PX}, bbox={init_bbox})")
                out_dict = tracker.initialize(
                    init_pil, init_mask=None, bbox=init_bbox)
        except Exception as e:
            import traceback
            log.error(f"AOT initialize crashed: {e!r}\n{traceback.format_exc()}")
            return
        mask0 = out_dict["pred_mask"]
        log.info(
            f"AOT init: mask shape={mask0.shape} px_on={int(mask0.sum())} "
            f"bbox={self._init_bbox.tolist()}")
        # Debug-dump init RGB + bbox + predicted mask so we can see what
        # EdgeTAM was shown when it failed to lock on. Saved to a stable
        # path under the episode log dir if EP_LOG_DIR is set.
        debug_dir = os.environ.get("EP_LOG_DIR")
        if debug_dir:
            try:
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                _x1, _y1, _w, _h = [int(v) for v in self._init_bbox.tolist()]
                _x2, _y2 = _x1 + _w, _y1 + _h
                rgb_with_box = self._init_rgb.copy()
                cv2.rectangle(rgb_with_box, (_x1, _y1), (_x2, _y2),
                              (0, 255, 0), 2)
                cv2.imwrite(f"{debug_dir}/edgetam_init_rgb.png",
                            rgb_with_box[..., ::-1])
                cv2.imwrite(f"{debug_dir}/edgetam_init_mask.png",
                            (mask0 * 255).astype(np.uint8))
            except Exception as e:
                log.warn(f"debug init dump failed: {e!r}")
        # Pass the wrapper's smart centroid (prob-weighted + EMA) into
        # _make_result instead of letting it compute a geometric mean
        # over the binary mask.
        self._push_result(0, self._make_result(mask0, out_dict.get("centroid_uv")))
        vis_n = 1 if mask0.any() else 0
        inv_n = 0 if mask0.any() else 1
        # Dump overlay PNGs for the first N propagated frames. Default 8
        # for quick smoke-debug; bump via AOT_DEBUG_FRAMES env var to
        # generate a per-frame video of the full episode.
        debug_frames_left = int(os.environ.get("AOT_DEBUG_FRAMES", "8"))

        # ---- Streaming loop: one track() per fresh ROS frame --------
        LOG_EVERY = 30
        # EdgeTAM stores every prepared image + per-frame predictor
        # features in inference_state[...]. With edgetam.pt at 1024x1024
        # that's ~3.5 GB / 30 frames if nothing is freed — we OOM in ~90 s.
        # Keep a rolling window of recent frames; drop the rest.
        STATE_KEEP_BEHIND = 16
        EMPTY_CACHE_EVERY = 30
        # The conditioning frame (frame 0) MUST stay — SAM2's propagation
        # re-reads its cached features each step. Everything else past
        # STATE_KEEP_BEHIND can go.
        frame_idx = 1
        # Worker-rate throttle. Pure-PyTorch AOT can saturate a CPU
        # core, which makes the BT/Nav2 thread (on the same host
        # network) lag. The BT ticks at ~20 Hz and Nav2 plans at
        # ~10 Hz, so 8-12 Hz of fresh detections is plenty. Cap the
        # worker so we leave headroom for the rest of the stack.
        # Set TRACKER_MAX_HZ=0 to disable the cap (run flat-out).
        tracker_max_hz = float(os.environ.get("TRACKER_MAX_HZ", "10"))
        min_period = (1.0 / tracker_max_hz) if tracker_max_hz > 0 else 0.0
        last_iter_t = time.time()
        try:
            while not self._stop.is_set() and frame_idx < MAX_FRAMES:
                # Pace the worker — sleep the remainder of the slot
                # if we finished a frame faster than min_period.
                if min_period > 0:
                    sleep_for = min_period - (time.time() - last_iter_t)
                    if sleep_for > 0.001:
                        time.sleep(sleep_for)
                last_iter_t = time.time()
                rgb, depth, K, stamp = self._snapshot_latest_blocking()
                if rgb is None:
                    return  # stop event or timeout

                with self._side_lock:
                    self._side_data[frame_idx] = (rgb, depth, K, stamp)
                    while len(self._side_data) > SIDE_DATA_KEEP:
                        self._side_data.popitem(last=False)

                pil = PILImage.fromarray(rgb)
                out_dict = tracker.track(pil, init=False)
                mask = out_dict["pred_mask"]
                if mask.any():
                    vis_n += 1
                else:
                    inv_n += 1
                self._push_result(
                    frame_idx,
                    self._make_result(mask, out_dict.get("centroid_uv")))
                if debug_frames_left > 0 and debug_dir:
                    try:
                        overlay = rgb.copy()
                        if mask.any():
                            overlay[mask > 0] = (
                                overlay[mask > 0] * 0.5
                                + np.array([255, 80, 80]) * 0.5).astype(np.uint8)
                        cv2.imwrite(
                            f"{debug_dir}/edgetam_track_f{frame_idx:03d}.png",
                            overlay[..., ::-1])
                        cv2.imwrite(
                            f"{debug_dir}/edgetam_track_f{frame_idx:03d}_mask.png",
                            (mask * 255).astype(np.uint8))
                    except Exception as e:
                        log.warn(f"debug track dump failed: {e!r}")
                    debug_frames_left -= 1

                self._evict_aot_state(
                    tracker, frame_idx, STATE_KEEP_BEHIND)

                if frame_idx % EMPTY_CACHE_EVERY == 0:
                    # gc.collect() before empty_cache() so Python releases
                    # any zero-refcount CUDA tensors back to the caching
                    # allocator first; empty_cache() can then actually
                    # return those segments to the OS.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if frame_idx % LOG_EVERY == 0:
                    alloc_gb = (
                        torch.cuda.memory_allocated() / 1e9
                        if torch.cuda.is_available() else 0.0)
                    log.info(
                        f"aot f={frame_idx} "
                        f"vis={vis_n}/{vis_n+inv_n} "
                        f"last_mask_px={int(mask.sum())} "
                        f"cuda alloc={alloc_gb:.2f}GB")

                frame_idx += 1
        except Exception as e:
            import traceback
            log.error(f"tracker_worker crashed: {e!r}\n{traceback.format_exc()}")

    def _evict_aot_state(
        self, tracker, frame_idx: int, keep_behind: int,
    ) -> None:
        """Drop per-frame state in EdgeTAM's inference_state + predictor
        caches for any frame older than (frame_idx - keep_behind), except
        frame 0 (the conditioning frame).

        Called after every track() step to keep VRAM bounded."""
        cutoff = frame_idx - keep_behind
        if cutoff <= 0:
            return
        state = getattr(tracker, "inference_state", None)
        if state is None:
            return
        # Prepared images live in state["images"][i].
        imgs = state.get("images")
        if isinstance(imgs, dict):
            for k in [k for k in imgs if 0 < k < cutoff]:
                del imgs[k]
        # SAM2 video predictor caches features per frame.
        for key in ("cached_features",):
            d = state.get(key)
            if isinstance(d, dict):
                for k in [k for k in d if 0 < k < cutoff]:
                    del d[k]
        # Output dicts — keep frame 0 (cond) and recent non-cond.
        out_dict = state.get("output_dict") or {}
        for sect in ("non_cond_frame_outputs",):
            d = out_dict.get(sect, {})
            for k in [k for k in d if 0 < k < cutoff]:
                del d[k]
        # Per-object output dict mirrors the global one.
        for obj_out in (state.get("output_dict_per_obj") or {}).values():
            d = obj_out.get("non_cond_frame_outputs", {})
            for k in [k for k in d if 0 < k < cutoff]:
                del d[k]
        # frames_already_tracked is a dict of (frame_idx -> dict).
        ft = state.get("frames_already_tracked")
        if isinstance(ft, dict):
            for k in [k for k in ft if 0 < k < cutoff]:
                del ft[k]

    # ------------------------------------------------------------------
    # AOT helpers — kept as instance methods so they're easy to test.
    # ------------------------------------------------------------------
    def _build_aot_streaming_tracker(self, log):
        """Build the AOT/DeAOT video object segmenter from
        yoxu515/aot-benchmark and wrap it in a streaming tracker that
        exposes the same .initialize / .track API the rest of this
        file expects.

        Mirrors the inference flow of `tools/demo.py` from the AOT
        repo (paper-faithful preprocessing, single-scale, decoder
        upsamples back to video resolution per frame). Two extras
        we layer on top of vanilla DeAOT here, both targeting the
        unbounded-memory failure mode that caps long videos:

          - DMAOT-style FIFO cap on the long-term memory bank.
            Vanilla DeAOT's `update_long_term_memory` concatenates
            features per layer along dim=0 every TEST_LONG_TERM_MEM_GAP
            frames with no eviction → GPU OOM at ~355 frames on the
            pure-PyTorch fallback. We monkey-patch the engine class
            to keep only the most-recent AOT_LT_MAX entries (oldest
            evicted first). DMAOT proper uses cosine-similarity
            dropout to keep the *most informative* frames; FIFO is the
            cheap bounded variant.
          - LONG_TERM_MEM_GAP override (env var AOT_LT_GAP), so
            ROS can use a denser default than the engine's stage cfg.

        We also skip the spatial_correlation_sampler dependency: the
        AOT source has try/except + pure-PyTorch fallbacks at every
        usage site (networks/layers/attention.py:213-219 et al.).
        That saves us a CUDA toolkit install in the Dockerfile."""
        import os
        import importlib
        import torch

        sys.path.insert(0, "/opt/aot-benchmark")

        # Config — picks model variant via env var so the same node
        # can drive r50_deaotl / deaott / deaotl / etc. AOT_CKPT
        # defaults to the matching checkpoint for AOT_MODEL so callers
        # don't have to set both; only override if they want a
        # non-standard checkpoint.
        aot_model = os.environ.get("AOT_MODEL", "r50_deaotl")
        aot_stage = os.environ.get("AOT_STAGE", "pre_ytb_dav")
        _ckpt_for_model = {
            "r50_deaotl": "R50_DeAOTL_PRE_YTB_DAV.pth",
            "deaott":     "DeAOTT_PRE_YTB_DAV.pth",
            "deaots":     "DeAOTS_PRE_YTB_DAV.pth",
            "deaotb":     "DeAOTB_PRE_YTB_DAV.pth",
            "deaotl":     "DeAOTL_PRE_YTB_DAV.pth",
            "aott":       "AOTT_PRE_YTB_DAV.pth",
            "aots":       "AOTS_PRE_YTB_DAV.pth",
            "aotb":       "AOTB_PRE_YTB_DAV.pth",
            "aotl":       "AOTL_PRE_YTB_DAV.pth",
            "r50_aotl":   "R50_AOTL_PRE_YTB_DAV.pth",
        }
        aot_ckpt  = os.environ.get(
            "AOT_CKPT",
            f"/opt/aot-benchmark/pretrain_models/"
            f"{_ckpt_for_model.get(aot_model, 'R50_DeAOTL_PRE_YTB_DAV.pth')}")
        aot_lt_gap = int(os.environ.get("AOT_LT_GAP", "5"))
        # Demo-validated default: gap=5 + lt_max=80 + batched eviction
        # produced 636/1001 non-empty (best of all variants tested) with
        # stable 5.8 GB / 2.1% frag. lt_max=30 worked but is more
        # aggressive than necessary on Grace Blackwell-class VRAM.
        aot_lt_max = int(os.environ.get("AOT_LT_MAX", "80"))
        aot_lt_batch_evict = int(os.environ.get("AOT_LT_BATCH_EVICT_EVERY", "50"))
        aot_lt_keep_ratio = float(os.environ.get("AOT_LT_KEEP_RATIO", "0.8"))
        aot_max_long_edge = int(os.environ.get("AOT_MAX_LONG_EDGE", "800"))

        log.info(
            f"AOT tracker model={aot_model} stage={aot_stage} ckpt={aot_ckpt}")

        engine_config_mod = importlib.import_module(f"configs.{aot_stage}")
        cfg = engine_config_mod.EngineConfig("ros", aot_model)
        cfg.TEST_GPU_ID = 0
        cfg.TEST_CKPT_PATH = aot_ckpt
        cfg.TEST_LONG_TERM_MEM_GAP = aot_lt_gap

        device = (torch.device("cuda")
                  if torch.cuda.is_available() else torch.device("cpu"))
        gpu_id = 0 if device.type == "cuda" else -1

        # Confirm whether the CUDA correlation kernel was importable.
        # If False we're on the slower pure-PyTorch fallback path.
        try:
            from networks.layers.attention import enable_corr as _enable_corr
        except Exception:
            _enable_corr = None
        log.info(
            f"AOT enable_corr (CUDA correlation kernel)={_enable_corr} "
            f"— {'fast path' if _enable_corr else 'pure-PyTorch fallback'}")
        log.info(
            f"AOT memory: LT_GAP={aot_lt_gap} (store every Nth frame), "
            f"LT_MAX={aot_lt_max} (cap), "
            f"BATCH_EVICT_EVERY={aot_lt_batch_evict} frames, "
            f"KEEP_RATIO={aot_lt_keep_ratio}")

        from networks.models import build_vos_model
        from networks.engines import build_engine
        from utils.checkpoint import load_network

        # ── Batched-eviction long-term memory cap ───────────────────
        # Demo-validated (1001-frame video, gap=5): batched eviction
        # every N frames keeping newest keep_ratio*max is friendlier to
        # the CUDA allocator than per-frame trimming (4.2% → 2.1% frag
        # drop across the run, vs growing frag with per-call slicing).
        # Hard safety cap at max*1.5 catches any pathological growth
        # between scheduled batch evictions. Newest entries are at the
        # FRONT of the concat (torch.cat([new, last], dim=0)), so we
        # slice [:keep*HW] = keep newest, drop oldest.
        # MODEL_ENGINE is 'aotengine' or 'deaotengine' (no underscore);
        # the file lives at networks/engines/{aot,deaot}_engine.py.
        if cfg.MODEL_ENGINE.startswith("deaot"):
            engine_module = importlib.import_module(
                "networks.engines.deaot_engine")
            engine_cls = engine_module.DeAOTEngine
            engine_cls_name = "DeAOTEngine"
        else:
            engine_module = importlib.import_module(
                "networks.engines.aot_engine")
            engine_cls = engine_module.AOTEngine
            engine_cls_name = "AOTEngine"

        # Reset prior monkey-patch (cap-only version) if present, so a
        # reload uses the new batched-eviction logic.
        if getattr(engine_cls, "_lt_cap_installed", False):
            log.info("Re-installing long-term memory cap with new strategy.")

        # Capture engine-class import for use inside the closure.
        AOTEngine_cls = engine_cls
        _hard_cap = int(aot_lt_max * 1.5) if aot_lt_max > 0 else 0
        _keep_frames = max(1, int(aot_lt_max * aot_lt_keep_ratio)) \
            if aot_lt_max > 0 else 0
        _evict_stats = {"calls": 0, "last_log": 0}

        def _batch_evict_update(self, new_long_term_memories,
                                _cap=aot_lt_max,
                                _hard=_hard_cap,
                                _keep=_keep_frames,
                                _every=aot_lt_batch_evict,
                                _stats=_evict_stats,
                                _log=log):
            if self.long_term_memories is None:
                self.long_term_memories = new_long_term_memories
                return

            # Token count per frame (HW) — uniform across all layers.
            HW = None
            for layer in new_long_term_memories:
                for e in layer:
                    if e is not None:
                        HW = e.shape[0]
                        break
                if HW is not None:
                    break

            do_batch = (
                _cap > 0 and _every > 0 and self.frame_step > 0
                and (self.frame_step % _every) == 0)

            updated = []
            evicted = False
            for new_lm, last_lm in zip(
                    new_long_term_memories, self.long_term_memories):
                layer_out = []
                for new_e, last_e in zip(new_lm, last_lm):
                    if new_e is None or last_e is None:
                        layer_out.append(None)
                        continue
                    cat = torch.cat([new_e, last_e], dim=0)
                    if HW is not None and _cap > 0:
                        T = cat.shape[0] // HW
                        # Hard safety cap — fires only if scheduled
                        # eviction was somehow missed.
                        if T > _hard:
                            cat = cat[: _hard * HW].contiguous()
                            evicted = True
                        elif do_batch and T > _cap:
                            cat = cat[: _keep * HW].contiguous()
                            evicted = True
                    layer_out.append(cat)
                updated.append(layer_out)
            self.long_term_memories = updated

            if evicted:
                _stats["calls"] += 1
                if (_stats["calls"] - _stats["last_log"]) >= 5 \
                        or _stats["calls"] == 1:
                    _stats["last_log"] = _stats["calls"]
                    _log.info(
                        f"AOT LT-cap batch evict "
                        f"(frame={self.frame_step}, kept={_keep}, "
                        f"max={_cap}, total_evicts={_stats['calls']})")

        engine_cls.update_long_term_memory = _batch_evict_update
        engine_cls._lt_cap_installed = True
        engine_cls._dmaot_capped = True  # legacy flag, keep for back-compat
        log.info(
            f"Installed batched-eviction long-term memory cap on "
            f"{engine_cls_name}.update_long_term_memory "
            f"(max={aot_lt_max}, batch_every={aot_lt_batch_evict}, "
            f"keep={_keep_frames}, hard_cap={_hard_cap})")

        # ── Build model + engine ────────────────────────────────────
        _t_build = time.time()
        model = build_vos_model(cfg.MODEL_VOS, cfg)
        if device.type == "cuda":
            model = model.cuda(gpu_id)
        else:
            model = model.to(device)
        model, _ = load_network(model, cfg.TEST_CKPT_PATH, gpu_id)
        model.eval()
        engine = build_engine(
            cfg.MODEL_ENGINE,
            phase="eval",
            aot_model=model,
            gpu_id=gpu_id,
            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
        )
        log.info(
            f"AOT predictor built in {time.time() - _t_build:.1f}s "
            f"(ready for first frame)")

        img_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=device)
        img_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=device)

        class _AOTStreamingTracker:
            """Thin streaming wrapper around the AOT engine.

            Preprocessing reproduces MultiRestrictSize + MultiToTensor
            from the AOT repo (max_long_edge cap, stride-16 alignment
            with the align_corners +1 trick, ImageNet normalization).
            Per-frame flow mirrors `tools/demo.py`:
              - frame 0 → engine.add_reference_frame(img, mask, ...)
              - frame N → engine.match_propogate_one_frame(img)
                          + engine.decode_current_logits((H, W))
            """

            def __init__(self):
                self.engine = engine
                self.model = model
                self.device = device
                self.img_mean = img_mean
                self.img_std = img_std
                self.max_long_edge = aot_max_long_edge
                self.align_corners = bool(cfg.MODEL_ALIGN_CORNERS)
                self.frame_index = 0
                self.video_h = 0
                self.video_w = 0
                self.input_h = 0
                self.input_w = 0
                # Centroid history for the prob-weighted + blob-tracking
                # + EMA smoothing pipeline in _logit_to_result. Set on
                # frame 0's add_reference_frame; updated each track().
                self._last_centroid_uv: Optional[Tuple[float, float]] = None

            def _aligned(self, n):
                """Round n to align with stride 16, mirroring
                MultiRestrictSize. With align_corners=True the
                AOT repo uses (n-1) % 16 == 0, otherwise n % 16 == 0."""
                stride = 16
                if self.align_corners:
                    return int(round((n - 1) / stride) * stride) + 1
                return int(round(n / stride) * stride)

            def _prep_image(self, image_pil):
                """PIL image → (1, 3, H', W') float tensor on device."""
                arr = np.array(image_pil)
                h, w = arr.shape[:2]
                long_edge = max(h, w)
                scale = (self.max_long_edge / long_edge
                         if long_edge > self.max_long_edge else 1.0)
                new_h = max(17, self._aligned(int(h * scale)))
                new_w = max(17, self._aligned(int(w * scale)))
                if (new_h, new_w) != (h, w):
                    arr = cv2.resize(arr, (new_w, new_h),
                                     interpolation=cv2.INTER_CUBIC)
                self.input_h, self.input_w = new_h, new_w
                t = (torch.from_numpy(arr).to(self.device)
                     .permute(2, 0, 1).float() / 255.0)
                t = (t - self.img_mean[:, None, None]) / self.img_std[:, None, None]
                return t.unsqueeze(0)

            def _prep_mask(self, mask_np):
                """uint8 HxW mask → (1, 1, H', W') int tensor on device,
                resized to match the prepared image."""
                if mask_np.shape != (self.input_h, self.input_w):
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8),
                        (self.input_w, self.input_h),
                        interpolation=cv2.INTER_NEAREST)
                t = (torch.from_numpy(mask_np.astype(np.int64))
                     .unsqueeze(0).unsqueeze(0).to(self.device).float())
                return t

            @torch.inference_mode()
            def initialize(self, image, init_mask=None, bbox=None):
                """Seed AOT on frame 0. Either init_mask (HxW uint8)
                or bbox ([x, y, w, h]) must be provided. Bbox is
                converted to a filled rectangle mask — AOT refines
                the silhouette within a few frames via memory
                attention. Best results come from a clean mask."""
                self.frame_index = 0
                self.video_w = image.width
                self.video_h = image.height
                self.engine.restart_engine()

                if init_mask is None:
                    if bbox is None:
                        raise ValueError(
                            "AOT init: neither bbox nor init_mask provided")
                    x, y, w, h = [int(v) for v in bbox]
                    init_mask = np.zeros(
                        (self.video_h, self.video_w), dtype=np.uint8)
                    x1, y1 = max(0, x), max(0, y)
                    x2 = min(self.video_w, x + w)
                    y2 = min(self.video_h, y + h)
                    init_mask[y1:y2, x1:x2] = 1
                else:
                    init_mask = (init_mask > 0).astype(np.uint8)

                img_t = self._prep_image(image)
                mask_t = self._prep_mask(init_mask)
                self.engine.add_reference_frame(
                    img_t, mask_t, obj_nums=[1], frame_step=0)

                # Reset the centroid history so frame-0 selection
                # is "biggest valid blob" (no last_centroid to compare).
                self._last_centroid_uv = None

                # Decode at the video resolution so the published mask
                # matches camera_info; the BT projection assumes that.
                logit = self.engine.decode_current_logits(
                    (self.video_h, self.video_w))
                m, cuv = self._logit_to_result(logit)
                return {"pred_mask": m, "centroid_uv": cuv}

            @torch.inference_mode()
            def track(self, image, init=False):
                """Advance one frame; AOT keeps its own memory state."""
                if init:
                    return self.initialize(image)
                self.frame_index += 1
                img_t = self._prep_image(image)
                self.engine.match_propogate_one_frame(img_t)
                logit = self.engine.decode_current_logits(
                    (self.video_h, self.video_w))
                m, cuv = self._logit_to_result(logit)
                if m is None or m.size == 0:
                    m = np.zeros(
                        (self.video_h, self.video_w), dtype=np.uint8)
                    cuv = None
                return {"pred_mask": m, "centroid_uv": cuv}

            def _logit_to_result(self, logit):
                """Decode AOT logits into (mask, centroid_uv).

                Three stacked improvements over plain `argmax` + geometric
                centroid:

                  1. Probability-weighted centroid. The geometric centre
                     of a thresholded binary mask treats every pixel as
                     equal — a few drifty pixels at the boundary shift
                     it frame-to-frame. Weighting by per-pixel fg_prob
                     pins the centroid on the high-confidence core
                     (head/torso) and ignores the noisy fringe.
                  2. Blob selection by proximity to last_centroid.
                     When multiple blobs survive the area threshold
                     (head + torso + a stray noise blob), picking the
                     largest can swing focus between them. Picking the
                     one closest to the previous centroid preserves
                     temporal continuity.
                  3. EMA smoothing on the published centroid, with
                     blend factor scaled by the chosen blob's area
                     (high-confidence detection → trust new value;
                     small/uncertain → move slowly). This is the
                     "last_seen weighted by recency + confidence"
                     pattern, but done in the tracker so we don't
                     need to touch the 2D project's BT.
                """
                presence    = float(os.environ.get("AOT_PRESENCE_THRESH", "0.5"))
                min_blob    = int(  os.environ.get("AOT_MIN_BLOB_PX",      "200"))
                normal_blob = float(os.environ.get("AOT_NORMAL_BLOB_PX",   "5000"))
                alpha_min   = float(os.environ.get("AOT_EMA_ALPHA_MIN",    "0.2"))
                alpha_max   = float(os.environ.get("AOT_EMA_ALPHA_MAX",    "0.9"))

                prob = torch.softmax(logit, dim=1)
                fg_prob_t = prob[0, 1]                      # (H, W) on device
                binary_t  = fg_prob_t > presence

                H, W = fg_prob_t.shape
                if not binary_t.any():
                    return np.zeros((H, W), dtype=np.uint8), None

                binary_np = binary_t.cpu().numpy().astype(np.uint8)
                fg_prob_np = fg_prob_t.cpu().numpy()

                n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary_np, connectivity=8)
                if n_cc <= 1:
                    return np.zeros_like(binary_np), None

                # stats columns: LEFT TOP WIDTH HEIGHT AREA; index 0 is
                # the background. Compute per-blob bbox centre as a
                # cheap proxy for blob centroid for the proximity check.
                stats_blobs = stats[1:]
                areas = stats_blobs[:, cv2.CC_STAT_AREA]
                bcx = stats_blobs[:, cv2.CC_STAT_LEFT] + stats_blobs[:, cv2.CC_STAT_WIDTH]  / 2.0
                bcy = stats_blobs[:, cv2.CC_STAT_TOP]  + stats_blobs[:, cv2.CC_STAT_HEIGHT] / 2.0

                valid = areas >= min_blob
                if not valid.any():
                    return np.zeros_like(binary_np), None

                # Pick the blob: closest to last centroid if we have
                # one (preserves identity through dual-blob frames),
                # otherwise the largest valid blob.
                if self._last_centroid_uv is not None:
                    lcx, lcy = self._last_centroid_uv
                    dists = np.hypot(bcx - lcx, bcy - lcy)
                    dists[~valid] = np.inf
                    chosen = int(np.argmin(dists))
                else:
                    areas_v = areas.astype(np.float64)
                    areas_v[~valid] = -1.0
                    chosen = int(np.argmax(areas_v))

                chosen_label = chosen + 1   # +1 because we skipped background
                mask = (labels == chosen_label).astype(np.uint8)

                # Prob-weighted centroid within the chosen blob.
                ys, xs = np.where(mask > 0)
                w = fg_prob_np[ys, xs]
                wsum = float(w.sum())
                if wsum <= 0.0:
                    return np.zeros_like(binary_np), None
                cx_now = float((xs * w).sum() / wsum)
                cy_now = float((ys * w).sum() / wsum)

                # EMA: trust new value more when the blob is bigger.
                # area at "normal" detection distance maps to alpha ≈ 1;
                # small/sparse blobs blend in slowly so a single noisy
                # frame can't yank the published centroid across the
                # image.
                blob_area = float(areas[chosen])
                alpha = float(np.clip(
                    blob_area / max(normal_blob, 1.0), alpha_min, alpha_max))
                if self._last_centroid_uv is not None:
                    lcx, lcy = self._last_centroid_uv
                    cx_pub = alpha * cx_now + (1.0 - alpha) * lcx
                    cy_pub = alpha * cy_now + (1.0 - alpha) * lcy
                else:
                    cx_pub, cy_pub = cx_now, cy_now

                self._last_centroid_uv = (cx_pub, cy_pub)
                return mask, (cx_pub, cy_pub)

        return _AOTStreamingTracker()

    def _snapshot_latest_blocking(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], object]:
        """Block until the RGB callback has produced a frame newer than the
        previous tracker step, then return (rgb, depth, K, stamp). Returns
        (None, ..., ...) if the stop event fires or we wait > 5 s."""
        wait_start = time.time()
        last_stamp = self._latest_stamp_seen_by_worker
        while not self._stop.is_set():
            with self._snap_lock:
                rgb = self._latest_rgb
                depth = self._latest_depth
                K = self._latest_K
                stamp = self._latest_stamp
            if (rgb is not None and depth is not None and K is not None
                    and stamp is not last_stamp):
                self._latest_stamp_seen_by_worker = stamp
                return rgb, depth, K, stamp
            if time.time() - wait_start > 5.0:
                return None, None, None, None
            time.sleep(0.005)
        return None, None, None, None

    def _make_result(
        self,
        mask_np: np.ndarray,
        centroid_uv: Optional[Tuple[float, float]] = None,
    ) -> TrackResult:
        coords = np.argwhere(mask_np)
        h, w = mask_np.shape
        min_area = max(
            20, int(h * w * PERCEPTION_CFG["min_mask_area_ratio"]))
        if len(coords) < min_area:
            return TrackResult(
                mask=None, confidence=0.0,
                centroid_uv=None, is_visible=False)
        if centroid_uv is not None:
            # Prefer the streaming wrapper's smart centroid
            # (prob-weighted + EMA across frames). Falling back to the
            # geometric mean gives a noisier signal that jumps with
            # fringe pixels — see _logit_to_result in the AOT wrapper.
            u, v = centroid_uv
        else:
            v = float(coords[:, 0].mean())
            u = float(coords[:, 1].mean())
        area_ratio = len(coords) / float(h * w)
        conf = float(min(1.0, area_ratio * 50.0))
        return TrackResult(
            mask=mask_np, confidence=conf,
            centroid_uv=(u, v), is_visible=True)

    def _push_result(self, frame_idx: int, result: TrackResult) -> None:
        while True:
            try:
                self._results_q.put((frame_idx, result), block=False)
                return
            except queue.Full:
                try:
                    self._results_q.get_nowait()
                except queue.Empty:
                    pass

    # ------------------------------------------------------------------
    # Main thread — drain results, project to body frame, publish
    # ------------------------------------------------------------------
    def _publish_results(self) -> None:
        latest: Optional[Tuple[int, object]] = None
        while True:
            try:
                latest = self._results_q.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return  # nothing new — don't republish stale

        frame_idx, result = latest

        with self._side_lock:
            side = self._side_data.get(frame_idx)

        out = Detection2DArray()
        # Honest stamp = the camera frame's stamp, not now(). Lets any
        # downstream freshness gate see the real perception age (typically
        # 100–300 ms behind real time at the tracker's rate).
        if side is not None:
            rgb, depth, K, header = side
            out.header.stamp = header.stamp
        else:
            out.header.stamp = self.get_clock().now().to_msg()
        # World-frame leader position. We do the body→world lift inside this
        # node — using the bot's odom pose at the *frame's* stamp — so the
        # consumer just reads world coords. The previous body-frame pattern
        # forced every consumer to re-multiply by their *current* odom pose,
        # which (because they ran a few ms behind us) introduced 4–8° of
        # spurious rotation per detection. Frame "follower/odom" matches
        # what world_odom_publisher.py emits on /follower/odom.
        out.header.frame_id = "follower/odom"

        if side is None:
            self.pub.publish(out)
            return

        # Always publish the overlay — even when result.is_visible is False
        # we want to see "SAM2 saw nothing this frame" in the debug stream.
        self._publish_overlay(rgb, result, frame_idx)

        if not result.is_visible or result.centroid_uv is None:
            self.pub.publish(out)
            return

        # Time-aligned lidar lookup: find the scan whose stamp is closest to
        # the camera frame's stamp. Prevents bot rotation between frame
        # capture and projection from biasing the bearing→beam mapping.
        frame_ns = _stamp_to_ns(header.stamp)
        scan, scan_skew_ms = self._scan_at(frame_ns)
        self._last_scan_skew_ms = scan_skew_ms
        body_xy_frame = self._project_to_base_link(
            result.centroid_uv, result.mask, depth, K, scan)
        if body_xy_frame is None:
            self.pub.publish(out)
            return

        # body→world at the *frame's* timestamp. If no pose is buffered
        # within the skew window (typical at startup before odom warms up),
        # publish the body-frame xy with frame_id="follower/base_link" so
        # the BT falls back to its own current-pose conversion. The BT
        # dispatches on header.frame_id, so this is safe.
        pose_frame, _frame_skew = self._pose_at(frame_ns)
        det = Detection2D()
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "leader"
        hyp.hypothesis.score = float(result.confidence)
        if pose_frame is None:
            out.header.frame_id = "follower/base_link"
            hyp.pose.pose.position.x = float(body_xy_frame[0])
            hyp.pose.pose.position.y = float(body_xy_frame[1])
            self._log_world_frame(
                None, None, self._latest_pose(),
                body_xy_frame, body_xy_frame)
        else:
            xf, yf, yawf = pose_frame
            cf, sf = math.cos(yawf), math.sin(yawf)
            wx = xf + cf * body_xy_frame[0] - sf * body_xy_frame[1]
            wy = yf + sf * body_xy_frame[0] + cf * body_xy_frame[1]
            world_xy = (wx, wy)
            hyp.pose.pose.position.x = float(wx)
            hyp.pose.pose.position.y = float(wy)
            self._log_world_frame(
                world_xy, pose_frame, self._latest_pose(),
                body_xy_frame, world_xy)
        det.results.append(hyp)
        out.detections.append(det)
        self.pub.publish(out)

    def _log_world_frame(
        self,
        world_xy: Optional[Tuple[float, float]],
        pose_frame: Optional[Tuple[float, float, float]],
        pose_now: Optional[Tuple[float, float, float]],
        body_xy_frame: np.ndarray,
        body_xy_pub,
    ) -> None:
        if self._centroid_log_n % 10 != 0:
            return  # piggyback on the centroid log cadence
        if world_xy is None or pose_frame is None:
            return
        xf, yf, yawf = pose_frame
        bx_f, by_f = float(body_xy_frame[0]), float(body_xy_frame[1])
        bx_p, by_p = float(body_xy_pub[0]),   float(body_xy_pub[1])
        wx, wy = world_xy
        if pose_now is not None:
            yawn = pose_now[2]
            dyaw_deg = math.degrees(
                (yawn - yawf + math.pi) % (2 * math.pi) - math.pi)
        else:
            dyaw_deg = 0.0
        self.get_logger().info(
            f"world: leader_w=({wx:+.2f},{wy:+.2f}) "
            f"bot_w_frame=({xf:+.2f},{yf:+.2f},{math.degrees(yawf):+.0f}°) "
            f"body_xy_frame=({bx_f:+.2f},{by_f:+.2f}) "
            f"body_xy_pub=({bx_p:+.2f},{by_p:+.2f}) "
            f"Δyaw_now-frame={dyaw_deg:+.1f}°")

    # ------------------------------------------------------------------
    def _publish_overlay(self, rgb: np.ndarray, result, frame_idx: int) -> None:
        """RGB + tinted mask + centroid dot + status text → /follower/camera/aot_overlay."""
        vis = rgb.copy()
        h, w = vis.shape[:2]

        if result.mask is not None and result.is_visible:
            mask = result.mask.astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # Red tint over mask: 50% blend on the masked pixels.
            tint = vis.copy()
            tint[mask > 0] = (tint[mask > 0] * 0.5 + np.array([255, 60, 60]) * 0.5).astype(np.uint8)
            vis = tint
            if result.centroid_uv is not None:
                u_s = int(float(result.centroid_uv[0]))
                v_s = int(float(result.centroid_uv[1]))
                cv2.circle(vis, (u_s, v_s), 6, (0, 255, 0), -1)
                cv2.circle(vis, (u_s, v_s), 8, (0, 0, 0),   1)

        label = (f"frame={frame_idx} conf={result.confidence:.2f} "
                 f"{'VIS' if result.is_visible else 'LOST'}")
        cv2.putText(vis, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "follower/camera_optical_frame"
        msg.height = h
        msg.width  = w
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = vis.tobytes()
        self.overlay_pub.publish(msg)

    # ------------------------------------------------------------------
    def _project_to_base_link(
        self,
        uv: Tuple[float, float],
        mask: Optional[np.ndarray],
        depth: np.ndarray,
        K: np.ndarray,
        scan: Optional[LaserScan],
    ) -> Optional[np.ndarray]:
        """Project mask centroid to body-frame (x, y), preferring lidar range
        at the bearing of the camera ray, with depth-inside-mask as fallback.

        Why hybrid: depth at the centroid pixel is brittle — when EdgeTAM's
        mask is U-shaped or split, the centroid lands on background and the
        7×7-patch median picks up wall/floor depth (8–12 m) instead of the
        leader, projecting the leader to a phantom point 4–8 m past where
        they actually are. Lidar at the same bearing gives a single robust
        scalar; sampling depth across the *whole mask* (not at one pixel)
        also avoids that failure. Cross-check between the two catches the
        residual case where EdgeTAM has drifted onto background.
        """
        h, w = depth.shape
        u_pix, v_pix = float(uv[0]), float(uv[1])
        u_i, v_i = int(u_pix), int(v_pix)
        if not (0 <= u_i < w and 0 <= v_i < h):
            return None

        fx, _fy, cx, _cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # Body-frame bearing of the camera ray (x fwd, y left, REP-103).
        # Optical x-right maps to body y-left negated.
        bearing = math.atan2(-(u_pix - cx), fx)

        rng_lidar, lidar_idx = self._lidar_range_with_idx(scan, bearing)
        rng_depth = self._depth_range_in_mask(mask, depth, u_pix, cx, fx)

        # Always trust depth-in-mask when available. Don't cross-check
        # against lidar: lidar_leader_filter intentionally replaces beams
        # that hit the leader with infinity, so at the leader's bearing
        # rng_lidar is *expected* to be much longer than rng_depth (it
        # sees whatever's behind the leader, not the leader itself).
        # The previous "drop if rng_depth + 0.5 < rng_lidar" gate killed
        # most close-range detections — when the leader exited frame on
        # one side, the BT's leader_world stayed pinned to the last
        # un-dropped reading and the bot didn't rotate to follow.
        if rng_depth is not None:
            self._proj_both_agree += 1
            rng = rng_depth
            self._log_centroid_lidar(
                "depth", uv, bearing, lidar_idx, rng_lidar, rng_depth)
        elif rng_lidar is not None:
            self._proj_lidar_only += 1
            rng = rng_lidar
            self._log_centroid_lidar(
                "lidar-only", uv, bearing, lidar_idx, rng_lidar, None)
        else:
            self._proj_neither += 1
            return None

        # Treat range as along the camera ray (small bias from the 0.23 m
        # camera→body offset; absorbed by the 0.5 m cross-check tolerance).
        body_x = CAM_OFFSET_X_BODY + rng * math.cos(bearing)
        body_y = rng * math.sin(bearing)
        return np.array([body_x, body_y])

    @staticmethod
    def _lidar_range_with_idx(
        scan: Optional[LaserScan], bearing: float,
    ) -> Tuple[Optional[float], Optional[int]]:
        """Find the finite lidar return closest in bearing to the camera ray.
        Returns (range, idx_used) — idx_used is None when no hit found.

        When the camera sees the leader, no obstacle blocks the line of
        sight — so a lidar beam pointing at the same bearing must also
        return the leader's range. The centroid-derived idx may be a beam
        or two off because of the camera/lidar mount offset and integer
        rounding (and an empty-world scan returns inf on every beam except
        the one(s) actually hitting the leader: a 0.4 m body at 5 m
        subtends < 1 bin at 5°/bin). So we expand outward from the
        centroid idx and take the first finite hit. Capped at ±5 bins
        (±25°) so we don't accidentally bind to an unrelated obstacle on
        the other side of the bot.
        """
        if scan is None or scan.angle_increment <= 0.0:
            return None, None
        n = len(scan.ranges)
        idx0 = int(round((bearing - scan.angle_min) / scan.angle_increment))
        if not (0 <= idx0 < n):
            return None, None
        for delta in range(6):
            for sign in (0,) if delta == 0 else (-1, 1):
                i = idx0 + sign * delta
                if 0 <= i < n:
                    r = scan.ranges[i]
                    if math.isfinite(r) and scan.range_min < r < scan.range_max:
                        return float(r), i
        return None, idx0

    def _log_centroid_lidar(
        self,
        tag: str,
        uv: Tuple[float, float],
        bearing: float,
        lidar_idx: Optional[int],
        rng_lidar: Optional[float],
        rng_depth: Optional[float],
    ) -> None:
        """Periodic log: centroid → bearing → lidar bin → range, with the
        time-skew between the camera frame and the chosen scan so the
        time-alignment is verifiable."""
        self._centroid_log_n += 1
        if self._centroid_log_n % 10 != 0:
            return
        rng_l_str = f"{rng_lidar:.2f}" if rng_lidar is not None else "—"
        rng_d_str = f"{rng_depth:.2f}" if rng_depth is not None else "—"
        idx_str = str(lidar_idx) if lidar_idx is not None else "—"
        self.get_logger().info(
            f"centroid: uv=({uv[0]:.0f},{uv[1]:.0f}) "
            f"bearing={math.degrees(bearing):+.1f}° "
            f"lidar_idx={idx_str} rng_lidar={rng_l_str}m "
            f"rng_depth={rng_d_str}m frame_vs_scan_dt_ms={self._last_scan_skew_ms:+.0f} "
            f"[{tag}]")

    @staticmethod
    def _depth_range_in_mask(
        mask: Optional[np.ndarray],
        depth: np.ndarray,
        u_pix: float, cx: float, fx: float,
    ) -> Optional[float]:
        if mask is None or mask.shape != depth.shape:
            return None
        m = mask.astype(bool)
        if not m.any():
            return None
        d = depth[m]
        d = d[np.isfinite(d) & (d > 0.1) & (d < 20.0)]
        if d.size < 5:
            return None
        # Median Z over the entire mask — robust to silhouette edges and
        # background pixels that the centroid-only patch was hitting.
        z_med = float(np.median(d))
        # Convert orthogonal Z (perpendicular to image plane) to range
        # along the ray from the camera origin: rng = Z·sec(bearing).
        return z_med * math.sqrt(1.0 + ((u_pix - cx) / fx) ** 2)

    def destroy_node(self) -> None:  # type: ignore[override]
        self._stop.set()
        self._init_event.set()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = AOTTracker()
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
