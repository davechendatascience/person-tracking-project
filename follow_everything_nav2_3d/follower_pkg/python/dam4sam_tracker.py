"""DAM4SAM tracker bridge.

Online streaming wrapper around the *full* DAM4SAM tracker (with DRM —
Distractor Rejection Module — and proper temporal smoothing). The earlier
version of this file ran a bespoke per-frame `_run_single_frame_inference`
loop with no DRM; on a low-poly Gazebo actor the mask collapsed within
~30 frames and never recovered. Replaced with `DAM4SAMTracker` from
/opt/DAM4SAM, which manages its own inference state, DRM bookkeeping,
and mask propagation.

The tracker is slow (sam2.1_hiera_large ≈ 5–8 Hz on a GB10), the camera
is fast (20 Hz). Coupling them naïvely backs up frames, so:

    - Lidar + oracle keep ticking at 20 Hz, untouched.
    - The detection topic /follower/camera/detections_dam4sam ticks at
      DAM4SAM's actual rate — we publish exactly when DAM4SAM yields a
      result.
    - On every track() call we snapshot the *latest* camera RGB at that
      moment. Frames that arrived while the tracker was busy are simply
      dropped. The tracker itself sees a contiguous frame index sequence.

Bootstrap: YOLO11 finds the highest-confidence 'person' box on the first
incoming RGB frame; that box becomes the DAM4SAM init prompt. Until then
the node publishes empty Detection2DArrays.
"""
import os
# Must be set before torch is imported anywhere. Tells the CUDA caching
# allocator to use expandable segments — fragmented reserved memory is
# returned to the OS instead of held forever, which keeps VRAM bounded
# under our pruning loop below.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
# sys.path injection so the follow_everything + DAM4SAM checkouts mounted
# at /opt/* (see docker-compose.yml) are importable. /opt is the parent of
# the `follow_everything` package; /opt/DAM4SAM goes on the path directly
# because we import its `sam2` subdir as a top-level module.
for _p in ("/ws", "/opt", "/opt/DAM4SAM"):
    if _p not in sys.path and Path(_p).is_dir():
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
SAM2_CFG = {
    # DAM4SAM ships its own sam21pp_*.yaml configs alongside /opt/DAM4SAM/sam2/.
    # We use the small (S) variant — ~3× faster than large at the cost of
    # slightly weaker masks. With a 320×240 camera and a sim actor that's
    # already low-detail, the smaller model's spatial accuracy is fine and
    # the higher tracker rate (~7-8 Hz vs ~2.5 Hz) cuts the temporal lag
    # the BT sees on `last_seen`.
    "model_cfg":  "sam21pp_hiera_s.yaml",
    "checkpoint": "/opt/sam2.1_hiera_small.pt",
    "device":     "cuda",
    "image_size": 1024,
}
# Heuristic gates for our TrackResult — DAM4SAM gives us the mask, we
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
CAM_OFFSET_X_BODY = 0.23
CAM_OFFSET_Z_BODY = 0.15

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
class Dam4SamTracker(Node):
    def __init__(self) -> None:
        super().__init__("dam4sam_tracker")

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
            Detection2DArray, "/follower/camera/detections_dam4sam", 10)
        # Debug overlay — RGB frame SAM2 actually processed, with mask
        # tinted red, centroid drawn, and frame_idx in the corner. View in
        # RViz with fixed_frame=follower/camera_optical_frame.
        self.overlay_pub = self.create_publisher(
            Image, "/follower/camera/dam4sam_overlay", 10)

        # 50 Hz drain — non-blocking. Whenever SAM2 has yielded, we publish.
        self.create_timer(0.02, self._publish_results)
        self.create_timer(2.0, self._log_proj_stats)
        self._worker = threading.Thread(target=self._tracker_worker, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"DAM4SAM tracker live. "
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
        pixels."""
        with self._snap_lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            K = None if self._latest_K is None else self._latest_K.copy()
        with self._gz_lock:
            f_xyy = self._gz_follower_xyy
            l_xy  = self._gz_leader_xy
        if depth is None or K is None or f_xyy is None or l_xy is None:
            return
        bbox = self._project_leader_bbox(rgb.shape, K, f_xyy, l_xy)
        if bbox is None:
            return
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
    ) -> Optional[Tuple[float, float, float, float]]:
        """Project an axis-aligned 3D box around the leader's body to image
        pixel coords. Returns (x1, y1, x2, y2) clipped to the image, or None
        if the leader is fully outside the FOV."""
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
        # Clip to image. If the projected box is fully outside the frame,
        # SAM2 has nothing to seed from.
        x1 = max(0.0, min(w - 1.0, u_min))
        y1 = max(0.0, min(h - 1.0, v_min))
        x2 = max(0.0, min(w - 1.0, u_max))
        y2 = max(0.0, min(h - 1.0, v_max))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)

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
        pose within 200 ms or if the buffer is empty."""
        with self._odom_lock:
            if not self._odom_buf:
                return None, 0.0
            ns, x, y, yaw = min(
                self._odom_buf, key=lambda kv: abs(kv[0] - target_ns))
        skew_ns = ns - target_ns
        if abs(skew_ns) > 200_000_000:
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
    # Worker thread — full DAM4SAM tracker, one track() per ROS frame.
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
            tracker = self._build_dam4sam_tracker(log)
        except Exception as e:
            import traceback
            log.error(f"DAM4SAM build failed: {e!r}\n{traceback.format_exc()}")
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
            # Prefer the depth-filtered oracle mask when available — it's
            # already restricted to leader pixels, so SAM2 doesn't have to
            # disambiguate from a wall-contaminated bbox.
            init_mask = self._init_mask
            init_bbox = (None if init_mask is not None
                         else self._init_bbox.astype(np.float32).tolist())
            out_dict = tracker.initialize(
                init_pil, init_mask=init_mask, bbox=init_bbox)
        except Exception as e:
            import traceback
            log.error(f"DAM4SAM initialize crashed: {e!r}\n{traceback.format_exc()}")
            return
        mask0 = out_dict["pred_mask"]
        log.info(
            f"DAM4SAM init: mask shape={mask0.shape} px_on={int(mask0.sum())} "
            f"bbox={self._init_bbox.tolist()}")
        self._push_result(0, self._make_result(mask0))
        vis_n = 1 if mask0.any() else 0
        inv_n = 0 if mask0.any() else 1

        # ---- Streaming loop: one track() per fresh ROS frame --------
        LOG_EVERY = 30
        # DAM4SAMTracker stores every prepared image + per-frame predictor
        # features in inference_state[...]. With sam2.1_hiera_l at 1024x1024
        # that's ~3.5 GB / 30 frames if nothing is freed — we OOM in ~90 s.
        # Keep a rolling window of recent frames; drop the rest.
        STATE_KEEP_BEHIND = 16
        EMPTY_CACHE_EVERY = 30
        # The conditioning frame (frame 0) MUST stay — DAM4SAM's DRM looks
        # it up directly. Everything else past STATE_KEEP_BEHIND can go.
        frame_idx = 1
        try:
            while not self._stop.is_set() and frame_idx < MAX_FRAMES:
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
                self._push_result(frame_idx, self._make_result(mask))

                self._evict_dam4sam_state(
                    tracker, frame_idx, STATE_KEEP_BEHIND)

                if frame_idx % EMPTY_CACHE_EVERY == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if frame_idx % LOG_EVERY == 0:
                    alloc_gb = (
                        torch.cuda.memory_allocated() / 1e9
                        if torch.cuda.is_available() else 0.0)
                    log.info(
                        f"dam4sam f={frame_idx} "
                        f"vis={vis_n}/{vis_n+inv_n} "
                        f"last_mask_px={int(mask.sum())} "
                        f"cuda alloc={alloc_gb:.2f}GB")

                frame_idx += 1
        except Exception as e:
            import traceback
            log.error(f"tracker_worker crashed: {e!r}\n{traceback.format_exc()}")

    def _evict_dam4sam_state(
        self, tracker, frame_idx: int, keep_behind: int,
    ) -> None:
        """Drop per-frame state in DAM4SAM's inference_state + predictor
        caches for any frame older than (frame_idx - keep_behind), except
        frame 0 (the DRM-anchored conditioning frame).

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
    # DAM4SAM helpers — kept as instance methods so they're easy to test.
    # ------------------------------------------------------------------
    def _build_dam4sam_tracker(self, log):
        """Instantiate the official DAM4SAMTracker, bypassing the parent's
        `determine_tracker` lookup (which assumes the checkpoint lives at
        /opt/DAM4SAM/checkpoints/sam2.1_hiera_large.pt — but our copy is
        at /opt/sam2.1_hiera_large.pt and /opt/DAM4SAM is read-only)."""
        import torch
        sys.path.insert(0, "/opt/DAM4SAM")
        from dam4sam_tracker import DAM4SAMTracker
        from sam2.build_sam import build_sam2_video_predictor

        device = SAM2_CFG["device"] if torch.cuda.is_available() else "cpu"
        log.info(
            f"DAM4SAM (full tracker, with DRM) device={device}, "
            f"building predictor (slow first time)...")

        class _LocalDAM4SAMTracker(DAM4SAMTracker):
            def __init__(self):
                self.checkpoint = SAM2_CFG["checkpoint"]
                self.model_cfg = SAM2_CFG["model_cfg"]
                self.input_image_size = SAM2_CFG["image_size"]
                self.img_mean = torch.tensor(
                    [0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
                self.img_std = torch.tensor(
                    [0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
                self.predictor = build_sam2_video_predictor(
                    self.model_cfg, self.checkpoint, device=device)
                self.tracking_times = []

        return _LocalDAM4SAMTracker()

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

    def _make_result(self, mask_np: np.ndarray) -> TrackResult:
        coords = np.argwhere(mask_np)
        h, w = mask_np.shape
        min_area = max(
            20, int(h * w * PERCEPTION_CFG["min_mask_area_ratio"]))
        if len(coords) < min_area:
            return TrackResult(
                mask=None, confidence=0.0,
                centroid_uv=None, is_visible=False)
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
        out.header.frame_id = "follower/base_link"

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

        # body→world at frame time, then world→body at "now". The BT does
        # body→world with its own current pose; without this hop the bot's
        # rotation between frame capture and BT consumption shifts the
        # leader's world position by up to 25° at 1.5 rad/s × 200 ms.
        pose_frame, _frame_skew = self._pose_at(frame_ns)
        pose_now = self._latest_pose()
        body_xy_pub = body_xy_frame
        world_xy: Optional[Tuple[float, float]] = None
        if pose_frame is not None:
            xf, yf, yawf = pose_frame
            cf, sf = math.cos(yawf), math.sin(yawf)
            wx = xf + cf * body_xy_frame[0] - sf * body_xy_frame[1]
            wy = yf + sf * body_xy_frame[0] + cf * body_xy_frame[1]
            world_xy = (wx, wy)
            if pose_now is not None:
                xn, yn, yawn = pose_now
                cn, sn = math.cos(yawn), math.sin(yawn)
                dx, dy = wx - xn, wy - yn
                # Inverse rotation by yawn (R^T).
                body_xy_pub = ( cn * dx + sn * dy,
                               -sn * dx + cn * dy)

        self._log_world_frame(
            world_xy, pose_frame, pose_now, body_xy_frame, body_xy_pub)

        det = Detection2D()
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "leader"
        hyp.hypothesis.score = float(result.confidence)
        hyp.pose.pose.position.x = float(body_xy_pub[0])
        hyp.pose.pose.position.y = float(body_xy_pub[1])
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
        """RGB + tinted mask + centroid dot + status text → /follower/camera/dam4sam_overlay."""
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

        Why hybrid: depth at the centroid pixel is brittle — when DAM4SAM's
        mask is U-shaped or split, the centroid lands on background and the
        7×7-patch median picks up wall/floor depth (8–12 m) instead of the
        leader, projecting the leader to a phantom point 4–8 m past where
        they actually are. Lidar at the same bearing gives a single robust
        scalar; sampling depth across the *whole mask* (not at one pixel)
        also avoids that failure. Cross-check between the two catches the
        residual case where DAM4SAM has drifted onto background.
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

        if rng_lidar is not None and rng_depth is not None:
            if abs(rng_lidar - rng_depth) > 1.0:
                # Tolerance widened from 0.5 to 1.0 m: at typical leader
                # distances the depth-in-mask median samples the body's
                # geometric center while lidar samples whatever surface
                # faces the bot, and the camera-vs-lidar mount offset
                # (~0.23 m) adds further normal variance. 1.0 m still
                # catches the failure mode (mask drift onto a wall reads
                # 7 m+, lidar at the same bearing reads 4 m).
                self._proj_disagree += 1
                self._log_centroid_lidar(
                    "DROP-disagree", uv, bearing, lidar_idx,
                    rng_lidar, rng_depth)
                return None
            self._proj_both_agree += 1
            rng = rng_lidar
            self._log_centroid_lidar(
                "agree", uv, bearing, lidar_idx, rng_lidar, rng_depth)
        elif rng_lidar is not None:
            self._proj_lidar_only += 1
            rng = rng_lidar
            self._log_centroid_lidar(
                "lidar-only", uv, bearing, lidar_idx, rng_lidar, None)
        elif rng_depth is not None:
            self._proj_depth_only += 1
            rng = rng_depth
            self._log_centroid_lidar(
                "depth-only", uv, bearing, lidar_idx, None, rng_depth)
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
    node = Dam4SamTracker()
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
