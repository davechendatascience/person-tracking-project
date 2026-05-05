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

import sys
import threading
import queue
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2  # used in _publish_overlay
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image
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
    # The DAM4SAM repo ships its own sam21pp_*.yaml configs alongside
    # /opt/DAM4SAM/sam2/. We load the SAM2 1.0 plus-plus large variant.
    "model_cfg":  "sam21pp_hiera_l.yaml",
    "checkpoint": "/opt/sam2.1_hiera_large.pt",
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

        self.create_subscription(Image, "/follower/camera/image", self._on_rgb, 10)
        self.create_subscription(Image, "/follower/camera/depth_image", self._on_depth, 10)
        self.create_subscription(CameraInfo, "/follower/camera/camera_info",
                                 self._on_camera_info, 10)
        self.pub = self.create_publisher(
            Detection2DArray, "/follower/camera/detections_dam4sam", 10)
        # Debug overlay — RGB frame SAM2 actually processed, with mask
        # tinted red, centroid drawn, and frame_idx in the corner. View in
        # RViz with fixed_frame=follower/camera_optical_frame.
        self.overlay_pub = self.create_publisher(
            Image, "/follower/camera/dam4sam_overlay", 10)

        # 50 Hz drain — non-blocking. Whenever SAM2 has yielded, we publish.
        self.create_timer(0.02, self._publish_results)
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

        if self._init_bbox is None:
            self._try_yolo_bootstrap(rgb, msg.header)

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
        bbox = xyxy[i].astype(np.float32)

        with self._snap_lock:
            depth = None if self._latest_depth is None else self._latest_depth.copy()
            K = None if self._latest_K is None else self._latest_K.copy()
        if depth is None or K is None:
            return  # need depth + K to do anything useful with a hit

        self._init_rgb   = rgb.copy()
        self._init_depth = depth
        self._init_K     = K
        self._init_stamp = header
        self._init_bbox  = bbox
        self._init_event.set()
        self.get_logger().info(
            f"YOLO bootstrap: init bbox {bbox.tolist()} (conf {conf[i]:.2f})")

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
            out_dict = tracker.initialize(
                init_pil, init_mask=None,
                bbox=self._init_bbox.astype(np.float32).tolist())
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

        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "follower/base_link"

        frame_idx, result = latest

        with self._side_lock:
            side = self._side_data.get(frame_idx)
        if side is None:
            self.pub.publish(out)
            return
        rgb, depth, K, _ = side

        # Always publish the overlay — even when result.is_visible is False
        # we want to see "SAM2 saw nothing this frame" in the debug stream.
        self._publish_overlay(rgb, result, frame_idx)

        if not result.is_visible or result.centroid_uv is None:
            self.pub.publish(out)
            return

        body_xy = self._project_to_base_link(result.centroid_uv, depth, K)
        if body_xy is None:
            self.pub.publish(out)
            return

        det = Detection2D()
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = "leader"
        hyp.hypothesis.score = float(result.confidence)
        hyp.pose.pose.position.x = float(body_xy[0])
        hyp.pose.pose.position.y = float(body_xy[1])
        det.results.append(hyp)
        out.detections.append(det)
        self.pub.publish(out)

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
        depth: np.ndarray,
        K: np.ndarray,
    ) -> Optional[np.ndarray]:
        h, w = depth.shape
        # `video_res_masks` is what produced this centroid; that path runs
        # `predictor._get_orig_video_res_output(...)`, which returns the
        # mask at the *original video resolution* — i.e. the same (w, h)
        # as the depth image. The centroid is already in pixel coords;
        # do NOT rescale by SAM2_CFG["image_size"]. (Earlier code did,
        # which projected u,v ≈ 50,28 instead of 160,120 for a centered
        # leader, hit the sky-pixel finite==0 fallback, and dropped
        # every detection.)
        u_pix = float(uv[0])
        v_pix = float(uv[1])
        u_i, v_i = int(u_pix), int(v_pix)
        if not (0 <= u_i < w and 0 <= v_i < h):
            return None

        r = 3
        patch = depth[max(0, v_i-r):v_i+r+1, max(0, u_i-r):u_i+r+1]
        finite = patch[np.isfinite(patch) & (patch > 0.1)]
        if finite.size == 0:
            return None
        d = float(np.median(finite))

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # Optical frame: x right, y down, z forward (REP-103 camera).
        opt_x = (u_pix - cx) * d / fx
        opt_y = (v_pix - cy) * d / fy
        opt_z = d
        # Hardcoded optical → base_link transform — mirrors oracle_camera's
        # body→optical math in reverse, since gz_bridge doesn't publish
        # follower/camera_optical_frame to /tf and tf2 lookup fails. The
        # camera is mounted (CAM_OFFSET_X_BODY, 0, CAM_OFFSET_Z_BODY)
        # forward + up of base_link, no rotation other than the standard
        # optical-to-body axis convention.
        body_x = opt_z + CAM_OFFSET_X_BODY
        body_y = -opt_x
        return np.array([body_x, body_y])

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
