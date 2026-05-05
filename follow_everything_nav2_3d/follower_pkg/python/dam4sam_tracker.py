"""DAM4SAM tracker bridge — Phase 4b-ii.

Online streaming SAM2/DAM4SAM tracker.

The tracker is **slow** (sam2.1_hiera_large is ~5–8 Hz on a GB10), the camera
is **fast** (20 Hz). Coupling them naïvely backs up frames. So:

    - Lidar + oracle keep ticking at 20 Hz, untouched.
    - The DAM4SAM topic /follower/camera/detections_dam4sam ticks at whatever
      SAM2 actually sustains — we publish exactly when SAM2 yields a result.
    - SAM2 still sees a *contiguous* index sequence (0, 1, 2, …) because
      `propagate_in_video` requires temporal continuity, but each slot is
      filled with whatever the latest camera frame is at the moment SAM2
      asks for it. Frames that arrive while SAM2 is busy on the previous
      one are dropped.

Implementation: a `LiveLoader` subclass of follow_everything's
`StreamingFrameLoader` that, on `__getitem__(idx)`, *first* asks the node to
write the current latest RGB to that path, *then* defers to the parent
loader's normal load.

Frame 0 is special: we use the snapshot of (rgb, depth, K) captured at the
moment YOLO produced the init bbox, so the bbox aligns with what SAM2 sees.

Bootstrap: YOLO11 finds the highest-confidence "person" box on the first
incoming RGB frame; that box becomes the SAM2 init prompt. Until then the
node publishes empty Detection2DArrays.
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
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import rclpy
from rclpy.node import Node

import tf2_ros
from geometry_msgs.msg import TransformStamped
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
    "model_cfg":  "configs/sam2.1/sam2.1_hiera_l.yaml",  # aliased to sam21pp
    "checkpoint": "/opt/sam2.1_hiera_large.pt",
    "device":     "cuda",
    # Pinned at 1024 — the hiera_l checkpoint was trained at this resolution
    # and the predictor asserts on shape. Memory pressure is mitigated below
    # by offload=True on the loader (image cache lives on CPU, not GPU).
    "image_size": 1024,
}
# Mirrors configs/follow_everything.yaml::perception so SAM2Tracker's DRM
# bookkeeping has every key it reads.
PERCEPTION_CFG = {
    "temporal_buffer_size": 10,
    "num_distance_bins":    5,
    "distance_bin_width":   1.0,
    "min_mask_confidence":  0.40,
    "min_mask_area_ratio":  0.0005,
    "fov_half_angle_deg":   90.0,
}

YOLO_WEIGHTS = "/opt/yolo11m.pt"
YOLO_CONF    = 0.5
PERSON_CLS   = 0  # COCO

# Pre-allocated paths for the loader; SAM2 stops when this many frames have
# been processed. At ~5 Hz that's ~5.5 hours; plenty for a Phase 4 demo.
MAX_FRAMES = 100_000
SIDE_DATA_KEEP = 64  # only need the most recent few for projection lookup


# ---------------------------------------------------------------------------
def _msg_to_rgb(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding.lower().startswith("bgr"):
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def _msg_to_depth(msg: Image) -> np.ndarray:
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)


def _quat_to_R(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


# ---------------------------------------------------------------------------
class Dam4SamTracker(Node):
    def __init__(self) -> None:
        super().__init__("dam4sam_tracker")

        self._tmpdir = Path(tempfile.mkdtemp(prefix="dam4sam_stream_"))
        self._frame_paths = [self._tmpdir / f"{i:08d}.jpg" for i in range(MAX_FRAMES)]

        # ---- Atomic snapshot of the latest ROS frame --------------------
        self._snap_lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_K: Optional[np.ndarray] = None
        self._latest_stamp = None
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

        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)
        self._T_base_optical: Optional[np.ndarray] = None

        # 50 Hz drain — non-blocking. Whenever SAM2 has yielded, we publish.
        self.create_timer(0.02, self._publish_results)
        self._worker = threading.Thread(target=self._tracker_worker, daemon=True)
        self._worker.start()

        self.get_logger().info(
            f"DAM4SAM tracker live. stream dir: {self._tmpdir}. "
            f"awaiting first 'person' YOLO detection (conf ≥ {YOLO_CONF}); "
            f"detection topic ticks at SAM2's own rate (slower than the camera).")

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
    # Always-latest loader hook
    # ------------------------------------------------------------------
    def _request_frame(self, idx: int) -> None:
        """Called by LiveLoader.__getitem__ before it loads img_paths[idx].

        For SAM2 frame 0 we hand it the YOLO-bootstrap snapshot so the bbox
        aligns. For later frames we snapshot whatever's freshest right now —
        this is what makes the tracker drop stale frames automatically.
        """
        path = self._frame_paths[idx]
        if path.exists():
            return  # already produced (e.g. retry / re-prompt path)

        if idx == 0:
            rgb, depth, K, stamp = (
                self._init_rgb, self._init_depth, self._init_K, self._init_stamp)
        else:
            with self._snap_lock:
                rgb = None if self._latest_rgb is None else self._latest_rgb
                depth = None if self._latest_depth is None else self._latest_depth
                K = None if self._latest_K is None else self._latest_K
                stamp = self._latest_stamp
            # Block briefly until the camera produces a new frame. We expect
            # this to be near-instant in steady state since cam is 20Hz and
            # SAM2 is slower.
            wait_start = time.time()
            while rgb is None or depth is None or K is None:
                if self._stop.is_set() or time.time() - wait_start > 5.0:
                    return
                time.sleep(0.005)
                with self._snap_lock:
                    rgb = self._latest_rgb
                    depth = self._latest_depth
                    K = self._latest_K
                    stamp = self._latest_stamp

        # Encode in memory + atomic rename. We can't use cv2.imwrite directly
        # to a `.part` tmp file — cv2 selects the encoder by file extension
        # and `.part` has no registered writer.
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1])
        if not ok:
            return
        tmp = path.with_name(path.stem + ".part")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, path)

        with self._side_lock:
            self._side_data[idx] = (rgb, depth, K, stamp)
            while len(self._side_data) > SIDE_DATA_KEEP:
                self._side_data.popitem(last=False)

    # ------------------------------------------------------------------
    # Worker thread — true online: one SAM2 step per ROS frame
    # ------------------------------------------------------------------
    def _tracker_worker(self) -> None:
        while not self._stop.is_set():
            if self._init_event.wait(timeout=0.5):
                break
        if self._stop.is_set():
            return

        # No propagate_in_video — that builds a fixed processing_order =
        # range(start, num_frames) up front, which is offline-flavored.
        # We call _run_single_frame_inference per ROS frame in our own loop:
        # explicitly pull the freshest RGB, run one SAM2 step, emit result,
        # advance our frame index. State accumulates exactly the same
        # cond + non_cond entries propagate_in_video would have produced,
        # but timing is driven by ROS, not by SAM2 walking a fixed range.
        from follow_everything.perception.sam2_tracker import (
            SAM2Tracker, StreamingFrameLoader, TrackResult,
        )

        import torch
        device = SAM2_CFG["device"] if torch.cuda.is_available() else "cpu"
        self.get_logger().info(
            f"SAM2 (online per-frame, no DRM) device={device}, "
            f"building predictor (slow first time)...")

        helper = SAM2Tracker(PERCEPTION_CFG, SAM2_CFG)
        helper._ensure_predictor()
        predictor = helper._predictor

        # Seed frame 0 BEFORE building inference state — _build_inference_state
        # warms up by loading frame 0 to discover video resolution.
        self._request_frame(0)

        loader = StreamingFrameLoader(
            img_paths=[str(p) for p in self._frame_paths],
            image_size=SAM2_CFG["image_size"],
            device=device,
            offload=True,  # image cache lives on CPU, not GPU
        )
        state = helper._build_inference_state(loader)

        NON_COND_KEEP     = 16
        EMPTY_CACHE_EVERY = 30
        # How far behind the active frame we keep the loader's CPU image cache
        # and the on-disk JPEG. Anything older gets evicted / unlinked.
        FILE_KEEP_BEHIND  = NON_COND_KEEP + 8
        # Periodically tear down + rebuild the SAM2 session, reseeding from
        # the latest mask. SAM2's track_step holds tensor refs in places we
        # can't reach via dict pruning alone — only a fresh state actually
        # releases everything. 300 frames ≈ 1 minute at 5 Hz.
        SESSION_FRAMES    = 300
        log = self.get_logger()

        def _prune(state, frame_idx):
            cutoff = frame_idx - NON_COND_KEEP
            if cutoff <= 0:
                return
            ncfo = state["output_dict"]["non_cond_frame_outputs"]
            stale = [k for k in ncfo if k < cutoff]
            for k in stale:
                del ncfo[k]
            state["consolidated_frame_inds"]["non_cond_frame_outputs"].difference_update(stale)
            for obj_out in state["output_dict_per_obj"].values():
                for k in [k for k in obj_out["non_cond_frame_outputs"] if k < cutoff]:
                    del obj_out["non_cond_frame_outputs"][k]
            for tmp in state.get("temp_output_dict_per_obj", {}).values():
                for sect in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    d = tmp.get(sect, {})
                    for k in [k for k in d if k < cutoff and k != 0]:
                        del d[k]
            # cached_features is keyed by frame_idx (int), not "image" — drop
            # entries older than cutoff so the per-frame backbone outputs
            # release. (Previous version had a typo'd key that pruned nothing.)
            for k in [k for k in state["cached_features"] if k < cutoff]:
                del state["cached_features"][k]
            for k in [k for k in state.get("frames_already_tracked", {}) if k < cutoff]:
                del state["frames_already_tracked"][k]
            for ft in state.get("frames_tracked_per_obj", {}).values():
                if isinstance(ft, dict):
                    for k in [k for k in ft if k < cutoff]:
                        del ft[k]

        def _mask_to_bbox(mask_np: np.ndarray, pad: int = 4) -> Optional[np.ndarray]:
            coords = np.argwhere(mask_np)
            if len(coords) < 20:
                return None
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            h, w = mask_np.shape
            x0 = max(0, int(x0) - pad)
            y0 = max(0, int(y0) - pad)
            x1 = min(w - 1, int(x1) + pad)
            y1 = min(h - 1, int(y1) + pad)
            return np.array([x0, y0, x1 + 1, y1 + 1], dtype=np.float32)

        def _make_result(mask_np: np.ndarray) -> "TrackResult":
            coords = np.argwhere(mask_np)
            h, w = mask_np.shape
            min_area = max(20, int(h * w * PERCEPTION_CFG["min_mask_area_ratio"]))
            if len(coords) < min_area:
                return TrackResult(
                    mask=None, confidence=0.0, centroid_uv=None, is_visible=False)
            v = float(coords[:, 0].mean())
            u = float(coords[:, 1].mean())
            area_ratio = len(coords) / float(h * w)
            conf = float(min(1.0, area_ratio * 50.0))
            return TrackResult(
                mask=mask_np, confidence=conf, centroid_uv=(u, v), is_visible=True)

        def _push_result(frame_idx: int, result: "TrackResult") -> None:
            while True:
                try:
                    self._results_q.put((frame_idx, result), block=False)
                    break
                except queue.Full:
                    try:
                        self._results_q.get_nowait()
                    except queue.Empty:
                        pass

        try:
            with torch.inference_mode(), \
                 torch.autocast(device, dtype=torch.bfloat16):

                # Frame 0: prompt with the YOLO bbox.
                _, _, init_masks = predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    box=self._init_bbox.astype(np.float32),
                )
                mask0 = (init_masks[0, 0] > 0.0).cpu().numpy()
                log.info(
                    f"sam2 frame 0 init mask: shape={mask0.shape} "
                    f"px_on={int(mask0.sum())} bbox={self._init_bbox.tolist()}")
                _push_result(0, _make_result(mask0))
                vis_n = 1 if mask0.any() else 0
                inv_n = 0 if mask0.any() else 1

                # Mandatory before any _run_single_frame_inference call —
                # consolidates conditioning frames into the right buffers.
                predictor.propagate_in_video_preflight(state)
                batch_size = predictor._get_obj_num(state)

                # True online loop: one ROS frame in, one SAM2 step, repeat.
                frame_idx = 1
                session_start = 0
                last_good_mask: Optional[np.ndarray] = mask0 if mask0.any() else None
                import gc
                while not self._stop.is_set() and frame_idx < MAX_FRAMES:
                    # 1. Pull the freshest RGB and write it at this index.
                    #    _request_frame blocks briefly until ROS produces
                    #    a new frame after the previous SAM2 step.
                    self._request_frame(frame_idx)

                    # 2. Run SAM2 on it.
                    current_out, pred_masks = predictor._run_single_frame_inference(
                        inference_state=state,
                        output_dict=state["output_dict"],
                        frame_idx=frame_idx,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=False,
                        run_mem_encoder=True,
                    )
                    # 3. Update state the way propagate_in_video would have.
                    state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
                    state["curr_out"] = current_out
                    predictor._add_output_per_object(
                        state, frame_idx, current_out, "non_cond_frame_outputs")
                    state["frames_already_tracked"][frame_idx] = {"reverse": False}

                    # 4. Mask in the original video resolution.
                    _, video_res_masks = predictor._get_orig_video_res_output(
                        state, pred_masks)
                    mask = (video_res_masks[0, 0] > 0.0).cpu().numpy()
                    if mask.any():
                        last_good_mask = mask
                        vis_n += 1
                    else:
                        inv_n += 1
                    _push_result(frame_idx, _make_result(mask))

                    # 5. Memory hygiene.
                    _prune(state, frame_idx)
                    # 5a. Evict StreamingFrameLoader's CPU image cache (it
                    #     keeps every loaded frame forever otherwise).
                    evict_idx = frame_idx - FILE_KEEP_BEHIND
                    if 0 < evict_idx < len(loader.images):
                        loader.images[evict_idx] = None
                    # 5b. Unlink the JPEG. Otherwise /tmp (tmpfs on most
                    #     Linux distros) accumulates every processed frame.
                    if 0 < evict_idx:
                        old_path = self._frame_paths[evict_idx]
                        if old_path.exists():
                            try:
                                old_path.unlink()
                            except OSError:
                                pass
                    if frame_idx % EMPTY_CACHE_EVERY == 0:
                        torch.cuda.empty_cache()
                        ncfo = state["output_dict"]["non_cond_frame_outputs"]
                        cfo  = state["output_dict"]["cond_frame_outputs"]
                        log.info(
                            f"sam2 online f={frame_idx} "
                            f"non_cond={len(ncfo)} cond={len(cfo)} "
                            f"cuda alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                            f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                            f"vis={vis_n}/{vis_n+inv_n} "
                            f"last_mask_px={int(mask.sum())}")

                    # 6. Periodic session restart — the only reliable way to
                    #    keep VRAM bounded across long runs. Note: we MUST
                    #    re-prompt at frame 0 (not the current frame_idx),
                    #    because DAM4SAM's select_closest_cond_frames has a
                    #    hardcoded `cond_frame_outputs[0]` lookup. So we
                    #    reset SAM2's frame counter to 0 on each restart.
                    if frame_idx - session_start >= SESSION_FRAMES:
                        new_bbox = (_mask_to_bbox(last_good_mask)
                                    if last_good_mask is not None else None)
                        if new_bbox is None:
                            log.warn(
                                f"sam2 restart skipped at f={frame_idx}: "
                                "no recent mask to derive bbox from")
                            session_start = frame_idx
                        else:
                            # Snapshot the RGB SAM2 just processed so we can
                            # reseed the new session at frame 0 with it.
                            with self._side_lock:
                                cur = self._side_data.get(frame_idx)
                            if cur is None:
                                log.warn(
                                    f"sam2 restart skipped at f={frame_idx}: "
                                    "missing side_data for current frame")
                                session_start = frame_idx
                            else:
                                cur_rgb, cur_depth, cur_K, cur_stamp = cur
                                log.info(
                                    f"sam2 session restart f={frame_idx}->0 "
                                    f"new_bbox={new_bbox.tolist()}")

                                # 1) drop everything holding tensors
                                del state, current_out, pred_masks, video_res_masks
                                gc.collect()
                                torch.cuda.empty_cache()

                                # 2) overwrite frame_paths[0] with current RGB
                                ok, buf = cv2.imencode(".jpg", cur_rgb[:, :, ::-1])
                                if ok:
                                    p0 = self._frame_paths[0]
                                    tmp = p0.with_name(p0.stem + ".part")
                                    tmp.write_bytes(buf.tobytes())
                                    os.replace(tmp, p0)

                                # 3) invalidate loader cache so it re-reads
                                for i in range(len(loader.images)):
                                    loader.images[i] = None
                                loader.video_height = None
                                loader.video_width  = None

                                # 4) reset side_data; new SAM2 frame 0 = current
                                with self._side_lock:
                                    self._side_data.clear()
                                    self._side_data[0] = (
                                        cur_rgb, cur_depth, cur_K, cur_stamp)

                                # 5) fresh state + prompt at frame 0
                                state = helper._build_inference_state(loader)
                                _, _, _ = predictor.add_new_points_or_box(
                                    state, frame_idx=0, obj_id=1, box=new_bbox)
                                predictor.propagate_in_video_preflight(state)

                                # 6) reset our counter — next iteration starts
                                #    fresh at SAM2 frame 1.
                                frame_idx = 0
                                session_start = 0
                                vis_n = 0
                                inv_n = 0

                    frame_idx += 1
        except Exception as e:
            import traceback
            self.get_logger().error(
                f"tracker_worker crashed: {e!r}\n{traceback.format_exc()}")

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
                u_s = int(float(result.centroid_uv[0]) * w / SAM2_CFG["image_size"])
                v_s = int(float(result.centroid_uv[1]) * h / SAM2_CFG["image_size"])
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
        # SAM2 internally resizes to image_size×image_size before producing
        # masks, so centroid_uv lives in that resized space.
        u_scaled = float(uv[0]) * w / SAM2_CFG["image_size"]
        v_scaled = float(uv[1]) * h / SAM2_CFG["image_size"]
        u_i, v_i = int(u_scaled), int(v_scaled)
        if not (0 <= u_i < w and 0 <= v_i < h):
            return None

        r = 3
        patch = depth[max(0, v_i-r):v_i+r+1, max(0, u_i-r):u_i+r+1]
        finite = patch[np.isfinite(patch) & (patch > 0.1)]
        if finite.size == 0:
            return None
        d = float(np.median(finite))

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p_opt = np.array([
            (u_scaled - cx) * d / fx,
            (v_scaled - cy) * d / fy,
            d,
            1.0,
        ])

        T = self._lookup_T_base_optical()
        if T is None:
            return None
        p_base = T @ p_opt
        return p_base[:2]

    def _lookup_T_base_optical(self) -> Optional[np.ndarray]:
        if self._T_base_optical is not None:
            return self._T_base_optical
        try:
            tfs: TransformStamped = self._tf_buf.lookup_transform(
                "follower/base_link", "follower/camera_optical_frame",
                rclpy.time.Time())
        except Exception:
            return None
        t = tfs.transform.translation
        R = _quat_to_R(tfs.transform.rotation)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [t.x, t.y, t.z]
        self._T_base_optical = T
        return T

    def destroy_node(self) -> None:  # type: ignore[override]
        self._stop.set()
        self._init_event.set()
        return super().destroy_node()


# Imported here to keep the top of the file readable.
import time  # noqa: E402


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
