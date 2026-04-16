"""SAM2-based leader tracker with Distance Frame Buffer.

Paper reference: "Follow Everything" §III-A

Two memory augmentations on top of SAM2VideoPredictor:
  1. Temporal buffer  B^T — top-N frames with highest mask confidence.
  2. Distance buffer  B^D — one best frame per distance bin.

When the leader leaves the FOV (confidence drops), we re-prompt SAM2
using the frame stored in the distance bin closest to the predicted
re-acquisition distance.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Iterator, Union
from pathlib import Path
import cv2
import numpy as np
import time

class StreamingFrameLoader:
    """A lazy-loading frame source for SAM2 that blocks until frames appear."""
    def __init__(self, img_paths, image_size, device, offload=False):
        import torch
        from PIL import Image
        self.img_paths = [Path(p) for p in img_paths]
        self.image_size = image_size
        self.device = device
        self.offload = offload
        self.images = [None] * len(img_paths)
        self.img_mean = torch.tensor((0.485, 0.456, 0.406), device=device)[:, None, None]
        self.img_std = torch.tensor((0.229, 0.224, 0.225), device=device)[:, None, None]
        self.video_height = None
        self.video_width = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        import time
        import torch
        from PIL import Image
        import numpy as np
        
        if self.images[index] is not None:
            return self.images[index]

        path = self.img_paths[index]
        wait_start = time.time()
        while not path.exists():
            if time.time() - wait_start > 60.0:
                raise TimeoutError(f"Missing frame at {path} after 60s")
            time.sleep(0.01)

        img_pil = Image.open(path)
        if self.video_height is None:
            self.video_width, self.video_height = img_pil.size
            
        img_np = np.array(img_pil.convert("RGB").resize((self.image_size, self.image_size)))
        img = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).float()
        img = (img.to(self.device) - self.img_mean) / self.img_std
        
        if self.offload:
            img = img.cpu()
        
        self.images[index] = img
        return img


@dataclass
class TrackResult:
    mask: Optional[np.ndarray]   # (H, W) bool, or None if leader not visible
    confidence: float            # 0–1 IoU-like score
    centroid_uv: Optional[Tuple[float, float]]  # pixel (u, v)
    is_visible: bool


@dataclass
class _BufferEntry:
    frame_idx: int
    bbox: np.ndarray          # [x1, y1, x2, y2] float32
    confidence: float
    distance_m: float


class SAM2Tracker:
    """Wraps SAM2VideoPredictor for offline sequence tracking with re-prompting.

    Usage::

        tracker = SAM2Tracker(cfg.perception, cfg.sam2)
        results = tracker.track_sequence(
            frames_dir   = "/tmp/crowdbot_frames/seq0",
            initial_bbox = np.array([120, 80, 200, 320]),
            depth_seq    = list_of_depth_arrays,   # or None
        )
        for frame_idx, result in results.items():
            if result.is_visible: ...
    """

    def __init__(self, perception_cfg, sam2_cfg):
        self._pcfg  = perception_cfg
        self._scfg  = sam2_cfg
        self._predictor = None

        # Distance frame buffer: {bin_idx: _BufferEntry}
        self._dist_buf:   Dict[int, _BufferEntry] = {}
        # Temporal buffer: sorted list of _BufferEntry, max size N
        self._temp_buf:   List[_BufferEntry] = []

    # ------------------------------------------------------------------
    def _ensure_predictor(self) -> None:
        """Build the SAM2 video predictor if not already initialised."""
        if self._predictor is not None:
            return
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        device = self._scfg["device"] if torch.cuda.is_available() else "cpu"
        vos_optimized = self._scfg.get("vos_optimized", False)
        if vos_optimized:
            print("[INFO] Building VOS-optimized predictor (torch.compile warm-up on first run ~1-3 min)...")
        self._predictor = build_sam2_video_predictor(
            config_file=self._scfg["model_cfg"],
            ckpt_path=self._scfg["checkpoint"],
            device=device,
            vos_optimized=vos_optimized,
        )

    def _build_inference_state(
        self, frames_dir: Union[str, Path, "StreamingFrameLoader"]
    ) -> dict:
        """Construct an SAM2 inference state for a directory or StreamingFrameLoader.

        Must be called inside a ``torch.inference_mode()`` context.
        """
        if not isinstance(frames_dir, StreamingFrameLoader):
            return self._predictor.init_state(video_path=str(frames_dir))

        state: dict = {}
        state["images"]               = frames_dir
        state["num_frames"]           = len(frames_dir)
        state["offload_video_to_cpu"] = frames_dir.offload
        state["offload_state_to_cpu"] = False
        state["video_height"]         = frames_dir.video_height or 0
        state["video_width"]          = frames_dir.video_width  or 0
        state["device"]               = self._predictor.device
        state["storage_device"]       = self._predictor.device
        state["point_inputs_per_obj"]       = {}
        state["mask_inputs_per_obj"]        = {}
        state["cached_features"]            = {}
        state["constants"]                  = {}
        state["obj_id_to_idx"]              = OrderedDict()
        state["obj_idx_to_id"]              = OrderedDict()
        state["obj_ids"]                    = []
        state["output_dict_per_obj"]        = {}
        state["temp_output_dict_per_obj"]   = {}
        state["frames_tracked_per_obj"]     = {}

        # Warm up backbone and resolve actual frame dimensions.
        self._predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        state["video_height"] = frames_dir.video_height
        state["video_width"]  = frames_dir.video_width
        return state

    # ------------------------------------------------------------------
    def track_sequence(
        self,
        frames_dir:   Union[str, Path, StreamingFrameLoader],
        initial_bbox: np.ndarray,
        depth_seq:    Optional[List[Optional[np.ndarray]]] = None,
        allow_reprompt: bool = True,
        max_reprompts:  int = 2,
    ) -> Iterator[Tuple[int, TrackResult]]:
        """Process an entire frame sequence with synchronous streaming.

        Yields:
            Tuple of (frame_idx, TrackResult).
        """
        import torch

        self._ensure_predictor()
        self._dist_buf.clear()
        self._temp_buf.clear()

        with torch.inference_mode(), \
             torch.autocast(self._scfg["device"], dtype=torch.bfloat16):

            state = self._build_inference_state(frames_dir)

            self._predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1,
                box=initial_bbox.astype(np.float32),
            )

            # Tracking passes
            reprompt_from = 0
            reprompt_count = 0
            
            while True:
                lost_at = None
                # Generator for propagation
                track_gen = self._predictor.propagate_in_video(state, start_frame_idx=reprompt_from)
                
                try:
                    for frame_idx, obj_ids, masks in track_gen:
                        mask_np = (masks[0, 0] > 0.0).cpu().numpy()
                        conf = self._compute_confidence(mask_np)
                        res = self._make_result(mask_np, conf)
                        
                        yield frame_idx, res

                        # Update buffers while tracking confidently
                        if conf >= self._pcfg["min_mask_confidence"]:
                            dist = 2.0
                            if depth_seq and frame_idx < len(depth_seq) \
                                    and depth_seq[frame_idx] is not None:
                                dist = self._depth_at_mask(mask_np, depth_seq[frame_idx])
                            bbox = self._mask_to_bbox(mask_np)
                            if bbox is not None:
                                self._update_buffer(frame_idx, bbox, dist, conf)
                        
                        # Detect loss
                        if conf < self._pcfg["min_mask_confidence"]:
                            if lost_at is None:
                                lost_at = frame_idx
                                if not allow_reprompt: break
                        else:
                            # If we regain confidence, reset loss marker (for local jitters)
                            lost_at = None

                except StopIteration:
                    pass

                # Try one re-prompt cycle if we detected loss and it's enabled
                if lost_at is not None and allow_reprompt and reprompt_count < max_reprompts:
                    entry = self._best_reprompt_entry(
                        expected_dist=self._predict_reprompt_distance()
                    )
                    if entry is not None and entry.frame_idx < lost_at:
                        reprompt_from = entry.frame_idx
                        reprompt_count += 1
                        # Restart generator from the re-prompt point
                        track_gen = self._predictor.propagate_in_video(state, start_frame_idx=reprompt_from)
                        continue
                break  # done

    # ------------------------------------------------------------------
    def track_sequence_multi(
        self,
        frames_dir:   Union[str, Path, "StreamingFrameLoader"],
        yolo_model_path: str = "yolo11m.pt",
        disappearance_timeout_frames: int = 300,
        iou_match_threshold: float = 0.3,
        yolo_conf_threshold: float = 0.45,
        confirm_frames: int = 2,
    ) -> Iterator[Tuple[int, Dict[int, "TrackResult"]]]:
        """Multi-person tracking: YOLO scans every frame for new persons,
        SAM2 tracks all active persons simultaneously.

        Two-stage registration: a new YOLO detection must appear in
        ``confirm_frames`` consecutive frames before SAM2 starts tracking it,
        which suppresses one-frame YOLO false positives.

        Persons absent for ``disappearance_timeout_frames`` consecutive frames
        are dropped.

        Yields:
            Tuple of (frame_idx, {obj_id: TrackResult}).
        """
        import torch
        from ultralytics import YOLO

        self._ensure_predictor()
        device = self._scfg["device"]
        yolo_model = YOLO(yolo_model_path)

        with torch.inference_mode(), \
             torch.autocast(device, dtype=torch.bfloat16):

            state = self._build_inference_state(frames_dir)
            num_frames = state["num_frames"]

            next_obj_id  = 1
            next_pend_id = 0
            # active_objs: obj_id → {last_seen, last_bbox (SAM2 mask), last_yolo_bbox}
            active_objs: Dict[int, dict] = {}
            # pending: cand_id → {bbox, first_frame}  — awaiting confirmation
            pending: Dict[int, dict] = {}

            for frame_idx in range(num_frames):
                # ----------------------------------------------------------
                # 1. Obtain the raw BGR frame for YOLO.
                # ----------------------------------------------------------
                if isinstance(frames_dir, StreamingFrameLoader):
                    frame_bgr = self._wait_and_read_bgr(frames_dir.img_paths[frame_idx])
                else:
                    frame_bgr = cv2.imread(
                        str(Path(frames_dir) / f"{frame_idx:06d}.jpg")
                    )

                if frame_bgr is None:
                    yield frame_idx, {}
                    continue

                # ----------------------------------------------------------
                # 2. YOLO detection.
                #    • conf threshold filters low-confidence detections
                #    • iou=0.45 makes YOLO's NMS more aggressive so a single
                #      person doesn't produce two overlapping boxes
                # ----------------------------------------------------------
                with torch.autocast(device, enabled=False):
                    yolo_bboxes: List[np.ndarray] = []
                    for r in yolo_model(
                        frame_bgr, verbose=False,
                        conf=yolo_conf_threshold, iou=0.45
                    ):
                        for box in r.boxes:
                            if int(box.cls) == 0:  # person
                                yolo_bboxes.append(
                                    box.xyxy[0].cpu().numpy().astype(np.float32)
                                )

                # ----------------------------------------------------------
                # 3. Match YOLO detections → active SAM2 objects.
                #    Reference: last_yolo_bbox (same detector scale) with
                #    last_bbox (SAM2 mask) as fallback when yolo ref is gone.
                # ----------------------------------------------------------
                active_ref: Dict[int, np.ndarray] = {
                    oid: (info.get("last_yolo_bbox") or info["last_bbox"])
                    for oid, info in active_objs.items()
                    if (info.get("last_yolo_bbox") or info.get("last_bbox")) is not None
                }
                matched_active, unmatched_indices = self._match_detections(
                    yolo_bboxes, active_ref, iou_match_threshold
                )
                # Keep last_yolo_bbox fresh for matched objects
                for oid, yi in matched_active.items():
                    active_objs[oid]["last_yolo_bbox"] = yolo_bboxes[yi]

                # ----------------------------------------------------------
                # 4. Two-frame confirmation window.
                #    Unmatched YOLO detections are checked against pending
                #    candidates. A second hit on the same candidate promotes
                #    it to an active SAM2 object; first-time detections just
                #    enter pending and wait.
                # ----------------------------------------------------------
                unmatched_bboxes = [yolo_bboxes[yi] for yi in unmatched_indices]
                pending_ref: Dict[int, np.ndarray] = {
                    cid: info["bbox"] for cid, info in pending.items()
                }
                matched_pending, truly_new_local = self._match_detections(
                    unmatched_bboxes, pending_ref, iou_match_threshold
                )

                # Confirmed candidates → register with SAM2
                for cid, local_yi in matched_pending.items():
                    orig_yi = unmatched_indices[local_yi]
                    self._predictor.add_new_points_or_box(
                        state,
                        frame_idx=frame_idx,
                        obj_id=next_obj_id,
                        box=yolo_bboxes[orig_yi],
                    )
                    active_objs[next_obj_id] = {
                        "last_seen":      frame_idx,
                        "last_bbox":      yolo_bboxes[orig_yi],
                        "last_yolo_bbox": yolo_bboxes[orig_yi],
                    }
                    next_obj_id += 1
                    del pending[cid]

                # Truly new detections → enter pending
                for local_yi in truly_new_local:
                    orig_yi = unmatched_indices[local_yi]
                    pending[next_pend_id] = {
                        "bbox":        yolo_bboxes[orig_yi],
                        "first_frame": frame_idx,
                    }
                    next_pend_id += 1

                # Drop stale pending entries that were never re-detected
                stale_age = confirm_frames - 1
                for cid in [
                    cid for cid, info in pending.items()
                    if frame_idx - info["first_frame"] >= stale_age
                ]:
                    del pending[cid]

                # ----------------------------------------------------------
                # 5. Run SAM2 for this single frame across all active objects.
                # ----------------------------------------------------------
                results: Dict[int, TrackResult] = {}
                if active_objs:
                    for _, obj_ids_out, masks_out in self._predictor.propagate_in_video(
                        state,
                        start_frame_idx=frame_idx,
                        max_frame_num_to_track=1,
                    ):
                        for i, oid in enumerate(obj_ids_out):
                            if oid not in active_objs:
                                continue  # already expired
                            mask_np = (masks_out[i, 0] > 0.0).cpu().numpy()
                            conf    = self._compute_confidence(mask_np)
                            res     = self._make_result(mask_np, conf)
                            results[oid] = res
                            if res.is_visible:
                                active_objs[oid]["last_seen"] = frame_idx
                                bbox = self._mask_to_bbox(mask_np)
                                if bbox is not None:
                                    active_objs[oid]["last_bbox"] = bbox

                # ----------------------------------------------------------
                # 6. Expire objects gone longer than the timeout.
                # ----------------------------------------------------------
                expired = [
                    oid for oid, info in active_objs.items()
                    if frame_idx - info["last_seen"] > disappearance_timeout_frames
                ]
                for oid in expired:
                    del active_objs[oid]

                yield frame_idx, results

    # ------------------------------------------------------------------
    @staticmethod
    def _wait_and_read_bgr(path: Path, timeout: float = 60.0) -> Optional[np.ndarray]:
        """Block until ``path`` exists (written by the extraction thread), then read it."""
        wait_start = time.time()
        while not path.exists():
            if time.time() - wait_start > timeout:
                return None
            time.sleep(0.01)
        return cv2.imread(str(path))

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        """IoU between two [x1, y1, x2, y2] boxes."""
        ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return float(inter / (area_a + area_b - inter))

    @staticmethod
    def _match_detections(
        query_bboxes: List[np.ndarray],
        ref_bboxes: Dict[int, np.ndarray],
        iou_threshold: float,
    ) -> Tuple[Dict[int, int], List[int]]:
        """Greedy IoU matching: query boxes vs. a reference dict of boxes.

        Args:
            query_bboxes:  list of [x1,y1,x2,y2] boxes to assign
            ref_bboxes:    {id → [x1,y1,x2,y2]} existing tracked positions
            iou_threshold: minimum IoU to accept a match

        Returns:
            matched:   {ref_id → query_index} for accepted pairs
            unmatched: query indices that had no match
        """
        if not query_bboxes or not ref_bboxes:
            return {}, list(range(len(query_bboxes)))

        ref_ids = list(ref_bboxes.keys())
        iou_mat = np.zeros((len(query_bboxes), len(ref_ids)), dtype=np.float32)
        for i, qb in enumerate(query_bboxes):
            for j, rid in enumerate(ref_ids):
                iou_mat[i, j] = SAM2Tracker._box_iou(qb, ref_bboxes[rid])

        matched: Dict[int, int] = {}
        used_q: set = set()
        used_r: set = set()

        while iou_mat.max() >= iou_threshold:
            qi, ri = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            matched[ref_ids[ri]] = int(qi)
            used_q.add(int(qi));  used_r.add(int(ri))
            iou_mat[qi, :] = -1.0;  iou_mat[:, ri] = -1.0

        unmatched = [i for i in range(len(query_bboxes)) if i not in used_q]
        return matched, unmatched

    # ------------------------------------------------------------------
    @staticmethod
    def write_frames(
        images: List[np.ndarray],
        out_dir: str | Path,
    ) -> Path:
        """Save a list of RGB uint8 arrays as zero-padded JPEGs for SAM2.

        Returns the directory path.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), bgr)
        return out_dir

    # ------------------------------------------------------------------
    def _compute_confidence(self, mask: np.ndarray) -> float:
        """Confidence proxy: penalise very small or very large masks.
        
        Smoothing logic:
        - 0.0 if ratio < 0.5 * min_ratio
        - Linear ramp from 0.0 to 1.0 between [0.5 * min_ratio, 2.0 * min_ratio]
        - Max 1.0 between [2.0 * min_ratio, 0.25]
        - Decay to 0.3 if ratio > 0.30
        """
        ratio = mask.sum() / max(mask.size, 1)
        min_r = self._pcfg["min_mask_area_ratio"]
        
        if ratio < 0.5 * min_r:
            return 0.0
        
        if ratio > 0.30:
            return 0.3
            
        # Smooth ramp logic
        if ratio < 2.0 * min_r:
            return (ratio - 0.5 * min_r) / (1.5 * min_r)
            
        return 1.0

    def _update_buffer(
        self, frame_idx: int, bbox: np.ndarray, dist: float, conf: float
    ) -> None:
        bin_idx = int(dist / self._pcfg["distance_bin_width"])
        entry = _BufferEntry(frame_idx, bbox.copy(), conf, dist)

        # Distance buffer — keep best per bin
        existing = self._dist_buf.get(bin_idx)
        if existing is None or conf > existing.confidence:
            self._dist_buf[bin_idx] = entry

        # Temporal buffer — keep top-N by confidence
        self._temp_buf.append(entry)
        self._temp_buf.sort(key=lambda e: e.confidence, reverse=True)
        if len(self._temp_buf) > self._pcfg["temporal_buffer_size"]:
            self._temp_buf.pop()

    def _best_reprompt_entry(
        self, expected_dist: float
    ) -> Optional[_BufferEntry]:
        """Return the most suitable buffer entry for re-prompting."""
        if not self._dist_buf:
            return self._temp_buf[0] if self._temp_buf else None
        target_bin = int(expected_dist / self._pcfg["distance_bin_width"])
        best_bin = min(self._dist_buf.keys(), key=lambda b: abs(b - target_bin))
        return self._dist_buf[best_bin]

    def _predict_reprompt_distance(self) -> float:
        """Estimate the distance at which we expect to re-acquire the leader."""
        if not self._temp_buf:
            return 2.0
        return float(np.mean([e.distance_m for e in self._temp_buf[:3]]))

    @staticmethod
    def _depth_at_bbox(bbox: np.ndarray, depth: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = depth.shape
        patch = depth[
            max(0, y1): min(h, y2),
            max(0, x1): min(w, x2),
        ]
        valid = patch[patch > 0.1]
        return float(np.median(valid)) if len(valid) else 2.0

    @staticmethod
    def _depth_at_mask(mask: np.ndarray, depth: np.ndarray) -> float:
        valid = depth[mask & (depth > 0.1)]
        return float(np.median(valid)) if len(valid) else 2.0

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

    @staticmethod
    def _make_result(mask: np.ndarray, conf: float) -> TrackResult:
        if conf <= 0.0 or not mask.any():
            return TrackResult(mask=None, confidence=0.0,
                               centroid_uv=None, is_visible=False)
        coords = np.argwhere(mask)
        cy, cx = coords.mean(axis=0)
        return TrackResult(
            mask=mask, confidence=conf,
            centroid_uv=(float(cx), float(cy)),
            is_visible=True,
        )
