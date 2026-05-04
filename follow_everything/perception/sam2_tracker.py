"""SAM2-based leader tracker with DAM4SAM Distractor-aware Replay Memory.

Paper references:
    "Follow Everything"   §III-A — distance frame buffer for re-prompting.
    "DAM4SAM"             §3    — Distractor-aware Replay Memory (DRM).

The tracker drives DAM4SAM's modified SAM2 video predictor frame-by-frame.
On each step the predictor returns the chosen mask plus its alternative-mask
hypotheses; when those hypotheses diverge from the chosen mask the frame is
pinned as a conditioning frame in the memory bank (`add_to_drm`), which
prevents the rolling FIFO memory from evicting frames where the target was
clearly distinguishable from distractors.

On top of that we keep:
  - Distance frame buffer  B^D — best frame per distance bin.
  - Temporal buffer        B^T — top-N frames by confidence.
for re-prompting when the target leaves the FOV.
"""

from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Resolve `import sam2` to DAM4SAM's modified package (which adds DRM and
# `return_all_masks` to the video predictor).  The DAM4SAM checkout sits next
# to this file under the project root.
# ---------------------------------------------------------------------------
_DAM4SAM_DIR = Path(__file__).resolve().parents[2] / "DAM4SAM"
if _DAM4SAM_DIR.is_dir() and str(_DAM4SAM_DIR) not in sys.path:
    sys.path.insert(0, str(_DAM4SAM_DIR))


# Map stock-style SAM2 config names to DAM4SAM equivalents so existing
# callers don't have to edit their YAML files. The DAM4SAM configs target
# the modified `sam2.sam2_video_predictor.SAM2VideoPredictor` class.
_DAM4SAM_CONFIG_ALIAS = {
    "configs/sam2.1/sam2.1_hiera_t.yaml":  "sam21pp_hiera_t.yaml",
    "configs/sam2.1/sam2.1_hiera_s.yaml":  "sam21pp_hiera_s.yaml",
    "configs/sam2.1/sam2.1_hiera_b+.yaml": "sam21pp_hiera_b+.yaml",
    "configs/sam2.1/sam2.1_hiera_l.yaml":  "sam21pp_hiera_l.yaml",
    "configs/sam2/sam2_hiera_t.yaml":      "sam2pp_hiera_t.yaml",
    "configs/sam2/sam2_hiera_s.yaml":      "sam2pp_hiera_s.yaml",
    "configs/sam2/sam2_hiera_b+.yaml":     "sam2pp_hiera_b+.yaml",
    "configs/sam2/sam2_hiera_l.yaml":      "sam2pp_hiera_l.yaml",
}


# DRM trigger thresholds (from DAM4SAM paper / reference implementation).
_DRM_MIN_PRED_IOU       = 0.8     # chosen-mask predicted IoU floor
_DRM_OBJ_RATIO_LO       = 0.8     # mask area must be within ±20% of recent median
_DRM_OBJ_RATIO_HI       = 1.2
_DRM_MIN_FRAME_GAP      = 5       # don't add to DRM more often than every N frames
_DRM_OBJ_HISTORY_WINDOW = 300     # frames to consider for the size-ratio median
_DRM_OBJ_HISTORY_RECENT = 10      # most-recent N nonzero sizes used in the median
_DRM_DISTRACTOR_IOU     = 0.7     # alt-vs-chosen bbox IoU below this → distractor


class StreamingFrameLoader:
    """A lazy-loading frame source for SAM2 that blocks until frames appear."""
    def __init__(self, img_paths, image_size, device, offload=False):
        import torch
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
        import torch
        from PIL import Image

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


@dataclass
class _DRMState:
    """Per-object state needed to decide whether to pin a frame into DRM."""
    object_sizes: List[int]
    last_added: int


class SAM2Tracker:
    """Wraps DAM4SAM's SAM2 video predictor for offline sequence tracking.

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
        """Build the DAM4SAM video predictor if not already initialised."""
        if self._predictor is not None:
            return
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        device = self._scfg["device"] if torch.cuda.is_available() else "cpu"
        cfg_name = self._scfg["model_cfg"]
        cfg_name = _DAM4SAM_CONFIG_ALIAS.get(cfg_name, cfg_name)
        # DAM4SAM's predictor expects no `vos_optimized` hook; ignore the flag.
        self._predictor = build_sam2_video_predictor(
            config_file=cfg_name,
            ckpt_path=self._scfg["checkpoint"],
            device=device,
        )

    def _build_inference_state(
        self, frames_dir: Union[str, Path, "StreamingFrameLoader"]
    ) -> dict:
        """Construct a DAM4SAM-compatible inference state.

        Accepts a directory of pre-extracted frames or a
        `StreamingFrameLoader`. Must be called inside a
        ``torch.inference_mode()`` context.
        """
        if not isinstance(frames_dir, StreamingFrameLoader):
            frames_dir = self._as_streaming_loader(frames_dir)

        state: dict = {}
        state["images"]               = frames_dir
        state["num_frames"]           = len(frames_dir)
        state["offload_video_to_cpu"] = frames_dir.offload
        state["offload_state_to_cpu"] = False
        state["video_height"]         = frames_dir.video_height or 0
        state["video_width"]          = frames_dir.video_width  or 0
        state["device"]               = self._predictor.device
        state["storage_device"]       = self._predictor.device

        # Per-object inputs and DRM bookkeeping (DAM4SAM addition).
        state["point_inputs_per_obj"]       = {}
        state["mask_inputs_per_obj"]        = {}
        state["adds_in_drm_per_obj"]        = {}

        state["cached_features"]            = {}
        state["constants"]                  = {}
        state["obj_id_to_idx"]              = OrderedDict()
        state["obj_idx_to_id"]              = OrderedDict()
        state["obj_ids"]                    = []
        state["curr_out"]                   = None

        # DAM4SAM's predictor expects the full output_dict / consolidated_frame_inds
        # bookkeeping that stock SAM2 also uses internally.
        state["output_dict"] = {
            "cond_frame_outputs":     {},
            "non_cond_frame_outputs": {},
        }
        state["output_dict_per_obj"]        = {}
        state["temp_output_dict_per_obj"]   = {}
        state["consolidated_frame_inds"]    = {
            "cond_frame_outputs":     set(),
            "non_cond_frame_outputs": set(),
        }
        state["tracking_has_started"]       = False
        state["frames_already_tracked"]     = {}
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
        """Process an entire frame sequence one frame at a time, with DRM.

        Mirrors DAM4SAM's reference flow: frame 0 is initialised via
        ``add_new_points_or_box`` (whose return value already gives the
        consolidated mask), and ``propagate_in_video`` is only called for
        frames 1..N-1.  This avoids a latent bug in DAM4SAM's predictor
        where propagating a conditioning frame raises ``UnboundLocalError``.

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
            num_frames = state["num_frames"]
            obj_id = 1
            drm_state = _DRMState(object_sizes=[], last_added=-1)

            # ---- Frame 0: prompt and yield directly (no propagate). ----
            _, _, init_masks = self._predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id,
                box=initial_bbox.astype(np.float32),
            )
            mask0 = (init_masks[0, 0] > 0.0).cpu().numpy()
            conf0 = self._compute_confidence(mask0)
            yield 0, self._make_result(mask0, conf0)
            self._update_buffer_if_visible(0, mask0, conf0, depth_seq)

            # ---- Frames 1..N-1: propagate one frame at a time. ----
            reprompt_count = 0
            frame_idx = 1

            while frame_idx < num_frames:
                lost_at = None
                while frame_idx < num_frames:
                    out = self._track_one_frame(state, frame_idx)
                    if out is None:
                        break
                    mask_np, alt_payload = out

                    conf = self._compute_confidence(mask_np)
                    yield frame_idx, self._make_result(mask_np, conf)
                    self._update_buffer_if_visible(frame_idx, mask_np, conf, depth_seq)

                    # DAM4SAM trigger — pin this frame as a conditioning
                    # frame when the model sees a credible distractor.
                    if alt_payload is not None:
                        self._maybe_add_to_drm(
                            state, obj_id, frame_idx, mask_np,
                            alt_payload, drm_state,
                        )

                    # Detect target loss.
                    if conf < self._pcfg["min_mask_confidence"]:
                        if lost_at is None:
                            lost_at = frame_idx
                            if not allow_reprompt:
                                frame_idx = num_frames
                                break
                    else:
                        lost_at = None

                    frame_idx += 1

                # One re-prompt cycle per loss event.
                if lost_at is not None and allow_reprompt and reprompt_count < max_reprompts:
                    entry = self._best_reprompt_entry(
                        expected_dist=self._predict_reprompt_distance()
                    )
                    if entry is not None and entry.frame_idx < lost_at:
                        # Re-seed the predictor's memory from the best buffered
                        # frame; resume propagation at the next frame so we do
                        # not re-propagate a (now) conditioning frame.
                        reprompt_count += 1
                        self._reseed_from_buffer(state, obj_id, entry)
                        frame_idx = entry.frame_idx + 1
                        continue
                break

    def _update_buffer_if_visible(
        self,
        frame_idx: int,
        mask: np.ndarray,
        conf: float,
        depth_seq: Optional[List[Optional[np.ndarray]]],
    ) -> None:
        """Update distance/temporal buffer when the frame is confidently tracked."""
        if conf < self._pcfg["min_mask_confidence"]:
            return
        dist = 2.0
        if depth_seq and frame_idx < len(depth_seq) and depth_seq[frame_idx] is not None:
            dist = self._depth_at_mask(mask, depth_seq[frame_idx])
        bbox = self._mask_to_bbox(mask)
        if bbox is not None:
            self._update_buffer(frame_idx, bbox, dist, conf)

    # ------------------------------------------------------------------
    def track_sequence_multi(
        self,
        frames_dir: Union[str, Path, "StreamingFrameLoader"],
        yolo_model_path: str = "yolo11m-seg.pt",
        yolo_conf_threshold: float = 0.45,
        **_kwargs,  # absorb legacy args (disappearance_timeout_frames, etc.)
    ) -> Iterator[Tuple[int, Dict[int, "TrackResult"]]]:
        """Multi-person tracking via N parallel single-object DAM4SAM trackers.

        DAM4SAM is fundamentally single-object: its `n_pixels_pos` and `iou`
        fields are stored as ``(batch_size, 1)`` tensors that get unpacked
        with ``if out['n_pixels_pos'] < 1:`` inside ``_prepare_memory_-
        conditioned_features``. With batch_size > 1 that comparison raises
        ``RuntimeError: Boolean value of Tensor with more than one value is
        ambiguous``. Rather than fight DAM4SAM's design, we run one
        inference state per person — each is a faithful single-object
        DAM4SAM tracker, and the DRM logic applies independently per person.

        Persons visible in the first frame are detected by YOLO and seeded
        with segmentation masks when available, bbox prompts otherwise.
        After that, no YOLO runs — the parallel trackers do the work.

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

            # Build a shared frame source so each per-person state reads the
            # same on-disk JPEGs / streamed tensors.
            shared_loader = self._as_streaming_loader(frames_dir)
            num_frames = len(shared_loader)

            # ----------------------------------------------------------
            # 1. Read frame 0 and detect all persons with YOLO.
            # ----------------------------------------------------------
            frame0_bgr = self._wait_and_read_bgr(shared_loader.img_paths[0])
            if frame0_bgr is None:
                for fi in range(num_frames):
                    yield fi, {}
                return

            with torch.autocast(device, enabled=False):
                yolo_out = yolo_model(
                    frame0_bgr, verbose=False,
                    conf=yolo_conf_threshold, iou=0.7,
                )

            # ----------------------------------------------------------
            # 2. Build one inference state per detected person.
            # ----------------------------------------------------------
            states: Dict[int, dict]      = {}
            drm_states: Dict[int, _DRMState] = {}
            frame0_masks: Dict[int, np.ndarray] = {}
            next_obj_id = 1

            for r in yolo_out:
                _seg_data = None
                if r.masks is not None and r.masks.data is not None:
                    _seg_data = r.masks.data.cpu().numpy()  # (N, H_m, W_m)

                for i, box in enumerate(r.boxes):
                    if int(box.cls) != 0:
                        continue
                    bbox = box.xyxy[0].cpu().numpy().astype(np.float32)

                    seg_mask = None
                    if _seg_data is not None and i < len(_seg_data):
                        m = cv2.resize(
                            _seg_data[i],
                            (frame0_bgr.shape[1], frame0_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        seg_mask = m > 0.5

                    state = self._build_inference_state(shared_loader)
                    if seg_mask is not None:
                        _, _, mask_logits = self._predictor.add_new_mask(
                            state, frame_idx=0, obj_id=1, mask=seg_mask,
                        )
                    else:
                        cx = float((bbox[0] + bbox[2]) / 2)
                        cy = float((bbox[1] + bbox[3]) / 2)
                        _, _, mask_logits = self._predictor.add_new_points_or_box(
                            state,
                            frame_idx=0,
                            obj_id=1,
                            box=bbox,
                            points=np.array([[cx, cy]], dtype=np.float32),
                            labels=np.array([1], dtype=np.int32),
                        )
                    mask_np = (mask_logits[0, 0] > 0.0).cpu().numpy()

                    states[next_obj_id]       = state
                    drm_states[next_obj_id]   = _DRMState(object_sizes=[], last_added=-1)
                    frame0_masks[next_obj_id] = mask_np
                    next_obj_id += 1

            if not states:
                print("[WARN] No persons detected in frame 0.")
                for fi in range(num_frames):
                    yield fi, {}
                return

            print(f"[INFO] Registered {len(states)} persons in frame 0.")

            # ---- Frame 0: yield per-tracker init masks. ----
            results0 = {
                oid: self._make_result(m, self._compute_confidence(m))
                for oid, m in frame0_masks.items()
            }
            yield 0, results0

            # ----------------------------------------------------------
            # 3. Frames 1..N-1: propagate every tracker independently and
            #    apply DRM per tracker. Backbone features are computed once
            #    per frame and shared across all states to avoid N forward
            #    passes through the image encoder.
            # ----------------------------------------------------------
            for fi in range(1, num_frames):
                self._share_image_features(states, fi)
                results: Dict[int, TrackResult] = {}
                for oid, state in states.items():
                    out = self._track_one_frame(state, fi)
                    if out is None:
                        continue
                    mask_np, alt_payload = out
                    conf = self._compute_confidence(mask_np)
                    results[oid] = self._make_result(mask_np, conf)
                    if alt_payload is not None:
                        self._maybe_add_to_drm(
                            state, 1, fi, mask_np,
                            alt_payload, drm_states[oid],
                        )
                yield fi, results

    # ------------------------------------------------------------------
    def _as_streaming_loader(
        self, frames_dir: Union[str, Path, "StreamingFrameLoader"]
    ) -> "StreamingFrameLoader":
        """Coerce a directory path into a StreamingFrameLoader."""
        if isinstance(frames_dir, StreamingFrameLoader):
            return frames_dir
        paths = sorted(Path(frames_dir).glob("*.jpg"))
        if not paths:
            paths = sorted(Path(frames_dir).glob("*.png"))
        return StreamingFrameLoader(
            paths,
            image_size=self._predictor.image_size,
            device=self._predictor.device,
        )

    def _share_image_features(self, states: Dict[int, dict], frame_idx: int) -> None:
        """Compute backbone features once for ``frame_idx`` and share across states.

        DAM4SAM's `_get_image_feature` caches `(image, backbone_out)` per
        state, replacing any prior frame's cache. Without sharing, N
        per-person states would each run the image encoder on the same
        frame. We compute features for the first state, then poke the same
        ``cached_features`` dict object into every other state so the
        downstream cache lookup hits.
        """
        if not states:
            return
        it = iter(states.values())
        first = next(it)
        # Trigger the encode + cache on the first state.
        self._predictor._get_image_feature(first, frame_idx, batch_size=1)
        shared = first["cached_features"]
        for s in it:
            s["cached_features"] = shared

    # ------------------------------------------------------------------
    # Frame-by-frame propagation helpers
    # ------------------------------------------------------------------
    def _track_one_frame(
        self,
        state: dict,
        frame_idx: int,
    ) -> Optional[Tuple[np.ndarray, Optional[Tuple[List[np.ndarray], np.ndarray]]]]:
        """Propagate a single frame via DAM4SAM's predictor.

        Returns (chosen_mask_bool, alt_payload) where ``alt_payload`` is
        ``(alternative_masks_bool_list, all_pred_ious)`` when available, else
        ``None``. The first frame yields no alternatives.
        """
        gen = self._predictor.propagate_in_video(
            state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=0,
            return_all_masks=True,
        )
        try:
            out = next(iter(gen))
        except StopIteration:
            return None

        if len(out) == 3:
            _, _, masks = out
            mask_np = (masks[0, 0] > 0.0).cpu().numpy()
            return mask_np, None

        _, _, masks, alt_payload = out
        mask_np = (masks[0, 0] > 0.0).cpu().numpy()
        all_masks_t, all_ious = alt_payload
        all_masks_np = [(m[0, 0] > 0.0).cpu().numpy() for m in all_masks_t]
        return mask_np, (all_masks_np, np.asarray(all_ious))

    # ------------------------------------------------------------------
    # DAM4SAM Distractor-aware Replay Memory
    # ------------------------------------------------------------------
    def _maybe_add_to_drm(
        self,
        state: dict,
        obj_id: int,
        frame_idx: int,
        chosen_mask: np.ndarray,
        alt_payload: Tuple[List[np.ndarray], np.ndarray],
        drm: _DRMState,
    ) -> None:
        """Decide whether to pin ``frame_idx`` into the predictor's DRM.

        Mirrors the trigger logic in ``DAM4SAM/dam4sam_tracker.py::track``:
        the chosen mask must look healthy (high predicted IoU, area within
        ±20% of recent median, and at least 5 frames since the last DRM add),
        and at least one alternative must diverge from the chosen mask
        (bbox IoU < 0.7 once overlap with the chosen mask is removed).
        """
        all_masks, all_ious = alt_payload
        if len(all_masks) == 0 or len(all_ious) == 0:
            return

        m_idx = int(np.argmax(all_ious))
        m_iou = float(all_ious[m_idx])
        # Drop the chosen mask from the alternatives list.
        alt_masks = [m for i, m in enumerate(all_masks) if i != m_idx]

        n_pixels = int(chosen_mask.sum())
        drm.object_sizes.append(n_pixels)

        if len(drm.object_sizes) > 1 and n_pixels >= 1:
            recent = [s for s in drm.object_sizes[-_DRM_OBJ_HISTORY_WINDOW:] if s >= 1]
            recent = recent[-_DRM_OBJ_HISTORY_RECENT:]
            obj_ratio = n_pixels / max(np.median(recent), 1.0) if recent else -1.0
        else:
            obj_ratio = -1.0

        gap_ok = (drm.last_added == -1) or (frame_idx - drm.last_added > _DRM_MIN_FRAME_GAP)
        if not (m_iou > _DRM_MIN_PRED_IOU
                and _DRM_OBJ_RATIO_LO <= obj_ratio <= _DRM_OBJ_RATIO_HI
                and n_pixels >= 1
                and gap_ok):
            return

        chosen_bbox = self._mask_to_bbox(chosen_mask)
        if chosen_bbox is None:
            return

        # For each alternative, subtract the chosen mask, keep the largest
        # connected component, then union with the chosen mask before
        # comparing bounding boxes.
        ious: List[float] = []
        for am in alt_masks:
            diff = np.logical_and(am, np.logical_not(chosen_mask))
            if diff.sum() < 1:
                continue
            largest = self._keep_largest_component(diff.astype(np.uint8))
            if largest.sum() < 1:
                continue
            unioned = np.logical_or(largest, chosen_mask)
            alt_bbox = self._mask_to_bbox(unioned)
            if alt_bbox is None:
                continue
            ious.append(self._box_iou(chosen_bbox, alt_bbox))

        if not ious:
            return
        if min(ious) <= _DRM_DISTRACTOR_IOU:
            drm.last_added = frame_idx
            self._predictor.add_to_drm(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
            )

    def _reseed_from_buffer(
        self, state: dict, obj_id: int, entry: _BufferEntry,
    ) -> None:
        """Re-prompt the predictor on a buffered frame.

        Pins the buffered bbox as a new conditioning input on
        ``entry.frame_idx``. The frame must already have been tracked once
        (so its image features are cached); this is true by construction
        because we only buffer frames after we have processed them.
        """
        self._predictor.add_new_points_or_box(
            state,
            frame_idx=entry.frame_idx,
            obj_id=obj_id,
            box=entry.bbox.astype(np.float32),
        )

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
    def _keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
        """Keep only the largest 8-connected component in a uint8 mask.

        Mirrors `DAM4SAM/utils/utils.py::keep_largest_component`.
        """
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8,
        )
        if stats.shape[0] <= 1:
            return mask_u8
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == keep).astype(np.uint8)

    @staticmethod
    def write_frames(
        images: List[np.ndarray],
        out_dir: str | Path,
    ) -> Path:
        """Save a list of RGB uint8 arrays as zero-padded JPEGs for SAM2."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"{i:06d}.jpg"), bgr)
        return out_dir

    # ------------------------------------------------------------------
    def _compute_confidence(self, mask: np.ndarray) -> float:
        """Smooth confidence proxy that penalises very small or very large masks.

        - 0.0 if ratio < 0.5 * min_ratio
        - Linear ramp 0→1 between [0.5 * min_ratio, 2.0 * min_ratio]
        - 1.0 between [2.0 * min_ratio, 0.30]
        - 0.3 if ratio > 0.30 (likely background or an over-segmentation).
        """
        ratio = mask.sum() / max(mask.size, 1)
        min_r = self._pcfg["min_mask_area_ratio"]

        if ratio < 0.5 * min_r:
            return 0.0
        if ratio > 0.30:
            return 0.3
        if ratio < 2.0 * min_r:
            return (ratio - 0.5 * min_r) / (1.5 * min_r)
        return 1.0

    def _update_buffer(
        self, frame_idx: int, bbox: np.ndarray, dist: float, conf: float
    ) -> None:
        bin_idx = int(dist / self._pcfg["distance_bin_width"])
        entry = _BufferEntry(frame_idx, bbox.copy(), conf, dist)

        existing = self._dist_buf.get(bin_idx)
        if existing is None or conf > existing.confidence:
            self._dist_buf[bin_idx] = entry

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
        return TrackResult(
            mask=mask, confidence=conf,
            centroid_uv=(float(cx), float(cy)),
            is_visible=True,
        )
