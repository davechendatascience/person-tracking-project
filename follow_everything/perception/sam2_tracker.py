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
        from sam2.build_sam import build_sam2_video_predictor

        if self._predictor is None:
            device = self._scfg["device"] if torch.cuda.is_available() else "cpu"
            self._predictor = build_sam2_video_predictor(
                config_file=self._scfg["model_cfg"],
                ckpt_path=self._scfg["checkpoint"],
                device=device,
            )

        self._dist_buf.clear()
        self._temp_buf.clear()

        # Build inference state
        with torch.inference_mode(), \
             torch.autocast(self._scfg["device"], dtype=torch.bfloat16):

            # Case 1: Standard directory (Offline)
            if not isinstance(frames_dir, StreamingFrameLoader):
                state = self._predictor.init_state(video_path=str(frames_dir))
            else:
                # Case 2: StreamingFrameLoader (Online)
                state = {}
                state["images"] = frames_dir
                state["num_frames"] = len(frames_dir)
                state["offload_video_to_cpu"] = frames_dir.offload
                state["offload_state_to_cpu"] = False
                state["video_height"] = frames_dir.video_height or 0
                state["video_width"] = frames_dir.video_width or 0
                state["device"] = self._predictor.device
                state["storage_device"] = self._predictor.device
                state["point_inputs_per_obj"] = {}
                state["mask_inputs_per_obj"] = {}
                state["cached_features"] = {}
                state["constants"] = {}
                state["obj_id_to_idx"] = OrderedDict()
                state["obj_idx_to_id"] = OrderedDict()
                state["obj_ids"] = []
                state["output_dict_per_obj"] = {}
                state["temp_output_dict_per_obj"] = {}
                state["frames_tracked_per_obj"] = {}
                
                # Warm up visual backbone
                self._predictor._get_image_feature(state, frame_idx=0, batch_size=1)
                
                # Sync height/width from loader after it loaded frame 0
                state["video_height"] = frames_dir.video_height
                state["video_width"] = frames_dir.video_width

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
                        if conf < self._pcfg["min_mask_confidence"] and frame_idx > 0:
                            if lost_at is None:
                                lost_at = frame_idx
                                if not allow_reprompt: break
                        elif lost_at is None and frame_idx > reprompt_from:
                            lost_at = frame_idx
                            if not allow_reprompt: pass

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

        return results

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
