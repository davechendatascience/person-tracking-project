"""SAM2 (EdgeTAM) tracking with AOT's Long-Short-Term memory idea grafted on.

The model — EdgeTAM's SAM2 video predictor — is kept **frozen (no retraining)**.
We only change *what lives in its memory bank and how it is managed*, porting
AOT/DeAOT's memory scheme onto SAM2's architecture.

────────────────────────────────────────────────────────────────────────────
WHY — what AOT does that plain SAM2 doesn't
────────────────────────────────────────────────────────────────────────────
Debugging the AOT path taught us its two memory behaviours:

  1. It writes the prediction into memory *every frame* (update_memory →
     update_short_term_memory, and every `lt_gap` frames update_long_term_memory).
  2. It keeps a *curated, bounded long-term bank* of past frames as identity
     anchors — separate from the 1-frame short-term memory.

SAM2 already does (1) natively: `propagate_in_video` runs the memory encoder
each frame and stores that frame's `maskmem_features` in
`output_dict["non_cond_frame_outputs"][N]`. So there is NO "update_memory"
omission to fix here (that was an AOT-port bug, not a SAM2 one).

What SAM2 lacks is (2). Reading `sam2_base._prepare_memory_conditioned_features`,
frame N attends to exactly:
  • EVERY frame in `output_dict["cond_frame_outputs"]`  (temporal pos t_pos=0,
    i.e. always attended — `max_cond_frames_in_attn == -1`, no cap), plus
  • the most recent `num_maskmem` (=7) frames of `non_cond_frame_outputs`
    (a temporal-stride window), plus object pointers from ~16 recent frames.
So SAM2's only *long-term* anchor is whatever sits in `cond_frame_outputs` —
and normally that's just the prompt frame (frame 0). The recent 7-frame window
is its short-term memory.

────────────────────────────────────────────────────────────────────────────
THE MAPPING (AOT → SAM2), and the graft
────────────────────────────────────────────────────────────────────────────
  AOT short-term memory  ↔  SAM2's recent `num_maskmem` window   (leave as-is)
  AOT long-term memory   ↔  SAM2's `cond_frame_outputs`           (we curate it)

Graft = give SAM2 the long-term tier AOT has, by **promoting** confident,
stable propagated frames from `non_cond_frame_outputs` into
`cond_frame_outputs`, periodically (every `mem_gap` frames), and bounding that
set by eviction. A promoted frame becomes an always-attended identity anchor —
exactly AOT's long-term role. This generalises DAM4SAM's distractor-aware
replay memory (which only *pins* distractor-separable frames) into a full
periodic-write + bounded-evict long-term bank.

────────────────────────────────────────────────────────────────────────────
WHY IT'S CORRECT (the careful part)
────────────────────────────────────────────────────────────────────────────
• Promotion = MOVE `output_dict["non_cond_frame_outputs"][N]` →
  `["cond_frame_outputs"][N]` (and mirror the per-object dicts). Memory is read
  from the GLOBAL `output_dict` (propagate runs batched over objects against it;
  the per-obj dicts are just slices), so the global move is the one that counts.

• Doing it *between* propagate calls is safe: `propagate_in_video_preflight`
  only consolidates *temp* outputs produced by new clicks/masks — which we never
  add after frame 0 — and asserts `consolidated_frame_inds == input_frames_inds`.
  Moving entries across the cond/non_cond dicts touches neither set, so the
  assertion holds. The one hard requirement it imposes: frame 0 (the prompt)
  must STAY in `cond_frame_outputs` (preflight asserts every consolidated cond
  frame is present). Hence frame 0 is never promoted and never evicted.

• A promoted frame carries a valid `maskmem_features` (it was propagated with
  `run_mem_encoder=True`), which is exactly the field cond-frame attention reads.

• All cond frames get t_pos=0 → the "oldest" temporal-position embedding
  (`maskmem_tpos_enc[num_maskmem-1]`). So promoted frames are encoded as
  temporally-distant anchors — semantically what a long-term memory should be.

• `max_cond_frames_in_attn == -1` means promoted frames are ALL attended every
  frame, so cost grows with the bank → the eviction cap is load-bearing, not
  optional. Short-term: SAM2 never auto-evicts `non_cond_frame_outputs`, so we
  also drop entries older than the recent window to keep VRAM bounded.

CAVEAT (out-of-distribution): SAM2 was trained with 1–2 cond (prompt) frames;
promoting up to `lt_max` of them pushes the model beyond its training regime.
DAM4SAM validates the broad idea and `-1` permits it, but quality is empirical
— keep `lt_max` modest (default 6).
"""
from __future__ import annotations

import os
# Allocator tuning (same as the AOT path) — set before torch import.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True",
)

import gc
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

# Resolve `import sam2` to EdgeTAM's fork (ships its own `sam2/` package).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EDGETAM_DIR = _PROJECT_ROOT / "EdgeTAM"
if _EDGETAM_DIR.is_dir() and str(_EDGETAM_DIR) not in sys.path:
    sys.path.insert(0, str(_EDGETAM_DIR))


@dataclass
class TrackResult:
    """Mirrors the AOT tracker / SAM2Tracker result so viz code matches."""
    mask: Optional[np.ndarray]            # (H, W) bool, or None if not visible
    confidence: float
    centroid_uv: Optional[Tuple[float, float]]
    is_visible: bool


class SAM2AOTMemoryTracker:
    """EdgeTAM SAM2 video predictor + AOT Long-Short-Term memory management.

    Usage::

        t = SAM2AOTMemoryTracker()
        for idx, res in t.track_sequence(frames, initial_bbox=box):
            if res.is_visible: ...
    """

    # Backbone presets. Both are built through EdgeTAM's `sam2` fork (the one
    # with the *global* output_dict / consolidated_frame_inds layout this
    # mechanism manipulates). The stock pip `sam2` is a newer per-object layout
    # and is intentionally NOT used here. The large Hiera checkpoint loads
    # cleanly into EdgeTAM's package.
    _BACKBONES = {
        "sam2-large": ("configs/sam2.1/sam2.1_hiera_l.yaml",
                       _PROJECT_ROOT / "sam2.1_hiera_large.pt"),
        "sam2-small": ("configs/sam2.1/sam2.1_hiera_s.yaml",
                       _PROJECT_ROOT / "sam2.1_hiera_small.pt"),
        "edgetam":    ("configs/edgetam.yaml",
                       _EDGETAM_DIR / "checkpoints" / "edgetam.pt"),
    }

    def __init__(self,
                 backbone: str = "sam2-small",
                 model_cfg: Optional[str] = None,
                 checkpoint: Optional[str] = None,
                 device: str = "cuda",
                 image_size: int = 1024,
                 # --- AOT long/short-term memory knobs ---
                 mem_gap: int = 10,            # promote a frame every N frames (AOT lt_gap)
                 lt_max: int = 6,              # max promoted long-term frames (+ frame 0)
                 keep_behind: int = 24,        # recent non-cond frames to retain (short-term)
                 promote_area_lo: float = 0.6,  # size-stability band vs recent median
                 promote_area_hi: float = 1.6,
                 min_area_ratio: float = 0.0005,
                 empty_cache_every: int = 200,
                 amp: str = "bf16",           # "bf16" | "fp16" | "none" — ~2.5x on this HW
                 compile_encoder: bool = True,  # torch.compile the image encoder (~1.4x)
                 verbose: bool = True):
        import torch
        if backbone not in self._BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; "
                             f"choose from {list(self._BACKBONES)}")
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                          "none": None}[amp]
        self.compile_encoder = compile_encoder
        self.backbone = backbone
        _cfg, _ckpt = self._BACKBONES[backbone]
        self.model_cfg = model_cfg or _cfg
        self.checkpoint = checkpoint or str(_ckpt)
        self.device = (torch.device(device)
                       if torch.cuda.is_available() else torch.device("cpu"))
        self.image_size = image_size
        self.mem_gap = mem_gap
        self.lt_max = lt_max
        self.keep_behind = keep_behind
        self.promote_area_lo = promote_area_lo
        self.promote_area_hi = promote_area_hi
        self.min_area_ratio = min_area_ratio
        self.empty_cache_every = empty_cache_every
        self.verbose = verbose
        self._predictor = None
        self._img_mean = None
        self._img_std = None
        # per-run state
        self._promoted: List[int] = []
        self._area_hist: List[int] = []
        self._n_promoted = 0
        self._n_evicted = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SAM2-AOTmem] {msg}")

    # ------------------------------------------------------------------
    def _build(self) -> None:
        if self._predictor is not None:
            return
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        sz = (os.path.getsize(self.checkpoint) / 1e6
              if os.path.exists(self.checkpoint) else -1)
        self._log(f"building {self.backbone} predictor cfg={self.model_cfg} "
                  f"ckpt={self.checkpoint} ({sz:.0f} MB) device={self.device}")
        t = time.time()
        self._predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device=self.device)
        # Free TF32 on Blackwell, and torch.compile the image encoder — it's a
        # fixed-shape (1024²) feed-forward graph, so it compiles cleanly (unlike
        # the stateful memory loop) for ~1.4x on the per-frame bottleneck. The
        # first frame pays a one-time compile cost (~tens of seconds).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if self.compile_encoder and self.device.type == "cuda":
            # Default inductor mode (op fusion), NOT max-autotune: this is a
            # unified-memory (Grace-Blackwell) box where max-autotune's parallel
            # GEMM-search workers spike RAM enough to OOM-kill a full-res run —
            # and it falls back anyway ("Not enough SMs to use max_autotune_gemm")
            # so the autotune cost buys nothing here. Cap compile threads to keep
            # the transient compile footprint small. Default mode also avoids
            # CUDA graphs, so SAM2's stored encoder outputs are never overwritten.
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
            self._predictor.image_encoder = torch.compile(
                self._predictor.image_encoder, dynamic=False)
            self._log("torch.compile(image_encoder, default mode) enabled "
                      "(first frame pays a one-time compile cost)")
        self._log(f"built in {time.time()-t:.1f}s; AOT memory: "
                  f"mem_gap={self.mem_gap} lt_max={self.lt_max} "
                  f"keep_behind={self.keep_behind}")
        self._img_mean = torch.tensor([0.485, 0.456, 0.406],
                                      device=self.device)[:, None, None]
        self._img_std = torch.tensor([0.229, 0.224, 0.225],
                                     device=self.device)[:, None, None]

    # ------------------------------------------------------------------
    def _prep_image(self, rgb: np.ndarray):
        import torch
        import torch.nn.functional as F
        t = (torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
             .permute(2, 0, 1).float() / 255.0)
        t = F.interpolate(t.unsqueeze(0),
                          size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False).squeeze(0)
        return (t - self._img_mean) / self._img_std

    def _new_state(self, h: int, w: int):
        return {
            "images": {}, "num_frames": 0,
            "offload_video_to_cpu": False, "offload_state_to_cpu": False,
            "video_height": h, "video_width": w,
            "device": self.device, "storage_device": self.device,
            "point_inputs_per_obj": {}, "mask_inputs_per_obj": {},
            "cached_features": {}, "constants": {},
            "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {},
                            "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(),
                                        "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {}, "frames_tracked_per_obj": {},
        }

    # ------------------------------------------------------------------
    # AOT memory management — operate on the predictor's output dicts.
    # ------------------------------------------------------------------
    def _should_promote(self, frame_idx: int, mask_px: int, hw: int) -> bool:
        """DMAOT-style informative gate: periodic, non-empty, size-stable.
        Rejecting size jumps keeps drift / distractor frames out of the
        long-term bank (a polluted anchor is worse than none)."""
        if frame_idx == 0 or self.mem_gap <= 0 or frame_idx % self.mem_gap != 0:
            return False
        if mask_px < max(20, int(hw * self.min_area_ratio)):
            return False
        if self._area_hist:
            med = float(np.median(self._area_hist[-10:]))
            if med > 0 and not (self.promote_area_lo * med
                                <= mask_px <= self.promote_area_hi * med):
                return False
        return True

    def _promote(self, state, frame_idx: int) -> None:
        """Move a propagated frame from short-term (non_cond) into the
        always-attended long-term tier (cond) — AOT's periodic LT write."""
        od = state["output_dict"]
        if frame_idx not in od["non_cond_frame_outputs"]:
            return
        od["cond_frame_outputs"][frame_idx] = \
            od["non_cond_frame_outputs"].pop(frame_idx)
        for obj in state["output_dict_per_obj"].values():
            if frame_idx in obj["non_cond_frame_outputs"]:
                obj["cond_frame_outputs"][frame_idx] = \
                    obj["non_cond_frame_outputs"].pop(frame_idx)
        self._promoted.append(frame_idx)
        self._n_promoted += 1

    def _evict_long_term(self, state) -> None:
        """Bounded long-term bank — drop oldest promoted frames beyond lt_max
        (FIFO). Frame 0 is never in self._promoted, so it is never evicted —
        required by SAM2's preflight, and it's the ground-truth anchor."""
        od = state["output_dict"]
        while len(self._promoted) > self.lt_max:
            victim = self._promoted.pop(0)
            od["cond_frame_outputs"].pop(victim, None)
            for obj in state["output_dict_per_obj"].values():
                obj["cond_frame_outputs"].pop(victim, None)
            self._n_evicted += 1

    def _evict_short_term(self, state, frame_idx: int) -> None:
        """Drop non-cond frames + cached features older than the recent window
        so VRAM stays bounded (SAM2 never auto-evicts non_cond). Promoted
        frames live in cond, so they are exempt."""
        cutoff = frame_idx - self.keep_behind
        if cutoff <= 0:
            return
        od = state["output_dict"]
        for k in [k for k in od["non_cond_frame_outputs"] if 0 < k < cutoff]:
            od["non_cond_frame_outputs"].pop(k, None)
        for obj in state["output_dict_per_obj"].values():
            for k in [k for k in obj["non_cond_frame_outputs"] if 0 < k < cutoff]:
                obj["non_cond_frame_outputs"].pop(k, None)
        for store in ("cached_features", "frames_already_tracked"):
            d = state.get(store)
            if isinstance(d, dict):
                for k in [k for k in d if 0 < k < cutoff]:
                    d.pop(k, None)

    # ------------------------------------------------------------------
    def _result(self, logit, h: int, w: int) -> Tuple[TrackResult, int]:
        import torch
        mask = (logit[0, 0] > 0).detach().cpu().numpy()
        px = int(mask.sum())
        if px < max(20, int(h * w * self.min_area_ratio)):
            return TrackResult(None, 0.0, None, False), px
        # probability-weighted centroid for a stable follow signal
        prob = torch.sigmoid(logit[0, 0]).detach().cpu().numpy()
        ys, xs = np.where(mask)
        wts = prob[ys, xs]
        s = float(wts.sum())
        cuv = ((float((xs * wts).sum() / s), float((ys * wts).sum() / s))
               if s > 0 else (float(xs.mean()), float(ys.mean())))
        conf = float(min(1.0, (px / float(h * w)) * 50.0))
        return TrackResult(mask.astype(bool), conf, cuv, True), px

    # ------------------------------------------------------------------
    def track_sequence(self,
                       frames: Iterable[np.ndarray],
                       initial_bbox: Optional[np.ndarray] = None,
                       initial_mask: Optional[np.ndarray] = None,
                       ) -> Iterator[Tuple[int, TrackResult]]:
        """Track one object across an RGB stream with AOT memory management.

        SAM2 locks more reliably on a box prompt than a sparse mask, so
        initial_bbox ([x1,y1,x2,y2]) is preferred; initial_mask is the fallback.
        Yields (frame_idx, TrackResult) per frame.
        """
        import contextlib
        import torch
        self._build()
        self._promoted.clear()
        self._area_hist.clear()
        self._n_promoted = self._n_evicted = 0
        predictor = self._predictor
        state = None

        # BF16/FP16 autocast speeds up the whole per-frame pipeline (encoder +
        # memory attention + decoder) ~2.5x on this hardware. ONNX export of the
        # encoder was slower than even FP32 PyTorch here, so this is the win.
        amp = (torch.autocast(self.device.type, dtype=self.amp_dtype)
               if (self.amp_dtype is not None and self.device.type == "cuda")
               else contextlib.nullcontext())
        with torch.inference_mode(), amp:
            for frame_idx, rgb in enumerate(frames):
                h, w = rgb.shape[:2]

                if frame_idx == 0:
                    state = self._new_state(h, w)
                    state["images"][0] = self._prep_image(rgb)
                    state["num_frames"] = 1
                    predictor.reset_state(state)
                    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
                    if initial_bbox is not None:
                        x1, y1, x2, y2 = [float(v) for v in initial_bbox]
                        _, _, logit = predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=0, obj_id=0,
                            box=np.array([x1, y1, x2, y2], dtype=np.float32))
                    elif initial_mask is not None:
                        _, _, logit = predictor.add_new_mask(
                            inference_state=state, frame_idx=0, obj_id=0,
                            mask=(initial_mask > 0).astype(np.uint8))
                    else:
                        raise ValueError("need initial_bbox or initial_mask")
                    state["images"].pop(0, None)
                    res, px = self._result(logit, h, w)
                    self._area_hist.append(px)
                    yield 0, res
                    continue

                # ---- propagate one frame: SAM2 writes memory natively here ----
                state["images"][frame_idx] = self._prep_image(rgb)
                state["num_frames"] = frame_idx + 1
                logit = None
                for out in predictor.propagate_in_video(
                        state, start_frame_idx=frame_idx,
                        max_frame_num_to_track=0):
                    logit = out[2]
                state["images"].pop(frame_idx, None)

                res, px = self._result(logit, h, w)
                self._area_hist.append(px)

                # ---- AOT long/short-term memory management ----
                if self._should_promote(frame_idx, px, h * w):
                    self._promote(state, frame_idx)
                    self._evict_long_term(state)
                    self._log(f"f{frame_idx}: promote→LT "
                              f"(LT={len(self._promoted)}/{self.lt_max}, "
                              f"px={px}, evicted={self._n_evicted})")
                self._evict_short_term(state, frame_idx)

                yield frame_idx, res

                if (self.empty_cache_every > 0
                        and frame_idx % self.empty_cache_every == 0):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
