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
        "sam2-tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", 
                        _PROJECT_ROOT / "sam2.1_hiera_tiny.pt"),
        "edgetam":    ("configs/edgetam.yaml",
                       _EDGETAM_DIR / "checkpoints" / "edgetam.pt"),
    }

    def __init__(self,
                 backbone: str = "sam2-tiny",
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
                 # --- appearance-gated promotion (anti-contamination) ---
                 promote_appearance_gate: bool = True,  # gate LT writes on obj_ptr identity
                 appearance_tau: float = 0.65,  # min cos(candidate, trusted ref) to promote
                 trusted_ref_k: int = 5,        # # early anchors that freeze the trusted ref
                 drift_floor: float = 0.5,      # cos below this counts toward a drift run
                 drift_patience: int = 12,      # consecutive low-cos frames -> freeze promotion
                 # --- short-term-vs-long-term coherence audit (anti-flip) ---
                 # The clean long-term anchors are the trusted identity; a frame that
                 # flips onto a distractor is an outlier vs that set. Evict such
                 # frames from SHORT-TERM memory (where the flip actually cascades)
                 # so later frames can't attend to them. Persistence avoids dropping
                 # brief hard poses of the tracked target. NOTE: obj_ptr conflates
                 # pose+identity so this is noisy; a pose-invariant descriptor
                 # (ReID/color) is the planned upgrade to the *signal* — the
                 # *structure* here is unchanged by that swap.
                 distractor_reject: bool = True,  # evict short-term frames incoherent with LT
                 st_audit_tau: float = 0.5,     # min median cos(frame, LT anchors) to keep
                 identity_vote_min: int = 2,    # (telemetry only) top-K trusted-anchor vote
                 reject_patience: int = 8,      # consecutive fails before evicting (skip brief dips)
                 # --- memory self-consistency audit (robust anti-contamination) ---
                 # Periodically evict long-term anchors that are outliers vs the
                 # coherent majority — the contaminated frames that slipped past the
                 # promotion gate. Judges the accumulated (stable) memory set, not a
                 # single noisy live frame, so it never touches good tracking.
                 audit_memory: bool = True,
                 audit_gap: int = 30,           # run the audit every N frames
                 audit_tau: float = 0.5,        # evict anchor if median cos to others < this
                 collect_telemetry: bool = False,  # record per-frame obj_ptr identity stats
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
        # --- appearance-gated promotion (anti-contamination) ---
        # Only promote a frame into long-term memory if its obj_ptr identity still
        # agrees with a *frozen* trusted reference (mean of the first few clean
        # anchors). Calibrated on two_girls_dance: correct frames score >=0.74,
        # contaminated/wrong-girl frames <=0.60 even when girl-sized (so the size
        # gate alone passes them). Anchoring to early clean frames — not the
        # rolling anchors — is what stops one bad promotion from poisoning the ref.
        self.promote_appearance_gate = promote_appearance_gate
        self.appearance_tau = appearance_tau
        self.trusted_ref_k = trusted_ref_k
        self.drift_floor = drift_floor
        self.drift_patience = drift_patience
        self.distractor_reject = distractor_reject
        self.st_audit_tau = st_audit_tau
        self.identity_vote_min = identity_vote_min
        self.reject_patience = reject_patience
        self._reject_run = 0            # consecutive frames failing the identity vote
        self._pending_fail: List[int] = []  # failing frames not yet suppressed
        self.audit_memory = audit_memory
        self.audit_gap = audit_gap
        self.audit_tau = audit_tau
        self._trusted_ref = None        # frozen identity reference (normalized)
        self._ref_ptrs: List = []       # accumulates first trusted_ref_k anchors
        self._drift_low = 0             # consecutive frames with cos < drift_floor
        # --- identity telemetry (calibration for appearance-gated promotion) ---
        # When on, every frame records obj_ptr cosine-to-seed / cosine-to-LT-mean,
        # centroid, area and the gate decisions. Used to pick promotion thresholds
        # from data before turning the appearance gate on. Zero cost when off.
        self.collect_telemetry = collect_telemetry
        self._telemetry: List[dict] = []
        self._seed_ptr = None

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

    def _audit_memory(self, state) -> None:
        """Self-consistency sweep of the long-term anchors: evict any promoted
        anchor whose obj_ptr is an outlier vs the coherent majority — a
        contaminated frame that slipped past the promotion gate and 'accidentally
        got in'. Operates on the accumulated (stable) memory set, not a noisy live
        frame, so it never disturbs good tracking. Frame 0 (seed) is protected.

        A clean anchor's median cosine to the other anchors is high (~0.7-0.9);
        a wrong-identity anchor scores low against all of them (~0.3-0.5)."""
        if len(self._promoted) < 3:
            return  # need a majority to judge outliers against
        import torch
        ptrs = {i: self._obj_ptr(state, i) for i in self._promoted}
        ids = [i for i in self._promoted if ptrs[i] is not None]
        if len(ids) < 3:
            return
        for i in list(ids):
            others = sorted(float((ptrs[i] * ptrs[j]).sum())
                            for j in ids if j != i)
            coh = others[len(others) // 2]          # median cos to the rest
            if coh < self.audit_tau:
                self._evict_anchor(state, i)
                self._log(f"audit: evict outlier anchor f{i} "
                          f"(coherence={coh:.3f} < {self.audit_tau})")

    def _evict_anchor(self, state, frame_idx: int) -> None:
        """Remove a single promoted long-term anchor (used by the audit)."""
        od = state["output_dict"]
        od["cond_frame_outputs"].pop(frame_idx, None)
        for obj in state["output_dict_per_obj"].values():
            obj["cond_frame_outputs"].pop(frame_idx, None)
        if frame_idx in self._promoted:
            self._promoted.remove(frame_idx)
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
    def _obj_ptr(self, state, frame_idx):
        """L2-normalized obj_ptr (256-d identity vector) for a frame, or None.

        This is SAM2's per-frame object descriptor; its cosine to the seed is a
        cheap drift signal — when the track jumps to a distractor the pointer
        moves toward that distractor's appearance and the cosine drops."""
        import torch
        # Frame 0's output (from the box/mask prompt) sits in the per-object /
        # temp dicts until the first propagate consolidates it into the global
        # output_dict, so search all of them.
        entries = []
        for store in ("cond_frame_outputs", "non_cond_frame_outputs"):
            entries.append(state["output_dict"].get(store, {}).get(frame_idx))
        for key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
            for obj in state.get(key, {}).values():
                for store in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    entries.append(obj.get(store, {}).get(frame_idx))
        for e in entries:
            if e is not None and e.get("obj_ptr") is not None:
                p = e["obj_ptr"].detach().float().reshape(-1)
                n = torch.linalg.norm(p)
                return (p / n) if n > 0 else p
        return None

    def _appearance_cos(self, state, frame_idx):
        """cos(candidate obj_ptr, frozen trusted reference), or None if either
        the reference isn't established yet or the frame has no obj_ptr."""
        if self._trusted_ref is None:
            return None
        ptr = self._obj_ptr(state, frame_idx)
        if ptr is None:
            return None
        return float((ptr * self._trusted_ref).sum())

    def _identity_vote(self, state, frame_idx):
        """Compare this frame's obj_ptr against EACH of the frozen top-K trusted
        anchors (not their mean). Returns (vote, median_cos): vote = how many of
        the K score >= appearance_tau. Calibrated on two_girls_dance: the tracked
        girl scores 4-5/5; a flip onto the distractor scores 0-2/5. Returns
        (None, nan) until the trusted set exists or if the frame has no obj_ptr."""
        if not self._ref_ptrs:
            return None, float("nan")
        ptr = self._obj_ptr(state, frame_idx)
        if ptr is None:
            return None, float("nan")
        cos = sorted(float((ptr * a).sum()) for a in self._ref_ptrs)
        vote = sum(1 for c in cos if c >= self.appearance_tau)
        return vote, cos[len(cos) // 2]

    def _lt_coherence(self, state, frame_idx):
        """Median cosine of this frame's obj_ptr to the CLEAN long-term anchors —
        the trusted-identity reference. High for the tracked target (it matches
        its own LT anchors across poses), low for a flip onto a distractor (it
        matches none of them). Returns None until enough anchors exist / no ptr.

        Using the accumulated LT set (which spans several poses, kept clean by the
        promotion gate) tolerates pose variation better than one frozen anchor."""
        if len(self._promoted) < 2:
            return None
        ptr = self._obj_ptr(state, frame_idx)
        if ptr is None:
            return None
        cos = []
        for k in self._promoted:
            a = self._obj_ptr(state, k)
            if a is not None:
                cos.append(float((ptr * a).sum()))
        if not cos:
            return None
        cos.sort()
        return cos[len(cos) // 2]

    def _suppress_frame(self, state, frame_idx):
        """Distractor rejection: drop this frame's short-term memory write so a
        flipped/wrong-identity mask can't be attended by later frames. This is
        what makes contamination REVERSIBLE — the clean anchors keep driving the
        track and it re-acquires the target when it reappears. Mirrors the
        short-term eviction pop, so it touches no consolidated (cond) set."""
        state["output_dict"]["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj in state["output_dict_per_obj"].values():
            obj["non_cond_frame_outputs"].pop(frame_idx, None)

    def _update_trusted_ref(self, state, frame_idx):
        """Build the trusted identity reference from the first trusted_ref_k
        promoted (early, clean) anchors, then freeze it."""
        if self._trusted_ref is not None:
            return
        ptr = self._obj_ptr(state, frame_idx)
        if ptr is not None:
            self._ref_ptrs.append(ptr)
        if len(self._ref_ptrs) >= self.trusted_ref_k:
            import torch
            m = torch.stack(self._ref_ptrs).mean(0)
            n = torch.linalg.norm(m)
            self._trusted_ref = (m / n) if n > 0 else m
            self._log(f"trusted identity reference frozen from "
                      f"{len(self._ref_ptrs)} anchors")

    def _record_telemetry(self, state, frame_idx, res, px, size_gate, promoted,
                          distractor=False):
        """Log per-frame identity stats for promotion-threshold calibration."""
        import torch
        ptr = self._obj_ptr(state, frame_idx)
        cos_seed = cos_lt = float("nan")
        if ptr is not None:
            if self._seed_ptr is not None:
                cos_seed = float((ptr * self._seed_ptr).sum())
            # mean cosine to the *other* long-term anchors (exclude self).
            anchors = [self._obj_ptr(state, k) for k in self._promoted
                       if k != frame_idx]
            anchors = [a for a in anchors if a is not None]
            if anchors:
                cos_lt = float(torch.stack([(ptr * a).sum()
                                            for a in anchors]).mean())
        # cosine to each of the frozen top-K trusted anchors (individually) —
        # the "compare against ≥5 frames to be sure" signal for a robust vote.
        cos5_med = cos5_min = float("nan")
        cos5_vote = 0
        if ptr is not None and self._ref_ptrs:
            c5 = sorted(float((ptr * a).sum()) for a in self._ref_ptrs)
            cos5_med = c5[len(c5) // 2]
            cos5_min = c5[0]
            cos5_vote = sum(1 for c in c5 if c >= self.appearance_tau)
        cx, cy = (res.centroid_uv if res.centroid_uv is not None
                  else (float("nan"), float("nan")))
        self._telemetry.append(dict(
            frame=frame_idx, cos_seed=cos_seed, cos_lt=cos_lt,
            cos5_med=cos5_med, cos5_min=cos5_min, cos5_vote=cos5_vote,
            cx=cx, cy=cy, area=px, visible=bool(res.is_visible),
            size_gate=bool(size_gate), promoted=bool(promoted),
            distractor=bool(distractor)))

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
        self._telemetry.clear()
        self._seed_ptr = None
        self._trusted_ref = None
        self._ref_ptrs.clear()
        self._drift_low = 0
        self._reject_run = 0
        self._pending_fail.clear()
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
                    # Seed identity anchor — the clean reference for drift checks.
                    self._seed_ptr = self._obj_ptr(state, 0)
                    if self.collect_telemetry:
                        self._record_telemetry(state, 0, res, px, False, False)
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

                # ---- live distractor rejection (top-K identity vote) ----
                # A flip onto the other person is a girl-sized mask that disagrees
                # with the frozen trusted anchors. To avoid dropping the odd hard
                # pose of the *correct* girl, we only suppress once the vote has
                # failed for reject_patience CONSECUTIVE frames (a real flip lasts
                # the whole distractor solo; a hard pose is a brief dip). When it
                # trips we also retroactively drop that run's earlier frames, so
                # later frames can't attend to the flip — this is what keeps
                # contamination REVERSIBLE. Active only once the trusted set is
                # frozen (early frames bootstrap it).
                vote, cos5 = self._identity_vote(state, frame_idx)  # telemetry
                lt_coh = self._lt_coherence(state, frame_idx)
                fail = (self.distractor_reject
                        and res.is_visible
                        and lt_coh is not None
                        and lt_coh < self.st_audit_tau)
                if fail:
                    self._reject_run += 1
                else:
                    self._reject_run = 0
                    self._pending_fail.clear()
                distractor = fail and self._reject_run >= self.reject_patience
                if fail and not distractor:
                    self._pending_fail.append(frame_idx)  # provisional, still tracked

                if not distractor:
                    self._area_hist.append(px)

                # ---- AOT long/short-term memory management ----
                size_gate = (not distractor
                             and self._should_promote(frame_idx, px, h * w))

                # Drift run: consecutive frames whose identity has fallen away from
                # the trusted reference (freezes promotion during a long solo).
                cos_ref = self._appearance_cos(state, frame_idx)
                if cos_ref is not None:
                    self._drift_low = (self._drift_low + 1
                                       if cos_ref < self.drift_floor else 0)

                promoted = False
                if size_gate:
                    if not self.promote_appearance_gate or self._trusted_ref is None:
                        # Gate off, or still bootstrapping the trusted reference
                        # from early (clean) anchors — promote on the size gate.
                        promoted = True
                    else:
                        # Identity gate: agree with the frozen reference AND not
                        # be in the middle of a sustained drift run.
                        promoted = (cos_ref is not None
                                    and cos_ref >= self.appearance_tau
                                    and self._drift_low < self.drift_patience)
                    if promoted:
                        self._promote(state, frame_idx)
                        self._evict_long_term(state)
                        if self.promote_appearance_gate:
                            self._update_trusted_ref(state, frame_idx)
                        self._log(f"f{frame_idx}: promote→LT "
                                  f"(LT={len(self._promoted)}/{self.lt_max}, "
                                  f"px={px}, cos={cos_ref if cos_ref is not None else float('nan'):.3f}, "
                                  f"evicted={self._n_evicted})")
                    elif size_gate:
                        self._log(f"f{frame_idx}: REJECT promote "
                                  f"(cos={cos_ref if cos_ref is not None else float('nan'):.3f}"
                                  f" < tau={self.appearance_tau}, drift={self._drift_low})")

                # Telemetry BEFORE the suppression pop, while obj_ptr still exists.
                if self.collect_telemetry:
                    self._record_telemetry(state, frame_idx, res, px,
                                           size_gate, promoted, distractor)

                # Apply distractor suppression: drop the memory write + emit empty.
                if distractor:
                    self._suppress_frame(state, frame_idx)
                    # On the frame the run first trips, retroactively drop the
                    # earlier frames of this run (they were provisionally kept)
                    # so the flip never had a foothold in short-term memory.
                    if self._pending_fail:
                        for fi in self._pending_fail:
                            self._suppress_frame(state, fi)
                        self._log(f"f{frame_idx}: ST-audit trip — evict run "
                                  f"{self._pending_fail[0]}..{frame_idx} from short-term "
                                  f"(lt_coh={lt_coh:.3f} < {self.st_audit_tau})")
                        self._pending_fail.clear()
                    res, px = TrackResult(None, 0.0, None, False), 0

                # Periodic memory self-consistency audit: sweep outlier anchors
                # (contaminated frames that slipped past the promotion gate).
                if (self.audit_memory and frame_idx > 0
                        and frame_idx % self.audit_gap == 0):
                    self._audit_memory(state)

                self._evict_short_term(state, frame_idx)

                yield frame_idx, res

                if (self.empty_cache_every > 0
                        and frame_idx % self.empty_cache_every == 0):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
