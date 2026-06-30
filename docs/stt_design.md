# Voice-Directed Person Tracking — Design

## 1. Goal

Let an operator pick a tracking target by **voice**:

> "Track the person in the red shirt."

The system transcribes the command (STT), uses an LLM to resolve *which*
person in the current camera view the command refers to, seeds a tracker on
that person, and then follows them frame-to-frame with **AOT/DeAOT** video
object segmentation.

This fuses the STT + LLM control module from `github/ward_ws` (where STT
currently drives an LLM that controls a robot arm) into this person-tracking
project, repointing the LLM's "actuator" from an arm to a **visual target
selector** that feeds the tracker.

## 2. End-to-end pipeline

```
 mic ──► STT ──► text command
                    │
                    ▼
          LLM target resolver ──────────────┐
            (command + frame +               │  "person, red shirt,
             detections/crops)               │   left of frame"
                    │                         │
                    ▼                         │
          target person → bbox  ◄────────────┘
                    │
                    ▼
          seed mask (frame 0)
                    │
                    ▼
        ┌────────────────────────┐
        │   AOT / DeAOT tracker  │  ◄── per-frame RGB
        │  (streaming, memory)   │
        └────────────────────────┘
                    │
                    ▼
          mask + centroid per frame ──► viz / robot follow
```

Stages:

1. **STT** — microphone audio → text (ported from `ward_ws`).
2. **LLM target resolver** — given the transcribed command plus the current
   frame (and/or YOLO person detections / crops), the LLM returns the target
   person, grounded back to a pixel-space bounding box.
3. **Seed** — convert the chosen bbox into an initial segmentation mask for
   frame 0 (AOT needs a silhouette, not a box — see §4.2).
4. **Track** — AOT/DeAOT propagates the mask across the video stream, emitting
   a mask + centroid per frame. This is the long-running, real-time stage.

## 3. Why AOT instead of SAM2

The existing `run_video.py` tracks with `SAM2Tracker` (DAM4SAM-wrapped SAM2).
We are switching the tracking stage to the **AOT family** (`yoxu515/aot-benchmark`
— DeAOT, R50-DeAOTL, …) because it has already been **optimized for streaming /
edge use** in `follow_everything_nav2_3d`:

- Pure-PyTorch path — no `spatial_correlation_sampler` CUDA extension required.
- **Batched-eviction long-term-memory cap** so the memory bank does not grow
  unbounded → OOM on long videos (vanilla DeAOT concats per layer with no
  eviction).
- CUDA allocator tuning (`PYTORCH_CUDA_ALLOC_CONF`) validated over a 1001-frame
  clip (fragmentation 4.2% → 2.1%).
- Prob-weighted + EMA-smoothed centroid for a stable follow signal.

That optimized configuration lives in
`follow_everything_nav2_3d/follower_pkg/python/aot_tracker.py` (the
`_AOTStreamingTracker` class and `_build_aot_streaming_tracker`). We **reuse
that exact setting**, stripped of its ROS coupling, rather than re-tuning AOT
from scratch.

## 4. Phase 1 (current focus): refactor `run_video.py` to track with AOT

Before any STT/LLM work, get the tracking backbone right: `run_video.py` should
track with AOT instead of SAM2. This isolates the tracker swap from the voice
front-end.

### 4.1 New module: `follow_everything/perception/aot_tracker.py`

A ROS-free port of the optimized AOT tracker from `follow_everything_nav2_3d`.
It exposes an API close to `SAM2Tracker.track_sequence` so `run_video.py`
changes are minimal:

```python
tracker = AOTTracker(model="r50_deaotl", lt_max=80, ...)
for frame_idx, result in tracker.track_sequence(frame_source,
                                                 initial_bbox=target_box,
                                                 initial_mask=seed_mask):
    ...   # result.mask (bool HxW), result.centroid_uv, result.is_visible
```

The per-frame inference path is **copied directly from
`aot-benchmark/tools/demo.py`** — a hand-rolled port (and the nav2_3d wrapper)
tracked worse because they omitted `engine.update_memory()` and ran at a lower
resolution. The module now uses:

- `_build_engine` — `build_vos_model` + `build_engine` + `load_network`, the
  **batched-eviction long-term-memory cap**, the allocator env var, and the
  repo's own `MultiRestrictSize` + `MultiToTensor` transforms at the demo
  resolution (`TEST_MAX_SIZE = max_resolution * 800/480 = 1040`).
- `track_sequence` — frame 0 → `add_reference_frame(img, label, ...)`; frame N
  → `match_propogate_one_frame` → `decode_current_logits((H,W))` → softmax →
  argmax → **`update_memory(pred_label)`** (interpolated to
  `engine.input_size_2d`). The `update_memory` step is the one the earlier port
  missed — it writes each propagated mask into AOT's memory.

Verified: IoU 0.97–0.99 against the official demo's own `pred_masks` at frames
50–300 on `videos/798511637.725509.mp4` (seq1).

Dropped (ROS-specific, not needed offline): odom/lidar buffers, TF projection,
oracle-pose bootstrap, depth filtering, ROS publishers.

Result type mirrors `SAM2Tracker.TrackResult` (`mask` as **bool** HxW,
`confidence`, `centroid_uv`, `is_visible`) so the `run_video.py` drawing code is
unchanged.

### 4.2 Seeding AOT (bbox → mask)

AOT is trained on dense per-object masks; a filled-rectangle from a bbox covers
background and degrades the first-frame conditioning. For offline video (no
depth) we seed with a real silhouette:

- **Default:** run YOLO-seg (`yolo11m-seg.pt`, already in the repo) on frame 0,
  take the person instance whose box best matches the target bbox, use its mask.
- **Fallback:** filled-rectangle from the bbox (matches the nav2_3d fallback)
  when no seg instance overlaps the target.

### 4.3 `run_video.py` changes

- Add `--tracker {aot,sam2}` (default `aot`). The SAM2 path is preserved.
- Single mode (`--mode single`, track one person picked by color): build the
  target bbox via `identify_person_by_color` (unchanged), seed per §4.2, then
  drive `AOTTracker.track_sequence`.
- The producer/consumer frame extraction, viz overlay, and video-export
  threading are reused as-is; AOT consumes the same extracted frames.
- Multi-mode AOT is out of scope for Phase 1 (AOT wrapper is single-object);
  `--mode multi` stays on SAM2 for now.

### 4.4 Phase 1 acceptance

- `python run_video.py --tracker aot --mode single --target-color red ...`
  produces a tracking video where the masked person matches the SAM2 baseline
  qualitatively, with bounded VRAM over a long clip.

## 5. Phase 1.5 — AOT memory idea on the SAM2 architecture (no retraining)

AOT gives us a working streaming tracker fast (Phase 1), but the longer-term
tracker we want is **SAM2's architecture managed with AOT's memory idea**: keep
SAM2's encoder / memory-attention / mask-decoder *frozen* (no retraining), and
replace only what lives in its memory bank and how it is managed with AOT's
**Long-Short-Term memory** scheme. This is a pure inference-time change — we
never backprop — so SAM2's weights are untouched.

### 5.1 Why this, not the alternatives

- **Full architectural fusion** (add an ID-embedding branch / swap in AOT's
  local-correlation attention) changes the weights' meaning → **requires
  retraining**. Excluded by the no-retrain constraint.
- **Closed-loop concept re-prompting** (re-verify with an open-vocab detector,
  re-prompt SAM2) is useful but orthogonal — it re-grounds from *outside* the
  tracker rather than improving the tracker's own memory.
- **AOT memory on SAM2** is the no-retrain slice that actually adopts an
  AOT *mechanism*: its memory management. Highest payoff for the cost.

### 5.2 What SAM2 has vs. what AOT adds

SAM2's memory bank already holds two things the current frame attends to:
- **Conditioning frames** — `output_dict["cond_frame_outputs"]` (prompted
  frames; always attended). A *primitive* long-term anchor — but only of
  user-prompted frames.
- **Recent FIFO** — the last `num_maskmem` (≈7) propagated frames + object
  pointers. This is the short-term memory.

AOT's memory idea adds a curated **long-term tier** between these: high-value
past frames written *automatically* (not just when prompted), periodically, and
bounded by eviction. Mapping AOT → SAM2 hooks:

1. **Periodic long-term writes** — every `mem_gap` frames (AOT's
   `TEST_LONG_TERM_MEM_GAP`), promote the current propagated frame's memory
   features into a long-term store the memory attention also reads.
2. **Informative selection (DMAOT)** — promote only confident, non-redundant
   frames (size-stable + distinct from existing long-term entries). This is a
   generalization of DAM4SAM's DRM trigger (which only pins
   distractor-separable frames).
3. **Bounded eviction** — cap the long-term store and evict oldest / least
   informative. **Reuse `_install_batch_evict_lt_cap` from
   `follow_everything/perception/aot_tracker.py`**, retargeted from AOT's
   `long_term_memories` tensor list to SAM2's frame-keyed memory dict.
4. **Short-term untouched** — SAM2's `num_maskmem` recent window stays as-is.

Net: SAM2's memory attention now reads {short-term recent window} ∪ {curated,
bounded long-term bank} — AOT's Long-Short-Term structure on SAM2's frozen
weights.

### 5.3 Relationship to DAM4SAM

This repo's SAM2 path (`follow_everything/perception/sam2_tracker.py`) is built
around **DAM4SAM** (Distractor-aware Replay Memory), whose DRM is exactly the
*primitive* of step 2 above. Phase 1.5 generalizes DRM into a full long-term
tier (periodic writes + DMAOT eviction). The implementation extends DAM4SAM's
memory hooks rather than replacing them.

### 5.4 Implementation (done — prototype A, against EdgeTAM)

Implemented in `follow_everything/perception/sam2_aot_memory.py`
(`SAM2AOTMemoryTracker`), wired into `run_video.py --tracker sam2-aotmem`.

- Drives EdgeTAM's SAM2 video predictor in streaming mode (manual
  `inference_state`, `propagate_in_video(start, max=0)` per frame) — the same
  pattern the nav2_3d EdgeTAM wrapper uses, no DAM4SAM dependency.
- **Promotion** (`_promote`): every `mem_gap` frames, a confident, size-stable
  propagated frame is moved from `non_cond_frame_outputs` →
  `cond_frame_outputs` (SAM2's always-attended tier) in both the global and
  per-object dicts. Safe between propagate calls — preflight only consolidates
  temp outputs from clicks, which we never add after frame 0.
- **Long-term eviction** (`_evict_long_term`): FIFO cap at `lt_max`; frame 0 /
  prompt frames are protected (never promoted, so never evicted). This is what
  bounds VRAM, since SAM2's `max_cond_frames_in_attn == -1` has no built-in cap.
- **Short-term eviction** (`_evict_short_term`): drop non-cond outputs +
  cached features older than `keep_behind` so the recent-window memory stays
  bounded too.
- Exposes the same `track_sequence(frames, initial_bbox, initial_mask)` API as
  `AOTTracker`; prefers a box prompt (SAM2 locks better on a box than a sparse
  mask). Run: `python run_video.py --tracker sam2-aotmem --mode single
  --target-color red --video <clip>`.
- Dependency: EdgeTAM's encoder needs `timm` (now installed in the venv).

The `DAM4SAM/` checkout that `sam2_tracker.py` expects is still absent; porting
these memory hooks onto DAM4SAM (so the `--tracker sam2` path also gets them) is
the follow-up once DAM4SAM is restored.

### 5.5 Status & remaining validation

**Proven:** builds and runs end-to-end; promotion + bounded long/short-term
eviction work (verified with small `mem_gap`/`lt_max`); the target mask holds
across a 60-frame real clip with stable VRAM.

**Not yet proven (next):** the *quality* claim — that AOT-memory beats plain
SAM2 (no promotion) on a **distractor clip** (someone crossing the target).
Needs an A/B on a clip with an occlusion/distractor: `sam2-aotmem` should
recover / hold identity at least as well as plain EdgeTAM, no retraining.

## 6. Later phases (voice front-end)

- **Phase 2 — STT submodule.** Create `stt/` (ported from `ward_ws`):
  mic capture + transcription, emitting text commands.
- **Phase 3 — LLM target resolver.** Map a command + current frame/detections
  to a target bbox; hand off to the seeding path of whichever tracker is active.
- **Phase 4 — integration.** Wire STT → LLM → tracker into a live loop (and
  eventually the robot-follow stack in `follow_everything_nav2_3d`).

## 7. File layout (target)

```
docs/stt_design.md                                  ← this doc
follow_everything/perception/aot_tracker.py         ← Phase 1 (new)
follow_everything/perception/sam2_aot_memory.py     ← Phase 1.5 (planned)
follow_everything/perception/sam2_tracker.py        ← SAM2/DAM4SAM base (extend)
run_video.py                                         ← Phase 1 (edited)
stt/                                                 ← Phase 2 (new)
llm/  (or stt/resolver.py)                            ← Phase 3 (new)
follow_everything_nav2_3d/follower_pkg/python/aot_tracker.py
                                                     ← source of the optimized
                                                       AOT setting (reference)
EdgeTAM/sam2/                                         ← present SAM2 predictor
                                                       (Phase-1.5 prototype target)
DAM4SAM/                                              ← expected by sam2_tracker.py;
                                                       not checked out (restore)
```
