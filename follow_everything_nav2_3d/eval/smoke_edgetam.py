"""Smoke test: load EdgeTAM's video predictor and run one frame end-to-end.

Verifies that:
  1. The EdgeTAM repo's `sam2` package is importable (sys.path).
  2. build_sam2_video_predictor accepts edgetam's cfg + checkpoint.
  3. init_state / add_new_points_or_box / propagate_in_video all run on a
     synthetic 320x240 RGB sequence without crashing.

Run inside the container:
    python3 eval/smoke_edgetam.py
"""
import sys
import time

# EdgeTAM ships its own `sam2` package; insert /opt/EdgeTAM at the front
# of sys.path so `from sam2 ...` resolves to EdgeTAM's fork.
sys.path.insert(0, "/opt/EdgeTAM")

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

CFG  = "configs/edgetam.yaml"
CKPT = "/opt/EdgeTAM/checkpoints/edgetam.pt"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}")

    t0 = time.time()
    predictor = build_sam2_video_predictor(CFG, CKPT, device=device)
    print(f"predictor built in {time.time() - t0:.1f}s")

    # Synthetic 4-frame "video" — random RGB at the camera resolution we use.
    h, w = 240, 320
    frames = [
        (np.random.rand(h, w, 3) * 255).astype(np.uint8) for _ in range(4)]

    # Most SAM2 video predictors expect either a video path or an
    # in-memory uint8 ndarray. Try the dict-of-frames form first; if the
    # API doesn't accept it, fall back to writing temp PNGs.
    t0 = time.time()
    try:
        state = predictor.init_state(video_path=None, async_loading_frames=False,
                                     offload_video_to_cpu=False,
                                     offload_state_to_cpu=False)
    except TypeError:
        # Older API: takes a single positional video path. We'll just dump
        # frames to /tmp and pass the directory.
        import tempfile, os, cv2
        tmp = tempfile.mkdtemp(prefix="edgetam_smoke_")
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), f[..., ::-1])
        state = predictor.init_state(video_path=tmp)
    print(f"init_state in {time.time() - t0:.1f}s")

    # Seed bbox at frame 0 — centre 100x100 box. Use whatever entry point
    # the predictor exposes for box prompts.
    try:
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=1,
            box=np.array([110, 70, 210, 170], dtype=np.float32))
    except Exception as e:
        print(f"add_new_points_or_box failed: {e!r}")
        return 1

    # Propagate.
    n = 0
    t0 = time.time()
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        n += 1
        # mask shape: (num_objs, H, W) on cuda; sanity print
        if n == 1:
            print(f"frame {frame_idx} obj_ids={obj_ids} mask.shape={tuple(masks.shape)}")
    print(f"propagated {n} frames in {time.time() - t0:.2f}s")
    print("EdgeTAM smoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
