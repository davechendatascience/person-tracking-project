"""Standalone offline smoke test for AOT/DeAOT (yoxu515/aot-benchmark).

Validates that the pure-PyTorch fallback path works without
spatial_correlation_sampler (the C++ CUDA extension). The AOT source
at /opt/aot-benchmark has a try/except around the correlation import
in networks/layers/attention.py — when import fails, every usage site
falls back to a `pad_and_unfold + einsum-style multiply` implementation.

Init flow: YOLO-seg on the first frame for a clean leader mask →
AOT engine.add_reference_frame → streaming.

Usage (inside the sim container, from /ws):
    python3 eval/smoke_tracker_aot.py <input.mp4> [output.mp4]

Default model: DeAOTT (the smallest / fastest variant — 5.7 M params).
Set AOT_MODEL=deaots/deaotb/etc. to try a larger model.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# AOT-benchmark source on the host is mounted read-only at /opt/aot-benchmark.
sys.path.insert(0, "/opt/aot-benchmark")
from configs.pre_ytb_dav import EngineConfig
from networks.models import build_vos_model
from networks.engines import build_engine
from utils.checkpoint import load_network

# Confirm we're on the pure-PyTorch fallback path. If this prints False,
# we're running on the much-slower-without-CUDA-kernel path; that's
# the whole point of this smoke test.
from networks.layers.attention import enable_corr as _AOT_ENABLE_CORR

from ultralytics import YOLO


# ---- Config -----------------------------------------------------------
AOT_MODEL    = os.environ.get("AOT_MODEL",    "deaott")
AOT_CKPT     = os.environ.get("AOT_CKPT",     "/opt/aot-benchmark/pretrain_models/DeAOTT_PRE_YTB_DAV.pth")
MAX_LONG_EDGE = int(os.environ.get("AOT_MAX_LONG_EDGE", "480"))  # shrink for speed without correlation

YOLO_SEG_WEIGHTS = os.environ.get("YOLO_SEG", "yolo11m-seg.pt")
YOLO_CONF        = float(os.environ.get("YOLO_CONF", "0.30"))
PERSON_CLS       = 0
INIT_SCAN_MAX    = 30

PALETTE = np.array([
    [255,  60,  60], [ 60, 255,  60], [ 60,  60, 255], [255, 220,  60],
    [255,  60, 220], [ 60, 220, 255], [220,  60, 255], [220, 255,  60],
], dtype=np.uint8)


# ---- AOT streaming wrapper -----------------------------------------
class AOTStreamingTracker:
    """Wraps the AOT engine with .initialize / .track stable across model
    variants. Image preprocessing: resize so the longest edge is
    MAX_LONG_EDGE (rounded down to a multiple of 16 — AOT's encoder
    expects stride-16 inputs), normalize with ImageNet mean/std."""

    def __init__(self, device):
        print(f"[aot] model={AOT_MODEL} ckpt={AOT_CKPT} device={device}")
        print(f"[aot] enable_corr (CUDA correlation kernel)={_AOT_ENABLE_CORR} "
              f"— {'fast' if _AOT_ENABLE_CORR else 'pure-PyTorch fallback'}")
        # Module path: configs/models/<aot_model>.py is lowercase; the
        # MODEL_NAME attribute inside is mixed-case (e.g. 'DeAOTT').
        cfg = EngineConfig(exp_name='smoke', model=AOT_MODEL)
        cfg.TEST_CKPT_PATH = AOT_CKPT
        self.cfg = cfg
        self.device = device
        gpu_id = 0 if device.type == 'cuda' else -1

        t0 = time.time()
        model = build_vos_model(cfg.MODEL_VOS, cfg)
        if device.type == 'cuda':
            model = model.cuda(gpu_id)
        else:
            model = model.to(device)
        model, _ = load_network(model, cfg.TEST_CKPT_PATH, gpu_id)
        model.eval()
        self.model = model
        self.engine = build_engine(
            cfg.MODEL_ENGINE,
            phase='eval',
            aot_model=model,
            gpu_id=gpu_id,
            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
        )
        print(f"[aot] built in {time.time() - t0:.1f}s "
              f"(LONG_TERM_MEM_GAP={cfg.TEST_LONG_TERM_MEM_GAP})")
        self.frame_index = 0
        self.video_h = 0
        self.video_w = 0

        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=device)[:, None, None]
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=device)[:, None, None]

    def _prep_image(self, rgb):
        """rgb HxWx3 uint8 → (1, 3, H', W') tensor on device.
        H' and W' are scaled to keep longest edge ≤ MAX_LONG_EDGE
        and rounded down to multiples of 16 (encoder stride)."""
        h, w = rgb.shape[:2]
        long_edge = max(h, w)
        scale = MAX_LONG_EDGE / long_edge if long_edge > MAX_LONG_EDGE else 1.0
        new_h = max(16, int(h * scale) // 16 * 16)
        new_w = max(16, int(w * scale) // 16 * 16)
        if (new_h, new_w) != (h, w):
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized = rgb
        t = torch.from_numpy(resized).to(self.device).permute(2, 0, 1).float() / 255.0
        t = (t - self._mean) / self._std
        return t.unsqueeze(0), (new_h, new_w)

    def _prep_mask(self, mask, new_h, new_w):
        """binary mask HxW uint8 → (1, 1, H', W') float tensor (resize NEAREST)."""
        h, w = mask.shape
        if (new_h, new_w) != (h, w):
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        t = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        return t.unsqueeze(0).unsqueeze(0)

    @torch.inference_mode()
    def initialize(self, rgb, mask):
        """Seed AOT with the leader mask at frame 0."""
        self.frame_index = 0
        self.video_h, self.video_w = rgb.shape[:2]
        self.engine.restart_engine()
        img_t, (new_h, new_w) = self._prep_image(rgb)
        mask_t = self._prep_mask(mask, new_h, new_w)
        self.engine.add_reference_frame(img_t, mask_t, obj_nums=[1], frame_step=0)
        # Decode the post-init mask at video resolution to confirm.
        logit = self.engine.decode_current_logits((self.video_h, self.video_w))
        return self._logit_to_binary(logit)

    @torch.inference_mode()
    def track(self, rgb):
        self.frame_index += 1
        img_t, _ = self._prep_image(rgb)
        self.engine.match_propogate_one_frame(img_t)
        logit = self.engine.decode_current_logits((self.video_h, self.video_w))
        return self._logit_to_binary(logit)

    def _logit_to_binary(self, logit):
        # logit shape (1, num_obj+1, H, W). Object IDs 1..N for the
        # tracked objects, 0 for background. Single-target → ID 1.
        prob = torch.softmax(logit, dim=1)
        argmax = prob.argmax(dim=1).squeeze(0)
        return (argmax.cpu().numpy() == 1).astype(np.uint8)


# ---- Helpers -------------------------------------------------------
def bbox_from_mask(mask):
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def yolo_seg_init(rgb, yolo_seg):
    """First-frame YOLO-seg → binary mask of the most confident person."""
    results = yolo_seg(rgb, conf=YOLO_CONF, classes=[PERSON_CLS], verbose=False)
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return None
    confs = results[0].boxes.conf.cpu().numpy()
    best_i = int(confs.argmax())
    raw = results[0].masks.data[best_i].cpu().numpy()
    H, W = rgb.shape[:2]
    mask = (cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    return mask, float(confs[best_i])


def annotate(rgb, mask, status, frame_idx, fps_now, mask_px):
    vis = rgb.copy()
    if mask is not None and mask.any():
        tinted = vis.copy()
        tinted[mask > 0] = (vis[mask > 0] * 0.5 + np.array([60, 60, 255]) * 0.5).astype(np.uint8)
        vis = tinted
        bb = bbox_from_mask(mask)
        if bb is not None:
            x1, y1, x2, y2 = bb
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
    color = {"TRACK": (0, 220, 0), "LOST": (0, 0, 255), "INIT": (255, 220, 0)}.get(status, (200, 200, 200))
    line1 = f"{status} f={frame_idx} mask_px={mask_px} ({fps_now:.1f} fps)"
    line2 = ("src: DeAOTT streaming (offline, no oracle, "
             f"correlation={'CUDA' if _AOT_ENABLE_CORR else 'pure-pytorch'})")
    for thickness, c in ((4, (0, 0, 0)), (2, color)):
        cv2.putText(vis, line1, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, thickness, cv2.LINE_AA)
    for thickness, c in ((4, (0, 0, 0)), (2, (255, 255, 255))):
        cv2.putText(vis, line2, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, thickness, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


class FfmpegWriter:
    def __init__(self, path, width, height, fps):
        import imageio
        self.writer = imageio.get_writer(
            str(path), format="ffmpeg", fps=float(fps),
            codec="libx264", pixelformat="yuv420p",
            macro_block_size=1, ffmpeg_log_level="warning")

    def write(self, bgr_frame):
        self.writer.append_data(bgr_frame[..., ::-1])

    def release(self):
        self.writer.close()


# ---- Main ----------------------------------------------------------
def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    in_path = Path(argv[1])
    if not in_path.exists():
        print(f"input not found: {in_path}", file=sys.stderr)
        return 2
    out_path = (Path(argv[2]) if len(argv) > 2
                else in_path.with_name(in_path.stem + "_annot_aot.mp4"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    tracker = AOTStreamingTracker(device)
    yolo_seg = YOLO(YOLO_SEG_WEIGHTS)  # auto-downloads on first call

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"failed to open {in_path}", file=sys.stderr)
        return 2
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5) or 30.0
    n_frames = int(cap.get(7))
    print(f"video: {w}x{h} @ {fps:.1f} fps, {n_frames} frames")
    writer = FfmpegWriter(out_path, w, h, fps)

    # Scan for first YOLO-seg person hit.
    init_rgb = None
    init_mask = None
    init_conf = 0.0
    skipped = 0
    while skipped < INIT_SCAN_MAX:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hit = yolo_seg_init(frame_rgb, yolo_seg)
        if hit is not None:
            init_mask, init_conf = hit
            init_rgb = frame_rgb
            break
        skipped += 1
    if init_rgb is None:
        print(f"YOLO-seg found no person in first {INIT_SCAN_MAX} frames", file=sys.stderr)
        return 2
    print(f"init: skipped {skipped}, conf={init_conf:.2f}, init_mask_px={int(init_mask.sum())}")

    aot_mask = tracker.initialize(init_rgb, init_mask)
    print(f"AOT init mask after first decode: px={int(aot_mask.sum())}")
    writer.write(annotate(init_rgb, aot_mask, "INIT", 0, 0.0, int(aot_mask.sum())))

    # Streaming loop.
    frame_idx = 1
    vis_n = 0
    lost_n = 0
    t_start = time.time()
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = tracker.track(rgb)
        if mask.any():
            vis_n += 1
            status = "TRACK"
        else:
            lost_n += 1
            status = "LOST"
        elapsed = time.time() - t_start
        fps_now = (frame_idx + 1) / max(elapsed, 1e-6)
        writer.write(annotate(rgb, mask, status, frame_idx, fps_now, int(mask.sum())))
        if frame_idx % 30 == 0:
            print(f"  f={frame_idx}/{n_frames - 1} status={status} "
                  f"vis={vis_n}/{vis_n + lost_n} mask_px={int(mask.sum())} "
                  f"({fps_now:.1f} fps)")
        frame_idx += 1

    cap.release()
    writer.release()
    total = time.time() - t_start
    print(f"\nDone in {total:.1f}s ({frame_idx / max(total, 1e-6):.1f} fps)")
    print(f"  total streamed frames: {frame_idx}")
    print(f"  vis: {vis_n}, lost: {lost_n}")
    print(f"  output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
