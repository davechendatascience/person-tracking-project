"""Streaming smoke test for EdgeTAM.

Mirrors DAM4SAM's init_state_tw + per-frame propagate_in_video pattern
(no video file, frames fed in one at a time) but using EdgeTAM directly.
If this runs end-to-end without crashing, we have everything we need to
write a streaming tracker that doesn't depend on DAM4SAM.

Run inside the container after `pip install --quiet timm`:
    python3 eval/smoke_edgetam_streaming.py
"""
import sys
import time
from collections import OrderedDict

sys.path.insert(0, "/opt/EdgeTAM")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from sam2.build_sam import build_sam2_video_predictor

CFG  = "configs/edgetam.yaml"
CKPT = "/opt/EdgeTAM/checkpoints/edgetam.pt"
INPUT_SIZE = 1024


def init_state_tw(device):
    """Reproduce DAM4SAM's init_state_tw with the keys EdgeTAM's predictor
    expects, but without going through SAM2's video-loading codepath."""
    state = {}
    state["images"] = {}
    state["num_frames"] = 0
    state["offload_video_to_cpu"] = False
    state["offload_state_to_cpu"] = False
    state["video_height"] = None
    state["video_width"] = None
    state["device"] = device
    state["storage_device"] = device
    state["point_inputs_per_obj"] = {}
    state["mask_inputs_per_obj"] = {}
    state["cached_features"] = {}
    state["constants"] = {}
    state["obj_id_to_idx"] = OrderedDict()
    state["obj_idx_to_id"] = OrderedDict()
    state["obj_ids"] = []
    state["output_dict"] = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    state["output_dict_per_obj"] = {}
    state["temp_output_dict_per_obj"] = {}
    state["consolidated_frame_inds"] = {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    state["tracking_has_started"] = False
    state["frames_already_tracked"] = {}
    state["frames_tracked_per_obj"] = {}
    return state


def prepare(image_pil, mean, std, device):
    arr = np.array(image_pil)
    t = torch.from_numpy(arr).to(device).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(t.unsqueeze(0), size=(INPUT_SIZE, INPUT_SIZE),
                      mode="bilinear", align_corners=False).squeeze(0)
    return (t - mean) / std


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, torch={torch.__version__}")

    t0 = time.time()
    predictor = build_sam2_video_predictor(CFG, CKPT, device=device)
    print(f"predictor built in {time.time() - t0:.1f}s")

    img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                            device=device)[:, None, None]
    img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                           device=device)[:, None, None]

    state = init_state_tw(device)

    # Frame 0 — synthetic blue square in the centre
    h, w = 240, 320
    f0 = (np.random.rand(h, w, 3) * 60 + 80).astype(np.uint8)
    f0[80:160, 130:210] = [40, 110, 220]  # "leader"
    pil0 = PILImage.fromarray(f0)
    state["images"][0] = prepare(pil0, img_mean, img_std, device)
    state["num_frames"] = 1
    state["video_height"] = h
    state["video_width"] = w
    predictor.reset_state(state)
    predictor._get_image_feature(state, frame_idx=0, batch_size=1)

    # Seed by mask (matches DAM4SAM's path)
    init_mask = np.zeros((h, w), dtype=np.uint8)
    init_mask[80:160, 130:210] = 1
    _, _, out_logits = predictor.add_new_mask(
        inference_state=state, frame_idx=0, obj_id=0, mask=init_mask)
    print(f"init mask propagated, out_logits.shape={tuple(out_logits.shape)}")

    # Stream 4 more frames
    t0 = time.time()
    for fi in range(1, 5):
        # Synthetic frame with the "leader" shifted right by 5 px each step
        fn = (np.random.rand(h, w, 3) * 60 + 80).astype(np.uint8)
        fn[80:160, 130 + 5 * fi:210 + 5 * fi] = [40, 110, 220]
        pil = PILImage.fromarray(fn)
        # Store as 3D (C, H, W) — EdgeTAM's predictor batches internally;
        # an extra leading dim feeds conv2d a 5D tensor and crashes.
        prepared = prepare(pil, img_mean, img_std, device)
        state["images"][fi] = prepared
        state["num_frames"] = fi + 1
        for out in predictor.propagate_in_video(
                state, start_frame_idx=fi, max_frame_num_to_track=0):
            out_frame_idx = out[0]
            mask_logits = out[2]
            mask = (mask_logits[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
        state["images"].pop(fi)
        print(f"  frame {fi}: mask px_on={int(mask.sum())}")
    print(f"propagated 4 streaming frames in {time.time() - t0:.2f}s")
    print("EdgeTAM streaming smoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
