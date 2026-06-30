import importlib
import sys
import os
import gc

# PyTorch CUDA allocator tweaks for long-video tracking — must be set
# BEFORE torch is imported, since the allocator reads this env var at
# initialisation. Keeps large blocks intact (max_split_size_mb=128),
# proactively releases free segments above 70% pressure
# (garbage_collection_threshold=0.7), and uses expandable segments to
# reduce fragmentation when the long-term memory bank grows.
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'max_split_size_mb:128,garbage_collection_threshold:0.7,expandable_segments:True',
)

# Make aot-benchmark's networks/ dataloaders/ utils/ importable regardless of
# the working directory (the original demo assumes cwd == aot-benchmark/tools).
from pathlib import Path as _Path
_AOT_DIR = None
for _p in _Path(__file__).resolve().parents:
    if (_p / "aot-benchmark").is_dir():
        _AOT_DIR = _p / "aot-benchmark"
        sys.path.insert(0, str(_AOT_DIR))
        break
sys.path.append('.')
sys.path.append('..')

import cv2
from PIL import Image
from skimage.morphology import dilation

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.models import build_vos_model
from networks.engines import build_engine
from utils.checkpoint import load_network

from dataloaders.eval_datasets import VOSTest
import dataloaders.video_transforms as tr
from utils.image import save_mask

_palette = [
    255, 0, 0, 0, 0, 139, 255, 255, 84, 0, 255, 0, 139, 0, 139, 0, 128, 128,
    128, 128, 128, 139, 0, 0, 218, 165, 32, 144, 238, 144, 160, 82, 45, 148, 0,
    211, 255, 0, 255, 30, 144, 255, 255, 218, 185, 85, 107, 47, 255, 140, 0,
    50, 205, 50, 123, 104, 238, 240, 230, 140, 72, 61, 139, 128, 128, 0, 0, 0,
    205, 221, 160, 221, 143, 188, 143, 127, 255, 212, 176, 224, 230, 244, 164,
    96, 250, 128, 114, 70, 130, 180, 0, 128, 0, 173, 255, 47, 255, 105, 180,
    238, 130, 238, 154, 205, 50, 220, 20, 60, 176, 48, 96, 0, 206, 209, 0, 191,
    255, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45,
    45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51,
    52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58,
    58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64,
    64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70,
    71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77,
    77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83,
    83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89,
    90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96,
    96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101,
    102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106,
    107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111,
    112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116,
    117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121,
    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126,
    127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131,
    132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136,
    137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141,
    142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146,
    147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151,
    152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156,
    157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161,
    162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166,
    167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171,
    172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176,
    177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181,
    182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186,
    187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191,
    192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196,
    197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201,
    202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206,
    207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211,
    212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216,
    217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221,
    222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226,
    227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231,
    232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236,
    237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241,
    242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246,
    247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251,
    252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255, 0, 0, 0
]
color_palette = np.array(_palette).reshape(-1, 3)

# ── device helper ──────────────────────────────────────────────────────────────
def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    print('WARNING: CUDA not available, falling back to CPU. Expect slow inference.')
    return torch.device('cpu')


def to_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool = False) -> torch.Tensor:
    """Move tensor to device; non_blocking is only meaningful for CUDA."""
    nb = non_blocking and device.type == 'cuda'
    return tensor.to(device, non_blocking=nb)
# ───────────────────────────────────────────────────────────────────────────────


def overlay(image, mask, colors=[255, 0, 0], cscale=1, alpha=0.4):
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        foreground = image * alpha + np.ones(
            image.shape) * (1 - alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        im_overlay[binary_mask] = foreground[binary_mask]

        countours = dilation(binary_mask) ^ binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


def demo(cfg):
    video_fps = 15
    gpu_id = cfg.TEST_GPU_ID
    device = get_device(gpu_id)

    # Load pre-trained model
    print('Build AOT model.')
    if device.type == 'cuda':
        model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
    else:
        model = build_vos_model(cfg.MODEL_VOS, cfg).to(device)

    print('Load checkpoint from {}'.format(cfg.TEST_CKPT_PATH))
    model, _ = load_network(model, cfg.TEST_CKPT_PATH, gpu_id if device.type == 'cuda' else -1)

    print('Build AOT engine.')
    engine = build_engine(
        cfg.MODEL_ENGINE,
        phase='eval',
        aot_model=model,
        gpu_id=gpu_id if device.type == 'cuda' else -1,
        long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
    )

    # Prepare datasets
    transform = transforms.Compose([
        tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                             cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                             cfg.MODEL_ALIGN_CORNERS),
        tr.MultiToTensor()
    ])
    image_root = os.path.join(cfg.TEST_DATA_PATH, 'images')
    label_root = os.path.join(cfg.TEST_DATA_PATH, 'masks')

    sequences = os.listdir(image_root)
    seq_datasets = []
    for seq_name in sequences:
        print('Build a dataset for sequence {}.'.format(seq_name))
        seq_images = np.sort(os.listdir(os.path.join(image_root, seq_name)))
        seq_labels = [seq_images[0].replace('jpg', 'png')]
        seq_dataset = VOSTest(image_root,
                              label_root,
                              seq_name,
                              seq_images,
                              seq_labels,
                              transform=transform)
        seq_datasets.append(seq_dataset)

    # Infer
    output_root = cfg.TEST_OUTPUT_PATH
    output_mask_root = os.path.join(output_root, 'pred_masks')
    os.makedirs(output_mask_root, exist_ok=True)

    for seq_dataset in seq_datasets:
        seq_name = seq_dataset.seq_name
        image_seq_root = os.path.join(image_root, seq_name)
        output_mask_seq_root = os.path.join(output_mask_root, seq_name)
        os.makedirs(output_mask_seq_root, exist_ok=True)

        print('Build a dataloader for sequence {}.'.format(seq_name))
        # pin_memory only helps with CUDA
        seq_dataloader = DataLoader(
            seq_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.TEST_WORKERS,
            pin_memory=(device.type == 'cuda'),
        )

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(
            output_root, '{}_{}fps.avi'.format(seq_name, video_fps))

        print('Start the inference of sequence {}:'.format(seq_name))
        model.eval()
        engine.restart_engine()
        with torch.no_grad():
            for frame_idx, samples in enumerate(seq_dataloader):
                sample = samples[0]
                img_name = sample['meta']['current_name'][0]

                obj_nums = sample['meta']['obj_num']
                output_height = sample['meta']['height']
                output_width = sample['meta']['width']
                obj_idx = sample['meta']['obj_idx']

                obj_nums = [int(obj_num) for obj_num in obj_nums]
                obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

                current_img = to_device(sample['current_img'], device, non_blocking=True)

                if frame_idx == 0:
                    videoWriter = cv2.VideoWriter(
                        output_video_path, fourcc, video_fps,
                        (int(output_width), int(output_height)))
                    print(
                        'Object number: {}. Inference size: {}x{}. Output size: {}x{}.'
                        .format(obj_nums[0],
                                current_img.size()[2],
                                current_img.size()[3],
                                int(output_height),
                                int(output_width)))
                    current_label = to_device(
                        sample['current_label'], device, non_blocking=True
                    ).float()
                    current_label = F.interpolate(current_label,
                                                  size=current_img.size()[2:],
                                                  mode="nearest")
                    engine.add_reference_frame(current_img,
                                               current_label,
                                               frame_step=0,
                                               obj_nums=obj_nums)
                else:
                    print('Processing image {}...'.format(img_name))
                    engine.match_propogate_one_frame(current_img)
                    pred_logit = engine.decode_current_logits(
                        (output_height, output_width))
                    pred_prob = torch.softmax(pred_logit, dim=1)
                    pred_label = torch.argmax(pred_prob, dim=1,
                                              keepdim=True).float()

                    # ── track-confidence gate ────────────────────────────
                    # When target leaves the frame, AOT can latch onto a
                    # similar-looking distractor. The distractor's mean
                    # softmax prob over the predicted region is usually
                    # noticeably lower than for the real target (which
                    # matches F_0 strongly). Mean-prob threshold filters
                    # these out. Also blocks pollution of memory.
                    if cfg.DEMO_CONF_THRESH > 0:
                        fg_mask = (pred_label > 0).squeeze(1)  # [B, H, W]
                        for obj_id in range(1, pred_prob.shape[1]):
                            obj_mask = (pred_label.squeeze(1) == obj_id)
                            n_px = obj_mask.float().sum().item()
                            if n_px == 0:
                                continue
                            mean_p = (pred_prob[:, obj_id] *
                                      obj_mask.float()).sum().item() / n_px
                            if mean_p < cfg.DEMO_CONF_THRESH:
                                pred_label[pred_label == obj_id] = 0
                                if not hasattr(cfg, '_conf_stats'):
                                    cfg._conf_stats = {'sup': 0, 'last': 0}
                                cfg._conf_stats['sup'] += 1
                                if (cfg._conf_stats['sup'] -
                                        cfg._conf_stats['last']) >= 20 or \
                                        cfg._conf_stats['sup'] == 1:
                                    cfg._conf_stats['last'] = cfg._conf_stats['sup']
                                    print(f"  [CONF] suppress obj{obj_id} "
                                          f"(mean_p={mean_p:.3f} < "
                                          f"{cfg.DEMO_CONF_THRESH}, "
                                          f"total_sup={cfg._conf_stats['sup']})")

                    _pred_label = F.interpolate(pred_label,
                                                size=engine.input_size_2d,
                                                mode="nearest")
                    engine.update_memory(_pred_label)

                    # Periodic cache cleanup + fragmentation monitor.
                    # Long-running inference accumulates allocator state;
                    # gc.collect() + empty_cache() periodically gives the
                    # allocator room to reclaim freed segments.
                    if (getattr(cfg, 'DEMO_CACHE_CLEAR_EVERY', 0) > 0
                            and frame_idx > 0
                            and frame_idx % cfg.DEMO_CACHE_CLEAR_EVERY == 0):
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if (getattr(cfg, 'DEMO_FRAG_LOG_EVERY', 0) > 0
                            and frame_idx > 0
                            and frame_idx % cfg.DEMO_FRAG_LOG_EVERY == 0
                            and torch.cuda.is_available()):
                        alloc = torch.cuda.memory_allocated() / 1024**2
                        resv = torch.cuda.memory_reserved() / 1024**2
                        frag = (1 - alloc / resv) if resv > 0 else 0.0
                        print(f"  [mem] frame={frame_idx} alloc={alloc:.1f}MB "
                              f"reserved={resv:.1f}MB frag={frag*100:.1f}%")

                    input_image_path = os.path.join(image_seq_root, img_name)
                    output_mask_path = os.path.join(
                        output_mask_seq_root,
                        img_name.split('.')[0] + '.png')

                    pred_label = Image.fromarray(
                        pred_label.squeeze(0).squeeze(0).cpu().numpy().astype(
                            'uint8')).convert('P')
                    pred_label.putpalette(_palette)
                    pred_label.save(output_mask_path)

                    input_image = Image.open(input_image_path)
                    overlayed_image = overlay(
                        np.array(input_image, dtype=np.uint8),
                        np.array(pred_label, dtype=np.uint8), color_palette)
                    videoWriter.write(overlayed_image[..., [2, 1, 0]])

        print('Save a visualization video to {}.'.format(output_video_path))
        videoWriter.release()


def _install_lt_cap(max_frames: int, batch_evict_every: int = 50,
                    keep_ratio: float = 0.8):
    """Hard cap on long-term memory with batch eviction.

    Replaces ``AOTEngine.update_long_term_memory`` to:
      * Always append the new memory at the head of the concat (newest first).
      * Hard safety cap at ``max_frames * 1.5`` — slice immediately if bank
        grows beyond that (prevents OOM between scheduled batch evictions).
      * Scheduled batch eviction every ``batch_evict_every`` frames once
        the bank exceeds ``max_frames`` — keep newest ``max_frames * keep_ratio``
        frames. Doing this in batches (vs every-frame trimming) is friendlier
        to the CUDA allocator, which is why the linked notes recommend it.

    Implementation-preserving: does not change the matching, encoder, or
    short-term memory; only the eviction policy. Both AOT and DeAOT
    inherit this through ``AOTEngine``.
    """
    from networks.engines.aot_engine import AOTEngine

    stats = {'evict_calls': 0, 'last_log': 0}

    def update_long_term_memory(self, new_long_term_memories):
        if self.long_term_memories is None:
            self.long_term_memories = new_long_term_memories
            return

        # Token count per frame (HW) — uniform across all layers/tensors.
        HW = None
        for layer in new_long_term_memories:
            for e in layer:
                if e is not None:
                    HW = e.shape[0]
                    break
            if HW is not None:
                break

        hard_cap_frames = int(max_frames * 1.5) if max_frames > 0 else 0
        keep_frames = max(1, int(max_frames * keep_ratio)) if max_frames > 0 else 0
        do_batch_evict = (
            max_frames > 0
            and batch_evict_every > 0
            and self.frame_step > 0
            and (self.frame_step % batch_evict_every) == 0
        )

        updated = []
        evicted_this_call = False
        for new_layer, last_layer in zip(new_long_term_memories, self.long_term_memories):
            updated_layer = []
            for new_e, last_e in zip(new_layer, last_layer):
                if new_e is None or last_e is None:
                    updated_layer.append(None)
                else:
                    cat = torch.cat([new_e, last_e], dim=0)
                    if HW is not None and max_frames > 0:
                        T = cat.shape[0] // HW
                        # Hard safety cap to prevent runaway growth between
                        # scheduled evictions (rare; only fires if the run
                        # somehow misses the batch_evict_every cadence).
                        if T > hard_cap_frames:
                            cat = cat[:hard_cap_frames * HW].contiguous()
                            evicted_this_call = True
                        elif do_batch_evict and T > max_frames:
                            cat = cat[:keep_frames * HW].contiguous()
                            evicted_this_call = True
                    updated_layer.append(cat)
            updated.append(updated_layer)
        self.long_term_memories = updated

        if evicted_this_call:
            stats['evict_calls'] += 1
            if (stats['evict_calls'] - stats['last_log']) >= 5 or \
                    stats['evict_calls'] == 1:
                stats['last_log'] = stats['evict_calls']
                print(f"  [LT-cap] batch evict "
                      f"(frame={self.frame_step}, kept={keep_frames}, "
                      f"max={max_frames}, total_evicts={stats['evict_calls']})")

    AOTEngine.update_long_term_memory = update_long_term_memory


# ===========================================================================
# Streaming tracker wrapper — exposes the demo's per-frame flow as a
# track_sequence(frames, initial_bbox/initial_mask) generator so run_video.py
# can drive it like sam2-aotmem. Same engine ops as demo(): frame 0 ->
# add_reference_frame; frame N -> match_propogate_one_frame -> decode ->
# update_memory. The LT-memory cap (_install_lt_cap) is applied too.
# ===========================================================================
from dataclasses import dataclass as _dataclass
from typing import Iterable as _Iterable, Iterator as _Iterator
from typing import Optional as _Optional, Tuple as _Tuple


@_dataclass
class TrackResult:
    """Mirrors the sam2-aotmem result so run_video's viz code is identical."""
    mask: _Optional[np.ndarray]            # (H, W) bool, or None if not visible
    confidence: float
    centroid_uv: _Optional[_Tuple[float, float]]
    is_visible: bool


_CKPT_FOR_MODEL = {
    "r50_deaotl":   "R50_DeAOTL_PRE_YTB_DAV.pth",
    "deaott":       "DeAOTT_PRE_YTB_DAV.pth",
    "swinb_deaotl": "SwinB_DeAOTL_PRE_YTB_DAV.pth",
    "r50_aotl":     "R50_AOTL_PRE_YTB_DAV.pth",
}


class AOTTracker:
    """Streaming AOT/DeAOT tracker built from the demo's exact inference path.

        tracker = AOTTracker(model="r50_deaotl")
        for idx, res in tracker.track_sequence(frames, initial_bbox=box):
            if res.is_visible: ...
    """

    def __init__(self, model="r50_deaotl", stage="pre_ytb_dav", ckpt=None,
                 lt_gap=10, lt_max=80, lt_batch_evict=50, lt_keep_ratio=0.8,
                 max_resolution=480 * 1.3, gpu_id=0, cache_clear_every=200,
                 min_area_ratio=0.0005, verbose=True):
        self.model = model
        self.stage = stage
        self.ckpt = ckpt or str(
            _AOT_DIR / "pretrain_models"
            / _CKPT_FOR_MODEL.get(model, "R50_DeAOTL_PRE_YTB_DAV.pth"))
        self.lt_gap = lt_gap
        self.lt_max = lt_max
        self.lt_batch_evict = lt_batch_evict
        self.lt_keep_ratio = lt_keep_ratio
        self.max_resolution = max_resolution
        self.gpu_id = gpu_id
        self.cache_clear_every = cache_clear_every
        self.min_area_ratio = min_area_ratio
        self.verbose = verbose
        self._engine = None
        self._cfg = None
        self._device = None
        self._transform = None

    def _log(self, m):
        if self.verbose:
            print(f"[AOT] {m}")

    def _build(self):
        if self._engine is not None:
            return
        if self.lt_max > 0:
            _install_lt_cap(self.lt_max, self.lt_batch_evict, self.lt_keep_ratio)
        cfg = importlib.import_module(
            "configs." + self.stage).EngineConfig("video", self.model)
        cfg.TEST_GPU_ID = self.gpu_id
        cfg.TEST_CKPT_PATH = self.ckpt
        cfg.TEST_MIN_SIZE = None
        cfg.TEST_MAX_SIZE = self.max_resolution * 800. / 480.   # demo formula -> 1040
        cfg.TEST_LONG_TERM_MEM_GAP = self.lt_gap
        cfg.DEMO_CONF_THRESH = 0.0
        cfg.DEMO_CACHE_CLEAR_EVERY = self.cache_clear_every
        cfg.DEMO_FRAG_LOG_EVERY = 0
        self._cfg = cfg
        device = get_device(self.gpu_id)
        self._device = device
        gid = self.gpu_id if device.type == "cuda" else -1
        self._log(f"model={self.model} ckpt={self.ckpt} "
                  f"TEST_MAX_SIZE={cfg.TEST_MAX_SIZE:.0f} LT_GAP={self.lt_gap} "
                  f"LT_MAX={self.lt_max}")
        net = build_vos_model(cfg.MODEL_VOS, cfg)
        net = net.cuda(gid) if device.type == "cuda" else net.to(device)
        net, _ = load_network(net, cfg.TEST_CKPT_PATH, gid)
        net.eval()
        self._engine = build_engine(
            cfg.MODEL_ENGINE, phase="eval", aot_model=net, gpu_id=gid,
            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)
        self._transform = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                 cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                 cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor(),
        ])

    def _apply_transform(self, rgb, label):
        """VOSTest-style sample construction + demo transform for one frame."""
        h, w = rgb.shape[:2]
        sample = {"current_img": rgb.astype(np.float32)}
        if label is not None:
            sample["current_label"] = label.astype(np.uint8)
        sample["meta"] = {"seq_name": "video", "frame_num": 0, "obj_num": 1,
                          "current_name": "frame", "height": h, "width": w,
                          "flip": False, "obj_idx": [0, 1]}
        s = self._transform(sample)[0]
        img_t = s["current_img"].unsqueeze(0).float()
        lbl_t = None
        if "current_label" in s and s["current_label"] is not None:
            lbl_t = s["current_label"].unsqueeze(0).float()
        return img_t, lbl_t

    @staticmethod
    def _bbox_to_mask(bbox, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        m = np.zeros((h, w), dtype=np.uint8)
        m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
        return m

    def _result_from_logit(self, logit, h, w):
        prob = torch.softmax(logit, dim=1)
        mask = (torch.argmax(prob, dim=1)[0] == 1).cpu().numpy()
        px = int(mask.sum())
        if px < max(20, int(h * w * self.min_area_ratio)):
            return TrackResult(None, 0.0, None, False)
        fg = prob[0, 1].detach().cpu().numpy()
        ys, xs = np.where(mask)
        ws = fg[ys, xs]
        s = float(ws.sum())
        cuv = ((float((xs * ws).sum() / s), float((ys * ws).sum() / s))
               if s > 0 else (float(xs.mean()), float(ys.mean())))
        conf = float(min(1.0, (px / float(h * w)) * 50.0))
        return TrackResult(mask.astype(bool), conf, cuv, True)

    def track_sequence(self, frames, initial_bbox=None, initial_mask=None):
        """Yield (frame_idx, TrackResult) for each RGB frame. Frame 0 is seeded
        from initial_mask (preferred) or a filled-rect initial_bbox."""
        self._build()
        self._engine.restart_engine()
        engine, device = self._engine, self._device
        with torch.no_grad():
            for frame_idx, rgb in enumerate(frames):
                h, w = rgb.shape[:2]
                if frame_idx == 0:
                    if initial_mask is not None:
                        label = (initial_mask > 0).astype(np.uint8)
                    elif initial_bbox is not None:
                        label = self._bbox_to_mask(initial_bbox, rgb.shape)
                    else:
                        raise ValueError("need initial_bbox or initial_mask")
                    img_t, lbl_t = self._apply_transform(rgb, label)
                    img_t = img_t.to(device)
                    lbl_t = F.interpolate(lbl_t.to(device),
                                          size=img_t.size()[2:], mode="nearest")
                    engine.add_reference_frame(img_t, lbl_t, frame_step=0,
                                               obj_nums=[1])
                    logit = engine.decode_current_logits((h, w))
                    yield 0, self._result_from_logit(logit, h, w)
                    continue

                img_t, _ = self._apply_transform(rgb, None)
                engine.match_propogate_one_frame(img_t.to(device))
                logit = engine.decode_current_logits((h, w))
                res = self._result_from_logit(logit, h, w)
                # write this frame's mask into memory (demo's update_memory)
                pred_label = torch.argmax(torch.softmax(logit, dim=1), dim=1,
                                          keepdim=True).float()
                _pred = F.interpolate(pred_label, size=engine.input_size_2d,
                                      mode="nearest")
                engine.update_memory(_pred)
                yield frame_idx, res

                if (self.cache_clear_every > 0
                        and frame_idx % self.cache_clear_every == 0):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AOT Demo")
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--stage', type=str, default='pre_ytb_dav')
    parser.add_argument('--model', type=str, default='r50_aotl')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='./datasets/Demo')
    parser.add_argument('--output_path', type=str, default='./demo_output')
    parser.add_argument('--ckpt_path', type=str,
                        default='./pretrain_models/R50_AOTL_PRE_YTB_DAV.pth')
    parser.add_argument('--max_resolution', type=float, default=480 * 1.3)
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=False)
    parser.add_argument('--lt_max', type=int, default=80,
                        help='Hard cap on long-term memory entries (frames). '
                             'Recommended 80 for long videos. 0 = unlimited.')
    parser.add_argument('--lt_batch_evict', type=int, default=50,
                        help='Trigger batch eviction every N frames once bank '
                             '> lt_max (keeps newest lt_max*keep_ratio). '
                             'Batched eviction is friendlier to the CUDA '
                             'allocator than per-frame trimming.')
    parser.add_argument('--lt_keep_ratio', type=float, default=0.8,
                        help='When evicting, keep this fraction of lt_max as '
                             'newest entries.')
    parser.add_argument('--lt_gap', type=int, default=10,
                        help='Override cfg.TEST_LONG_TERM_MEM_GAP. Config '
                             'default is 5; 10 is recommended for long videos '
                             '(roughly halves long-term write rate, no '
                             'noticeable accuracy loss in practice).')
    parser.add_argument('--cache_clear_every', type=int, default=200,
                        help='Run gc.collect() + torch.cuda.empty_cache() every '
                             'N frames. 0 = disable.')
    parser.add_argument('--frag_log_every', type=int, default=200,
                        help='Print CUDA allocated/reserved/fragmentation '
                             'every N frames. 0 = disable.')
    parser.add_argument('--conf_thresh', type=float, default=0.0,
                        help='Track-confidence threshold: if mean softmax prob '
                             'over a predicted-object region < thresh, suppress '
                             'that object. 0 = disable (did not help in our '
                             'distractor-heavy test, kept for opt-in).')

    args = parser.parse_args()

    if args.lt_max > 0:
        _install_lt_cap(args.lt_max, args.lt_batch_evict, args.lt_keep_ratio)
        print(f'[LT-cap] enabled: lt_max={args.lt_max}, '
              f'batch_evict_every={args.lt_batch_evict}, '
              f'keep_ratio={args.lt_keep_ratio}')

    engine_config = importlib.import_module('configs.' + args.stage)
    cfg = engine_config.EngineConfig(args.exp_name, args.model)

    cfg.TEST_GPU_ID = args.gpu_id
    cfg.TEST_CKPT_PATH = args.ckpt_path
    cfg.TEST_DATA_PATH = args.data_path
    cfg.TEST_OUTPUT_PATH = args.output_path
    cfg.TEST_MIN_SIZE = None
    cfg.TEST_MAX_SIZE = args.max_resolution * 800. / 480.
    cfg.DEMO_CONF_THRESH = args.conf_thresh
    if args.conf_thresh > 0:
        print(f'[CONF] enabled: threshold={args.conf_thresh}')
    if args.lt_gap > 0:
        print(f'[lt_gap] override: {cfg.TEST_LONG_TERM_MEM_GAP} → {args.lt_gap}')
        cfg.TEST_LONG_TERM_MEM_GAP = args.lt_gap
    cfg.DEMO_CACHE_CLEAR_EVERY = args.cache_clear_every
    cfg.DEMO_FRAG_LOG_EVERY = args.frag_log_every
    print(f'[allocator] PYTORCH_CUDA_ALLOC_CONF='
          f'{os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(unset)")}')

    # AMP is CUDA-only; skip gracefully on CPU
    use_amp = args.amp and torch.cuda.is_available()
    if args.amp and not torch.cuda.is_available():
        print('WARNING: --amp requested but CUDA not available; running without AMP.')

    if use_amp:
        with torch.amp.autocast(device_type='cuda', enabled=True):
            demo(cfg)
    else:
        demo(cfg)

if __name__ == '__main__':
    try:
        import spatial_correlation_sampler
    except Exception as inst:
        print(inst)
        print(" For better efficiency, please install it.")
    main()
    