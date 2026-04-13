import sys
import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F

# Add sambamotr to path
SAMBA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sambamotr')
sys.path.append(SAMBA_ROOT)

from models import build_model
from models.utils import load_checkpoint
from models.runtime_tracker import RuntimeTracker
from utils.utils import yaml_to_dict
from utils.nested_tensor import tensor_list_to_nested_tensor
from structures.track_instances import TrackInstances
from utils.box_ops import box_cxcywh_to_xyxy

class SambaMOTRTracker:
    def __init__(self, model_path, config_path, device='cuda'):
        self.device = device
        self.config = yaml_to_dict(config_path)
        self.model = build_model(self.config)
        load_checkpoint(self.model, model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tracker = RuntimeTracker(
            det_score_thresh=0.4, 
            track_score_thresh=0.4,
            miss_tolerance=30,
            use_motion=False,
            use_dab=self.config["USE_DAB"],
        )
        
        self.tracks = [TrackInstances(
            hidden_dim=self.model.hidden_dim,
            num_classes=self.model.num_classes,
            state_dim=getattr(self.model.query_updater, "state_dim", 0),
            expand=getattr(self.model.query_updater, "expand", 0),
            num_layers=getattr(self.model.query_updater, "num_layers", 0),
            conv_dim=getattr(self.model.query_updater, "conv_dim", 0),
            use_dab=self.config["USE_DAB"]
        ).to(self.device)]
        
        self.result_score_thresh = 0.4

    def _process_image(self, image):
        h, w = image.shape[:2]
        scale = 800 / min(h, w)
        if max(h, w) * scale > 1536:
            scale = 1536 / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image_resized = cv2.resize(image, (target_w, target_h))
        # BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_tensor = F.normalize(F.to_tensor(image_rgb), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image_tensor, h, w

    @torch.no_grad()
    def update(self, frame):
        image_tensor, ori_h, ori_w = self._process_image(frame)
        frame_nested = tensor_list_to_nested_tensor([image_tensor]).to(self.device)
        
        # Inference
        res = self.model(frame=frame_nested, tracks=self.tracks)
        previous_tracks, new_tracks = self.tracker.update(
            model_outputs=res,
            tracks=self.tracks
        )
        
        # Postprocess
        self.tracks = self.model.postprocess_single_frame(previous_tracks, new_tracks, None, intervals=[1])
        
        # Extract results
        tracks_result = self.tracks[0].to("cpu")
        
        # Filter by score
        keep = torch.max(tracks_result.scores, dim=-1).values > self.result_score_thresh
        tracks_result = tracks_result[keep]
        
        # Convert boxes [cx, cy, w, h] -> [x1, y1, x2, y2] in pixel coordinates
        boxes = box_cxcywh_to_xyxy(tracks_result.boxes)
        boxes = boxes * torch.as_tensor([ori_w, ori_h, ori_w, ori_h], dtype=torch.float)
        
        return boxes.numpy(), tracks_result.ids.numpy()
