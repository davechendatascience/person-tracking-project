import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import tempfile
import shutil
import yaml

from crowdbot.dataset import VideoSequence
from follow_everything.perception.sam2_tracker import SAM2Tracker

def identify_person_by_color(image_rgb, color='red', model_path='yolo11m.pt'):
    """Find the person matching the target color."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    if color == 'red':
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        target_mask = mask1 | mask2
    elif color == 'black':
        # Black: low value (brightness)
        target_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
        
        # Also create a mask for yellow to penalize/exclude
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    elif color == 'any':
        target_mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
    else:
        # Default to neutral mask if color not supported
        target_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    model = YOLO(model_path)
    results = model(image_bgr, verbose=False)
    
    persons = []
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0: # person
                persons.append(box.xyxy[0].cpu().numpy())
    
    print(f"Found {len(persons)} people candidates.")
    
    if not persons:
        return None
    
    best_box = None
    max_score = -1
    
    img_h, img_w = image_rgb.shape[:2]
    img_center_x = img_w / 2
    
    for i, box in enumerate(persons):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        
        roi_mask = target_mask[y1:y2, x1:x2]
        if roi_mask.size == 0: continue
        
        color_ratio = np.mean(roi_mask) / 255.0
        
        # Penalty for yellow (if searching for black)
        yellow_penalty = 1.0
        if color == 'black':
            y_roi = yellow_mask[y1:y2, x1:x2]
            y_ratio = np.mean(y_roi) / 255.0
            yellow_penalty = max(0.1, 1.0 - y_ratio * 5.0) # Strong penalty for yellow
            
        # Combine color ratio with box area
        area = (x2 - x1) * (y2 - y1)
        
        # Center bias: favor targets near the horizontal center
        box_center_x = (x1 + x2) / 2
        center_dist = abs(box_center_x - img_center_x) / img_w
        center_bias = 1.0 - center_dist 
        
        score = color_ratio * np.log1p(area) * center_bias * yellow_penalty
        
        print(f"  Candidate {i}: box={box}, color_ratio={color_ratio:.3f}, yellow_ratio={y_ratio if color=='black' else 0:.3f}, score={score:.3f}")
        
        if score > max_score:
            max_score = score
            best_box = [x1, y1, x2, y2]
            
    return best_box

def run_video_tracking():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="video-data-mot/MOT17-11-SDP-raw.webm")
    parser.add_argument("--output", type=str, default="results/video_mot17_11")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--yolo-model", type=str, default="yolo11m.pt")
    parser.add_argument("--target-color", type=str, default="red", choices=["red", "black", "any"])
    parser.add_argument("--no-reprompt", action="store_true", help="Disable re-propagation passes")
    parser.add_argument("--max-reprompts", type=int, default=2, help="Max number of re-propagation passes")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Dataset
    seq = VideoSequence(args.video)
    
    # 2. Setup Config & SAM2
    config_path = Path("configs/follow_everything.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        # Fallback minimal config
        cfg = {
            "sam2": {"device": "cuda" if torch.cuda.is_available() else "cpu", 
                     "checkpoint": "sam2.1_hiera_base_plus.pt", 
                     "model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml"},
            "perception": {"temporal_buffer_size": 10, "min_mask_confidence": 0.4, 
                           "min_mask_area_ratio": 0.0005, "distance_bin_width": 1.0}
        }
    
    # Override with relevant bits for the demo if needed
    cfg["sam2"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    tracker = SAM2Tracker(cfg["perception"], cfg["sam2"])
    
    # 3. Automatically find the "guy in red" in the first frame
    first_frame_data = seq[args.start_frame]
    if first_frame_data.image is None:
        print("[ERROR] Could not load the first frame.")
        return

    print(f"[INFO] Identifying the person in {args.target_color} shirt...")
    target_box = identify_person_by_color(first_frame_data.image, args.target_color, args.yolo_model)
    
    if target_box is None:
        print("[ERROR] Could not find a person in red.")
        return
    
    print(f"[INFO] Found target at {target_box}. Starting tracking...")

    # 4. Prepare sequence for SAM2
    frames_to_track = list(range(args.start_frame, min(args.start_frame + args.num_frames, len(seq))))
    image_seq = []
    for i in tqdm(frames_to_track, desc="Loading Frames"):
        img = seq[i].image
        if img is not None:
            image_seq.append(img)
    
    # 5. Run SAM2 Tracking
    # SAM2 init_state expects a directory of JPGs
    temp_frames_dir = Path(tempfile.mkdtemp())
    try:
        print(f"[INFO] Preparing temporary frames in {temp_frames_dir}...")
        for i, img in enumerate(image_seq):
            # SAM2 expects frames to be named in a way it can sort
            cv2.imwrite(str(temp_frames_dir / f"{i:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        prompts = {0: [target_box]} 
        results = tracker.track_sequence(
            temp_frames_dir, 
            np.array(target_box), 
            allow_reprompt=not args.no_reprompt,
            max_reprompts=args.max_reprompts
        )
    finally:
        # We'll keep it for now for debugging if needed, but shutil.rmtree(temp_frames_dir) is better
        pass
    
    # 6. Visualization (Faster with CV2)
    print("[INFO] Saving visualization frames...")
    for i, res in enumerate(tqdm(results.values() if isinstance(results, dict) else results, desc="Visualizing")):
        frame_idx = frames_to_track[i]
        img_rgb = image_seq[i]
        
        # Convert RGB to BGR for CV2
        viz_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        if res.is_visible and res.mask is not None:
            mask = res.mask
            # Draw red mask overlay efficiently
            colored_mask = np.zeros_like(viz_img)
            colored_mask[mask] = [0, 0, 255] # Red in BGR
            cv2.addWeighted(viz_img, 1.0, colored_mask, 0.4, 0, viz_img)
            
            if res.centroid_uv:
                cx, cy = map(int, res.centroid_uv)
                cv2.drawMarker(viz_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        # Draw initial target box on first frame
        if i == 0:
            x1, y1, x2, y2 = map(int, target_box)
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(viz_img, "Target Identified", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(viz_img, f"Frame {frame_idx}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imwrite(str(viz_dir / f"{frame_idx:06d}.jpg"), viz_img)

    # 7. Compile to Video
    video_out_path = output_dir / "tracking_result.mp4"
    if len(image_seq) > 0:
        h, w = image_seq[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, 30.0, (w, h))
        
        print(f"[INFO] Compiling frames into video: {video_out_path}")
        for i in range(len(image_seq)):
            frame_path = viz_dir / f"{frames_to_track[i]:06d}.jpg"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                video_writer.write(frame)
        video_writer.release()

    print(f"[INFO] Finished. Results saved to {output_dir}")

if __name__ == "__main__":
    run_video_tracking()
