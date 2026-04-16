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
import os
import torch

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
    parser.add_argument("--show", action="store_true", help="Show real-time visualization window")
    parser.add_argument("--out-video", type=str, default=None, help="Designated output video file path")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Process every Nth frame (e.g. --frame-stride 2 halves frames and ~doubles throughput)")
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

    # 4. Producer-Consumer Pipeline Setup
    import threading
    import time
    from follow_everything.perception.sam2_tracker import StreamingFrameLoader

    if os.path.exists("/dev/shm"):
        base_temp_dir = Path("/dev/shm")
    else:
        base_temp_dir = Path(tempfile.gettempdir())
    
    temp_frames_dir = base_temp_dir / f"sam2_{os.getpid()}_{int(time.time())}"
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    _all_frames = list(range(args.start_frame, min(args.start_frame + args.num_frames, len(seq))))
    frames_to_track = _all_frames[::args.frame_stride]
    total_expected = len(frames_to_track)
    img_paths = [temp_frames_dir / f"{i:06d}.jpg" for i in range(total_expected)]

    extraction_done = threading.Event()
    stop_extraction = threading.Event()
    
    def extraction_worker():
        print(f"[INFO] Extraction thread started.")
        for i, frame_idx in enumerate(frames_to_track):
            if stop_extraction.is_set(): break
            fd = seq[frame_idx]
            if fd.image is not None:
                # Optimized write: write to tmp then rename for atomicity
                tmp_path = temp_frames_dir / f"{i:06d}_tmp.jpg"
                cv2.imwrite(str(tmp_path), cv2.cvtColor(fd.image, cv2.COLOR_RGB2BGR))
                tmp_path.replace(temp_frames_dir / f"{i:06d}.jpg")
        extraction_done.set()
        print("[INFO] Extraction thread finished.")

    extractor_thread = threading.Thread(target=extraction_worker, daemon=True)
    extractor_thread.start()

    # 5. Run & Pipeline Tracking + Visualization + Video Export
    import queue as _queue

    if args.out_video:
        video_out_path = Path(args.out_video)
        video_out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_out_path = output_dir / "tracking_result.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Async write thread: offloads cv2.imwrite + H.264 encoding off SAM2's critical path
    _write_q = _queue.Queue(maxsize=16)

    def _write_worker():
        _vw = None
        while True:
            item = _write_q.get()
            if item is None:
                break
            frame_img, orig_frame_num = item
            cv2.imwrite(str(viz_dir / f"{orig_frame_num:06d}.jpg"), frame_img)
            if _vw is None:
                h, w = frame_img.shape[:2]
                _vw = cv2.VideoWriter(str(video_out_path), fourcc, 30.0, (w, h))
            _vw.write(frame_img)
            _write_q.task_done()
        if _vw:
            _vw.release()

    _write_thread = threading.Thread(target=_write_worker, daemon=False)
    _write_thread.start()

    try:
        # Initialize the Streaming Loader (will block on index access if file not ready)
        loader = StreamingFrameLoader(
            img_paths=img_paths,
            image_size=tracker._scfg.get("image_size", 1024),
            device=tracker._scfg["device"]
        )

        print("[INFO] Starting synchronized tracking & visualization...")
        tracker_gen = tracker.track_sequence(
            loader,
            np.array(target_box),
            allow_reprompt=not args.no_reprompt,
            max_reprompts=args.max_reprompts
        )

        for i, (f_idx, res) in enumerate(tqdm(tracker_gen, total=total_expected, desc="Pipelined Tracking")):
            if f_idx >= len(img_paths): continue
            # Load frame for visualization
            frame_path = img_paths[f_idx]
            viz_img = cv2.imread(str(frame_path))
            if viz_img is None: continue

            # Apply Visualization
            if res.is_visible and res.mask is not None:
                mask = res.mask
                colored_mask = np.zeros_like(viz_img)
                colored_mask[mask] = [0, 0, 255] # Red in BGR
                cv2.addWeighted(viz_img, 1.0, colored_mask, 0.4, 0, viz_img)
                if res.centroid_uv:
                    cx, cy = map(int, res.centroid_uv)
                    cv2.drawMarker(viz_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            if i == 0:
                x1, y1, x2, y2 = map(int, target_box)
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(viz_img, "Target Identified", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(viz_img, f"Frame {frames_to_track[f_idx]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Show Window (Optional — must stay on main thread)
            if getattr(args, 'show', False):
                cv2.imshow("Follow Everything Real-Time", viz_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    args.show = False
                    stop_extraction.set()

            # Hand off disk writes to background thread
            _write_q.put((viz_img, frames_to_track[f_idx]))

    finally:
        _write_q.put(None)  # signal writer to flush and exit
        _write_thread.join()
        stop_extraction.set()
        if 'extractor_thread' in locals():
            print("[INFO] Waiting for extraction thread to finish...")
            extractor_thread.join(timeout=5.0)
        cv2.destroyAllWindows()

        # Clean up temp frames
        if 'temp_frames_dir' in locals() and temp_frames_dir.exists():
            shutil.rmtree(temp_frames_dir)

    print(f"[INFO] Processing finished. Results saved to {output_dir}")

if __name__ == "__main__":
    run_video_tracking()
