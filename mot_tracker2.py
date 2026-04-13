import cv2
import sys
import argparse
import os
import torch
import numpy as np
import time
from ultralytics import YOLO
from samba_wrapper import SambaMOTRTracker

def get_youtube_cap(url, quality='720p'):
    try:
        import yt_dlp
        ydl_opts = {
            'format': f'best[height<={quality[:-1]}][vcodec^=avc1]',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return cv2.VideoCapture(info['url'])
    except Exception as e:
        print(f"Error resolving YouTube URL: {e}")
        return None

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def main():
    parser = argparse.ArgumentParser(description="Hybrid YOLOv11 + SambaMOTR Tracker")
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--quality", type=str, default="720p", help="YouTube quality")
    parser.add_argument("--model_path", type=str, default="pretrained/sambamotr/sambamotr_dancetrack.pth")
    parser.add_argument("--config_path", type=str, default="sambamotr/configs/sambamotr/dancetrack/def_detr/train_residual_masking_sync_longer.yaml")
    parser.add_argument("--yolo_model", type=str, default="yolo11m.pt", help="Ultralytics YOLO model")
    parser.add_argument("--output", type=str, default=None, help="Save output to MP4 file")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--duration", type=int, default=0, help="Stop after N seconds")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    print(f"Initializing YOLOv11 Detector and SambaMOTR Tracker on {device}...")
    # Load YOLOv11 model (will download automatically)
    yolo_model = YOLO(args.yolo_model).to(device)
    tracker = SambaMOTRTracker(args.model_path, args.config_path, device=device)

    cap = None
    if args.source.startswith('http'):
        cap = get_youtube_cap(args.source, args.quality)
    else:
        try: source = int(args.source)
        except: source = args.source
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        return

    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    start_time = time.time()
    seen_ids = set()
    print("Starting Hybrid Tracking Loop (YOLOv11 + SambaMOTR)...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        if args.duration > 0 and (time.time() - start_time) > args.duration:
            break

        # 1. Primary Detections (YOLOv11)
        yolo_results = yolo_model.predict(frame, classes=[0], conf=0.25, device=device, verbose=False) # Lowered conf for safety
        yolo_bboxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        # 2. SambaMOTR Tracking
        samba_bboxes, samba_ids = tracker.update(frame)
        
        if frame_count < 10:
            print(f"\n[Frame {frame_count}] YOLO found {len(yolo_bboxes)}, Samba found {len(samba_bboxes)}")
            if len(samba_bboxes) > 0:
                print(f"  Samba Box 0: {samba_bboxes[0]}")
            if len(yolo_bboxes) > 0:
                print(f"  YOLO Box 0: {yolo_bboxes[0]}")

        final_draw_bboxes = []
        final_draw_ids = []
        
        for i in range(len(samba_bboxes)):
            sb = samba_bboxes[i]
            sid = int(samba_ids[i])
            
            if sid in seen_ids:
                final_draw_bboxes.append(sb)
                final_draw_ids.append(sid)
            else:
                is_verified = False
                for yb in yolo_bboxes:
                    iou = calculate_iou(sb, yb)
                    if frame_count < 10:
                        print(f"    Check SID {sid} against YOLO box: IoU={iou:.4f}")
                    if iou > 0.01: # Extremely lenient for debugging
                        is_verified = True
                        break
                
                if is_verified:
                    seen_ids.add(sid)
                    final_draw_bboxes.append(sb)
                    final_draw_ids.append(sid)

        # 4. Rendering
        # ALWAYS draw a small dot in corner to verify frame is being processed
        cv2.circle(frame, (10, 10), 5, (0, 255, 0), -1)

        for bbox, obj_id in zip(final_draw_bboxes, final_draw_ids):
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # DEBUG: Also draw YOLO boxes in red if nothing is showing
        if len(final_draw_bboxes) == 0:
            for yb in yolo_bboxes:
                x1, y1, x2, y2 = map(int, yb[:4])
                cv2.rectangle(frame, (0, 0, 255), (x1, y1), (x2, y2), 1) # Red for raw YOLO

        if writer:
            writer.write(frame)
            
        if not args.no_display:
            cv2.imshow("Hybrid YOLOv11 + SambaMOTR", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {args.output}")

if __name__ == "__main__":
    main()
