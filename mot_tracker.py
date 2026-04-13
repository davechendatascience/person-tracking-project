import cv2
import sys
import argparse
import numpy as np
import torch
import yt_dlp
from rtmlib import Wholebody, draw_skeleton

# Import SambaMOTR Wrapper
try:
    from samba_wrapper import SambaMOTRTracker
except ImportError as e:
    print(f"Error importing SambaMOTRTracker: {e}")
    sys.exit(1)

def get_youtube_cap(url, quality='best'):
    """
    Get a cv2.VideoCapture object from a YouTube URL while bypassing AV1 codec.
    """
    print(f"Extracting compatible YouTube stream: {url}")
    ydl_opts = {
        'format': f'best[vcodec^=avc1]/best[vcodec!=av01]/best',
        'quiet': True,
        'no_warnings': True,
    }
    if quality != 'best' and quality != 'worst':
        res = quality.replace('p', '')
        ydl_opts['format'] = f'best[height<={res}][vcodec^=avc1]/best[height<={res}][vcodec!=av01]/best'

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info['url']
            return cv2.VideoCapture(stream_url)
    except Exception as e:
        print(f"Error extracting YouTube stream: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="rtmlib + SambaMOTR Multi-Object Tracking")
    parser.add_argument("--source", type=str, default="0", help="Video source (e.g. video.mp4, or 0 for webcam)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda, cpu)")
    parser.add_argument("--quality", type=str, default="720p", help="Max video quality for YouTube")
    parser.add_argument("--model_path", type=str, default="pretrained/sambamotr/sambamotr_dancetrack.pth", help="SambaMOTR weights")
    parser.add_argument("--config_path", type=str, default="sambamotr/configs/sambamotr/dancetrack/def_detr/train_residual_masking_sync_longer.yaml", help="SambaMOTR config")
    parser.add_argument("--output", type=str, default=None, help="Save output to MP4 file (e.g. output.mp4)")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow window (headless mode)")
    parser.add_argument("--duration", type=int, default=0, help="Stop after N seconds (0 = unlimited)")
    args = parser.parse_args()

    # Normalize device
    device = 'cuda' if args.device == 'gpu' else args.device

    # Handle YouTube Source
    if isinstance(args.source, str) and (args.source.startswith('http') or args.source.startswith('www')):
        from urllib.parse import urlparse, parse_qs, urlunparse
        url = args.source
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc and '/watch' in parsed_url.path:
            query = parse_qs(parsed_url.query)
            if 'v' in query:
                new_query = f"v={query['v'][0]}"
                url = urlunparse(parsed_url._replace(query=new_query))
        cap = get_youtube_cap(url, args.quality)
    else:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        cap = cv2.VideoCapture(source)

    if cap is None or not cap.isOpened():
        print(f"Error opening video source: {args.source}")
        sys.exit(1)

    # Set up video writer if --output specified
    writer = None
    if args.output:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output} ({width}x{height} @ {fps:.1f} fps)")

    print(f"Initializing RTMLib (Pose) and SambaMOTR (Tracking) on {device}...")
    try:
        pose_model = Wholebody(device=device, backend='onnxruntime').pose_model
    except Exception as e:
        print(f"Error initializing RTMLib: {e}. Falling back to CPU...")
        pose_model = Wholebody(device='cpu', backend='onnxruntime').pose_model

    # Initialize SambaMOTR
    try:
        tracker = SambaMOTRTracker(args.model_path, args.config_path, device=device)
    except Exception as e:
        print(f"Error initializing SambaMOTR: {e}")
        sys.exit(1)

    print("Starting video loop, press 'q' to quit.")
    
    import time
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check duration
        if args.duration > 0 and (time.time() - start_time) > args.duration:
            print(f"Duration {args.duration}s reached. Stopping.")
            break

        # 1. SambaMOTR Detection & Tracking
        tracked_bboxes, track_ids = tracker.update(frame)

        if len(tracked_bboxes) > 0:
            # 2. Pose Estimation on tracked boxes
            keypoints, scores = pose_model(frame, bboxes=tracked_bboxes)

            # 3. Visualization
            frame = draw_skeleton(frame, keypoints, scores)

            for i, kpts in enumerate(keypoints):
                x, y = int(kpts[0, 0]), int(kpts[0, 1])
                cv2.putText(frame, f"ID: {track_ids[i]}", (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow('rtmlib + SambaMOTR', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {args.output}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

