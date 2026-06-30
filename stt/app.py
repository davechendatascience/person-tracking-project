#!/usr/bin/env python3
"""Voice/text -> multimodal LLM -> target person bbox on a frame.

The simple end-to-end app: pick a frame (image or video), describe who to
track (typed with --text, or spoken with --voice), and the multimodal LLM
returns the target's bounding box. The box is printed and drawn onto an output
image — and is exactly what seeds the tracker in run_video.

Examples:
  # typed description on a still image
  python -m stt.app --image frame.jpg --text "the person in the red shirt"

  # typed description on a video frame
  python -m stt.app --video videos/798511637.725509.mp4 --frame 0 \
      --text "追蹤穿紅色衣服的人"

  # speak the description (USB mic + faster-whisper)
  python -m stt.app --video videos/798511637.725509.mp4 --frame 0 --voice
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stt.target_resolver import resolve_target


def _load_frame(args) -> np.ndarray:
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"could not read image: {args.image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if args.video:
        from crowdbot.dataset import VideoSequence
        seq = VideoSequence(args.video)
        fd = seq[args.frame]
        if fd.image is None:
            raise SystemExit(f"could not read frame {args.frame} of {args.video}")
        return fd.image  # RGB
    raise SystemExit("provide --image or --video")


def _get_description(args) -> str:
    if args.text:
        return args.text
    if args.voice:
        from stt.speech_to_text import SpeechToText
        result = {}
        stt = SpeechToText(status_cb=lambda m: print(m, flush=True))
        print(">>> Speak the target description (one phrase)…")

        def _on_text(text, lang):
            result["text"] = text
            result["lang"] = lang
            stt.stop()
        stt.run(_on_text)
        return result.get("text", "")
    raise SystemExit("provide --text or --voice")


def main():
    ap = argparse.ArgumentParser(description="Voice/text -> LLM -> target bbox")
    src = ap.add_argument_group("frame source")
    src.add_argument("--image", type=str, help="path to a still image")
    src.add_argument("--video", type=str, help="path to a video")
    src.add_argument("--frame", type=int, default=0, help="frame index for --video")
    desc = ap.add_argument_group("target description")
    desc.add_argument("--text", type=str, help="typed description of who to track")
    desc.add_argument("--voice", action="store_true", help="speak the description (mic)")
    ap.add_argument("--out", type=str, default="stt_target.jpg",
                    help="output image with the bbox drawn")
    args = ap.parse_args()

    frame_rgb = _load_frame(args)
    description = _get_description(args)
    if not description.strip():
        raise SystemExit("empty description")
    print(f"[target] description: {description!r}")

    bbox = resolve_target(frame_rgb, description)
    if bbox is None:
        print("[target] LLM found no matching person.")
        return
    x1, y1, x2, y2 = bbox
    print(f"[target] bbox = [{x1}, {y1}, {x2}, {y2}]  (feed this to the tracker seed)")

    vis = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(vis, description[:40], (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(args.out, vis)
    print(f"[target] wrote {args.out}")


if __name__ == "__main__":
    main()
