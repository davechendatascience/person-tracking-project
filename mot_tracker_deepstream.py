#!/usr/bin/env python3
import sys
import argparse
import numpy as np
import cv2
import gi
import yt_dlp

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import pyds

# Try to import RTMLib
try:
    from rtmlib import Wholebody, draw_skeleton
except ImportError:
    print("RTMLib is not installed.")
    sys.exit(1)

# Try to import SAM2 (optional — only needed for mask overlay)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

pose_model = None   # Global RTMLib pose model
sam2_predictor = None  # Global SAM2 predictor (optional)

# Distinct colors per track ID for mask overlays
ID_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

def get_id_color(track_id):
    return ID_COLORS[int(track_id) % len(ID_COLORS)]

def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Get NvDs metadata (still travels through the pipeline even in system memory)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # Get frame dimensions from the pad's current caps
    caps = pad.get_current_caps()
    if not caps:
        return Gst.PadProbeReturn.OK
    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")

    # Map the system-memory buffer directly (avoids pyds.get_nvds_buf_surface crash)
    success, map_info = gst_buffer.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
    if not success:
        return Gst.PadProbeReturn.OK

    try:
        # Build an RGBA numpy view of the raw buffer bytes
        n_frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 4).copy()
        frame_bgr = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2BGR)

        # Collect tracked bboxes and IDs from NvDs metadata
        tracked_bboxes = []
        track_ids = []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                if obj_meta.class_id == args.track_class_id:  # configurable class filter
                    r = obj_meta.rect_params
                    tracked_bboxes.append([r.left, r.top, r.left + r.width, r.top + r.height])
                    track_ids.append(obj_meta.object_id)
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        # Run SAM2 mask overlay if available
        if sam2_predictor is not None and tracked_bboxes:
            sam2_predictor.set_image(frame_bgr)
            for i, bbox in enumerate(tracked_bboxes):
                masks, scores, _ = sam2_predictor.predict(
                    box=np.array(bbox, dtype=np.float32),
                    multimask_output=False
                )
                mask = masks[0]  # (H, W) bool
                color = get_id_color(track_ids[i])
                overlay = frame_bgr.copy()
                overlay[mask] = (
                    np.array(frame_bgr[mask], dtype=np.float32) * 0.5
                    + np.array(color, dtype=np.float32) * 0.5
                ).astype(np.uint8)
                frame_bgr = overlay

        # Run RTMLib pose estimation and draw skeletons
        if tracked_bboxes and pose_model is not None:
            bboxes_np = np.array(tracked_bboxes, dtype=np.float32)
            keypoints, scores = pose_model(frame_bgr.copy(), bboxes=bboxes_np)
            frame_bgr = draw_skeleton(frame_bgr, keypoints, scores, kpt_thr=0.4)
            for i, kpts in enumerate(keypoints):
                if len(kpts) > 0:
                    x, y = int(kpts[0, 0]), int(kpts[0, 1])
                    cv2.putText(frame_bgr, f"ID:{track_ids[i]}", (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the drawn frame back into the buffer memory
        result_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        map_info.data[:] = result_rgba.tobytes()
    finally:
        gst_buffer.unmap(map_info)

    return Gst.PadProbeReturn.OK
def resolve_source(source, quality='720p'):
    """Resolve a YouTube URL to a direct stream URL, or return the original path."""
    if 'youtube.com' in source or 'youtu.be' in source:
        print(f"Extracting YouTube stream URL from: {source}")
        res = quality.replace('p', '')
        ydl_opts = {
            'format': f'best[height<={res}][vcodec^=avc1]/best[vcodec^=avc1]/best[vcodec!=av01]/best',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=False)
            url = info['url']
            print(f"Resolved to stream: {url[:80]}...")
            return url
    return source


def main(args):
    # Resolve YouTube URLs to direct stream URLs
    resolved_uri = resolve_source(args.source, args.quality)

    global pose_model

    Gst.init(None)

    # Initialize rtmlib pose model on specified device
    print(f"Initializing RTMLib on {args.device}...")
    pose_model = Wholebody(device=args.device, backend='onnxruntime').pose_model
    # Initialize SAM2 (optional) — use HuggingFace Hub loading to avoid Hydra config issues
    if args.sam2_checkpoint and SAM2_AVAILABLE:
        print(f"Loading SAM2 from {args.sam2_checkpoint} on {args.sam2_device}...")
        global sam2_predictor
        sam2_predictor = SAM2ImagePredictor.from_pretrained(
            args.sam2_checkpoint, device=args.sam2_device
        )
        print("SAM2 ready.")
    elif args.sam2_checkpoint and not SAM2_AVAILABLE:
        print("SAM2 not installed — skipping mask overlay.")

    print("Creating Pipeline...")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline\n")
        sys.exit(1)

    # Create streamux
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux\n")
        sys.exit(1)
    
    # Set streammux properties
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)

    # Source — uridecodebin handles both file:// and rtsp:// / http://
    source = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not source:
        sys.stderr.write(" Unable to create uridecodebin\n")
        sys.exit(1)

    # Normalise the URI (prepend file:// for local paths)
    uri = resolved_uri
    if not uri.startswith("rtsp://") and not uri.startswith("http://") \
            and not uri.startswith("https://") and not uri.startswith("file://"):
        uri = f"file://{uri}"
    source.set_property("uri", uri)

    def cb_newpad(decodebin, decoder_src_pad, data):
        caps = decoder_src_pad.get_current_caps()
        if not caps:
            return
        gstname = caps.get_structure(0).get_name()
        if "video" in gstname:
            sinkpad = streammux.get_request_pad("sink_0")
            if sinkpad and not sinkpad.is_linked():
                decoder_src_pad.link(sinkpad)

    source.connect("pad-added", cb_newpad, streammux)


    # Setup the rest of pipeline: pgie -> tracker -> nvvidconv(NVMM->sys RGBA) -> capsfilter -> nvdsosd -> sink
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    # First converter: NVMM GPU -> system RGBA (needed for pyds numpy access)
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    # Capsfilter: system memory RGBA so get_nvds_buf_surface can map it to numpy
    nvvidconv_caps = Gst.ElementFactory.make("capsfilter", "nvvidconv_caps")
    nvvidconv_caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
    # Second converter: system RGBA -> NVMM for nvosd rendering
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    if args.output:
        # File output: nvvideoconvert -> H264 encoder -> mp4mux -> filesink
        enc_conv = Gst.ElementFactory.make("nvvideoconvert", "enc_convertor")
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "h264-encoder")
        h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
        mp4mux = Gst.ElementFactory.make("mp4mux", "mux")
        sink = Gst.ElementFactory.make("filesink", "file-output")
        if not enc_conv or not encoder or not h264parse or not mp4mux or not sink:
            sys.stderr.write(" Unable to create file output elements\n")
            sys.exit(1)
        sink.set_property("location", args.output)
        sink.set_property("sync", False)
    elif args.no_display:
        enc_conv = encoder = h264parse = mp4mux = None
        sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
    else:
        enc_conv = encoder = h264parse = mp4mux = None
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    
    if not pgie or not tracker or not nvvidconv or not nvvidconv_caps or not nvvidconv2 or not nvosd or not sink:
        sys.stderr.write(" Unable to create all elements\n")
        sys.exit(1)

    # --- Properties ---
    # Primary GIE config
    pgie.set_property('config-file-path', args.pgie_config)
    
    # Tracker config
    # E.g. using NvDCF: ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
    # ll-config-file=config_tracker_NvDCF_perf.yml
    # tracker-width=640 tracker-height=384
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', args.tracker_config)

    # --- Pipeline linkages ---
    print("Adding elements to Pipeline")
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvvidconv_caps)
    pipeline.add(nvvidconv2)
    pipeline.add(nvosd)
    if args.output:
        pipeline.add(enc_conv)
        pipeline.add(encoder)
        pipeline.add(h264parse)
        pipeline.add(mp4mux)
    pipeline.add(sink)

    print("Linking elements in Pipeline")
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvvidconv_caps)
    nvvidconv_caps.link(nvvidconv2)
    nvvidconv2.link(nvosd)
    if args.output:
        nvosd.link(enc_conv)
        enc_conv.link(encoder)
        encoder.link(h264parse)
        h264parse.link(mp4mux)
        mp4mux.link(sink)
    else:
        nvosd.link(sink)

    # --- Attach Probe ---
    # IMPORTANT: probe must be on nvvidconv2 SINK pad — that's where the buffer is still
    # in system (CPU-accessible) memory from the capsfilter. Probing nvosd.sink would give
    # GPU/NVMM memory which get_nvds_buf_surface cannot map.
    probe_pad = nvvidconv2.get_static_pad("sink")
    if not probe_pad:
        sys.stderr.write(" Unable to get sink pad of nvvidconv2 \n")
    else:
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
        print("Successfully attached probe for RTMLib.")

    # Start loop
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    
    def bus_call(bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err}: {debug}")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()
        return True

    bus.connect("message", bus_call, loop)

    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)

    # Optional duration limit
    if args.duration > 0:
        print(f"Will stop after {args.duration} seconds.")
        def stop_pipeline():
            print("Duration reached — sending EOS.")
            pipeline.send_event(Gst.Event.new_eos())
            return False  # Don't repeat
        GObject.timeout_add_seconds(args.duration, stop_pipeline)

    try:
        loop.run()
    except:
        pass

    print("Exiting pipeline")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepStream + RTMLib Tracker Script')
    parser.add_argument('--source', required=True, help='Video file, RTSP/HTTP URI, or YouTube URL')
    parser.add_argument('--quality', default='720p', help='YouTube stream quality (e.g. 720p, 1080p)')
    parser.add_argument('--device', default='cuda', help='Device for RTMLib (e.g. cuda, cpu)')
    parser.add_argument('--pgie_config', default='dstest2_pgie_config.txt', help='Path to nvinfer config file')
    parser.add_argument('--tracker_config', default='dstest2_tracker_config.txt', help='Path to nvtracker config file')
    parser.add_argument('--no-display', action='store_true', help='Disable display (fakesink)')
    parser.add_argument('--output', type=str, default=None, help='Save output to MP4 file (e.g. output.mp4)')
    parser.add_argument('--track-class-id', type=int, default=0,
                        help='Only track objects of this class ID (default: 0 = person)')
    parser.add_argument('--duration', type=int, default=0,
                        help='Stop pipeline after N seconds (0 = unlimited)')
    parser.add_argument('--sam2-checkpoint', type=str, default=None,
                        help='SAM2 HuggingFace model ID (e.g. facebook/sam2.1-hiera-base-plus)')
    parser.add_argument('--sam2-device', type=str, default='cpu',
                        help='Device for SAM2 inference (default: cpu; use cuda if torch has CUDA)')
    args = parser.parse_args()
    main(args)

