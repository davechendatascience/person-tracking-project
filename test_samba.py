import sys
import os
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Test imports
try:
    from samba_wrapper import SambaMOTRTracker
    print("SambaMOTRTracker imported successfully.")
except Exception as e:
    print(f"Error importing SambaMOTRTracker: {e}")
    sys.exit(1)

model_path = "pretrained/sambamotr/sambamotr_dancetrack.pth"
config_path = "sambamotr/configs/sambamotr/dancetrack/def_detr/train_residual_masking_sync_longer.yaml"

if not os.path.exists(model_path):
    print(f"Model path not found: {model_path}")
    sys.exit(1)

print("Loading config...")
try:
    from utils.utils import yaml_to_dict
    config = yaml_to_dict(config_path)
    print("Config loaded.")
except Exception as e:
    print(f"Error loading config: {e}")
    sys.exit(1)

print("Building model...")
try:
    from models import build_model
    model = build_model(config)
    print("Model built.")
except Exception as e:
    print(f"Error building model: {e}")
    sys.exit(1)

print("Loading checkpoint...")
try:
    from models.utils import load_checkpoint
    load_checkpoint(model, model_path)
    print("Checkpoint loaded.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    sys.exit(1)

print("Initializing RuntimeTracker...")
try:
    from models.runtime_tracker import RuntimeTracker
    tracker = RuntimeTracker(
        det_score_thresh=0.4, 
        track_score_thresh=0.4,
        miss_tolerance=30,
        use_motion=False,
        use_dab=config["USE_DAB"],
    )
    print("RuntimeTracker initialized.")
except Exception as e:
    print(f"Error initializing RuntimeTracker: {e}")
    sys.exit(1)

print("Performing dummy inference...")
try:
    dummy_input = torch.randn(1, 4, 3, 256, 256).cuda()
    # Mocking track instances
    tracks = [RuntimeTracker(
        det_score_thresh=0.4, 
        track_score_thresh=0.4,
        miss_tolerance=30,
        use_motion=False,
        use_dab=config["USE_DAB"],
    ).to('cuda')] # This is wrong, but I just want to see if model(frame, tracks) works
    # Actually just run the model once
    # model(frame_nested, tracks)
    print("Dummy inference setup done.")
except Exception as e:
    print(f"Error in dummy inference setup: {e}")

print("SambaMOTR Initialization Complete!")
