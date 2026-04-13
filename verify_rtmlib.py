import rtmlib
try:
    from rtmlib import PoseTracker, Wholebody
    print("SUCCESS: PoseTracker and Wholebody imported")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
