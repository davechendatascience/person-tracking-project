import rtmlib
from rtmlib import PoseTracker, Wholebody
import inspect

print(f"PoseTracker __init__ signature: {inspect.signature(PoseTracker.__init__)}")
print(f"Wholebody __init__ signature: {inspect.signature(Wholebody.__init__)}")
