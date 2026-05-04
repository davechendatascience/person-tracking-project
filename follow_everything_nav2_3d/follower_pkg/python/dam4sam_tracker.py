"""DAM4SAM tracker bridge — Phase 4a skeleton.

Subscribes to the follower's RGB-D camera and publishes a Detection2DArray on
/follower/camera/detections_dam4sam (kept distinct from the oracle topic so the
two can run side-by-side and be compared in shadow mode).

This skeleton wires the topology end-to-end with a NO-OP detector that always
publishes an empty detections array. Phase 4b drops in the real SAM2/DAM4SAM
tracker (../../follow_everything/perception/sam2_tracker.py), YOLO bootstrap
for first-frame init, and depth back-projection for body-frame (x, y).

Wiring this in shadow mode now means:
  - the camera/topic plumbing is verified before we add CUDA + ML deps,
  - the follower can already subscribe to detections_dam4sam,
  - replacing the no-op _track_frame() in Phase 4b is a localised change.
"""
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray


class Dam4SamTracker(Node):
    def __init__(self) -> None:
        super().__init__("dam4sam_tracker")
        self._latest_rgb: Image | None = None
        self._latest_depth: Image | None = None
        self._camera_info: CameraInfo | None = None

        self.create_subscription(
            Image, "/follower/camera/image", self._on_rgb, 10)
        self.create_subscription(
            Image, "/follower/camera/depth_image", self._on_depth, 10)
        self.create_subscription(
            CameraInfo, "/follower/camera/camera_info", self._on_camera_info, 10)

        self.pub = self.create_publisher(
            Detection2DArray, "/follower/camera/detections_dam4sam", 10)
        self.create_timer(0.05, self._tick)  # 20 Hz — match oracle rate

        self._frames_seen = 0
        self.get_logger().info(
            "DAM4SAM tracker (Phase 4a stub) running. _track_frame is a no-op; "
            "Phase 4b will integrate sam2_tracker.SAM2Tracker.")

    def _on_rgb(self, msg: Image) -> None:
        self._latest_rgb = msg
        self._frames_seen += 1
        if self._frames_seen % 100 == 1:
            self.get_logger().info(
                f"rgb {msg.width}x{msg.height} frames seen: {self._frames_seen}")

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _track_frame(self) -> Detection2DArray:
        """No-op for Phase 4a. Phase 4b: run SAM2/DAM4SAM, project mask
        centroid through depth + camera_info to produce body-frame (x, y)."""
        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "follower/base_link"
        return out

    def _tick(self) -> None:
        if self._latest_rgb is None or self._camera_info is None:
            return
        self.pub.publish(self._track_frame())


def main() -> None:
    rclpy.init()
    node = Dam4SamTracker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
