"""Pure-Python ROS1 bag reader for CrowdBot_v2 RGB-D camera data.

Uses the `rosbags` package (pip install rosbags) — no ROS installation needed.

Topics read (configurable, these are the standard RealSense ROS defaults):
  /camera/color/image_raw        — sensor_msgs/Image (rgb8 or bgr8)
  /camera/depth/image_rect_raw   — sensor_msgs/Image (16UC1, mm)
  /camera/color/camera_info      — sensor_msgs/CameraInfo
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class CameraIntrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.width, self.height = width, height

    @classmethod
    def realsense_d435_default(cls) -> "CameraIntrinsics":
        """Fallback intrinsics for RealSense D435 at 640×480."""
        return cls(fx=612.0, fy=612.0, cx=320.0, cy=240.0, width=640, height=480)


class BagImageReader:
    """Extract time-aligned RGB and depth frames from a ROS1 bag file.

    Frames are indexed by the order they appear in the bag.  On first
    access the entire bag is scanned and frame timestamps cached so that
    subsequent random access is O(1).
    """

    RGB_TOPIC   = "/camera/color/image_raw"
    DEPTH_TOPIC = "/camera/depth/image_rect_raw"
    INFO_TOPIC  = "/camera/color/camera_info"

    def __init__(
        self,
        bag_path: str | Path,
        rgb_topic:   str = RGB_TOPIC,
        depth_topic: str = DEPTH_TOPIC,
        info_topic:  str = INFO_TOPIC,
    ):
        from rosbags.rosbag1 import Reader

        self._bag_path   = Path(bag_path)
        self._rgb_topic   = rgb_topic
        self._depth_topic = depth_topic
        self._info_topic  = info_topic

        self._reader = Reader(self._bag_path)
        self._reader.open()

        # Cache: list of (timestamp_ns, rawdata, msgtype) per topic
        self._rgb_frames:   list[tuple] = []
        self._depth_frames: list[tuple] = []
        self._intrinsics: Optional[CameraIntrinsics] = None

        self._scan_bag()

    # ------------------------------------------------------------------
    def __del__(self):
        try:
            self._reader.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self._rgb_frames)

    # ------------------------------------------------------------------
    def get_frame(self, idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (rgb_uint8_HxWx3, depth_float32_HxW_metres) for frame *idx*.

        Depth is aligned to the closest depth message by timestamp.
        Returns (None, None) if the frame index is out of range.
        """
        if idx >= len(self._rgb_frames):
            return None, None

        ts_rgb, raw_rgb, msgtype_rgb = self._rgb_frames[idx]
        rgb = self._decode_image(raw_rgb, msgtype_rgb)

        # Find nearest depth frame by timestamp
        depth = None
        if self._depth_frames:
            depth_ts = np.array([f[0] for f in self._depth_frames])
            nearest = int(np.argmin(np.abs(depth_ts - ts_rgb)))
            _, raw_d, msgtype_d = self._depth_frames[nearest]
            depth_raw = self._decode_image(raw_d, msgtype_d)
            if depth_raw is not None:
                depth = depth_raw.astype(np.float32) / 1000.0   # mm → metres

        return rgb, depth

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            return CameraIntrinsics.realsense_d435_default()
        return self._intrinsics

    # ------------------------------------------------------------------
    def _scan_bag(self) -> None:
        """One-pass scan to cache frame pointers and camera intrinsics."""
        from rosbags.serde import deserialize_cdr

        connections_by_topic = {
            c.topic: c
            for c in self._reader.connections
            if c.topic in (self._rgb_topic, self._depth_topic, self._info_topic)
        }

        for connection, timestamp, rawdata in self._reader.messages(
            connections=list(connections_by_topic.values())
        ):
            topic = connection.topic
            if topic == self._rgb_topic:
                self._rgb_frames.append((timestamp, rawdata, connection.msgtype))
            elif topic == self._depth_topic:
                self._depth_frames.append((timestamp, rawdata, connection.msgtype))
            elif topic == self._info_topic and self._intrinsics is None:
                try:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    K = msg.K  # row-major 3×3
                    self._intrinsics = CameraIntrinsics(
                        fx=K[0], fy=K[4], cx=K[2], cy=K[5],
                        width=msg.width, height=msg.height,
                    )
                except Exception:
                    pass

    @staticmethod
    def _decode_image(rawdata: bytes, msgtype: str) -> Optional[np.ndarray]:
        """Decode a raw sensor_msgs/Image into a numpy array."""
        try:
            from rosbags.serde import deserialize_cdr
            msg = deserialize_cdr(rawdata, msgtype)
            h, w = msg.height, msg.width
            enc = msg.encoding.lower()
            data = np.frombuffer(msg.data, dtype=np.uint8)

            if enc in ("rgb8",):
                return data.reshape(h, w, 3)
            if enc in ("bgr8",):
                return data.reshape(h, w, 3)[:, :, ::-1].copy()
            if enc in ("16uc1", "mono16"):
                return np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            if enc in ("32fc1",):
                return np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
            # Fallback: try as uint8
            return data.reshape(h, w, -1)
        except Exception:
            return None
