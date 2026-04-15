"""CrowdBot_v2 dataset loader.

Actual extracted structure (one zip = one day):
    {day_dir}/                              e.g. 0410_rds_defaced_processed/
    ├── lidars/
    │   └── {recording}/                   e.g. defaced_2021-04-10-10-38-36_filtered_lidar_odom/
    │       ├── 00000.npy                  (N, 3) XYZ in robot frame
    │       └── ...
    ├── alg_res/
    │   ├── tracks/{recording}.npy         0-d array wrapping dict {frame_idx: (K,8)}
    │   └── detections/{recording}.npy     0-d array wrapping dict {frame_idx: (M,7)}
    └── source_data/ ...

Track / detection columns  (LiDAR frame: x-fwd, y-left, z-up):
    [x, y, z, length, width, height, theta, track_id]   (tracks, 8 cols)
    [x, y, z, length, width, height, theta]              (detections, 7 cols)

Pass --processed-dir  as the day directory (e.g. data/crowdbot/0410_rds_defaced_processed)
and  --sequence       as the recording name (list with: python run.py --processed-dir ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class PedestrianTrack:
    track_id: int
    x: float        # metres forward in robot/LiDAR frame
    y: float        # metres left
    z: float        # metres up
    length: float
    width: float
    height: float
    theta: float    # yaw, radians


@dataclass
class FrameData:
    frame_idx: int
    timestamp: float                        # seconds (index / 20 Hz approx)
    lidar: np.ndarray                       # (N, 3) XYZ robot frame
    robot_pose: np.ndarray = field(        # (3,) [x, y, yaw] world frame
        default_factory=lambda: np.zeros(3))
    tracks: List[PedestrianTrack] = field(default_factory=list)
    detections: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    image: Optional[np.ndarray] = None     # (H, W, 3) uint8 RGB from rosbag
    depth: Optional[np.ndarray] = None     # (H, W) float32 metres from rosbag


# ---------------------------------------------------------------------------
class CrowdBotSequence:
    """Iterate over one CrowdBot_v2 recording.

    Args:
        day_dir:       Path to the *_processed day directory,
                       e.g. ``data/crowdbot/0410_rds_defaced_processed``.
        recording:     Recording name (subdirectory under lidars/),
                       e.g. ``defaced_2021-04-10-10-38-36_filtered_lidar_odom``.
        load_images:   If True, try to open the companion rosbag for RGB/depth.
    """

    def __init__(
        self,
        day_dir: str | Path,
        recording: str,
        load_images: bool = True,
    ):
        self.day_dir   = Path(day_dir)
        self.recording = recording

        lidar_dir = self.day_dir / "lidars" / recording
        if not lidar_dir.exists():
            raise FileNotFoundError(
                f"LiDAR directory not found: {lidar_dir}\n"
                f"Run with --processed-dir alone to list available recordings."
            )

        self._lidar_files: List[Path] = sorted(lidar_dir.glob("*.npy"))
        if not self._lidar_files:
            raise FileNotFoundError(f"No .npy files in {lidar_dir}")

        self._tracks: Dict[int, np.ndarray] = self._load_npy_dict(
            self.day_dir / "alg_res" / "tracks" / f"{recording}.npy"
        )
        self._dets: Dict[int, np.ndarray] = self._load_npy_dict(
            self.day_dir / "alg_res" / "detections" / f"{recording}.npy"
        )
        self._robot_poses: np.ndarray = self._load_robot_poses()

        self._bag_reader = None
        if load_images:
            self._bag_reader = self._open_bag_reader()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._lidar_files)

    def __iter__(self) -> Iterator[FrameData]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> FrameData:
        path = self._lidar_files[idx]
        frame_idx = int(path.stem)
        lidar = np.load(path).astype(np.float32)

        robot_pose = (
            self._robot_poses[idx] if idx < len(self._robot_poses)
            else np.zeros(3)
        )
        fd = FrameData(
            frame_idx=frame_idx,
            timestamp=float(idx) / 20.0,
            lidar=lidar,
            robot_pose=robot_pose,
            tracks=self._parse_tracks(frame_idx),
            detections=self._dets.get(frame_idx, np.zeros((0, 7), dtype=np.float32)),
        )

        if self._bag_reader is not None:
            try:
                img, depth = self._bag_reader.get_frame(idx)
                fd.image = img
                fd.depth = depth
            except Exception:
                pass

        return fd

    # ------------------------------------------------------------------
    def list_track_ids(self, frame_idx: int) -> List[int]:
        arr = self._tracks.get(frame_idx)
        if arr is None or arr.shape[0] == 0:
            return []
        return [int(row[7]) for row in arr]

    def get_track(self, frame_idx: int, track_id: int) -> Optional[PedestrianTrack]:
        arr = self._tracks.get(frame_idx)
        if arr is None:
            return None
        for row in arr:
            if int(row[7]) == track_id:
                return _row_to_track(row)
        return None

    def first_frame_with_track(self, track_id: int) -> Optional[int]:
        for path in self._lidar_files:
            fidx = int(path.stem)
            if track_id in self.list_track_ids(fidx):
                return fidx
        return None

    def all_track_ids(self) -> List[int]:
        """Return all unique track IDs visible across the whole recording."""
        ids: set = set()
        for path in self._lidar_files:
            ids.update(self.list_track_ids(int(path.stem)))
        return sorted(ids)

    # ------------------------------------------------------------------
    def _parse_tracks(self, frame_idx: int) -> List[PedestrianTrack]:
        arr = self._tracks.get(frame_idx)
        if arr is None or arr.shape[0] == 0:
            return []
        return [_row_to_track(row) for row in arr]

    def _load_robot_poses(self) -> np.ndarray:
        """Return (N, 3) array of [x, y, yaw] world-frame poses, one per LiDAR frame."""
        pose_path = (
            self.day_dir / "source_data" / "tf_qolo"
            / f"{self.recording}_tfqolo_sampled.npy"
        )
        if not pose_path.exists():
            return np.zeros((len(self._lidar_files), 3))
        raw = np.load(pose_path, allow_pickle=True).item()
        positions    = raw["position"]     # (N, 3) x, y, z world frame
        orientations = raw["orientation"]  # (N, 4) x, y, z, w quaternion
        yaws = np.arctan2(
            2 * (orientations[:, 3] * orientations[:, 2]
                 + orientations[:, 0] * orientations[:, 1]),
            1 - 2 * (orientations[:, 1] ** 2 + orientations[:, 2] ** 2),
        )
        return np.column_stack([positions[:, 0], positions[:, 1], yaws])

    def _open_bag_reader(self) -> Optional[BagImageReader]:
        from .bag_reader import BagImageReader
        
        # Try a few common locations/names for the bag file
        search_paths = [
            self.day_dir / f"{self.recording}.bag",
            self.day_dir / "source_data" / f"{self.recording}.bag",
            # If recording name has suffixes, try stripping them
            self.day_dir / (self.recording.split("_")[0] + ".bag"),
        ]
        
        # Also check for bags in a 'rosbags' or 'raw' subdirectory
        for sub in ["rosbags", "raw", "source_data"]:
            search_paths.append(self.day_dir / sub / f"{self.recording}.bag")

        for p in search_paths:
            if p.exists():
                try:
                    return BagImageReader(p)
                except Exception as e:
                    print(f"[WARNING] Failed to open bag at {p}: {e}")
        
        # Last resort: recursive search in day_dir (might be slow if day_dir is huge)
        # Only do this if we haven't found anything yet
        try:
            bags = list(self.day_dir.rglob("*.bag"))
            if bags:
                # Pick the one that most closely matches the recording name
                best_bag = None
                best_score = -1
                for b in bags:
                    score = len(set(b.stem.split("_")) & set(self.recording.split("_")))
                    if score > best_score:
                        best_score = score
                        best_bag = b
                if best_bag:
                    return BagImageReader(best_bag)
        except Exception:
            pass

        return None

    @staticmethod
    def _load_npy_dict(path: Path) -> Dict[int, np.ndarray]:
        if not path.exists():
            return {}
        raw = np.load(path, allow_pickle=True)
        return raw.item() if raw.ndim == 0 else {}


# ---------------------------------------------------------------------------
def _row_to_track(row: np.ndarray) -> PedestrianTrack:
    return PedestrianTrack(
        x=float(row[0]), y=float(row[1]), z=float(row[2]),
        length=float(row[3]), width=float(row[4]), height=float(row[5]),
        theta=float(row[6]),
        track_id=int(row[7]) if len(row) > 7 else -1,
    )


# ---------------------------------------------------------------------------
class VideoSequence:
    """Iterate over frames from a video file."""

    def __init__(self, video_path: str | Path):
        import cv2
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __len__(self) -> int:
        return self.num_frames
        
    def __getitem__(self, idx: int) -> FrameData:
        import cv2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            # Fallback for failed read
            return FrameData(frame_idx=idx, timestamp=idx/(self.fps or 1.0), lidar=np.zeros((0,3)))
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return FrameData(
            frame_idx=idx,
            timestamp=float(idx) / (self.fps or 1.0),
            lidar=np.zeros((0, 3)), # No LiDAR in video
            image=frame_rgb,
            depth=None # No depth in video
        )

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


# ---------------------------------------------------------------------------
def list_recordings(day_dir: str | Path) -> List[str]:
    """Return all recording names found under day_dir/lidars/."""
    lidar_root = Path(day_dir) / "lidars"
    if not lidar_root.exists():
        return []
    return sorted(d.name for d in lidar_root.iterdir() if d.is_dir())
