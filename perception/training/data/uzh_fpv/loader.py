"""UZH-FPV dataset loader for covisibility pipeline.

Format (after extraction + rectification + depth generation):
  uzh-fpv/rectified/<scene>/left/000000.png  — rectified grayscale (640×480)
  uzh-fpv/depth/<scene>/depth/depth_000000.npy — float32 meters
  uzh-fpv/rectified/<scene>/rectification_params.json — rectified intrinsics
  uzh-fpv/rectified/<scene>/groundtruth.txt — [timestamp tx ty tz qx qy qz qw]

Same interface as ReplicaScene: n_frames, intrinsics(), poses(), rgb(idx),
depth(idx), depths().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


class UZHFPVScene:
    """Lazy loader for a single UZH-FPV sequence (post-processing)."""

    def __init__(
        self, data_dir: str | Path, scene_id: str,
        calib_path: str | Path | None = None,
    ) -> None:
        """Load a UZH-FPV scene.

        Args:
            data_dir: Root UZH-FPV data directory (e.g. ~/corvidx/data/uzh-fpv).
            scene_id: Sequence name (e.g. "indoor_forward_3_snapdragon").
            calib_path: Path to Kalibr camchain YAML. If None, auto-detected
                        from data_dir/calib/<env>_calib_snapdragon/.
        """
        self.data_dir = Path(data_dir)
        self.scene_id = scene_id

        self._rect_dir = self.data_dir / "rectified" / scene_id
        self._depth_dir = self.data_dir / "depth" / scene_id / "depth"

        if not self._rect_dir.exists():
            raise FileNotFoundError(f"Rectified directory not found: {self._rect_dir}")
        if not self._depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {self._depth_dir}")

        # Discover all frames
        all_rgb = sorted((self._rect_dir / "left").glob("*.png"))
        all_depth = sorted(self._depth_dir.glob("depth_*.npy"))

        # Load rectified intrinsics
        params_path = self._rect_dir / "rectification_params.json"
        with open(params_path) as f:
            self._rect_params = json.load(f)

        self._K = np.array([
            [self._rect_params["fx"], 0, self._rect_params["cx"]],
            [0, self._rect_params["fy"], self._rect_params["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)

        # Load T_cam_imu from calibration
        self._T_cam_imu = self._load_T_cam_imu(calib_path)

        # Load poses — this also filters frames to GT coverage
        self._poses, self._valid_indices = self._load_poses(all_rgb, all_depth)

        # Filter RGB/depth paths to only valid frames
        self._rgb_paths = [all_rgb[i] for i in self._valid_indices]
        self._depth_paths = [all_depth[i] for i in self._valid_indices]
        self._n_frames = len(self._valid_indices)

        if self._n_frames == 0:
            raise ValueError(f"No frames with GT coverage found in {self._rect_dir}")

        log.info(
            "Loaded %s: %d/%d frames with GT coverage",
            scene_id, self._n_frames, len(all_rgb),
        )

    def _load_T_cam_imu(self, calib_path: Path | None) -> np.ndarray:
        """Load T_cam_imu from Kalibr calibration YAML."""
        if calib_path is None:
            # Auto-detect: scene_id like "indoor_forward_3_snapdragon"
            # env is "indoor_forward"
            import re
            match = re.match(r"(.+?)_\d+_snapdragon", self.scene_id)
            if match:
                env = match.group(1)
                calib_dir = self.data_dir / "calib" / f"{env}_calib_snapdragon"
                candidates = list(calib_dir.glob("camchain-imucam-*.yaml"))
                if candidates:
                    calib_path = candidates[0]

        if calib_path is None or not Path(calib_path).exists():
            log.warning("No calibration found, using identity T_cam_imu")
            return np.eye(4, dtype=np.float64)

        import yaml
        with open(calib_path) as f:
            calib = yaml.safe_load(f)

        T = np.array(calib["cam0"]["T_cam_imu"], dtype=np.float64).reshape(4, 4)
        return T

    def _load_poses(
        self, all_rgb: list, all_depth: list,
    ) -> tuple[np.ndarray, list[int]]:
        """Load GT poses, filter to GT coverage, apply T_cam_imu.

        Returns:
            poses: (M, 4, 4) camera-to-world transforms for valid frames.
            valid_indices: List of original frame indices that have GT coverage.
        """
        gt_path = self._rect_dir / "groundtruth.txt"
        if not gt_path.exists():
            gt_path = self.data_dir / "extracted" / self.scene_id / "groundtruth.txt"

        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found for {self.scene_id}")

        # Load GT: [timestamp_ns, tx, ty, tz, qx, qy, qz, qw]
        gt_data = np.loadtxt(str(gt_path))
        gt_timestamps = gt_data[:, 0]
        gt_positions = gt_data[:, 1:4]
        gt_quats = gt_data[:, 4:8]  # [qx, qy, qz, qw]

        # Load frame timestamps
        ts_path = self._rect_dir / "left_timestamps.txt"
        if not ts_path.exists():
            ts_path = self.data_dir / "extracted" / self.scene_id / "left_timestamps.txt"

        if ts_path.exists():
            frame_timestamps = np.loadtxt(str(ts_path))
        else:
            log.warning("No frame timestamps found, using GT timestamps directly")
            frame_timestamps = gt_timestamps[:len(all_rgb)]

        n_total = min(len(all_rgb), len(all_depth), len(frame_timestamps))
        gt_start, gt_end = gt_timestamps[0], gt_timestamps[-1]

        # Max allowable time gap to nearest GT pose (2ms at 500Hz GT rate)
        max_gap_ns = 2_000_000

        # Filter frames to GT coverage and find nearest GT pose
        T_imu_cam = np.linalg.inv(self._T_cam_imu)
        valid_indices = []
        poses_list = []

        for i in range(n_total):
            t = frame_timestamps[i]

            # Skip frames outside GT coverage
            if t < gt_start - max_gap_ns or t > gt_end + max_gap_ns:
                continue

            idx = np.searchsorted(gt_timestamps, t)
            idx = np.clip(idx, 0, len(gt_timestamps) - 1)

            # Check both neighbors for closest
            if idx > 0 and abs(gt_timestamps[idx - 1] - t) < abs(gt_timestamps[idx] - t):
                idx = idx - 1

            gap = abs(gt_timestamps[idx] - t)
            if gap > max_gap_ns:
                continue

            pos = gt_positions[idx]
            quat = gt_quats[idx]
            R = Rotation.from_quat(quat).as_matrix()

            # T_world_body (body/IMU in world frame)
            T_w_body = np.eye(4, dtype=np.float64)
            T_w_body[:3, :3] = R
            T_w_body[:3, 3] = pos

            # T_world_cam = T_world_body @ inv(T_cam_imu)
            T_w_cam = T_w_body @ T_imu_cam

            valid_indices.append(i)
            poses_list.append(T_w_cam)

        poses = np.array(poses_list, dtype=np.float64)
        return poses, valid_indices

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def intrinsics(self) -> np.ndarray:
        """Return (3, 3) rectified intrinsic matrix."""
        return self._K.copy()

    def poses(self) -> np.ndarray:
        """Return (N, 4, 4) camera-to-world transforms."""
        return self._poses

    def rgb(self, idx: int) -> np.ndarray:
        """Load rectified frame as (H, W, 3) uint8 BGR.

        UZH-FPV Snapdragon images are grayscale; we replicate to 3 channels.
        """
        img = cv2.imread(str(self._rgb_paths[idx]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read {self._rgb_paths[idx]}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def depth(self, idx: int) -> np.ndarray:
        """Load depth map as (H, W) float32 in meters."""
        depth = np.load(str(self._depth_paths[idx]))
        return depth.astype(np.float32)

    def depths(self, indices: list[int] | None = None) -> list[np.ndarray]:
        """Load multiple depth maps. If indices is None, loads all."""
        if indices is None:
            indices = list(range(self._n_frames))
        return [self.depth(i) for i in indices]

    @classmethod
    def available_scenes(cls, data_dir: str | Path) -> list[str]:
        """List scenes that have both rectified images and depth maps."""
        data_dir = Path(data_dir)
        rect_dir = data_dir / "rectified"
        depth_dir = data_dir / "depth"
        scenes = []
        if rect_dir.exists():
            for d in sorted(rect_dir.iterdir()):
                if d.is_dir() and (depth_dir / d.name / "depth").exists():
                    scenes.append(d.name)
        return scenes
