"""EuRoC MAV dataset loader for covisibility pipeline.

Format (after rectification + FoundationStereo depth):
  euroc/rectified/<seq>/left/000000.png
  euroc/depth/<seq>/depth/depth_000000.npy
  euroc/rectified/<seq>/rectification_params.json
  euroc/rectified/<seq>/groundtruth.csv
  euroc/rectified/<seq>/calibration.json
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


class EuRoCScene:
    """Lazy loader for a single EuRoC sequence (post-processing)."""

    def __init__(self, data_dir: str | Path, scene_id: str) -> None:
        self.data_dir = Path(data_dir)
        self.scene_id = scene_id

        self._rect_dir = self.data_dir / "rectified" / scene_id
        self._depth_dir = self.data_dir / "depth" / scene_id / "depth"

        if not self._rect_dir.exists():
            raise FileNotFoundError(f"Rectified dir not found: {self._rect_dir}")
        if not self._depth_dir.exists():
            raise FileNotFoundError(f"Depth dir not found: {self._depth_dir}")

        all_rgb = sorted((self._rect_dir / "left").glob("*.png"))
        all_depth = sorted(self._depth_dir.glob("depth_*.npy"))

        # Load rectified intrinsics
        with open(self._rect_dir / "rectification_params.json") as f:
            self._rect_params = json.load(f)

        self._K = np.array([
            [self._rect_params["fx"], 0, self._rect_params["cx"]],
            [0, self._rect_params["fy"], self._rect_params["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)

        # Load T_BS (body-to-sensor for cam0)
        calib_path = self._rect_dir / "calibration.json"
        if calib_path.exists():
            with open(calib_path) as f:
                calib = json.load(f)
            self._T_BS0 = np.array(calib["T_BS0"], dtype=np.float64).reshape(4, 4)
        else:
            self._T_BS0 = np.eye(4, dtype=np.float64)

        # Load poses, filter to GT coverage
        self._poses, self._valid_indices = self._load_poses(all_rgb, all_depth)
        self._rgb_paths = [all_rgb[i] for i in self._valid_indices]
        self._depth_paths = [all_depth[i] for i in self._valid_indices]
        self._n_frames = len(self._valid_indices)

        if self._n_frames == 0:
            raise ValueError(f"No frames with GT coverage in {scene_id}")

        log.info("Loaded %s: %d/%d frames with GT", scene_id, self._n_frames, len(all_rgb))

    def _load_poses(self, all_rgb, all_depth) -> tuple[np.ndarray, list[int]]:
        """Load GT poses, filter to coverage, convert to camera frame."""
        gt_path = self._rect_dir / "groundtruth.csv"
        if not gt_path.exists():
            raise FileNotFoundError(f"GT not found for {self.scene_id}")

        # EuRoC GT: timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, ...
        gt_data = np.loadtxt(str(gt_path), delimiter=",", skiprows=1)
        gt_ts = gt_data[:, 0]
        gt_pos = gt_data[:, 1:4]
        gt_quat_wxyz = gt_data[:, 4:8]  # q_w, q_x, q_y, q_z

        # Load frame timestamps
        ts_path = self._rect_dir / "timestamps.txt"
        frame_ts = np.loadtxt(str(ts_path))

        n_total = min(len(all_rgb), len(all_depth), len(frame_ts))
        gt_start, gt_end = gt_ts[0], gt_ts[-1]
        max_gap_ns = 10_000_000  # 10ms (EuRoC GT at 200Hz)

        T_BS0_inv = np.linalg.inv(self._T_BS0)

        valid_indices = []
        poses_list = []

        for i in range(n_total):
            t = frame_ts[i]
            if t < gt_start - max_gap_ns or t > gt_end + max_gap_ns:
                continue

            idx = np.searchsorted(gt_ts, t)
            idx = np.clip(idx, 0, len(gt_ts) - 1)
            if idx > 0 and abs(gt_ts[idx - 1] - t) < abs(gt_ts[idx] - t):
                idx = idx - 1
            if abs(gt_ts[idx] - t) > max_gap_ns:
                continue

            pos = gt_pos[idx]
            # EuRoC: q_w first, scipy wants [q_x, q_y, q_z, q_w]
            qw, qx, qy, qz = gt_quat_wxyz[idx]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            # T_world_body
            T_wb = np.eye(4, dtype=np.float64)
            T_wb[:3, :3] = R
            T_wb[:3, 3] = pos

            # T_world_cam = T_world_body @ T_body_sensor = T_wb @ inv(T_BS0)
            # (T_BS = sensor_in_body, so body_to_sensor = inv(T_BS))
            # Actually T_BS maps body→sensor, so T_world_sensor = T_world_body @ T_BS
            T_wc = T_wb @ self._T_BS0

            valid_indices.append(i)
            poses_list.append(T_wc)

        return np.array(poses_list, dtype=np.float64), valid_indices

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def intrinsics(self) -> np.ndarray:
        return self._K.copy()

    def poses(self) -> np.ndarray:
        return self._poses

    def rgb(self, idx: int) -> np.ndarray:
        img = cv2.imread(str(self._rgb_paths[idx]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read {self._rgb_paths[idx]}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def depth(self, idx: int) -> np.ndarray:
        return np.load(str(self._depth_paths[idx])).astype(np.float32)

    def depths(self, indices: list[int] | None = None) -> list[np.ndarray]:
        if indices is None:
            indices = list(range(self._n_frames))
        return [self.depth(i) for i in indices]

    @classmethod
    def available_scenes(cls, data_dir: str | Path) -> list[str]:
        data_dir = Path(data_dir)
        rect_dir = data_dir / "rectified"
        depth_dir = data_dir / "depth"
        scenes = []
        if rect_dir.exists():
            for d in sorted(rect_dir.iterdir()):
                if d.is_dir() and (depth_dir / d.name / "depth").exists():
                    scenes.append(d.name)
        return scenes
