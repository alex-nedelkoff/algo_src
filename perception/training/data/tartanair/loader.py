"""TartanAir dataset loader for covisibility pipeline.

Format (v1, Easy difficulty):
  tartanair/<env>/Easy/<env>/Easy/P00X/image_left/000000_left.png  — RGB 640×480
  tartanair/<env>/Easy/<env>/Easy/P00X/depth_left/000000_left_depth.npy  — float32 meters
  tartanair/<env>/Easy/<env>/Easy/P00X/pose_left.txt  — [tx ty tz qx qy qz qw] per line

Pools all trajectories in an environment, subsampled by frame_skip for
cross-trajectory covisibility computation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import K, MAX_DEPTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


class TartanAirScene:
    """Loader for a single TartanAir environment (all trajectories pooled).

    Subsamples frames by frame_skip to keep covisibility tractable
    while preserving cross-trajectory pairs.
    """

    def __init__(
        self,
        data_dir: str | Path,
        env_id: str,
        frame_skip: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.env_id = env_id
        self.frame_skip = frame_skip

        # Detect v1 vs v2 layout:
        #   v1: <env>/Easy/<env>/Easy/P00X/image_left/
        #   v2: <env>/Data_easy/<env>/Data_easy/P00X/image_lcam_front/
        self._version = None
        env_dir = None

        for difficulty, img_sub, dep_sub, pose_name, version in [
            ("Data_easy", "image_lcam_front", "depth_lcam_front", "pose_lcam_front.txt", "v2"),
            ("Easy", "image_left", "depth_left", "pose_left.txt", "v1"),
        ]:
            # Check with double nesting (zip extracts create this)
            candidate = self.data_dir / env_id / difficulty / env_id / difficulty
            if not candidate.exists():
                candidate = self.data_dir / env_id / difficulty
            if candidate.exists() and any(
                d.is_dir() and d.name.startswith("P") for d in candidate.iterdir()
            ):
                env_dir = candidate
                self._version = version
                self._img_subdir = img_sub
                self._dep_subdir = dep_sub
                self._pose_name = pose_name
                break

        if env_dir is None:
            raise FileNotFoundError(
                f"Environment not found. Tried v1 (Easy) and v2 (Data_easy) "
                f"under {self.data_dir / env_id}"
            )

        # Discover trajectories
        self._rgb_paths = []
        self._depth_paths = []
        self._poses_list = []
        self._traj_ids = []

        trajs = sorted(
            d for d in env_dir.iterdir()
            if d.is_dir() and d.name.startswith("P")
        )

        if not trajs:
            raise ValueError(f"No trajectories found in {env_dir}")

        for traj_dir in trajs:
            img_dir = traj_dir / self._img_subdir
            dep_dir = traj_dir / self._dep_subdir
            pose_file = traj_dir / self._pose_name

            if not all(p.exists() for p in [img_dir, dep_dir, pose_file]):
                log.warning("Skipping %s (missing data)", traj_dir.name)
                continue

            poses_raw = np.loadtxt(str(pose_file))
            imgs = sorted(img_dir.glob("*.png"))
            # v1 uses .npy depth, v2 uses .png (RGBA-encoded float32)
            deps = sorted(dep_dir.glob("*.npy"))
            if not deps:
                deps = sorted(dep_dir.glob("*.png"))
            n = min(len(poses_raw), len(imgs), len(deps))

            for i in range(0, n, frame_skip):
                pos = poses_raw[i, :3]
                quat = poses_raw[i, 3:7]  # qx qy qz qw
                R = Rotation.from_quat(quat).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = pos

                self._rgb_paths.append(imgs[i])
                self._depth_paths.append(deps[i])
                self._poses_list.append(T)
                self._traj_ids.append(traj_dir.name)

        self._n_frames = len(self._rgb_paths)
        self._poses = np.array(self._poses_list, dtype=np.float64)
        # v2 uses 640x640, intrinsics: fx=fy=320, cx=cy=320
        # v1 uses 640x480, intrinsics: fx=fy=320, cx=320, cy=240
        # Detect from first image
        test_img = cv2.imread(str(self._rgb_paths[0]))
        h, w = test_img.shape[:2]
        self._K = np.array([
            [320.0, 0.0, w / 2.0],
            [0.0, 320.0, h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        n_trajs = len(set(self._traj_ids))
        log.info(
            "Loaded %s: %d frames from %d trajectories (skip=%d)",
            env_id, self._n_frames, n_trajs, frame_skip,
        )

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def traj_ids(self) -> list[str]:
        return self._traj_ids

    def intrinsics(self) -> np.ndarray:
        return self._K.copy()

    def poses(self) -> np.ndarray:
        return self._poses

    def rgb(self, idx: int) -> np.ndarray:
        img = cv2.imread(str(self._rgb_paths[idx]), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read {self._rgb_paths[idx]}")
        return img

    def depth(self, idx: int) -> np.ndarray:
        path = self._depth_paths[idx]
        if path.suffix == ".npy":
            d = np.load(str(path))
        else:
            # v2: RGBA PNG encodes float32 depth
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            d = raw.view(np.float32)[:, :, 0]
        return np.clip(d, 0, MAX_DEPTH).astype(np.float32)

    def depths(self, indices: list[int] | None = None) -> list[np.ndarray]:
        if indices is None:
            indices = list(range(self._n_frames))
        return [self.depth(i) for i in indices]

    @classmethod
    def available_scenes(cls, data_dir: str | Path) -> list[str]:
        data_dir = Path(data_dir)
        scenes = []
        for d in sorted(data_dir.iterdir()):
            if not d.is_dir():
                continue
            # Check v1 (Easy) and v2 (Data_easy)
            for difficulty in ["Data_easy", "Easy"]:
                env_dir = d / difficulty / d.name / difficulty
                if not env_dir.exists():
                    env_dir = d / difficulty
                if env_dir.exists() and any(
                    p.is_dir() and p.name.startswith("P")
                    for p in env_dir.iterdir()
                ):
                    scenes.append(d.name)
                    break
        return scenes
