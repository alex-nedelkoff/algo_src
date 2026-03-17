"""Replica dataset loader (NICE-SLAM version).

Format:
  Replica/<scene>/results/frame000000.jpg  — RGB (1200x680)
  Replica/<scene>/results/depth000000.png  — 16-bit PNG, meters = value / 6553.5
  Replica/<scene>/traj.txt                 — 16 floats per line → 4x4 c2w
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .constants import DEPTH_SCALE, K, SCENES


class ReplicaScene:
    """Lazy loader for a single Replica scene."""

    def __init__(self, data_dir: str | Path, scene_id: str) -> None:
        self.data_dir = Path(data_dir)
        self.scene_id = scene_id
        self._scene_dir = self.data_dir / scene_id
        self._results_dir = self._scene_dir / "results"

        if not self._results_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {self._results_dir}")

        # Discover frames by globbing RGB images
        self._rgb_paths = sorted(self._results_dir.glob("frame*.jpg"))
        self._depth_paths = sorted(self._results_dir.glob("depth*.png"))
        self._n_frames = len(self._rgb_paths)

        if self._n_frames == 0:
            raise ValueError(f"No frames found in {self._results_dir}")
        if len(self._depth_paths) != self._n_frames:
            raise ValueError(
                f"Frame count mismatch: {self._n_frames} RGB vs {len(self._depth_paths)} depth"
            )

        # Load poses eagerly (small: N x 16 floats in a text file)
        self._poses = self._load_poses()

    def _load_poses(self) -> np.ndarray:
        """Load traj.txt → (N, 4, 4) c2w poses.

        The Replica renderer stores poses and renders depth/RGB in a consistent
        coordinate system.  The Y/Z flip applied by NICE-SLAM is specific to
        their internal pipeline and should NOT be applied here — the raw poses
        already match the intrinsics and depth maps for reprojection.
        """
        traj_path = self._scene_dir / "traj.txt"
        poses = np.loadtxt(str(traj_path)).reshape(-1, 4, 4)

        if len(poses) != self._n_frames:
            raise ValueError(
                f"Pose count {len(poses)} != frame count {self._n_frames}"
            )

        return poses

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def intrinsics(self) -> np.ndarray:
        """Return (3, 3) intrinsic matrix (same for all Replica scenes)."""
        return K.copy()

    def poses(self) -> np.ndarray:
        """Return (N, 4, 4) camera-to-world transforms."""
        return self._poses

    def rgb(self, idx: int) -> np.ndarray:
        """Load RGB frame as (H, W, 3) uint8 BGR (OpenCV convention)."""
        img = cv2.imread(str(self._rgb_paths[idx]), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read {self._rgb_paths[idx]}")
        return img

    def depth(self, idx: int) -> np.ndarray:
        """Load depth map as (H, W) float32 in meters."""
        raw = cv2.imread(str(self._depth_paths[idx]), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Failed to read {self._depth_paths[idx]}")
        return raw.astype(np.float32) / DEPTH_SCALE

    def depths(self, indices: list[int] | None = None) -> list[np.ndarray]:
        """Load multiple depth maps. If indices is None, loads all."""
        if indices is None:
            indices = list(range(self._n_frames))
        return [self.depth(i) for i in indices]

    @classmethod
    def available_scenes(cls) -> list[str]:
        return list(SCENES)

    @classmethod
    def load_all(cls, data_dir: str | Path) -> list[ReplicaScene]:
        """Load all available scenes from data_dir."""
        scenes = []
        for scene_id in SCENES:
            scene_dir = Path(data_dir) / scene_id / "results"
            if scene_dir.exists():
                scenes.append(cls(data_dir, scene_id))
        return scenes
