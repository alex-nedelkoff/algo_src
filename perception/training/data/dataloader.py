"""PyTorch dataset with covisibility-binned sampling for VPR training."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .pairs import PairTable


class CovisibilityVPRDataset(Dataset):
    """VPR image pair dataset with covisibility-binned sampling.

    Loads pair tables from one or more datasets and samples pairs
    uniformly across overlap bins each epoch.

    Args:
        pair_paths: List of Parquet file paths containing pair tables.
        scene_dirs: Mapping from (dataset, scene_id) → local scene directory
                    containing rgb/ and depth/ subdirectories.
        samples_per_epoch: Number of pairs to sample per epoch.
        bins: Overlap bins for uniform sampling.
        image_size: Optional (H, W) to resize images.
        seed: Random seed.
    """

    def __init__(
        self,
        pair_paths: list[str | Path],
        scene_dirs: dict[tuple[str, str], Path],
        samples_per_epoch: int = 10_000,
        bins: list[tuple[float, float]] | None = None,
        image_size: tuple[int, int] | None = None,
        seed: int = 42,
    ) -> None:
        tables = [PairTable.load(p) for p in pair_paths]
        self._full_table = PairTable.merge(tables) if len(tables) > 1 else tables[0]
        self._scene_dirs = scene_dirs
        self._samples_per_epoch = samples_per_epoch
        self._bins = bins
        self._image_size = image_size
        self._rng = np.random.default_rng(seed)

        # Initial sampling
        self.reshuffle()

    def reshuffle(self) -> None:
        """Resample pairs for a new epoch."""
        sampled = self._full_table.sample_uniform(
            self._samples_per_epoch, bins=self._bins, rng=self._rng
        )
        self._datasets = sampled.column("dataset").to_pylist()
        self._scene_ids = sampled.column("scene_id").to_pylist()
        self._frames_i = sampled.column("frame_i").to_pylist()
        self._frames_j = sampled.column("frame_j").to_pylist()
        self._scores = sampled.column("overlap_score").to_pylist()

    def __len__(self) -> int:
        return len(self._datasets)

    def _load_image(self, dataset: str, scene_id: str, frame_idx: int) -> torch.Tensor:
        """Load and preprocess an RGB image."""
        scene_dir = self._scene_dirs[(dataset, scene_id)]
        img_path = scene_dir / "rgb" / f"frame{frame_idx:06d}.jpg"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self._image_size is not None:
            img = cv2.resize(img, (self._image_size[1], self._image_size[0]))

        # HWC uint8 → CHW float32 [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor

    def __getitem__(self, idx: int) -> dict:
        dataset = self._datasets[idx]
        scene_id = self._scene_ids[idx]
        frame_i = self._frames_i[idx]
        frame_j = self._frames_j[idx]
        score = self._scores[idx]

        img_i = self._load_image(dataset, scene_id, frame_i)
        img_j = self._load_image(dataset, scene_id, frame_j)

        return {
            "img_i": img_i,
            "img_j": img_j,
            "overlap_score": torch.tensor(score, dtype=torch.float32),
            "dataset": dataset,
            "scene_id": scene_id,
            "frame_i": frame_i,
            "frame_j": frame_j,
        }
