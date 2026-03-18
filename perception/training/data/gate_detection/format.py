"""Shared format module for gate detection training data.

Defines the canonical on-disk format that all data sources produce:
  rgb.png             — (H, W, 3) uint8 BGR image
  gate_mask.png       — (H, W) uint8 instance mask (0=bg, 1+=gate IDs)
  obstacle_mask.png   — (H, W) uint8 obstacle mask
  corner_heatmaps.npy — (4, H//stride, W//stride) float32 CenterNet heatmaps
  corner_coords.json  — list of gate corner annotations
  metadata.json       — source info, camera intrinsics, etc.

Corner coords format::

    [
        {
            "gate_id": int,
            "corners": [[x, y], [x, y], [x, y], [x, y]],  # TL, TR, BR, BL
            "confidence": float,
        },
        ...
    ]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Corner order convention: TL=0, TR=1, BR=2, BL=3
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def generate_corner_heatmaps(
    corner_coords: list[dict],
    height: int,
    width: int,
    stride: int = 4,
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate CenterNet-style Gaussian heatmaps for gate corners.

    One channel per corner type (TL=0, TR=1, BR=2, BL=3).  Multiple gates
    produce multiple peaks per channel (element-wise max, not sum).

    Args:
        corner_coords: List of gate annotations, each with ``"corners"``
            as ``[[x, y], ...]`` in pixel coords (TL, TR, BR, BL order).
        height: Original image height.
        width: Original image width.
        stride: Downsampling factor for the heatmap grid.
        sigma: Standard deviation of the Gaussian kernel (in grid cells).

    Returns:
        (4, H // stride, W // stride) float32 array with values in [0, 1].
    """
    hm_h = height // stride
    hm_w = width // stride
    heatmaps = np.zeros((4, hm_h, hm_w), dtype=np.float32)

    if not corner_coords:
        return heatmaps

    # Pre-compute a coordinate grid for the Gaussian
    ys = np.arange(hm_h, dtype=np.float32)
    xs = np.arange(hm_w, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (hm_h, hm_w)

    two_sigma_sq = 2.0 * sigma * sigma

    for gate in corner_coords:
        corners = gate["corners"]  # [[x, y] | None, ...] length 4
        for ch, corner in enumerate(corners):
            if corner is None:
                continue  # invisible corner — no heatmap peak
            cx, cy = corner
            # Map pixel coords to heatmap grid coords, snap to nearest cell
            # (standard CenterNet convention — peak is always exactly 1.0)
            gx = int(round(cx / stride))
            gy = int(round(cy / stride))

            # Clamp to valid grid range
            gx = min(max(gx, 0), hm_w - 1)
            gy = min(max(gy, 0), hm_h - 1)

            # Gaussian centered at integer grid cell (gx, gy)
            gaussian = np.exp(
                -((grid_x - gx) ** 2 + (grid_y - gy) ** 2) / two_sigma_sq
            )

            # Element-wise max so overlapping gates don't exceed 1.0
            np.maximum(heatmaps[ch], gaussian, out=heatmaps[ch])

    return heatmaps


def render_gate_mask(
    corner_coords: list[dict],
    height: int,
    width: int,
) -> np.ndarray:
    """Render filled quadrilateral gate instance masks.

    Args:
        corner_coords: List of gate annotations. Each must have ``"corners"``
            as ``[[x, y], ...]`` with 4 vertices.
        height: Image height.
        width: Image width.

    Returns:
        (H, W) uint8 mask where 0 = background and i = gate instance *i*
        (1-indexed, in the order gates appear in *corner_coords*).
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for instance_id, gate in enumerate(corner_coords, start=1):
        corners = gate["corners"]
        # Skip gates with any None corners (partial visibility)
        if any(c is None for c in corners):
            continue
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=int(instance_id))

    return mask


def save_sample(
    sample_dir: str | Path,
    *,
    rgb: np.ndarray,
    gate_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    corner_coords: list[dict],
    corner_heatmaps: np.ndarray,
    metadata: dict,
) -> None:
    """Persist a single training sample to disk.

    Creates *sample_dir* (and parents) if it does not exist, then writes:

    - ``rgb.png`` — BGR uint8 image
    - ``gate_mask.png`` — uint8 instance mask
    - ``obstacle_mask.png`` — uint8 obstacle mask
    - ``corner_heatmaps.npy`` — float32 array
    - ``corner_coords.json`` — gate corner annotations
    - ``metadata.json`` — arbitrary metadata dict

    Args:
        sample_dir: Directory to write files into.
        rgb: (H, W, 3) uint8 array.
        gate_mask: (H, W) uint8 array.
        obstacle_mask: (H, W) uint8 array.
        corner_coords: List of gate corner dicts.
        corner_heatmaps: (4, hm_H, hm_W) float32 array.
        metadata: Arbitrary JSON-serializable metadata.
    """
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(sample_dir / "rgb.png"), rgb)
    cv2.imwrite(str(sample_dir / "gate_mask.png"), gate_mask)
    cv2.imwrite(str(sample_dir / "obstacle_mask.png"), obstacle_mask)
    np.save(str(sample_dir / "corner_heatmaps.npy"), corner_heatmaps)

    with open(sample_dir / "corner_coords.json", "w") as f:
        json.dump(corner_coords, f, indent=2)

    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.debug("Saved sample to %s", sample_dir)


def load_sample(sample_dir: str | Path) -> dict:
    """Load a training sample from disk.

    Args:
        sample_dir: Directory previously written by :func:`save_sample`.

    Returns:
        Dict with keys: ``rgb``, ``gate_mask``, ``obstacle_mask``,
        ``corner_heatmaps``, ``corner_coords``, ``metadata``.
    """
    sample_dir = Path(sample_dir)

    rgb = cv2.imread(str(sample_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise IOError(f"Failed to read {sample_dir / 'rgb.png'}")

    gate_mask = cv2.imread(str(sample_dir / "gate_mask.png"), cv2.IMREAD_GRAYSCALE)
    if gate_mask is None:
        raise IOError(f"Failed to read {sample_dir / 'gate_mask.png'}")

    obstacle_mask = cv2.imread(
        str(sample_dir / "obstacle_mask.png"), cv2.IMREAD_GRAYSCALE
    )
    if obstacle_mask is None:
        raise IOError(f"Failed to read {sample_dir / 'obstacle_mask.png'}")

    corner_heatmaps = np.load(str(sample_dir / "corner_heatmaps.npy"))

    with open(sample_dir / "corner_coords.json") as f:
        corner_coords = json.load(f)

    with open(sample_dir / "metadata.json") as f:
        metadata = json.load(f)

    return {
        "rgb": rgb,
        "gate_mask": gate_mask,
        "obstacle_mask": obstacle_mask,
        "corner_heatmaps": corner_heatmaps,
        "corner_coords": corner_coords,
        "metadata": metadata,
    }
