"""B-fixture validation harness: compare PyBullet renders vs AirSim fixtures.

Renders the warehouse + gates in PyBullet at each fixture's camera pose
and intrinsics, builds silhouette masks for both renders, computes
per-pose silhouette IoU, and writes an HTML report.

Pass criteria (per spec): mean IoU ≥ 0.85, no individual pose < 0.70.

Run after fixtures are captured (Task 14 on the Linux box) and after
the build pipeline has produced sim/assets/warehouse_v1/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union for two boolean silhouette masks.

    If both masks are empty, returns 1.0 (degenerate but interpretable).
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)
