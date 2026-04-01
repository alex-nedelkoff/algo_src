"""Cubic spline through gate centers for reward shaping.

Fits a closed 3D cubic spline through gate center positions. Provides
fast nearest-point queries for computing spline proximity rewards.
The spline is used only for reward shaping during training — it is NOT
part of the deployed policy's observation or inference path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


class GateSpline:
    """Closed 3D cubic spline through gate centers.

    Args:
        gate_positions: (N, 3) array of gate center positions.
        samples_per_segment: Number of sample points between each gate pair.
    """

    def __init__(self, gate_positions: NDArray[np.float64], samples_per_segment: int = 50) -> None:
        n_gates = len(gate_positions)
        if n_gates < 2:
            raise ValueError(f"Need at least 2 gates, got {n_gates}")

        closed = np.vstack([gate_positions, gate_positions[0:1]])
        diffs = np.diff(closed, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        # Ensure strictly increasing knots — zero-length segments break CubicSpline
        seg_lengths = np.maximum(seg_lengths, 1e-6)
        t_knots = np.concatenate([[0], np.cumsum(seg_lengths)])

        self._spline_x = CubicSpline(t_knots, closed[:, 0], bc_type="periodic")
        self._spline_y = CubicSpline(t_knots, closed[:, 1], bc_type="periodic")
        self._spline_z = CubicSpline(t_knots, closed[:, 2], bc_type="periodic")

        n_samples = samples_per_segment * n_gates
        t_dense = np.linspace(0, t_knots[-1], n_samples, endpoint=False)
        self._samples = np.column_stack([
            self._spline_x(t_dense),
            self._spline_y(t_dense),
            self._spline_z(t_dense),
        ])

        dx = self._spline_x(t_dense, 1)
        dy = self._spline_y(t_dense, 1)
        dz = self._spline_z(t_dense, 1)
        tangents = np.column_stack([dx, dy, dz])
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        self._tangents = tangents / norms

        self._t_max = t_knots[-1]

    def distance_to_nearest(self, point: NDArray[np.float64]) -> float:
        diffs = self._samples - point
        dists_sq = np.sum(diffs ** 2, axis=1)
        return float(np.sqrt(np.min(dists_sq)))

    def distance_to_nearest_batch(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        diffs = points[:, None, :] - self._samples[None, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        return np.sqrt(np.min(dists_sq, axis=1))

    def nearest_point_and_tangent(self, point: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        diffs = self._samples - point
        dists_sq = np.sum(diffs ** 2, axis=1)
        idx = int(np.argmin(dists_sq))
        return self._samples[idx].copy(), self._tangents[idx].copy()

    def nearest_tangent_batch(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        diffs = points[:, None, :] - self._samples[None, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        indices = np.argmin(dists_sq, axis=1)
        return self._tangents[indices]
