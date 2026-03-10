"""Mock detector for RL training with configurable noise and dropout.

Produces ground-truth gate detections corrupted by Gaussian noise and
stochastic dropout, simulating imperfect perception during policy training.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from core.interfaces import Detector
from perception.detectors.base import GateDetection


class MockDetector(Detector):
    """Detector that returns noisy ground-truth gate positions.

    Used during RL training to simulate imperfect perception without
    running a real CNN.

    Args:
        gate_positions: List of (3,) arrays — 3D gate center positions in
            world frame.
        noise_std: Standard deviation of Gaussian noise added to 2D corner
            coordinates.  Defaults to 0.1.
        dropout_rate: Probability that each gate is independently dropped
            from the detection list.  Defaults to 0.0.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        gate_positions: list[NDArray[np.float64]],
        noise_std: float = 0.1,
        dropout_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.gate_positions = [np.asarray(p, dtype=np.float64) for p in gate_positions]
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self._rng = np.random.default_rng(seed)

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _gate_to_corners_2d(position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate canonical 2D corners from a 3D gate center.

        Uses a simple projection: x,y components of the gate center as the
        centroid, with a fixed half-size of 0.5 (matching 1 m gate).  This
        is intentionally a *mock* — real projection lives in ``QuadGate``.
        """
        cx, cy = position[0], position[1]
        half = 0.5
        return np.array(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ],
            dtype=np.float64,
        )

    # --------------------------------------------------------------------- #
    # Detector interface
    # --------------------------------------------------------------------- #

    def detect(self, image: NDArray[np.uint8]) -> list[GateDetection]:
        """Return noisy, dropout-affected gate detections.

        ``image`` is accepted for interface compatibility but ignored —
        detections are synthesized from ``self.gate_positions``.
        """
        detections: list[GateDetection] = []
        for gate_id, pos in enumerate(self.gate_positions):
            # Dropout: independently skip each gate with probability dropout_rate
            if self._rng.random() < self.dropout_rate:
                continue

            corners = self._gate_to_corners_2d(pos)

            # Add Gaussian noise to corner coordinates
            if self.noise_std > 0:
                corners = corners + self._rng.normal(
                    scale=self.noise_std, size=corners.shape
                )

            detections.append(
                GateDetection(
                    corners_2d=corners,
                    confidence=1.0,
                    gate_id=gate_id,
                )
            )
        return detections
