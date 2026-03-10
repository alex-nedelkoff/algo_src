"""Base detector interface and GateDetection dataclass.

Re-exports ``core.interfaces.Detector`` and defines the lightweight
``GateDetection`` data structure that is the canonical output of every
detector in the perception pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.interfaces import Detector

__all__ = ["Detector", "GateDetection"]


@dataclass
class GateDetection:
    """A single gate detection from a detector.

    Attributes:
        corners_2d: (4, 2) array of 2D pixel coordinates for each gate corner.
        confidence: Detection confidence score in [0, 1].
        gate_id: Integer identifier for the detected gate.
    """

    corners_2d: NDArray[np.float64]
    confidence: float
    gate_id: int

    def __post_init__(self) -> None:
        self.corners_2d = np.asarray(self.corners_2d, dtype=np.float64)
        if self.corners_2d.shape != (4, 2):
            raise ValueError(
                f"corners_2d must have shape (4, 2), got {self.corners_2d.shape}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
