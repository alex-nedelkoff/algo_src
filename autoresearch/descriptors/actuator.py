"""Axis 1: Actuator utilization (aggressiveness) descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def actuator_utilization(motor_rpms: NDArray[np.float64], max_rpm: float) -> float:
    """Compute mean actuator utilization across a trajectory.

    Returns the mean L2-norm of motor RPMs divided by the L2-norm of max RPMs.
    Range: [0, 1]. Values near 1 indicate near-saturation operation.
    """
    norms = np.linalg.norm(motor_rpms, axis=1)
    max_norm = np.linalg.norm(np.full(4, max_rpm))
    return float(np.mean(norms / max_norm))
