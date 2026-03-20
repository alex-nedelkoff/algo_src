"""Axis 2: Control smoothness (command rate of change) descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def control_smoothness(
    motor_rpms: NDArray[np.float64],
    max_rpm: float,
    control_freq: float,
) -> float:
    """Compute mean normalized rate of change of motor commands.

    Returns the mean L2-norm of d(motor_RPMs)/dt, normalized by
    max_rpm * control_freq to produce a dimensionless [0, 1] value.
    """
    if motor_rpms.shape[0] < 2:
        return 0.0

    d_rpm = np.diff(motor_rpms, axis=0) * control_freq
    d_rpm_norms = np.linalg.norm(d_rpm, axis=1)

    max_rate_norm = np.linalg.norm(np.full(4, max_rpm)) * control_freq

    return float(np.clip(np.mean(d_rpm_norms / max_rate_norm), 0.0, 1.0))
