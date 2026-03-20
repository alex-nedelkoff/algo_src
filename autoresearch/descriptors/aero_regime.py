"""Axis 3: Aerodynamic regime (speed x angle-of-attack) descriptor."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _body_z_from_quaternion(quaternions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract body z-axis direction from quaternions (w,x,y,z convention).

    Returns (T, 3) array of unit vectors.
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    bz_x = 2.0 * (x * z + w * y)
    bz_y = 2.0 * (y * z - w * x)
    bz_z = 1.0 - 2.0 * (x * x + y * y)
    return np.column_stack([bz_x, bz_y, bz_z])


def aero_regime_index(
    velocities: NDArray[np.float64],
    quaternions: NDArray[np.float64],
) -> float:
    """Compute mean(||v|| * sin(alpha)) where alpha is angle between velocity and body z.

    Higher values indicate more time spent in aerodynamically complex regimes.
    """
    speeds = np.linalg.norm(velocities, axis=1)

    body_z = _body_z_from_quaternion(quaternions)

    nonzero = speeds > 1e-6
    cos_alpha = np.zeros_like(speeds)
    if np.any(nonzero):
        v_hat = velocities[nonzero] / speeds[nonzero, np.newaxis]
        cos_alpha[nonzero] = np.sum(v_hat * body_z[nonzero], axis=1)
        cos_alpha[nonzero] = np.clip(cos_alpha[nonzero], -1.0, 1.0)

    sin_alpha = np.sqrt(1.0 - cos_alpha ** 2)
    return float(np.mean(speeds * sin_alpha))
