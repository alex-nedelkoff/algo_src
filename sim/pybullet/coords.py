"""NED↔ENU coordinate conversions.

NED: x=North, y=East, z=Down (AirSim, Cosys-AirSim).
ENU: x=East,  y=North, z=Up   (PyBullet world convention used here).

The transform is: (x, y, z)_enu = (y, x, -z)_ned. Equivalent to swapping
the X/Y axes and negating Z. Quaternions are converted by re-expressing
the rotation in the new basis: (w, x, y, z)_enu = (w, y, x, -z)_ned.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ned_to_enu_position(pos_ned: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert position(s) from NED to ENU.

    Works on a single (3,) vector or a batch (N, 3).
    """
    pos = np.asarray(pos_ned, dtype=np.float64)
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 1]
    out[..., 1] = pos[..., 0]
    out[..., 2] = -pos[..., 2]
    return out


def ned_to_enu_quaternion(q_ned_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a quaternion (w, x, y, z) from NED to ENU.

    The rotation is the same physical rotation, expressed in the new basis.
    For the (x, y, z) → (y, x, -z) basis change, the imaginary parts swap
    X/Y and negate Z; the scalar part is unchanged.
    """
    q = np.asarray(q_ned_wxyz, dtype=np.float64)
    return np.array([q[0], q[2], q[1], -q[3]])
