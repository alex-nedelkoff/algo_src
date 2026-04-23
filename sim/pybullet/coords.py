"""NED↔ENU and UE↔NED coordinate conversions.

NED: x=North, y=East, z=Down (AirSim, Cosys-AirSim).
ENU: x=East,  y=North, z=Up   (PyBullet world convention used here).
UE:  Unreal Engine left-handed Z-up, units in centimetres.

NED↔ENU transform: (x, y, z)_enu = (y, x, -z)_ned. Equivalent to swapping
the X/Y axes and negating Z. Quaternions are converted by re-expressing
the rotation in the new basis: (w, x, y, z)_enu = (w, y, x, -z)_ned.

UE→NED transform: UE uses a left-handed coordinate system (Y is mirrored
relative to a right-handed frame). The basis flip matrix S = diag(1, -1, -1)
converts between UE and NED orientation; positions are divided by 100 to
convert centimetres to metres.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

# Basis-flip matrix: converts between UE left-handed and NED right-handed.
# S = diag(1, -1, -1)  →  R_ned = S @ R_ue_rel @ S.T
_S = np.diag([1.0, -1.0, -1.0])


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


def ue_to_ned_position(
    pos_ue_cm: NDArray[np.float64],
    playerstart_ue_cm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a UE world position (cm) to NED metres, relative to PlayerStart.

    UE is left-handed Z-up; NED is right-handed Z-down. The handedness flip is
    captured by negating Y and Z after computing the PlayerStart-relative offset.

    Args:
        pos_ue_cm: UE world-frame position in centimetres, shape (3,).
        playerstart_ue_cm: UE world-frame PlayerStart position in cm, shape (3,).

    Returns:
        NED position in metres, shape (3,).
    """
    rel = np.asarray(pos_ue_cm, dtype=np.float64) - np.asarray(
        playerstart_ue_cm, dtype=np.float64
    )
    return np.array([rel[0], -rel[1], -rel[2]], dtype=np.float64) / 100.0


def ue_to_ned_quaternion(
    rpy_deg: NDArray[np.float64],
    playerstart_rpy_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a UE RPY rotation (degrees) to a NED quaternion (w, x, y, z).

    UE Editor shows rotation as (Roll, Pitch, Yaw) applied as intrinsic
    rotations in order Roll(X) → Pitch(Y) → Yaw(Z), matching scipy's 'xyz'
    convention.  To express the gate rotation relative to PlayerStart, the
    PlayerStart rotation is right-multiplied out (R_rel = R_gate * R_ps⁻¹).
    The left-to-right handedness flip is then applied via
    R_ned = S @ R_rel @ S.T where S = diag(1, -1, -1).

    Args:
        rpy_deg: Gate rotation (Roll, Pitch, Yaw) in degrees.
        playerstart_rpy_deg: PlayerStart rotation (Roll, Pitch, Yaw) in degrees.

    Returns:
        NED quaternion as (w, x, y, z), unit norm, shape (4,).
    """
    R_gate = Rotation.from_euler("xyz", rpy_deg, degrees=True)
    R_ps = Rotation.from_euler("xyz", playerstart_rpy_deg, degrees=True)
    R_rel = R_gate * R_ps.inv()
    R_ned_mat = _S @ R_rel.as_matrix() @ _S.T
    R_ned = Rotation.from_matrix(R_ned_mat)
    # scipy returns (x, y, z, w); reorder to (w, x, y, z)
    q_xyzw = R_ned.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
