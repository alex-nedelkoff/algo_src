"""NED↔ENU and UE↔NED coordinate conversions.

NED: x=North, y=East, z=Down (AirSim, Cosys-AirSim).
ENU: x=East,  y=North, z=Up   (PyBullet world convention used here).
UE:  Unreal Engine left-handed Z-up, units in centimetres.

NED↔ENU transform: (x, y, z)_enu = (y, x, -z)_ned. Equivalent to swapping
the X/Y axes and negating Z. Quaternions are converted by re-expressing
the rotation in the new basis: (w, x, y, z)_enu = (w, y, x, -z)_ned.

UE→NED transform: UE is left-handed Z-up; NED is right-handed Z-down. A
true handedness flip needs an ODD number of axis sign flips. The basis
flip matrix S = diag(1, 1, -1) (only Z negated) gives det = -1, which is
the actual UE↔NED handedness change. Positions are divided by 100 to
convert centimetres to metres.

Historical note: an earlier version of this module used S = diag(1, -1, -1)
(two sign flips, det = +1), which was NOT a handedness flip. That bug
caused warehouse-mesh vertices and gate positions to end up Z-mirrored
relative to each other when loaded into PyBullet. Fixed 2026-04-27.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

# Basis-flip matrix: converts UE left-handed Z-up to NED right-handed Z-down.
# Only Z is negated → det = -1, a true handedness flip.
_S = np.diag([1.0, 1.0, -1.0])


def ned_to_enu_position(pos_ned: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert position(s) from NED to PyBullet-world.

    Combined transform: standard NED→ENU `(a, b, c) → (b, a, -c)` followed
    by a Ry(-90) rotation `(x, y, z) → (-z, y, x)` that brings the mesh's
    "up" axis (which empirically lands on world +X without it) onto world
    +Z. Net result: `(a, b, c) → (c, a, b)`.

    The Ry(-90) tail was added 2026-04-27 after interactive verification
    showed the warehouse roof was rendering along +X instead of +Z. Applied
    in BOTH this position helper and `flip_mesh_ned_to_enu` so the mesh
    and the gate positions stay in the same frame.

    Works on a single (3,) vector or a batch (N, 3).
    """
    pos = np.asarray(pos_ned, dtype=np.float64)
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 2]   # world X = NED.z (was NED.y)
    out[..., 1] = pos[..., 0]   # world Y = NED.x
    out[..., 2] = pos[..., 1]   # world Z = NED.y (was -NED.z)
    return out


def ned_to_enu_quaternion(q_ned_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a quaternion (w, x, y, z) from NED to PyBullet-world.

    Re-expresses the rotation under the basis change `(a, b, c) → (c, a, b)`
    (a cyclic axis permutation, det = +1, handedness preserved). The scalar
    part is unchanged; the imaginary part is permuted accordingly:
    `(w, x, y, z)_world = (w, z, x, y)_ned`.
    """
    q = np.asarray(q_ned_wxyz, dtype=np.float64)
    return np.array([q[0], q[3], q[1], q[2]])


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
    # True handedness flip: only Z is negated (UE +Z up → NED +Z down).
    return np.array([rel[0], rel[1], -rel[2]], dtype=np.float64) / 100.0


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
    R_ned = S @ R_rel @ S.T where S = diag(1, 1, -1) (true handedness flip).

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
