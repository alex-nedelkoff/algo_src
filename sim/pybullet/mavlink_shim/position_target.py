"""Parsed SET_POSITION_TARGET_LOCAL_NED → PositionTarget (ENU world frame).

Honours the 12-bit type_mask: bits set ⇒ ignore that field group. Fields
disabled by the mask become zero (or current state, where applicable to the
PositionController).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


# MAVLink SET_POSITION_TARGET_LOCAL_NED type_mask bits (ignore flags).
_BIT_IGNORE_POS_X = 0x001
_BIT_IGNORE_POS_Y = 0x002
_BIT_IGNORE_POS_Z = 0x004
_BIT_IGNORE_VEL_X = 0x008
_BIT_IGNORE_VEL_Y = 0x010
_BIT_IGNORE_VEL_Z = 0x020
_BIT_IGNORE_ACC_X = 0x040
_BIT_IGNORE_ACC_Y = 0x080
_BIT_IGNORE_ACC_Z = 0x100
_BIT_FORCE_FLAG   = 0x200  # accel field encodes force in N (we treat as accel; warn only)
_BIT_IGNORE_YAW   = 0x400
_BIT_IGNORE_YAWR  = 0x800

_MASK_POS = _BIT_IGNORE_POS_X | _BIT_IGNORE_POS_Y | _BIT_IGNORE_POS_Z
_MASK_VEL = _BIT_IGNORE_VEL_X | _BIT_IGNORE_VEL_Y | _BIT_IGNORE_VEL_Z
_MASK_ACC = _BIT_IGNORE_ACC_X | _BIT_IGNORE_ACC_Y | _BIT_IGNORE_ACC_Z


@dataclass
class PositionTarget:
    """A parsed SET_POSITION_TARGET_LOCAL_NED in ENU world frame.

    The use_* booleans invert the MAVLink type_mask's ignore-sense:
    use_position is True when the position bits are zero (i.e. the
    position field is ACTIVE, not ignored). Downstream consumers
    (PositionController) read the use_* flags; the raw type_mask is
    discarded after parsing.
    """
    pos_enu: NDArray[np.float64]    # (3,) m
    vel_enu: NDArray[np.float64]    # (3,) m/s
    accel_enu: NDArray[np.float64]  # (3,) m/s²
    yaw_enu: float                  # rad
    use_position: bool
    use_velocity: bool
    use_accel: bool
    use_yaw: bool


def _ned_to_enu_xyz(x: float, y: float, z: float) -> NDArray[np.float64]:
    """Standard NED→ENU for MAVLink fields: (a,b,c) → (b,a,-c)."""
    return np.array([y, x, -z], dtype=np.float64)


def parse_set_position_target_local_ned(msg: Any) -> PositionTarget:
    """Convert a pymavlink SET_POSITION_TARGET_LOCAL_NED → PositionTarget (ENU).

    type_mask bits set ⇒ ignore the corresponding field group (zeroed out).
    Yaw flips sign across the NED↔ENU basis change.
    """
    tm = int(msg.type_mask)
    if tm & _BIT_FORCE_FLAG:
        warnings.warn(
            "SET_POSITION_TARGET_LOCAL_NED FORCE bit set; treating accel field as accel (m/s²), "
            "not force (N). Mass-scaling not applied.",
            stacklevel=2,
        )
    use_pos = (tm & _MASK_POS) == 0
    use_vel = (tm & _MASK_VEL) == 0
    use_acc = (tm & _MASK_ACC) == 0
    use_yaw = (tm & _BIT_IGNORE_YAW) == 0

    pos = _ned_to_enu_xyz(msg.x, msg.y, msg.z) if use_pos else np.zeros(3)
    vel = _ned_to_enu_xyz(msg.vx, msg.vy, msg.vz) if use_vel else np.zeros(3)
    acc = _ned_to_enu_xyz(msg.afx, msg.afy, msg.afz) if use_acc else np.zeros(3)
    yaw = -float(msg.yaw) if use_yaw else 0.0

    return PositionTarget(
        pos_enu=pos, vel_enu=vel, accel_enu=acc, yaw_enu=yaw,
        use_position=use_pos, use_velocity=use_vel,
        use_accel=use_acc, use_yaw=use_yaw,
    )
