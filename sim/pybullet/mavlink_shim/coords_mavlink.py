"""MAVLink-specific coordinate adapters (ENU↔NED for MAVLink message fields).

Builds on sim/pybullet/coords.py for the basic flips. Adds:
  - enu_quat_to_ned_euler: ATTITUDE message uses (roll, pitch, yaw)
    Euler angles in NED frame.
  - enu_quat_to_ned_quat_xyzw: ODOMETRY uses NED quaternion in xyzw
    order (MAVLink convention).
  - ned_quat_wxyz_to_enu_quat: inverse, used for round-trip tests.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.pybullet.coords import ned_to_enu_quaternion


def _quat_to_euler_zyx(q_wxyz: NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert quaternion (wxyz) to (roll, pitch, yaw) using Z-Y-X intrinsic."""
    w, x, y, z = q_wxyz
    # roll (X axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    # pitch (Y axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = float(np.arcsin(sinp))
    # yaw (Z axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def enu_quat_to_ned_euler(q_enu_wxyz: NDArray[np.float64]) -> tuple[float, float, float]:
    """ENU quaternion → NED Euler (roll, pitch, yaw) in radians."""
    # Step 1: re-express the same physical rotation in NED basis.
    # Inverse of ned_to_enu_quaternion: (w,x,y,z)_ned = (w,y,x,-z)_enu
    w, x, y, z = q_enu_wxyz
    q_ned_wxyz = np.array([w, y, x, -z])
    return _quat_to_euler_zyx(q_ned_wxyz)


def enu_quat_to_ned_quat_xyzw(q_enu_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """ENU quaternion (wxyz) → NED quaternion (xyzw) for MAVLink ODOMETRY."""
    w, x, y, z = q_enu_wxyz
    # NED wxyz, then reorder to xyzw
    return np.array([y, x, -z, w])


def ned_quat_wxyz_to_enu_quat(q_ned_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """NED quaternion (wxyz) → ENU quaternion (wxyz). Same math as ned_to_enu_quaternion."""
    return ned_to_enu_quaternion(q_ned_wxyz)
