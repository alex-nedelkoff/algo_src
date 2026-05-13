"""MAVLink-specific coordinate adapters (ENU↔NED for MAVLink message fields).

All conversions here use the **standard ENU↔NED basis change**:
    (a, b, c)_ned ↔ (b, a, -c)_enu
which is its own inverse. Re-expressing a rotation under that basis change
permutes the imaginary part of the quaternion the same way the position
basis is permuted; the scalar stays put.

NOTE: sim/pybullet/coords.py:ned_to_enu_quaternion uses a *different*
cyclic-permutation transform (a,b,c)→(c,a,b) that exists only to land the
warehouse mesh's "up" axis on world +Z. Do not call it from MAVLink code:
mixing the two frames means a NED-pitch attitude command is interpreted
as an ENU-yaw command by the controller, and the drone yaws instead of
pitching.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    w, x, y, z = q_enu_wxyz
    q_ned_wxyz = np.array([w, y, x, -z])  # standard ENU→NED basis change
    return _quat_to_euler_zyx(q_ned_wxyz)


def enu_quat_to_ned_quat_xyzw(q_enu_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """ENU quaternion (wxyz) → NED quaternion (xyzw) for MAVLink ODOMETRY."""
    w, x, y, z = q_enu_wxyz
    return np.array([y, x, -z, w])


def ned_quat_wxyz_to_enu_quat(q_ned_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """NED quaternion (wxyz) → ENU quaternion (wxyz), standard basis change."""
    w, x, y, z = q_ned_wxyz
    return np.array([w, y, x, -z])
