"""Tests for MAVLink-specific coordinate adapters."""
import numpy as np
import pytest

from sim.pybullet.mavlink_shim.coords_mavlink import (
    enu_quat_to_ned_euler,
    enu_quat_to_ned_quat_xyzw,
    ned_quat_wxyz_to_enu_quat,
)


def test_enu_quat_identity_to_ned_euler_is_zero():
    q_enu_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    roll, pitch, yaw = enu_quat_to_ned_euler(q_enu_wxyz)
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw) < 1e-9


def test_enu_quat_yaw_90deg_to_ned_euler_yaw_negative_90():
    # 90° yaw around ENU +Z axis (East→North) corresponds to -90° yaw in NED
    # (because NED yaw is around +Z_NED = -Z_ENU, so signs flip).
    half = np.sqrt(0.5)
    q_enu_wxyz = np.array([half, 0.0, 0.0, half])  # 90° rotation about +Z_ENU
    roll, pitch, yaw = enu_quat_to_ned_euler(q_enu_wxyz)
    assert abs(roll) < 1e-9
    assert abs(pitch) < 1e-9
    assert abs(yaw - (-np.pi / 2)) < 1e-6


def test_enu_quat_to_ned_quat_xyzw_round_trip():
    # Take a non-trivial ENU quaternion, convert to NED xyzw, convert back; same.
    q_enu_wxyz = np.array([0.5, 0.5, 0.5, 0.5])  # arbitrary
    q_enu_wxyz /= np.linalg.norm(q_enu_wxyz)

    q_ned_xyzw = enu_quat_to_ned_quat_xyzw(q_enu_wxyz)

    # Reorder NED xyzw → wxyz, then NED→ENU
    q_ned_wxyz = np.array([q_ned_xyzw[3], q_ned_xyzw[0], q_ned_xyzw[1], q_ned_xyzw[2]])
    q_back_enu_wxyz = ned_quat_wxyz_to_enu_quat(q_ned_wxyz)

    np.testing.assert_array_almost_equal(q_back_enu_wxyz, q_enu_wxyz)


def test_enu_to_ned_quat_xyzw_returns_unit_quaternion():
    q_enu_wxyz = np.array([0.7071, 0.0, 0.7071, 0.0])
    q_ned_xyzw = enu_quat_to_ned_quat_xyzw(q_enu_wxyz)
    assert abs(np.linalg.norm(q_ned_xyzw) - 1.0) < 1e-4
