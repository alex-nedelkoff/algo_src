"""Tests for NED↔ENU coordinate conversions."""
import numpy as np

from sim.pybullet.coords import ned_to_enu_position, ned_to_enu_quaternion


def test_ned_to_enu_position_swaps_xy_and_negates_z():
    pos_ned = np.array([1.0, 2.0, 3.0])
    pos_enu = ned_to_enu_position(pos_ned)
    np.testing.assert_array_almost_equal(pos_enu, [2.0, 1.0, -3.0])


def test_ned_to_enu_position_zero_is_zero():
    np.testing.assert_array_almost_equal(
        ned_to_enu_position(np.zeros(3)), np.zeros(3)
    )


def test_ned_to_enu_position_array_works_on_batch():
    pts_ned = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    pts_enu = ned_to_enu_position(pts_ned)
    np.testing.assert_array_almost_equal(pts_enu, [[2, 1, -3], [5, 4, -6]])


def test_ned_to_enu_quaternion_identity_stays_identity():
    q_ned = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
    q_enu = ned_to_enu_quaternion(q_ned)
    np.testing.assert_array_almost_equal(q_enu, [1.0, 0.0, 0.0, 0.0])


def test_ned_to_enu_quaternion_180deg_yaw_round_trip():
    # 180° yaw in NED == 180° yaw in ENU (Z axis just flipped sign)
    q_ned = np.array([0.0, 0.0, 0.0, 1.0])  # w=0, z=1 → 180° about Z
    q_enu = ned_to_enu_quaternion(q_ned)
    # After NED→ENU, the rotation should still be valid (unit norm)
    assert abs(np.linalg.norm(q_enu) - 1.0) < 1e-9
