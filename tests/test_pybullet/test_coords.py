"""Tests for NED↔ENU and UE↔NED coordinate conversions."""
import numpy as np

from sim.pybullet.coords import (
    ned_to_enu_position,
    ned_to_enu_quaternion,
    ue_to_ned_position,
    ue_to_ned_quaternion,
)


# ---------------------------------------------------------------------------
# NED → ENU (verbatim from predecessor plan)
# ---------------------------------------------------------------------------

def test_ned_to_enu_position_swaps_xy_and_negates_z():
    """NED → PyBullet-world is now the cyclic permutation (a, b, c) → (c, a, b)
    (standard NED→ENU then Ry(-90) to put the mesh's 'up' onto world +Z)."""
    pos_ned = np.array([1.0, 2.0, 3.0])
    pos_world = ned_to_enu_position(pos_ned)
    np.testing.assert_array_almost_equal(pos_world, [3.0, 1.0, 2.0])


def test_ned_to_enu_position_zero_is_zero():
    np.testing.assert_array_almost_equal(
        ned_to_enu_position(np.zeros(3)), np.zeros(3)
    )


def test_ned_to_enu_position_array_works_on_batch():
    pts_ned = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    pts_world = ned_to_enu_position(pts_ned)
    np.testing.assert_array_almost_equal(pts_world, [[3, 1, 2], [6, 4, 5]])


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


# ---------------------------------------------------------------------------
# UE → NED position
# ---------------------------------------------------------------------------

def test_ue_to_ned_position_identity_at_origin():
    """Gate at same UE world pos as PlayerStart → NED zero."""
    pos_ue = np.array([100.0, 200.0, 300.0])
    playerstart = np.array([100.0, 200.0, 300.0])
    ned = ue_to_ned_position(pos_ue, playerstart)
    np.testing.assert_array_almost_equal(ned, [0.0, 0.0, 0.0])


def test_ue_to_ned_position_gate01_hand_computed():
    """Gate_01 from warehouse_fab_v1_gates_ue.yaml.

    PlayerStart UE: (7580, 470, 142) cm
    Gate_01 UE:     (7570, 270, 150) cm
    UE-cm relative: (-10, -200, 8)
    NED-m (true handedness flip: only Z negated, /100): (-0.10, -2.00, -0.08)
    """
    pos_ue_cm = np.array([7570.0, 270.0, 150.0])
    playerstart_cm = np.array([7580.0, 470.0, 142.0])
    ned = ue_to_ned_position(pos_ue_cm, playerstart_cm)
    np.testing.assert_array_almost_equal(ned, [-0.10, -2.00, -0.08], decimal=6)


def test_ue_to_ned_position_cm_to_m_scaling():
    """100 cm offset in UE X → 1.0 m in NED X."""
    pos_ue_cm = np.array([100.0, 0.0, 0.0])
    playerstart_cm = np.array([0.0, 0.0, 0.0])
    ned = ue_to_ned_position(pos_ue_cm, playerstart_cm)
    np.testing.assert_array_almost_equal(ned, [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# UE → NED quaternion
# ---------------------------------------------------------------------------

def test_ue_to_ned_quaternion_identity():
    """Gate and PlayerStart both at zero rotation → identity quaternion."""
    q = ue_to_ned_quaternion([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(q, [1.0, 0.0, 0.0, 0.0], decimal=6)


def test_ue_to_ned_quaternion_is_unit_norm():
    """Result is always a unit quaternion."""
    q = ue_to_ned_quaternion([0.0, 0.0, 90.0], [0.0, 0.0, -90.0])
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_ue_to_ned_quaternion_gate01_hand_computed():
    """Gate_01 yaw=90, PlayerStart yaw=-90 → relative yaw=180 → still 180° about Z.

    With the corrected basis change S=diag(1,1,-1), a 180° Z rotation in UE
    becomes a 180° Z rotation in NED with the SAME sign on the Z imaginary
    part (single Z flip in S still preserves a 180° Z rotation).
    Quaternion (w,x,y,z) ≈ (0, 0, 0, ±1).
    """
    q = ue_to_ned_quaternion([0.0, 0.0, 90.0], [0.0, 0.0, -90.0])
    np.testing.assert_array_almost_equal(abs(q[3]), 1.0, decimal=6)
    np.testing.assert_array_almost_equal(q[0], 0.0, decimal=6)
    np.testing.assert_array_almost_equal(q[1], 0.0, decimal=6)
    np.testing.assert_array_almost_equal(q[2], 0.0, decimal=6)


def test_ue_to_ned_quaternion_pure_yaw_90():
    """Gate at 90° UE yaw, PlayerStart at 0° → 90° about NED Z (same sign).

    With basis flip S=diag(1,1,-1), a pure Z rotation R commutes with S
    (only the Z-axis basis vector flips, leaving the X/Y rotation block
    intact). So a +90° Z rotation in UE is a +90° Z rotation in NED.
    Quaternion: (cos45, 0, 0, +sin45).
    """
    q = ue_to_ned_quaternion([0.0, 0.0, 90.0], [0.0, 0.0, 0.0])
    expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    np.testing.assert_array_almost_equal(q, expected, decimal=6)
