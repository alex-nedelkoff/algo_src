import numpy as np
from aigp.flight_telemetry import tilt_deg, euler_rpy_deg, sideslip_deg, along_cross


def q_roll(deg):   # wxyz quaternion for a pure roll about body-x
    h = np.radians(deg) / 2
    return np.array([np.cos(h), np.sin(h), 0.0, 0.0])


def q_yaw(deg):    # wxyz quaternion for a pure yaw about body-z
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def test_tilt_level_is_zero():
    assert np.isclose(tilt_deg(np.array([1.0, 0, 0, 0])), 0.0, atol=1e-6)


def test_tilt_roll_90_is_90():
    assert np.isclose(tilt_deg(q_roll(90)), 90.0, atol=1e-4)


def test_tilt_roll_30():
    assert np.isclose(tilt_deg(q_roll(30)), 30.0, atol=1e-4)


def test_euler_yaw_only():
    r, p, y = euler_rpy_deg(q_yaw(40))
    assert np.isclose(y, 40.0, atol=1e-4)
    assert np.isclose(r, 0.0, atol=1e-4) and np.isclose(p, 0.0, atol=1e-4)


def test_euler_roll_only():
    r, p, y = euler_rpy_deg(q_roll(25))
    assert np.isclose(r, 25.0, atol=1e-4)


# camera-forward (nose) = -body_x. At yaw=180deg, body_x=[-1,0] (NED) so camera points NORTH [+1,0].
def test_sideslip_zero_when_nose_first():
    # camera north, velocity north -> no slip
    assert np.isclose(abs(sideslip_deg(q_yaw(180), np.array([2.0, 0.0, 0.0]))), 0.0, atol=1e-3)


def test_sideslip_90_when_strafing():
    # camera north, velocity east -> ~90 deg slip
    assert np.isclose(abs(sideslip_deg(q_yaw(180), np.array([0.0, 2.0, 0.0]))), 90.0, atol=1e-3)


def test_sideslip_180_when_backward():
    # camera north, velocity south -> flying tail-first
    assert np.isclose(abs(sideslip_deg(q_yaw(180), np.array([-2.0, 0.0, 0.0]))), 180.0, atol=1e-3)


def test_sideslip_zero_when_stationary():
    assert sideslip_deg(q_yaw(0), np.array([0.0, 0.0, 0.0])) == 0.0


def test_along_cross_decomposition():
    along, cross = along_cross(np.array([2.0, 1.0, 5.0]), np.array([1.0, 0.0]))  # tangent = north
    assert np.isclose(along, 2.0) and np.isclose(cross, 1.0)   # lat = [-0,1] -> east component


def test_along_cross_pure_cross():
    along, cross = along_cross(np.array([0.0, 3.0, 0.0]), np.array([1.0, 0.0]))
    assert np.isclose(along, 0.0) and np.isclose(cross, 3.0)
