# tests/test_aigp/test_gate_traj.py
import numpy as np
from gate_traj import GateTrajectory


def _straight():
    # straight line along +N at constant z
    gates = np.array([[0, 0, -2.0], [10, 0, -2.0], [20, 0, -2.0]])
    return GateTrajectory(gates, v_cruise=2.5)


def test_straight_tangent_and_cruise():
    t = _straight()
    r = t.sample(t.s_max * 0.5)
    np.testing.assert_allclose(r["tang"][:2], [1.0, 0.0], atol=1e-3)   # points +N
    assert abs(r["kappa"]) < 1e-3                                       # straight
    assert abs(r["v"] - 2.5) < 1e-6                                     # full cruise on a straight


def test_sample_yaw_is_camera_forward():
    # body_x opposite travel (camera-forward): yaw = atan2(-tang)
    t = _straight()
    r = t.sample(t.s_max * 0.5)
    assert abs(((r["yaw"] - np.arctan2(-r["tang"][1], -r["tang"][0]) + np.pi) % (2*np.pi)) - np.pi) < 1e-6


def test_nearest_s_projects_along_path():
    t = _straight()
    s_near = t.nearest_s(np.array([5.0, 0.5, -2.0]))   # next to the line at ~5 m along
    assert 3.0 < s_near < 7.0


def test_curve_lowers_scheduled_speed():
    straight = _straight().sample(0.0)["v"]
    curve = GateTrajectory(np.array([[0,0,-2.0],[10,0,-2.0],[14,8,-2.0],[6,12,-2.0]]),
                           v_cruise=8.0, phi_max_deg=35.0)
    vmin = min(curve.sample(s)["v"] for s in np.linspace(0, curve.s_max, 40))
    assert vmin < 8.0                                   # curvature caps speed below cruise
