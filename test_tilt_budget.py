import numpy as np
from gate_traj import GateTrajectory, G

GATES = np.array([[0, 0, -3], [12, 0, -3], [24, 0, -3], [36, 0, -3]], float)


def _traj(v_cruise=12.0, **kw):
    return GateTrajectory(GATES, v_cruise=v_cruise, **kw)


def test_speed_at_straight_is_drag_ceiling():
    tr = _traj(tilt_budget_deg=25.0, c_drag=0.057, margin=0.6)
    expected = np.sqrt(G * np.tan(np.radians(25.0)) * 0.6 / 0.057)   # ~6.94
    assert abs(tr.speed_at(0.0) - expected) < 1e-6
    assert 6.5 < tr.speed_at(0.0) < 7.5


def test_curvature_lowers_speed():
    tr = _traj(tilt_budget_deg=25.0, c_drag=0.057, margin=0.6)
    assert tr.speed_at(0.5) < tr.speed_at(0.0)
    assert tr.speed_at(2.0) < tr.speed_at(0.5)


def test_capped_at_v_cruise():
    tr = _traj(v_cruise=3.0, tilt_budget_deg=25.0, c_drag=0.057, margin=0.6)
    assert tr.speed_at(0.0) == 3.0          # 6.9 budget ceiling > 3.0 cap


def test_higher_budget_higher_ceiling():
    lo = _traj(tilt_budget_deg=18.0); hi = _traj(tilt_budget_deg=30.0)
    ratio = hi.speed_at(0.0) / lo.speed_at(0.0)
    assert abs(ratio - np.sqrt(np.tan(np.radians(30)) / np.tan(np.radians(18)))) < 1e-6
