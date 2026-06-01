import numpy as np
from aigp.dyn_model import quat_integrate, step, default_params

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _state(**kw):
    s = dict(pos=np.zeros(3), vel=np.zeros(3), quat=IDENT.copy(),
             omega=np.zeros(3), motor=np.zeros(4))
    s.update(kw)
    return s


def test_quat_integrate_zero_omega_is_identity():
    q = quat_integrate(IDENT, np.zeros(3), 0.01)
    assert np.allclose(q, IDENT)


def test_quat_integrate_yaw_rate():
    q = quat_integrate(IDENT, np.array([0, 0, 1.0]), 0.01)
    assert q[3] > 0 and np.isclose(np.linalg.norm(q), 1.0)


def test_hover_motor_gives_zero_vertical_accel():
    p = default_params()
    m = np.sqrt(p["g"] / (4 * p["c_T"]))
    s = _state(motor=np.full(4, m))
    s2 = step(s, np.full(4, m), p, 0.01)
    assert abs(s2["vel"][2]) < 1e-6


def test_more_thrust_accelerates_up():
    p = default_params()
    m = np.sqrt(p["g"] / (4 * p["c_T"]))
    s = _state(motor=np.full(4, m))
    s2 = step(s, np.full(4, m * 1.5), p, 0.02)
    assert s2["vel"][2] < 0


def test_roll_mix_gives_roll_rate():
    p = default_params()
    m = np.sqrt(p["g"] / (4 * p["c_T"]))
    s = _state(motor=np.full(4, m))
    u = np.full(4, m) + p["mix"][0] * 0.05 * m
    s2 = step(s, u, p, 0.02)
    assert s2["omega"][0] > 0
