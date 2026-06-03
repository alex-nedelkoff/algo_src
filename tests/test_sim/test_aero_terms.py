"""Linear rotor drag + weathervane moment terms in NumpyQuadDynamics (VQ-drone aero).

The two terms the stock sim lacked (it had only quadratic drag + motor-differential torque). Values
come from the VQ parameter pack; magnitudes are domain-randomized, so these verify the STRUCTURAL
wiring (coefficients applied to the right velocity/axis), not a physical sign. Motors off -> body force
is aero-only; thrust acts along body-z so it never touches the x/y drag or the weathervane checks.
"""
import numpy as np
from sim.dynamics.numpy_quad import NumpyQuadDynamics
from sim.dynamics.params import VehicleParams

DT = 0.001


def _params(**kw):
    base = dict(mass=1.0, inertia=np.diag([0.02, 0.02, 0.02]), arm_length=0.1,
                k_thrust=1e-5, k_torque=1e-7, max_rpm=30000.0)
    base.update(kw)
    return VehicleParams(**base)


def _accel(s0, s1):
    return (s1[:, 3:6] - s0[:, 3:6]) / DT


def test_linear_drag_is_minus_coeff_times_v():
    """Body x-velocity -> accel_x = -D_x * v_x (LINEAR, not quadratic)."""
    sim = NumpyQuadDynamics(_params(linear_drag_coeff=np.array([0.5, 0.0, 0.0])))
    s = sim.reset(1); s[0, 3:6] = [2.0, 0.0, 0.0]
    a = _accel(s, sim.step(s.copy(), np.zeros((1, 4)), dt=DT))
    assert np.isclose(a[0, 0], -0.5 * 2.0, atol=1e-4), a[0]      # -1.0 (linear); quad would be -2.0
    assert abs(a[0, 1]) < 1e-6, a[0]


def test_linear_drag_scales_linearly():
    """Doubling v doubles the linear drag accel (ratio 2, not 4)."""
    sim = NumpyQuadDynamics(_params(linear_drag_coeff=np.array([0.4, 0.0, 0.0])))
    s = sim.reset(2); s[0, 3] = 1.0; s[1, 3] = 2.0
    a = _accel(s, sim.step(s.copy(), np.zeros((2, 4)), dt=DT))
    assert np.isclose(a[1, 0] / a[0, 0], 2.0, atol=1e-4), (a[0, 0], a[1, 0])


def test_weathervane_yaw_from_sideslip():
    """weathervane_coeff=[_,_,wv_z]: sideslip v_y -> yaw accel wv_z*v_y/Izz."""
    Izz = 0.02
    sim = NumpyQuadDynamics(_params(inertia=np.diag([0.02, 0.02, Izz]),
                                    weathervane_coeff=np.array([0.0, 0.0, 0.1])))
    s = sim.reset(1); s[0, 3:6] = [0.0, 3.0, 0.0]; s[0, 10:13] = 0.0
    s2 = sim.step(s.copy(), np.zeros((1, 4)), dt=DT)
    assert np.isclose(s2[0, 12], (0.1 * 3.0 / Izz) * DT, rtol=1e-3), s2[0, 12]


def test_weathervane_couples_correct_axes():
    """[wv_x, wv_y, 0]: roll from v_y, pitch from v_x, no leak to yaw."""
    sim = NumpyQuadDynamics(_params(weathervane_coeff=np.array([0.2, 0.3, 0.0])))
    s = sim.reset(1); s[0, 3:6] = [1.0, 1.0, 0.0]; s[0, 10:13] = 0.0
    s2 = sim.step(s.copy(), np.zeros((1, 4)), dt=DT)
    assert s2[0, 10] > 1e-6 and s2[0, 11] > 1e-6, s2[0, 10:13]
    assert abs(s2[0, 12]) < 1e-9, s2[0, 12]


def test_aero_terms_domain_randomizable():
    """linear_drag_coeff + weathervane_coeff are registered for domain randomization."""
    from sim.domain_randomization import DomainRandomizer
    p = _params(linear_drag_coeff=np.array([0.52, 0.36, 0.1]),
                weathervane_coeff=np.array([0.05, 0.05, 0.05]))
    dr = DomainRandomizer.from_percentage(0.3, params=["linear_drag_coeff", "weathervane_coeff"])
    p2 = dr.apply(p, np.random.default_rng(0))
    assert not np.allclose(p2.linear_drag_coeff, p.linear_drag_coeff), p2.linear_drag_coeff
    assert not np.allclose(p2.weathervane_coeff, p.weathervane_coeff), p2.weathervane_coeff
    assert p2.linear_drag_coeff.shape == (3,) and p2.weathervane_coeff.shape == (3,)


def test_aero_terms_default_off():
    """No aero coeffs -> no horizontal force (unchanged from stock sim)."""
    sim = NumpyQuadDynamics(_params())
    s = sim.reset(1); s[0, 3:6] = [2.0, 1.0, 0.0]
    a = _accel(s, sim.step(s.copy(), np.zeros((1, 4)), dt=DT))
    assert abs(a[0, 0]) < 1e-9 and abs(a[0, 1]) < 1e-9, a[0]
