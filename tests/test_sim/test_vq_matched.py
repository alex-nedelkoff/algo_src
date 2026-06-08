"""Unit tests for VQMatchedDynamics (the VQ-matched ACRO rate-loop sim).

Self-contained: uses an inline model dict mirroring sysid/vq_model.json structure so the test
runs without the (laptop-only) VQ data. Equivalence to the validated scalar reference (vq_sim.py)
is checked separately on the laptop (test_vq_matched.py there: max |dpos|+|dvel| = 0.0 over 1s).
"""
import numpy as np
import pytest

from sim.dynamics.vq_matched import VQMatchedDynamics, GRAVITY, quat_multiply_batch

# Inline grey-box params (subset of sysid/vq_model.json, COR-96)
MODEL = {
    "drag_linear_body": {"Dx": 0.2287, "Dy": 0.2287, "Dz": 0.0},
    "thrust": {"f0": 5.17486, "df_dthr": -56.01702},
    "rate_loop": {
        "roll": {"tau_ms": 26.073, "gain_G": -2.53796},
        "pitch": {"tau_ms": 24.465, "gain_G": 2.53212},
        "yaw": {"tau_ms": 44.520, "gain_G": -2.28876},
    },
    "weathervane": {"wv_coeff": -0.14881},
}
THR_HOVER = 0.2675   # -f0/df-ish: c=-(f0+df*thr)=g  => zero vertical accel


def test_hover_equilibrium():
    """Level drone at hover thrust, zero rate cmd -> ~zero translational accel."""
    dyn = VQMatchedDynamics(MODEL, roll_wv=True)
    s = dyn.reset(4)
    s2 = dyn.step(s, np.tile([THR_HOVER, 0.0, 0.0, 0.0], (4, 1)))
    # velocity change per step should be ~0 (hover); allow small thrust-curve rounding
    dv = s2[:, 3:6] - s[:, 3:6]
    assert np.all(np.abs(dv) < 1e-2), f"hover not in equilibrium: dv={dv[0]}"


def test_rate_loop_zoh_steady_state():
    """Holding a body-rate command drives omega -> G * wcmd (steady state), per axis."""
    dyn = VQMatchedDynamics(MODEL, dt=1 / 250.0, roll_wv=False)
    dyn.yaw_wv = 0.0  # isolate the rate loop from the weathervane coupling
    s = dyn.reset(1)
    wcmd = np.array([0.3, -0.2, 0.4])
    act = np.array([[THR_HOVER, *wcmd]])
    for _ in range(2000):  # >> tau, converge
        s = dyn.step(s, act)
        s[:, 3:6] = 0.0  # freeze translation so sideslip never builds (pure rate-loop test)
    G = dyn.Gax
    np.testing.assert_allclose(s[0, 10:13], G * wcmd, atol=2e-3)


def test_rate_loop_zoh_alpha():
    """One step of the rate loop matches the exact ZOH coefficient a=exp(-dt/tau)."""
    dt = 1 / 72.0
    dyn = VQMatchedDynamics(MODEL, dt=dt, roll_wv=False)
    s = dyn.reset(1)
    wcmd = np.array([0.5, 0.0, 0.0])
    s2 = dyn.step(s, np.array([[THR_HOVER, *wcmd]]))
    a = np.exp(-dt / dyn.tau[0])
    expect = a * 0.0 + (1 - a) * dyn.Gax[0] * wcmd[0]
    assert abs(s2[0, 10] - expect) < 1e-9


def test_quaternion_stays_normalized():
    dyn = VQMatchedDynamics(MODEL)
    s = dyn.reset(8); s[:, 3] = 2.0  # forward velocity excites weathervane
    rng = np.random.default_rng(0)
    for _ in range(500):
        act = np.column_stack([np.full(8, 0.3), rng.uniform(-1, 1, (8, 3))])
        s = dyn.step(s, act)
    norms = np.linalg.norm(s[:, 6:10], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    assert np.isfinite(s).all()


def test_batch_determinism_and_independence():
    """Identical inputs -> identical envs; perturbing one env doesn't affect others."""
    dyn = VQMatchedDynamics(MODEL)
    s = dyn.reset(16); s[:, 3] = 1.0
    act = np.tile([0.3, 0.1, 0.0, 0.0], (16, 1))
    s2 = dyn.step(s, act)
    assert np.allclose(s2, s2[0])
    act2 = act.copy(); act2[5, 1] = 0.9
    s3 = dyn.step(s, act2)
    others = np.delete(np.arange(16), 5)
    assert np.allclose(s3[others], s2[others])
    assert not np.allclose(s3[5], s2[5])


def test_actuation_latency_buffer():
    """With latency, the command takes `delay` steps to reach the rate loop."""
    dt = 1 / 100.0
    dyn = VQMatchedDynamics(MODEL, dt=dt, latency_s=3 * dt, roll_wv=False)
    s = dyn.reset(1)
    act = np.array([[THR_HOVER, 1.0, 0.0, 0.0]])
    rates = []
    for _ in range(6):
        s = dyn.step(s, act); rates.append(s[0, 10])
    # first `delay` steps: zero command applied -> omega stays ~0
    assert abs(rates[0]) < 1e-9 and abs(rates[1]) < 1e-9 and abs(rates[2]) < 1e-9
    assert abs(rates[3]) > 1e-6  # command now arrives


def _B_state(s):
    """Apply B=diag(1,-1,-1) (180 deg about body-x) to a (N,13) state: NED/FRD <-> ENU/FLU.

    pos/vel/omega transform as B@v; quat conjugates by b=[0,1,0,0] (q' = b (x) q (x) b*).
    """
    out = s.copy()
    out[:, 0:3] *= np.array([1.0, -1.0, -1.0])    # pos
    out[:, 3:6] *= np.array([1.0, -1.0, -1.0])    # vel
    out[:, 10:13] *= np.array([1.0, -1.0, -1.0])  # omega (body rates)
    n = s.shape[0]
    b = np.tile([0.0, 1.0, 0.0, 0.0], (n, 1)); bc = np.tile([0.0, -1.0, 0.0, 0.0], (n, 1))
    out[:, 6:10] = quat_multiply_batch(quat_multiply_batch(b, s[:, 6:10]), bc)
    return out


def _B_action(a):
    """Action [thr,wx,wy,wz] under B: thr & wx unchanged, wy,wz negate."""
    out = a.copy(); out[:, 2] *= -1.0; out[:, 3] *= -1.0
    return out


def test_ned_enu_frame_equivalence():
    """ENU model == NED model conjugated by B (proves the frame sign-flips are correct, incl. yaw).

    Non-circular: the NED model is the one validated bit-for-bit against vq_sim/VQ on the laptop;
    this checks the ENU constants reproduce it under the change of basis.
    """
    ned = VQMatchedDynamics(MODEL, frame="NED", roll_wv=True)
    enu = VQMatchedDynamics(MODEL, frame="ENU", roll_wv=True)
    rng = np.random.default_rng(7)
    s = np.zeros((32, 13))
    s[:, 0:3] = rng.uniform(-3, 3, (32, 3))
    s[:, 3:6] = rng.uniform(-4, 4, (32, 3))
    q = rng.uniform(-1, 1, (32, 4)); s[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)
    s[:, 10:13] = rng.uniform(-2, 2, (32, 3))
    a = np.column_stack([rng.uniform(0.1, 0.5, 32), rng.uniform(-1, 1, (32, 3))])
    # roll forward a few steps in both frames, comparing each step
    s_enu = _B_state(s)
    for _ in range(20):
        s_ned_next = ned.step(s, a)
        s_enu_next = enu.step(s_enu, _B_action(a))
        np.testing.assert_allclose(_B_state(s_ned_next), s_enu_next, atol=1e-9)
        s, s_enu = s_ned_next, s_enu_next


def test_enu_hover_and_yaw_sense():
    """ENU mode: hover holds (z-up), and +yaw-rate cmd yaws CCW about +z (right-handed)."""
    dyn = VQMatchedDynamics(MODEL, frame="ENU", roll_wv=False)
    s = dyn.reset(1)  # level, z-up
    s2 = dyn.step(s, np.array([[THR_HOVER, 0.0, 0.0, 0.0]]))
    assert abs(s2[0, 5]) < 1e-2  # zero vertical velocity change at hover (vel_z)
    # +yaw-rate command -> omega_z > 0 (CCW about up), and gain sign matches NED yaw gain
    s3 = dyn.step(s, np.array([[THR_HOVER, 0.0, 0.0, 1.0]]))
    assert np.sign(s3[0, 12]) == np.sign(dyn.Gax[2])


def test_directional_roll_weathervane_sign():
    """Forward sideslip (vbx>0, vby>0) produces a roll-rate moment (directional, COR-127)."""
    dyn = VQMatchedDynamics(MODEL, roll_wv=True)
    off = VQMatchedDynamics(MODEL, roll_wv=False)
    s = np.zeros((1, 13)); s[0, 6] = 1.0
    s[0, 3] = 2.0; s[0, 4] = 1.0  # world vel; level => body vel = same
    act = np.array([[THR_HOVER, 0.0, 0.0, 0.0]])
    s_on = dyn.step(s.copy(), act); s_off = off.step(s.copy(), act)
    assert s_on[0, 10] != s_off[0, 10]  # roll rate differs only via weathervane
