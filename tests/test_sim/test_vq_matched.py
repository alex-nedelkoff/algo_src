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


def test_17_state_passthrough():
    """GateRaceEnv uses a 17-wide state ([...,13:17]=motor); step must propagate 0:13 and
    leave the motor slice untouched (rate loop subsumes motor dynamics)."""
    dyn = VQMatchedDynamics(MODEL, frame="ENU")
    s13 = np.zeros((4, 13)); s13[:, 6] = 1.0; s13[:, 3] = 1.5
    s17 = np.zeros((4, 17)); s17[:, :13] = s13; s17[:, 13:17] = 7.0  # sentinel motor speeds
    a = np.tile([THR_HOVER, 0.1, -0.2, 0.3], (4, 1))
    out13 = dyn.step(s13, a); out17 = dyn.step(s17, a)
    np.testing.assert_allclose(out17[:, :13], out13, atol=1e-12)
    np.testing.assert_allclose(out17[:, 13:17], 7.0)  # motor slice preserved


def test_first_order_thrust_lag():
    """thrust_lag_s makes the applied thrust first-order toward the command (state in motor slot 13)."""
    dt = 0.01; tau = 0.085
    dyn = VQMatchedDynamics(MODEL, dt=dt, frame="ENU", thrust_lag_s=tau, roll_wv=False)
    s = np.zeros((1, 17)); s[:, 6] = 1.0; s[0, 13] = THR_HOVER
    a = np.array([[1.0, 0.0, 0.0, 0.0]])  # command max thrust
    thr = []
    for _ in range(40):
        s = dyn.step(s, a); thr.append(s[0, 13])
    thr = np.array(thr); a_thr = np.exp(-dt / tau)
    assert abs(thr[0] - (a_thr * THR_HOVER + (1 - a_thr) * 1.0)) < 1e-6  # exact first-order step
    assert thr[0] < thr[10] < thr[-1] and thr[-1] > 0.9                  # monotone rise, ~converged


def test_directional_roll_weathervane_sign():
    """Forward sideslip (vbx>0, vby>0) produces a roll-rate moment (directional, COR-127)."""
    dyn = VQMatchedDynamics(MODEL, roll_wv=True)
    off = VQMatchedDynamics(MODEL, roll_wv=False)
    s = np.zeros((1, 13)); s[0, 6] = 1.0
    s[0, 3] = 2.0; s[0, 4] = 1.0  # world vel; level => body vel = same
    act = np.array([[THR_HOVER, 0.0, 0.0, 0.0]])
    s_on = dyn.step(s.copy(), act); s_off = off.step(s.copy(), act)
    assert s_on[0, 10] != s_off[0, 10]  # roll rate differs only via weathervane


# --- MIMO rate loop (COR-127): roll<->yaw cross-coupling the diagonal loop misses in a turn ---
MODEL_MIMO = {**MODEL, "rate_loop_mimo": {
    "A": [[0.9, 0.0, 0.1], [0.0, 0.8, 0.0], [0.05, 0.0, 0.95]],
    "B": [[-1.0, 0.0, 0.2], [0.0, 1.0, 0.0], [0.3, 0.0, -0.9]]}}


def test_mimo_rate_loop_applied():
    """When model has rate_loop_mimo, om[k+1] = A@om + B@wcmd (+weathervane); != the diagonal loop."""
    A = np.array(MODEL_MIMO["rate_loop_mimo"]["A"]); B = np.array(MODEL_MIMO["rate_loop_mimo"]["B"])
    dyn = VQMatchedDynamics(MODEL_MIMO, dt=1 / 72.0, roll_wv=True)
    s = np.zeros((3, 13)); s[:, 6] = 1.0                          # level, vel=0 => vb=0 => weathervane=0
    s[:, 10:13] = [[0.3, -0.2, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, -1.0]]
    a = np.array([[0.27, 0.4, -0.3, 0.6], [0.27, 0.0, 0.0, 0.0], [0.27, -1.0, 0.5, 0.8]])
    out = dyn.step(s, a)
    np.testing.assert_allclose(out[:, 10:13], s[:, 10:13] @ A.T + a[:, 1:4] @ B.T, atol=1e-12)
    diag = VQMatchedDynamics(MODEL, dt=1 / 72.0, roll_wv=True)    # diagonal loop gives a different result
    assert not np.allclose(diag.step(s, a)[:, 10:13], out[:, 10:13])


def test_mimo_ned_enu_equivalence():
    """MIMO A,B conjugate correctly under B=diag(1,-1,-1): ENU == B(NED) step-for-step."""
    ned = VQMatchedDynamics(MODEL_MIMO, frame="NED", roll_wv=True)
    enu = VQMatchedDynamics(MODEL_MIMO, frame="ENU", roll_wv=True)
    rng = np.random.default_rng(11)
    s = np.zeros((16, 13))
    s[:, 0:3] = rng.uniform(-3, 3, (16, 3)); s[:, 3:6] = rng.uniform(-4, 4, (16, 3))
    q = rng.uniform(-1, 1, (16, 4)); s[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)
    s[:, 10:13] = rng.uniform(-2, 2, (16, 3))
    a = np.column_stack([rng.uniform(0.1, 0.5, 16), rng.uniform(-1, 1, (16, 3))])
    s_enu = _B_state(s)
    for _ in range(20):
        sn = ned.step(s, a); se = enu.step(s_enu, _B_action(a))
        np.testing.assert_allclose(_B_state(sn), se, atol=1e-9)
        s, s_enu = sn, se


# --- weathervane v2 (COR-127, 06-10): speed-dependent coeff, replaces const-yaw + vbx-roll ---
MODEL_WV2 = {**MODEL, "weathervane_v2": {
    "roll": {"a": -0.603, "b": 0.104}, "yaw": {"a": -0.128, "b": 0.047}, "vclip": 8.0}}


def test_wv2_speed_dependent_coeff():
    """v2 applies (a + b*min(|v|,vclip))*vby*dt on roll & yaw; coeff sign flips with speed."""
    wv2 = MODEL_WV2["weathervane_v2"]
    dyn = VQMatchedDynamics(MODEL_WV2, dt=1 / 72.0, roll_wv=True)
    base = VQMatchedDynamics(MODEL_WV2, dt=1 / 72.0, roll_wv=True)
    s = np.zeros((3, 13)); s[:, 6] = 1.0
    s[0, 3:5] = [0.5, 1.0]            # slow: |v|~1.1 -> roll coeff ~-0.49 (negative)
    s[1, 3:5] = [7.5, 3.0]            # fast: |v|~8.1 -> clipped at 8 -> roll coeff +0.229
    s[2, 3:5] = [0.5, 1.0]
    a = np.tile([0.27, 0.0, 0.0, 0.0], (3, 1))
    out = dyn.step(s, a)
    for i in range(2):                # level attitude: vb == vel; om started at 0
        vb = s[i, 3:6]; spd = min(np.linalg.norm(vb), 8.0)
        exp_r = (wv2["roll"]["a"] + wv2["roll"]["b"] * spd) * vb[1] / 72.0
        exp_y = (wv2["yaw"]["a"] + wv2["yaw"]["b"] * spd) * vb[1] / 72.0
        np.testing.assert_allclose(out[i, 10], exp_r, atol=1e-12)
        np.testing.assert_allclose(out[i, 12], exp_y, atol=1e-12)
    assert out[0, 10] < 0 < out[1, 10]        # roll wv flips sign low->high speed (vby>0 both)
    # roll_wv=False zeroes the roll axis only
    off = VQMatchedDynamics(MODEL_WV2, dt=1 / 72.0, roll_wv=False)
    o2 = off.step(s, a)
    np.testing.assert_allclose(o2[:, 10], 0.0, atol=1e-12)
    np.testing.assert_allclose(o2[:, 12], out[:, 12], atol=1e-12)


def test_wv2_ned_enu_equivalence():
    """wv2 term is basis-correct: ENU == B(NED) step-for-step (|v| invariant, vby & yaw flip)."""
    ned = VQMatchedDynamics(MODEL_WV2, frame="NED", roll_wv=True)
    enu = VQMatchedDynamics(MODEL_WV2, frame="ENU", roll_wv=True)
    rng = np.random.default_rng(7)
    s = np.zeros((12, 13))
    s[:, 0:3] = rng.uniform(-3, 3, (12, 3)); s[:, 3:6] = rng.uniform(-6, 6, (12, 3))
    q = rng.uniform(-1, 1, (12, 4)); s[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)
    s[:, 10:13] = rng.uniform(-2, 2, (12, 3))
    a = np.column_stack([rng.uniform(0.1, 0.5, 12), rng.uniform(-1, 1, (12, 3))])
    s_enu = _B_state(s)
    for _ in range(20):
        sn = ned.step(s, a); se = enu.step(s_enu, _B_action(a))
        np.testing.assert_allclose(_B_state(sn), se, atol=1e-9)
        s, s_enu = sn, se


# --- DR disturbance field (COR-127 BRAKE-WV-01): tilt-scaled correlated rate bias ---
MODEL_DIST = {**MODEL, "dr_rate_disturbance": {
    "bins": [{"tilt_deg": [0, 10], "sigma": [1.0, 0.5, 0.6]},
             {"tilt_deg": [10, 90], "sigma": [4.0, 1.2, 3.0]}],
    "block_s": 0.5, "scale": 1.0}}


def test_disturbance_off_by_default():
    """Canonical file carries no 'scale' key -> field inactive, dynamics deterministic."""
    m = {**MODEL, "dr_rate_disturbance": {k: v for k, v in MODEL_DIST["dr_rate_disturbance"].items()
                                          if k != "scale"}}
    dyn = VQMatchedDynamics(m, dt=1 / 72.0, roll_wv=False)
    s = dyn.reset(4); a = np.tile([0.27, 0.1, -0.1, 0.05], (4, 1))
    o1 = dyn.step(s, a)
    dyn2 = VQMatchedDynamics(m, dt=1 / 72.0, roll_wv=False)
    dyn2.reset(4)
    np.testing.assert_allclose(o1, dyn2.step(s, a), atol=0)


def test_disturbance_injects_correlated_bias():
    """scale=1: omega deviates from nominal; bias constant within a block, resampled across blocks."""
    np.random.seed(3)
    dyn = VQMatchedDynamics(MODEL_DIST, dt=1 / 72.0, roll_wv=False)
    nom = VQMatchedDynamics(MODEL, dt=1 / 72.0, roll_wv=False)
    s = dyn.reset(8); nom.reset(8)
    a = np.tile([0.27, 0.0, 0.0, 0.0], (8, 1))
    o_d = dyn.step(s, a); o_n = nom.step(s, a)
    d1 = o_d[:, 10:13] - o_n[:, 10:13]
    assert np.abs(d1).max() > 0                      # bias present
    b_first = dyn._dist_bias.copy()
    for _ in range(10):                              # within the 36-step block: bias unchanged
        o_d = dyn.step(o_d, a)
    np.testing.assert_allclose(dyn._dist_bias, b_first, atol=0)
    for _ in range(40):                              # crosses the block boundary: resampled
        o_d = dyn.step(o_d, a)
    assert not np.allclose(dyn._dist_bias, b_first)


def test_quad_drag_off_by_default():
    """No 'drag_quadratic_body' key -> exact old translational behavior."""
    dyn = VQMatchedDynamics(MODEL, roll_wv=False)
    s = dyn.reset(2)
    s[:, 3] = 10.0   # fast forward flight
    a = np.tile([THR_HOVER, 0.0, 0.0, 0.0], (2, 1))
    o = dyn.step(s, a)
    m2 = {**MODEL, "drag_quadratic_body": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "Dz": 0.0}}
    dyn2 = VQMatchedDynamics(m2, roll_wv=False)
    dyn2.reset(2)
    np.testing.assert_allclose(o, dyn2.step(s, a), atol=0)


def test_quad_drag_caps_terminal_speed():
    """REPLAY-01: with q>0 sustained-thrust speed must saturate near sqrt-law, far below the
    linear plant (which reached 92.7 m/s on the runaway replay)."""
    m = {**MODEL, "drag_quadratic_body": {"qx": 0.032, "qy": 0.032, "qz": 0.032, "Dz": 0.0},
         "drag_linear_body": {"Dx": 0.10, "Dy": 0.10, "Dz": 0.0},
         "thrust": {**MODEL["thrust"], "thr_floor": 0.0924}}
    dyn_q = VQMatchedDynamics(m, dt=1 / 72.0, roll_wv=False)
    dyn_l = VQMatchedDynamics(MODEL, dt=1 / 72.0, roll_wv=False)
    a = np.tile([0.9, 0.0, 0.0, 0.0], (1, 1))
    sq = dyn_q.reset(1); sl = dyn_l.reset(1)
    sq[0, 6:10] = sl[0, 6:10] = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]  # pitched 90deg: thrust horizontal
    for _ in range(72 * 6):
        sq = dyn_q.step(sq, a); sl = dyn_l.step(sl, a)
    vq = np.linalg.norm(sq[0, 3:6]); vl = np.linalg.norm(sl[0, 3:6])
    assert vq < 45.0 < vl


def test_thrust_floor_clamps_negative_collective():
    """thr below -f0/df extrapolates to props pushing DOWN on the linear map; the floor stops it."""
    m = {**MODEL, "thrust": {**MODEL["thrust"], "thr_floor": 0.0924}}
    dyn_f = VQMatchedDynamics(m, roll_wv=False)
    dyn_0 = VQMatchedDynamics(MODEL, roll_wv=False)
    a = np.tile([0.0, 0.0, 0.0, 0.0], (1, 1))    # zero throttle
    sf = dyn_f.reset(1); s0 = dyn_0.reset(1)
    of = dyn_f.step(sf, a); o0 = dyn_0.step(s0, a)
    # NED: +z down. Floored plant falls at ~g (floor 0.0924 vs exact -f0/df leaves ~1e-3 m/s^2);
    # unfloored falls faster (negative thrust).
    assert np.isclose(of[0, 5], GRAVITY * dyn_f.dt, atol=1e-4)
    assert o0[0, 5] > of[0, 5]
