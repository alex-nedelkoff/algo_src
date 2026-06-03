import numpy as np
import pandas as pd
from aigp.aero_dataset import build_targets, world_to_body_vel, finite_diff_filtered

def test_world_to_body_vel_identity():
    R = np.eye(3); v_world = np.array([1.0, 2.0, 3.0])
    assert np.allclose(world_to_body_vel(v_world, R), v_world)

def test_finite_diff_filtered_constant_rate():
    t = np.linspace(0, 1, 101)
    w = np.stack([2.0 * t, 0 * t, -1.0 * t], axis=1)
    a = finite_diff_filtered(t, w)
    assert np.allclose(np.median(a[5:-5], axis=0), [2.0, 0.0, -1.0], atol=0.1)

def test_build_targets_coast_force_is_specific_force():
    df = pd.DataFrame({
        "t": [0.0, 0.004], "vx": [5.0, 5.0], "vy": [0, 0], "vz": [0, 0],
        "qw": [1, 1], "qx": [0, 0], "qy": [0, 0], "qz": [0, 0],
        "wx": [0, 0], "wy": [0, 0], "wz": [0, 0],
        "fx": [-1.2, -1.2], "fy": [0, 0], "fz": [0, 0],
        "thrust_accel": [0.0, 0.0], "coast": [True, True],
    })
    out = build_targets(df, I_ratio=np.array([3.7, 1.0, 1.0]))
    assert out["a_aero"].shape[1] == 3
    assert np.allclose(out["a_aero"][0], [-1.2, 0.0, 0.0])
    assert out["coast"].all()

def test_build_targets_powered_thrust_subtracted():
    """Powered (non-coast) sample: thrust_accel=T, identity attitude.
    a_aero[i] == f_body[i] - [0, 0, -T]  (thrust subtracted along body -z)."""
    T = 9.81
    df = pd.DataFrame({
        "t": [0.0, 0.004], "vx": [5.0, 5.0], "vy": [0, 0], "vz": [0, 0],
        "qw": [1, 1], "qx": [0, 0], "qy": [0, 0], "qz": [0, 0],
        "wx": [0, 0], "wy": [0, 0], "wz": [0, 0],
        "fx": [-1.2, -1.2], "fy": [0.3, 0.3], "fz": [T - 0.5, T - 0.5],
        "thrust_accel": [T, T], "coast": [False, False],
    })
    out = build_targets(df, I_ratio=np.array([3.7, 1.0, 1.0]))
    expected_aero = np.array([-1.2, 0.3, T - 0.5]) - np.array([0.0, 0.0, -T])
    assert np.allclose(out["a_aero"][0], expected_aero), \
        f"Expected {expected_aero}, got {out['a_aero'][0]}"
    assert not out["coast"].any()

def test_build_targets_al_aero_gyro_coupling():
    """Constant angular rate -> filtered dw/dt ~ 0 -> al_aero ~ omega x (Ir*omega)."""
    N = 200
    t = np.linspace(0, 1, N)
    omega_val = np.array([1.0, 2.0, 0.5])
    Ir = np.array([3.7, 1.0, 1.0])
    expected_coupling = np.cross(omega_val, Ir * omega_val)
    df = pd.DataFrame({
        "t": t,
        "vx": np.ones(N), "vy": np.zeros(N), "vz": np.zeros(N),
        "qw": np.ones(N), "qx": np.zeros(N), "qy": np.zeros(N), "qz": np.zeros(N),
        "wx": np.full(N, omega_val[0]),
        "wy": np.full(N, omega_val[1]),
        "wz": np.full(N, omega_val[2]),
        "fx": np.zeros(N), "fy": np.zeros(N), "fz": np.zeros(N),
        "thrust_accel": np.zeros(N), "coast": np.ones(N, dtype=bool),
    })
    out = build_targets(df, I_ratio=Ir)
    # Interior samples (away from edges) should have near-zero alpha
    mid = out["al_aero"][N//4 : 3*N//4]
    assert np.allclose(mid, expected_coupling, atol=1e-4), \
        f"Expected gyro coupling {expected_coupling}, got mid mean {mid.mean(axis=0)}"


# ============================================================
# NEW TESTS — spline_deriv, leverarm_correct, build_targets_cl
# ============================================================
from aigp.aero_dataset import spline_deriv, leverarm_correct, build_targets_cl
from aigp import motor_model


def test_spline_deriv_sinusoid():
    t = np.linspace(0, 2, 400)
    w = np.sin(3 * t)
    dw = spline_deriv(t, w)
    expected = 3 * np.cos(3 * t)
    # check interior (skip 5 samples each side)
    assert np.allclose(dw[5:-5], expected[5:-5], atol=0.05), \
        f'Max error: {np.max(np.abs(dw[5:-5] - expected[5:-5]))}'


def test_leverarm_zero_r_identity():
    rng = np.random.default_rng(42)
    N = 30
    a_imu = rng.standard_normal((N, 3))
    omega = rng.standard_normal((N, 3))
    omega_dot = rng.standard_normal((N, 3))
    r = np.zeros(3)
    a_cg = leverarm_correct(a_imu, omega, omega_dot, r)
    assert np.allclose(a_cg, a_imu), 'zero lever arm should be identity'


def test_leverarm_pure_spin():
    N = 10
    wz = 2.0
    rx = 0.05
    omega = np.tile([0.0, 0.0, wz], (N, 1))
    omega_dot = np.zeros((N, 3))
    a_imu = np.zeros((N, 3))
    r = np.array([rx, 0.0, 0.0])
    a_cg = leverarm_correct(a_imu, omega, omega_dot, r)
    # -w x (w x r): w=[0,0,wz], r=[rx,0,0]
    # w x r = [0, wz*rx, 0]
    # w x (w x r) = [-wz^2*rx, 0, 0]
    # -w x (w x r) = [wz^2*rx, 0, 0]
    # a_cg = a_imu - w x (w x r) - wdot x r = 0 - (-[wz^2*rx,0,0]) - 0 = [wz^2*rx, 0, 0]
    expected_correction = np.array([wz**2 * rx, 0.0, 0.0])
    assert np.allclose(a_cg[0], expected_correction, atol=1e-10), \
        f'Expected {expected_correction}, got {a_cg[0]}'


def _make_hover_df(N=50, u_hover=0.5, dt=1/250):
    t = np.arange(N) * dt
    data = {
        't': t,
        'vx': np.zeros(N), 'vy': np.zeros(N), 'vz': np.zeros(N),
        'qw': np.ones(N),  'qx': np.zeros(N), 'qy': np.zeros(N), 'qz': np.zeros(N),
        'wx': np.zeros(N), 'wy': np.zeros(N), 'wz': np.zeros(N),
        'fx': np.zeros(N), 'fy': np.zeros(N), 'fz': np.full(N, -9.81),
        'u0': np.full(N, u_hover), 'u1': np.full(N, u_hover),
        'u2': np.full(N, u_hover), 'u3': np.full(N, u_hover),
    }
    return pd.DataFrame(data)


def _hover_motor_params(u_hover=0.5, g=9.81):
    geom = dict(motor_model.DEFAULT_GEOM)
    # quadratic: g(u)=u^2; 4 motors; k_f * 4 * u^2 = g => k_f = g / (4 * u^2)
    k_f = g / (4 * u_hover**2)
    return dict(geom, k_f=k_f, k_q=k_f * 0.02, form='quadratic')


def test_build_targets_cl_hover():
    u_hover = 0.5
    N = 50
    df = _make_hover_df(N=N, u_hover=u_hover)
    motor_params = _hover_motor_params(u_hover)
    Ir = np.array([3.7, 1.0, 1.0])
    kappa = 1.0
    out = build_targets_cl(df, Ir, kappa, motor_params)

    # T ~ 9.81 for all samples
    assert np.allclose(out['T'], 9.81, atol=0.01), f'T mean={out["T"].mean()}'

    # tau_aero ~ -tau_motor ~ 0 at symmetric hover (all 4 motors same output)
    T_val, tau_motor_hover = motor_model.motor_outputs_to_wrench(
        np.full(4, u_hover), motor_params)
    assert np.allclose(out['tau_aero'], -tau_motor_hover, atol=0.1), \
        f'tau_aero mean={out["tau_aero"].mean(axis=0)}'

    # a_aero ~ [0, 0, -9.81] (specific force, no lever arm correction at r=0)
    assert np.allclose(out['a_aero'], [0.0, 0.0, -9.81], atol=0.01), \
        f'a_aero mean={out["a_aero"].mean(axis=0)}'


def test_build_targets_cl_gyro_term():
    u_hover = 0.5
    N = 60
    dt = 1/250
    t = np.arange(N) * dt
    omega_val = np.array([0.5, 0.3, 0.0])
    Ir = np.array([3.7, 1.0, 1.0])
    kappa = 1.2

    data = {
        't': t,
        'vx': np.zeros(N), 'vy': np.zeros(N), 'vz': np.zeros(N),
        'qw': np.ones(N),  'qx': np.zeros(N), 'qy': np.zeros(N), 'qz': np.zeros(N),
        'wx': np.full(N, omega_val[0]),
        'wy': np.full(N, omega_val[1]),
        'wz': np.full(N, omega_val[2]),
        'fx': np.zeros(N), 'fy': np.zeros(N), 'fz': np.full(N, -9.81),
        'u0': np.full(N, u_hover), 'u1': np.full(N, u_hover),
        'u2': np.full(N, u_hover), 'u3': np.full(N, u_hover),
    }
    df = pd.DataFrame(data)
    motor_params = _hover_motor_params(u_hover)
    out = build_targets_cl(df, Ir, kappa, motor_params)

    # omega_dot ~ 0 (constant rate); tau_aero = kappa*(Ir*0 + cross(omega,Ir*omega)) - tau_motor
    _, tau_motor_hover = motor_model.motor_outputs_to_wrench(
        np.full(4, u_hover), motor_params)
    gyro_cross = np.cross(omega_val, Ir * omega_val)
    expected_tau = kappa * gyro_cross - tau_motor_hover
    # check interior samples (spline edge artifacts)
    interior = out['tau_aero'][10:-10]
    assert np.all(np.isfinite(interior)), 'tau_aero must be finite'
    assert np.allclose(interior, expected_tau, atol=0.05), \
        f'Expected tau_aero={expected_tau}, got mean={interior.mean(axis=0)}'
