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
