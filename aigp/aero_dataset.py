"""Raw flight log -> regression dataset for aero sysID (pure numpy/pandas).
Input columns (per sample): t, vx/vy/vz (world NED), qw/qx/qy/qz (body->world), wx/wy/wz (gyro),
fx/fy/fz (HIGHRES_IMU specific force, body), thrust_accel (commanded thrust accel magnitude),
coast (bool). Output: aero force/moment targets + the coast mask."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from .geometry import quat_to_R


def world_to_body_vel(v_world, R) -> np.ndarray:
    """Transform world-frame velocity to body frame using rotation matrix R (body->world)."""
    return np.asarray(R, float).T @ np.asarray(v_world, float)


def finite_diff_filtered(t, w, win: int = 9) -> np.ndarray:
    """Angular acceleration = d(omega)/dt, central difference + moving-average smoothing."""
    t = np.asarray(t, float)
    w = np.asarray(w, float)
    a = np.gradient(w, t, axis=0)
    if win > 1 and len(a) > win:
        a = uniform_filter1d(a, size=win, axis=0, mode="nearest")
    return a


def build_targets(df: pd.DataFrame, I_ratio) -> dict:
    """Per-sample body velocity, aero force target, aero moment target, coast mask.

    a_aero  = f_body - thrust_body        (thrust along body -z)
    al_aero = alpha + omega x (I_ratio*omega)   (motor_torque omitted: rely on coast/zero-motor samples)
    """
    if len(df) == 0:
        raise ValueError("build_targets: empty DataFrame")

    t = df["t"].to_numpy()
    # I_ratio must be a 3-element vector (Ix/Iz, Iy/Iz, 1)
    Ir = np.asarray(I_ratio, float).ravel()
    assert Ir.shape == (3,), f"I_ratio must have 3 elements, got {Ir.shape}"
    n = len(df)
    vb = np.zeros((n, 3))
    aero_f = np.zeros((n, 3))
    w = df[["wx", "wy", "wz"]].to_numpy()
    f = df[["fx", "fy", "fz"]].to_numpy()

    # thrust_accel is an m/s^2 magnitude (non-negative); thrust acts along body -z
    assert (df["thrust_accel"].to_numpy() >= 0).all(), \
        "thrust_accel must be a non-negative magnitude (m/s^2)"

    for i in range(n):
        q = df[["qw", "qx", "qy", "qz"]].iloc[i].to_numpy()
        R = quat_to_R(q)
        v_world = df[["vx", "vy", "vz"]].iloc[i].to_numpy()
        vb[i] = world_to_body_vel(v_world, R)
        thrust_body = np.array([0.0, 0.0, -float(df["thrust_accel"].iloc[i])])
        aero_f[i] = f[i] - thrust_body

    if not (np.isfinite(vb).all() and np.isfinite(aero_f).all()):
        raise ValueError("NaN/Inf in body velocity or aero force — check quaternion/IMU columns")

    alpha = finite_diff_filtered(t, w)
    gyro_coupling = np.cross(w, Ir * w)
    aero_m = alpha + gyro_coupling

    return {
        "v_body": vb,
        "omega": w,
        "a_aero": aero_f,
        "al_aero": aero_m,
        "coast": df["coast"].to_numpy().astype(bool),
    }


def spline_deriv(t, w):
    # Cubic-spline differentiation. Works for w (N,) or (N,3).
    from scipy.interpolate import CubicSpline
    t = np.asarray(t, float)
    w = np.asarray(w, float)
    cs = CubicSpline(t, w, axis=0)
    return cs(t, 1)


def leverarm_correct(a_imu, omega, omega_dot, r):
    # Lever-arm correction: a_cg = a_imu - omega x (omega x r) - omega_dot x r
    r = np.asarray(r, float).ravel()
    a_imu = np.asarray(a_imu, float)
    omega = np.asarray(omega, float)
    omega_dot = np.asarray(omega_dot, float)
    if np.all(r == 0.0):
        return a_imu.copy()
    centripetal = np.cross(omega, np.cross(omega, r))
    euler = np.cross(omega_dot, r)
    return a_imu - centripetal - euler


def build_targets_cl(df, I_ratio, kappa, motor_params, r=None, lag=0):
    # Closed-loop aero targets using measured motor outputs to reconstruct tau_motor.
    # Returns dict: v_body, omega, omega_dot, tau_aero, a_aero, T
    from .motor_model import motor_outputs_to_wrench

    if r is None:
        r = np.zeros(3)
    r = np.asarray(r, float).ravel()

    I_ratio = np.asarray(I_ratio, float).ravel()
    assert I_ratio.shape == (3,), f'I_ratio must have 3 elements, got {I_ratio.shape}'

    t = df['t'].to_numpy()
    omega = df[['wx', 'wy', 'wz']].to_numpy()
    f_imu = df[['fx', 'fy', 'fz']].to_numpy()
    N = len(df)

    # Body velocity: rotate each world velocity into body frame
    v_body = np.zeros((N, 3))
    for i in range(N):
        q = df[['qw', 'qx', 'qy', 'qz']].iloc[i].to_numpy()
        R = quat_to_R(q)
        v_world = df[['vx', 'vy', 'vz']].iloc[i].to_numpy()
        v_body[i] = world_to_body_vel(v_world, R)

    # Angular acceleration via spline
    omega_dot = spline_deriv(t, omega)

    # Reconstruct motor torques and thrust per sample
    tau_motor = np.zeros((N, 3))
    T_arr = np.zeros(N)
    u_cols = df[['u0', 'u1', 'u2', 'u3']].to_numpy()
    for i in range(N):
        T_i, tau_i = motor_outputs_to_wrench(u_cols[i], motor_params)
        tau_motor[i] = tau_i
        T_arr[i] = T_i

    # Aero moment: kappa*(I_ratio*omega_dot + omega x (I_ratio*omega)) - tau_motor
    gyro_term = np.cross(omega, I_ratio * omega)
    tau_aero = kappa * (I_ratio * omega_dot + gyro_term) - tau_motor

    # IMU lever-arm correction for specific force
    a_cg = leverarm_correct(f_imu, omega, omega_dot, r)

    # Latency alignment: positive lag -> v_body is lag samples later
    if lag != 0:
        if lag > 0:
            v_body = v_body[lag:]
            omega = omega[:N - lag]
            omega_dot = omega_dot[:N - lag]
            tau_aero = tau_aero[:N - lag]
            a_cg = a_cg[:N - lag]
            T_arr = T_arr[:N - lag]
        else:
            nl = -lag
            v_body = v_body[:N - nl]
            omega = omega[nl:]
            omega_dot = omega_dot[nl:]
            tau_aero = tau_aero[nl:]
            a_cg = a_cg[nl:]
            T_arr = T_arr[nl:]

    assert np.isfinite(v_body).all(), 'NaN/Inf in v_body'
    assert np.isfinite(tau_aero).all(), 'NaN/Inf in tau_aero'
    assert np.isfinite(a_cg).all(), 'NaN/Inf in a_cg'
    assert np.isfinite(T_arr).all(), 'NaN/Inf in T'

    return {
        'v_body': v_body,
        'omega': omega,
        'omega_dot': omega_dot,
        'tau_aero': tau_aero,
        'a_aero': a_cg,
        'T': T_arr,
    }
