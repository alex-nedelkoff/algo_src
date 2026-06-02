"""Raw flight log -> regression dataset for aero sysID (pure numpy/pandas).
Input columns (per sample): t, vx/vy/vz (world NED), qw/qx/qy/qz (body->world), wx/wy/wz (gyro),
fx/fy/fz (HIGHRES_IMU specific force, body), thrust_accel (commanded thrust accel magnitude),
coast (bool). Output: aero force/moment targets + the coast mask."""
from __future__ import annotations
import numpy as np
import pandas as pd
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
        k = np.ones(win) / win
        a = np.stack([np.convolve(a[:, j], k, mode="same") for j in range(a.shape[1])], axis=1)
    return a


def build_targets(df: pd.DataFrame, I_ratio) -> dict:
    """Per-sample body velocity, aero force target, aero moment target, coast mask.

    a_aero  = f_body - thrust_body        (thrust along body -z)
    al_aero = alpha + omega x (I_ratio*omega)   (motor_torque omitted: rely on coast/zero-motor samples)
    """
    t = df["t"].to_numpy()
    Ir = np.asarray(I_ratio, float)
    n = len(df)
    vb = np.zeros((n, 3))
    aero_f = np.zeros((n, 3))
    w = df[["wx", "wy", "wz"]].to_numpy()
    f = df[["fx", "fy", "fz"]].to_numpy()

    for i in range(n):
        q = df[["qw", "qx", "qy", "qz"]].iloc[i].to_numpy()
        R = quat_to_R(q)
        v_world = df[["vx", "vy", "vz"]].iloc[i].to_numpy()
        vb[i] = world_to_body_vel(v_world, R)
        thrust_body = np.array([0.0, 0.0, -float(df["thrust_accel"].iloc[i])])
        aero_f[i] = f[i] - thrust_body

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
