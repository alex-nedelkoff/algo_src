"""Pure least-squares fitters for lumped dynamics sysID."""
from __future__ import annotations

import numpy as np


def fit_gain(x, y) -> float:
    """Through-origin least-squares slope of y vs x."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    d = float(x @ x)
    return float((x @ y) / d) if d > 0 else 0.0


def fit_specific_thrust(power_sum, up_accel) -> float:
    """c_T = slope of measured specific thrust (up accel) vs summed motor power."""
    return fit_gain(power_sum, up_accel)


def select_power(u_sum, u_sq_sum, up_accel):
    """Choose thrust ~ input (1) vs input^2 (2) by residual; return (power, c_T)."""
    up = np.asarray(up_accel, float)
    best = None
    for power, P in ((1, np.asarray(u_sum, float)), (2, np.asarray(u_sq_sum, float))):
        c = fit_gain(P, up)
        resid = float(np.sum((up - c * P) ** 2))
        if best is None or resid < best[2]:
            best = (power, c, resid)
    return best[0], best[1]


def angular_accel(t, gyro) -> np.ndarray:
    """Finite-difference angular acceleration from body-rate samples, shape (N,3)."""
    t = np.asarray(t, float)
    gyro = np.asarray(gyro, float)
    return np.gradient(gyro, t, axis=0)


def fit_axis_torque(mix_regressor, ang_accel_axis) -> float:
    """c_axis = slope of angular accel vs the axis's (mix . p) regressor."""
    return fit_gain(mix_regressor, ang_accel_axis)


def fit_motor_lag(t, response, taus=None) -> float:
    """Grid-search the first-order time constant of a step response
    response(t) = A * (1 - exp(-(t - t0) / tau)); returns tau."""
    t = np.asarray(t, float)
    t = t - t[0]
    y = np.asarray(response, float)
    if taus is None:
        taus = np.linspace(0.003, 0.12, 80)
    best = (None, np.inf)
    for tau in taus:
        basis = 1.0 - np.exp(-t / tau)
        a = fit_gain(basis, y)
        resid = float(np.sum((y - a * basis) ** 2))
        if resid < best[1]:
            best = (float(tau), resid)
    return best[0]


def fit_drag(vel, residual_accel) -> float:
    """Linear drag coeff: residual_accel = -drag * vel."""
    return -fit_gain(vel, residual_accel)


_MIX_ROWS = {
    "roll":  np.array([-1.0, 1.0, 1.0, -1.0]),
    "pitch": np.array([-1.0, -1.0, 1.0, 1.0]),
    "yaw":   np.array([1.0, -1.0, 1.0, -1.0]),
}


def fit_from_log(samples, n_motors=4) -> dict:
    """Assemble regressors from probe samples and fit lumped coefficients.

    Uses the collective segment for c_T + power law (specific thrust = |imu up|),
    and diff_<axis> segments for c_L/c_M/c_N (angular accel vs mix.p). Returns a
    dict of identified coefficients."""
    out = {"n_motors": n_motors}

    coll = [s for s in samples if s["segment"] == "collective" and "imu_acc" in s]
    if coll:
        u = np.array([s["u"][:n_motors] for s in coll], float)
        up = np.array([abs(s["imu_acc"][2]) for s in coll])   # body-up specific thrust
        u_sum = u.sum(axis=1)
        u_sq_sum = (u ** 2).sum(axis=1)
        power, c_T = select_power(u_sum, u_sq_sum, up)
        out["power"] = power
        out["c_T"] = c_T

    power = out.get("power", 2)
    for ax, key in (("roll", "c_L"), ("pitch", "c_M"), ("yaw", "c_N")):
        seg = [s for s in samples if s["segment"] == f"diff_{ax}" and "imu_gyro" in s]
        if len(seg) < 3:
            continue
        t = np.array([s["t"] for s in seg])
        gyro = np.array([s["imu_gyro"] for s in seg])
        aa = angular_accel(t, gyro)
        axis_idx = {"roll": 0, "pitch": 1, "yaw": 2}[ax]
        u = np.array([s["u"][:n_motors] for s in seg], float)
        reg = (u ** power) @ _MIX_ROWS[ax][:n_motors]
        out[key] = fit_axis_torque(reg, aa[:, axis_idx])

    return out
