"""Offline fits over probe logs -> sim_response constants. Pure numpy + CLI."""
from __future__ import annotations

import numpy as np

G = 9.81


def finite_diff(values, t) -> np.ndarray:
    return np.gradient(np.asarray(values, float), np.asarray(t, float))


def fit_thrust_map(thrust_norm, thrust_up_accel, g: float = G) -> dict:
    """Linear fit thrust_up_accel = k_a*thrust_norm + b; hover where thrust_up = g."""
    tn = np.asarray(thrust_norm, float)
    y = np.asarray(thrust_up_accel, float)
    A = np.vstack([tn, np.ones_like(tn)]).T
    k_a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"hover_thrust": float((g - b) / k_a), "k_a": float(k_a)}


def fit_rate_gain(cmd, meas) -> float:
    """Least-squares gain (through origin) of measured vs commanded body rate."""
    cmd = np.asarray(cmd, float); meas = np.asarray(meas, float)
    denom = float(cmd @ cmd)
    return float((cmd @ meas) / denom) if denom > 0 else 0.0


_AXIS = {"roll": 0, "pitch": 1, "yaw": 2}


def summarize(samples, g: float = G) -> dict:
    """Reduce tagged probe samples to sim_response constants."""
    sweep = [s for s in samples if s["segment"] == "thrust_sweep"]
    t = np.array([s["t"] for s in sweep])
    tn = np.array([s["thrust_norm"] for s in sweep])
    vz = np.array([s["vel_ned"][2] for s in sweep])
    net_up = -finite_diff(vz, t)            # upward net accel
    thrust_up = net_up + g
    tmap = fit_thrust_map(tn, thrust_up, g)

    rate_gain = {}
    for name, axis in _AXIS.items():
        seg = [s for s in samples if s["segment"] == f"rate_{name}"]
        if not seg:
            continue
        cmd = np.array([s["cmd_rates"][axis] for s in seg])
        meas = np.array([s["omega"][axis] for s in seg])
        rate_gain[name] = fit_rate_gain(cmd, meas)

    return {"hover_thrust": tmap["hover_thrust"], "k_a": tmap["k_a"],
            "rate_gain": rate_gain}
