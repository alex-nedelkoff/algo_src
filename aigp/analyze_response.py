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


def _rate_gains(samples) -> dict:
    rate_gain = {}
    for name, axis in _AXIS.items():
        seg = [s for s in samples if s["segment"] == f"rate_{name}"]
        if not seg:
            continue
        cmd = np.array([s["cmd_rates"][axis] for s in seg])
        meas = np.array([s["omega"][axis] for s in seg])
        rate_gain[name] = fit_rate_gain(cmd, meas)
    return rate_gain


def summarize_pulses(samples, g: float = G, skip: int = 2) -> dict:
    """Reduce pulse-from-rest samples to sim_response constants.

    For each thrust level, fit the slope of vz over its pulse (skipping the
    first `skip` transient samples) -> one (thrust, accel) point; fit those."""
    levels: dict = {}
    for s in samples:
        if s["segment"] == "thrust_pulse":
            levels.setdefault(s["thrust_norm"], []).append(s)

    tn_list, thrust_up_list = [], []
    for tn, ss in sorted(levels.items()):
        ss = ss[skip:]
        if len(ss) < 2:
            continue
        t = np.array([x["t"] for x in ss])
        vz = np.array([x["vel_ned"][2] for x in ss])
        slope = np.polyfit(t, vz, 1)[0]      # dvz/dt (down positive)
        tn_list.append(tn)
        thrust_up_list.append(-slope + g)    # thrust_up = net_up + g
    tmap = fit_thrust_map(tn_list, thrust_up_list, g)
    return {"hover_thrust": tmap["hover_thrust"], "k_a": tmap["k_a"],
            "rate_gain": _rate_gains(samples)}


def _main():
    import argparse
    import json
    import pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to probe log.jsonl")
    ap.add_argument("--out", default=None, help="output sim_response.json")
    args = ap.parse_args()
    samples = [json.loads(l) for l in open(args.log)]
    is_pulse = any(s.get("segment") == "thrust_pulse" for s in samples)
    out = summarize_pulses(samples) if is_pulse else summarize(samples)
    dest = args.out or str(pathlib.Path(args.log).with_name("sim_response.json"))
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    _main()
