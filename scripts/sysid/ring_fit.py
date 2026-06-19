"""Fit the rate-loop model to recordings by command-replay least squares, per axis.
Pure numpy/scipy."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scripts.sysid.ring_model import RateLoopParams, propagate_nl
from scripts.sysid.ring_loader import RunSeries


_LB = [0.2, 3.0, 0.02, -0.2]
_UB = [3.0, 120.0, 2.0, 0.2]
_X0 = [[1.0, 25.0, 0.3, 0.0], [1.0, 28.0, 0.2, 0.0],
       [1.0, 25.0, 0.15, -0.02], [1.0, 30.0, 0.3, -0.05], [1.0, 40.0, 0.2, 0.1]]


def _residual(theta, runs, axis, dt, sign, delay):
    k, wn, z0, z1 = theta
    p = RateLoopParams(k=k, wn=wn, zeta0=z0, zeta1=z1, delay=int(delay), sign=float(sign))
    res = []
    for rs in runs:
        pred = propagate_nl(rs.cmd[:, axis], dt, p)
        res.append(pred - rs.omega[:, axis])
    return np.concatenate(res)


def _solve(runs, axis, dt, sign, delay, starts, max_nfev=600):
    best = None
    for x0 in starts:
        try:
            s = least_squares(_residual, x0, bounds=(_LB, _UB), args=(runs, axis, dt, sign, delay),
                              method="trf", max_nfev=max_nfev)
            if best is None or s.cost < best.cost:
                best = s
        except ValueError:
            pass
    return best


def fit_axis(runs, axis, dt, x0=None, signs=(1.0, -1.0), max_delay=3) -> RateLoopParams:
    """Fit rate-loop params by least-squares command-replay error, discovering the per-axis
    SIGN (logged cmd vs gyro can be frame-flipped) and the integer transport DELAY (samples).

    Strategy: rank every (sign, delay) with one default start, then run the full multi-start
    (to escape zeta1 local minima) at the winning (sign, delay). If x0 is given, a single solve
    at sign=+1, delay=0 is run (caller-controlled, no search)."""
    if x0 is not None:
        sol = _solve(runs, axis, dt, 1.0, 0, [x0])
        k, wn, z0, z1 = sol.x
        return RateLoopParams(k=float(k), wn=float(wn), zeta0=float(z0), zeta1=float(z1))
    # coarse: rank (sign, delay) with a cheap solve (enough to rank, not to converge)
    best_key = None
    for sign in signs:
        for delay in range(int(max_delay) + 1):
            s = _solve(runs, axis, dt, sign, delay, [_X0[0]], max_nfev=120)
            if s is not None and (best_key is None or s.cost < best_key[0]):
                best_key = (s.cost, sign, delay)
    _, sign, delay = best_key
    # fine: full multi-start at the winning (sign, delay)
    sol = _solve(runs, axis, dt, sign, delay, _X0) or _solve(runs, axis, dt, 1.0, 0, [_X0[0]])
    k, wn, z0, z1 = sol.x
    return RateLoopParams(k=float(k), wn=float(wn), zeta0=float(z0), zeta1=float(z1),
                          delay=int(delay), sign=float(sign))


def holdout_r2(p, runs, axis, dt, tilt_edges) -> np.ndarray:
    """Compute per-tilt-bin R² of propagate_nl prediction vs measured omega on held-out runs.

    Args:
        p: RateLoopParams instance
        runs: list of RunSeries
        axis: 0=roll, 1=pitch, 2=yaw
        dt: sampling interval (seconds)
        tilt_edges: bin edges for tilt_deg (array-like)

    Returns:
        r2_by_bin: (n_bins,) array of R² values per tilt bin; nan where no samples exist.
                   R² = 1 - SS_res/SS_tot computed over samples whose tilt falls in the bin.
    """
    tilt_edges = np.asarray(tilt_edges, float)
    nb = len(tilt_edges) - 1
    pred_all, meas_all, tilt_all = [], [], []
    for rs in runs:
        pred_all.append(propagate_nl(rs.cmd[:, axis], dt, p))
        meas_all.append(rs.omega[:, axis])
        tilt_all.append(rs.tilt_deg)
    pred = np.concatenate(pred_all)
    meas = np.concatenate(meas_all)
    tilt = np.concatenate(tilt_all)
    out = np.full(nb, np.nan)
    bi = np.digitize(tilt, tilt_edges) - 1
    for b in range(nb):
        m = bi == b
        if m.sum() < 10:
            continue
        ss_res = float(np.sum((meas[m] - pred[m]) ** 2))
        ss_tot = float(np.sum((meas[m] - meas[m].mean()) ** 2)) + 1e-12
        out[b] = 1.0 - ss_res / ss_tot
    return out


def envelope(r2_by_bin, tilt_edges, thresh=0.9) -> dict:
    """Extract the contiguous tilt range from 0 up to the first bin below thresh.

    Args:
        r2_by_bin: (n_bins,) array of R² values per tilt bin
        tilt_edges: bin edges for tilt_deg (array-like)
        thresh: R² threshold (default 0.9)

    Returns:
        dict with keys:
            max_tilt_deg: maximum tilt (upper edge of last valid bin)
            valid_bins: list of (lower, upper) tilt ranges for bins above thresh
    """
    tilt_edges = np.asarray(tilt_edges, float)
    valid, max_tilt = [], 0.0
    for b, r2 in enumerate(r2_by_bin):
        if np.isnan(r2) or r2 < thresh:
            break
        valid.append((float(tilt_edges[b]), float(tilt_edges[b + 1])))
        max_tilt = float(tilt_edges[b + 1])
    return {"max_tilt_deg": max_tilt, "valid_bins": valid}
