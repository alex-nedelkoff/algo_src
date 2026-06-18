"""Fit the rate-loop model to recordings by command-replay least squares, per axis.
Pure numpy/scipy."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scripts.sysid.ring_model import RateLoopParams, propagate_nl
from scripts.sysid.ring_loader import RunSeries


def _residual(theta, runs, axis, dt):
    k, wn, z0, z1 = theta
    p = RateLoopParams(k=k, wn=wn, zeta0=z0, zeta1=z1)
    res = []
    for rs in runs:
        pred = propagate_nl(rs.cmd[:, axis], dt, p)
        res.append(pred - rs.omega[:, axis])
    return np.concatenate(res)


def fit_axis(runs, axis, dt, x0=None) -> RateLoopParams:
    """Fit rate-loop params via least-squares replay error minimization.

    If x0 is None, tries multiple initial guesses to escape local minima.
    """
    lb = [0.2, 3.0, 0.02, -0.2]
    ub = [3.0, 120.0, 2.0, 0.2]

    if x0 is None:
        # Try multiple initial guesses; keep the one with lowest final cost
        x0_candidates = [
            [1.0, 25.0, 0.3, 0.0],
            [1.0, 25.0, 0.2, 0.0],
            [1.0, 28.0, 0.2, 0.0],
            [1.0, 25.0, 0.15, -0.02],
            [1.0, 30.0, 0.3, -0.05],
        ]
        best_sol = None
        best_cost = float('inf')
        for x0_try in x0_candidates:
            try:
                sol_try = least_squares(_residual, x0_try, bounds=(lb, ub),
                                        args=(runs, axis, dt), method="trf", max_nfev=600)
                if sol_try.cost < best_cost:
                    best_cost = sol_try.cost
                    best_sol = sol_try
            except ValueError:
                pass
        if best_sol is not None:
            sol = best_sol
        else:
            # All candidates raised ValueError; fall back to a single default solve.
            sol = least_squares(_residual, [1.0, 25.0, 0.3, 0.0], bounds=(lb, ub),
                                args=(runs, axis, dt), method="trf", max_nfev=600)
    else:
        sol = least_squares(_residual, x0, bounds=(lb, ub), args=(runs, axis, dt),
                            method="trf", max_nfev=600)

    k, wn, z0, z1 = sol.x
    return RateLoopParams(k=float(k), wn=float(wn), zeta0=float(z0), zeta1=float(z1))


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
