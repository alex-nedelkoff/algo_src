"""Fit the rate-loop model to recordings by command-replay least squares, per axis.
Pure numpy/scipy."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scripts.sysid.ring_model import RateLoopParams, propagate_nl


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
                                        args=(runs, axis, dt), method="trf", max_nfev=400)
                if sol_try.cost < best_cost:
                    best_cost = sol_try.cost
                    best_sol = sol_try
            except Exception:
                pass
        sol = best_sol if best_sol is not None else \
              least_squares(_residual, [1.0, 25.0, 0.3, 0.0], bounds=(lb, ub),
                          args=(runs, axis, dt), method="trf", max_nfev=400)
    else:
        sol = least_squares(_residual, x0, bounds=(lb, ub), args=(runs, axis, dt),
                            method="trf", max_nfev=1000)

    k, wn, z0, z1 = sol.x
    return RateLoopParams(k=float(k), wn=float(wn), zeta0=float(z0), zeta1=float(z1))
