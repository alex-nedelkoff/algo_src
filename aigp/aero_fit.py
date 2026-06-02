"""Fit the parametric aero model by least-squares (linear in theta). Residual MLP added in Task 4."""
from __future__ import annotations
import numpy as np
from .aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS


def _stack(feature_fn, V, W=None):
    rows = []
    for i in range(len(V)):
        rows.append(feature_fn(V[i]) if W is None else feature_fn(V[i], W[i]))
    return np.vstack(rows)                       # (3n, n_params)


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean(0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_parametric(V, W, a_force, a_moment, weights=None) -> dict:
    """V,W:(n,3) body vel & rates. a_force,a_moment:(n,3) targets. weights: optional (n,) per-sample.
    Returns theta_F, theta_M, force_cols, moment_cols, r2_force, r2_moment."""
    V = np.asarray(V, float); W = np.asarray(W, float)
    PhiF = _stack(force_features, V)
    PhiM = _stack(moment_features, V, W)
    yF = np.asarray(a_force, float).reshape(-1); yM = np.asarray(a_moment, float).reshape(-1)
    if weights is not None:
        sw = np.sqrt(np.repeat(np.asarray(weights, float), 3))
        PhiF, yF = PhiF * sw[:, None], yF * sw
        PhiM, yM = PhiM * sw[:, None], yM * sw
    thF, *_ = np.linalg.lstsq(PhiF, yF, rcond=None)
    thM, *_ = np.linalg.lstsq(PhiM, yM, rcond=None)
    return {
        "theta_F": thF, "theta_M": thM,
        "force_cols": FORCE_COLS, "moment_cols": MOMENT_COLS,
        "r2_force": _r2(np.asarray(a_force).reshape(-1), _stack(force_features, V) @ thF),
        "r2_moment": _r2(np.asarray(a_moment).reshape(-1), _stack(moment_features, V, W) @ thM),
    }
