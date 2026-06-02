"""Held-out validation + deliverable serialization for the aero sysID."""
from __future__ import annotations
import json
import numpy as np
from .aero_model import force_features, moment_features


def _rmse(y, yhat): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))
def _r2(y, yhat):
    y = np.asarray(y); ss = float(np.sum((y - yhat) ** 2)); tot = float(np.sum((y - y.mean(0)) ** 2))
    return 1.0 - ss / tot if tot > 0 else 0.0


def _predict_stack(feature_fn, V, theta, W=None):
    return np.array([(feature_fn(V[i]) if W is None else feature_fn(V[i], W[i])) @ theta
                     for i in range(len(V))])


def single_step_metrics(V, W, a_force, a_moment, theta_F, theta_M, residual=None) -> dict:
    """Per-axis force/moment RMSE + R^2 on held-out data. residual: optional ResidualMLP."""
    pf = _predict_stack(force_features, V, theta_F)
    pm = _predict_stack(moment_features, V, theta_M, W)
    if residual is not None:
        pf = pf + residual.predict(V, W)
    return {
        "rmse_force": _rmse(a_force, pf), "r2_force": _r2(a_force, pf),
        "rmse_moment": _rmse(a_moment, pm), "r2_moment": _r2(a_moment, pm),
        "rmse_force_axes": [_rmse(a_force[:, j], pf[:, j]) for j in range(3)],
    }


def term_contributions(V, theta, feature_fn, cols) -> dict:
    """Variance each parametric column contributes to the predicted force/moment (per-term breakdown)."""
    out = {}
    for k, name in enumerate(cols):
        tk = np.zeros_like(theta); tk[k] = theta[k]
        out[name] = float(np.var(_predict_stack(feature_fn, V, tk)))
    return out


def save_model(path, theta_F, theta_M, force_cols, moment_cols, meta=None):
    json.dump({"theta_F": list(map(float, theta_F)), "theta_M": list(map(float, theta_M)),
               "force_cols": list(force_cols), "moment_cols": list(moment_cols),
               "meta": meta or {}}, open(path, "w"), indent=2)
