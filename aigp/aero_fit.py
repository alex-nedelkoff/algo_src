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
    import warnings
    for _name, _Phi in (("PhiF", PhiF), ("PhiM", PhiM)):
        _nc = _Phi.shape[1]
        if np.linalg.matrix_rank(_Phi) < _nc:
            warnings.warn(
                f"Rank-deficient design matrix {_name} "
                f"(rank {np.linalg.matrix_rank(_Phi)} < {_nc}): "
                "under-excited flight log — some velocity/rate axis never exercised.")
    thF, *_ = np.linalg.lstsq(PhiF, yF, rcond=None)
    thM, *_ = np.linalg.lstsq(PhiM, yM, rcond=None)
    return {
        "theta_F": thF, "theta_M": thM,
        "force_cols": FORCE_COLS, "moment_cols": MOMENT_COLS,
        "r2_force": _r2(np.asarray(a_force).reshape(-1), _stack(force_features, V) @ thF),
        "r2_moment": _r2(np.asarray(a_moment).reshape(-1), _stack(moment_features, V, W) @ thM),
    }


import torch
import torch.nn as nn


class ResidualMLP:
    """Small MLP r(v_body, omega) -> (dF or dTau) 3-vector. Differentiable, drops into a replica later."""
    def __init__(self, net):
        self._net = net

    def predict(self, V, W):
        x = torch.tensor(np.hstack([np.asarray(V, float), np.asarray(W, float)]), dtype=torch.float32)
        with torch.no_grad():
            return self._net(x).numpy()


def fit_residual(V, W, residual, hidden=64, epochs=400, lr=1e-3):
    """Fit an MLP to the parametric residual. Returns (ResidualMLP, info{residual_var_share}).
    Note: residual_var_share is in-sample (optimistic) — use aero_validate.single_step_metrics on held-out data for an unbiased number."""
    X = torch.tensor(np.hstack([np.asarray(V, float), np.asarray(W, float)]), dtype=torch.float32)
    Y = torch.tensor(np.asarray(residual, float), dtype=torch.float32)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, 3))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net(X) - Y) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        leftover = (net(X) - Y).numpy()
    share = float(np.var(leftover) / (np.var(np.asarray(residual)) + 1e-12))
    return ResidualMLP(net), {"residual_var_share": share}


# ---------------------------------------------------------------------------
# Closed-loop (CL) parametric fit — Task 4
# ---------------------------------------------------------------------------

def fit_parametric_cl(V, W, T, a_force, a_moment, weights=None) -> dict:
    """Fit the CL aero model (thrust-coupled drag + v_y weathervane yaw).

    Args:
        V: (N, 3) body-frame velocities
        W: (N, 3) body-frame angular rates
        T: (N,)  total thrust scalars
        a_force:  (N, 3) force acceleration targets
        a_moment: (N, 3) moment acceleration targets
        weights:  optional (N,) per-sample weights

    Returns dict with keys:
        theta_F (9,), theta_M (6,), force_cols, moment_cols, r2_force, r2_moment
    """
    from .aero_model import force_features_cl, moment_features_cl, FORCE_COLS_CL, MOMENT_COLS_CL
    import warnings

    V = np.asarray(V, float); W = np.asarray(W, float); T = np.asarray(T, float)
    N = len(V)

    # Stack per-sample feature matrices
    PhiF_rows = [force_features_cl(V[i], T[i]) for i in range(N)]
    PhiM_rows = [moment_features_cl(V[i], W[i]) for i in range(N)]
    PhiF = np.vstack(PhiF_rows)   # (3N, 9)
    PhiM = np.vstack(PhiM_rows)   # (3N, 6)

    yF = np.asarray(a_force, float).reshape(-1)    # (3N,)
    yM = np.asarray(a_moment, float).reshape(-1)   # (3N,)

    if weights is not None:
        sw = np.sqrt(np.repeat(np.asarray(weights, float), 3))
        PhiF, yF = PhiF * sw[:, None], yF * sw
        PhiM, yM = PhiM * sw[:, None], yM * sw

    for _name, _Phi in (("PhiF_cl", PhiF), ("PhiM_cl", PhiM)):
        _nc = _Phi.shape[1]
        if np.linalg.matrix_rank(_Phi) < _nc:
            warnings.warn(
                f"Rank-deficient design matrix {_name} "
                f"(rank {np.linalg.matrix_rank(_Phi)} < {_nc}): "
                "under-excited flight log — some velocity/rate axis never exercised.")

    thF, *_ = np.linalg.lstsq(PhiF, yF, rcond=None)
    thM, *_ = np.linalg.lstsq(PhiM, yM, rcond=None)

    # Re-stack without weights for R2 computation
    PhiF0 = np.vstack([force_features_cl(V[i], T[i]) for i in range(N)])
    PhiM0 = np.vstack([moment_features_cl(V[i], W[i]) for i in range(N)])
    yF0 = np.asarray(a_force, float).reshape(-1)
    yM0 = np.asarray(a_moment, float).reshape(-1)

    return {
        "theta_F":     thF,
        "theta_M":     thM,
        "force_cols":  FORCE_COLS_CL,
        "moment_cols": MOMENT_COLS_CL,
        "r2_force":    _r2(yF0, PhiF0 @ thF),
        "r2_moment":   _r2(yM0, PhiM0 @ thM),
    }


# ---------------------------------------------------------------------------
# Per-axis yaw moment fit (wv_z weathervane) — decoupled kappa
# ---------------------------------------------------------------------------

def fit_yaw_axis(W, WD, V, tau_motor_z, I_ratio, kappa=None) -> dict:
    """Fit the yaw (z-axis) moment balance in isolation to recover the sideslip
    weathervane wv_z, decoupled from the roll/pitch axes that dominate (and bias)
    the joint kappa fit.

    Yaw moment balance:
        tau_motor_z = kappa*inertia_z + d_z*w_z - wv_z*v_y
        inertia_z   = wd_z + (I_ratio_y - I_ratio_x)*w_x*w_y   (yaw gyroscopic coupling)

    kappa is None -> fit [kappa, d_z, wv_z] jointly (design [inertia_z, w_z, -v_y]).
    kappa given   -> fix it (move kappa*inertia_z to LHS), fit [d_z, wv_z]
                     (design [w_z, -v_y]). This breaks the kappa/damping collinearity
                     that washes wv_z out in the joint oscillatory-doublet fit.

    Returns dict(kappa, d_z, wv_z, r2, cond, n) where cond is the design-matrix
    condition number (high cond = collinear excitation, wv_z poorly determined).
    """
    W = np.asarray(W, float); WD = np.asarray(WD, float); V = np.asarray(V, float)
    y = np.asarray(tau_motor_z, float).ravel()
    I_ratio = np.asarray(I_ratio, float)
    inertia_z = WD[:, 2] + (I_ratio[1] - I_ratio[0]) * W[:, 0] * W[:, 1]
    w_z = W[:, 2]; v_y = V[:, 1]
    if kappa is None:
        Phi = np.stack([inertia_z, w_z, -v_y], axis=1)        # -> [kappa, d_z, wv_z]
        th, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        kap, d_z, wv_z = float(th[0]), float(th[1]), float(th[2])
        pred = Phi @ th
    else:
        kap = float(kappa)
        Phi = np.stack([w_z, -v_y], axis=1)                   # -> [d_z, wv_z]
        th, *_ = np.linalg.lstsq(Phi, y - kap * inertia_z, rcond=None)
        d_z, wv_z = float(th[0]), float(th[1])
        pred = Phi @ th + kap * inertia_z
    ss = float(np.sum((y - pred) ** 2)); tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss / tot if tot > 0 else 0.0
    cond = float(np.linalg.cond(Phi))
    return {"kappa": kap, "d_z": d_z, "wv_z": wv_z, "r2": r2, "cond": cond, "n": int(len(y))}
