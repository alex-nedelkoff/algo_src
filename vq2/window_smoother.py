"""Sliding-window least-squares pose smoother (T6c-lite, numpy only).

Why a smoother (measured, 2026-07-17/18): the sequential EKF cannot
disentangle yaw/position/landmark-identity errors — every acceptance gate
references the state the fix is supposed to repair (four circular-gate
failures logged), and 0.45-1.0 s perception staleness is applied as if
current. A window re-explains the past: a pillar fix at t re-estimates the
yaw that has been wrong since t-3s and the velocity error it banked, and
every factor is stamped at CAPTURE time (T6c "late anchors folded
retroactively").

States: x_k = [x, y, z, yaw] at a fixed cadence over the window.
Roll/pitch stay a trusted input (the campaign-proven wfix chain, optionally
VP-corrected) — they parameterize the measurement models, not the state.

Factors (all Huber-robustified):
  * odometry between consecutive states: local-frame position delta +
    yaw delta from the DR chain (IMU+flow), sigma scaled by the interval's
    measured flow quality (information-driven, not clock-driven)
  * position unaries (gate anchors), capture-time-stamped
  * pillar bearing+elevation-range factors: couple position AND yaw
    jointly; DUPLICATE station numbers (aisle twins) enter as a
    min-mixture — the solver keeps whichever twin fits the whole window,
    no est-referenced pre-gate
  * line-corridor unaries: controller-intent prior while line-following
    (lateral distance to the commanded line), the RELOC const-position
    trick generalized

Solved by damped Gauss-Newton on the dense normal equations — a 3-4 s
window at 10 Hz is ~160 unknowns; solve cost is milliseconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .camera import CX, CY, FX, M_BODY_CAM, R_level_body


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _huber_w(r_norm: float, delta: float) -> float:
    """Huber IRLS weight for a residual block of norm r_norm."""
    return 1.0 if r_norm <= delta else delta / r_norm


@dataclass
class OdomFactor:
    i: int                  # from state index
    j: int                  # to state index
    dp_local: np.ndarray    # (3,) position delta in frame of state i (yaw-relative)
    dyaw: float
    sigma_p: float = 0.25
    sigma_yaw: float = 0.02


@dataclass
class PosUnary:
    k: int                  # state index (capture-time-nearest)
    p: np.ndarray           # (3,) measured position
    sigma: float = 1.0
    xy_only: bool = True    # obs-Z quarantine convention


@dataclass
class PillarFactor:
    """One identity-read panel obs. ray_level = unit ray in the LEVEL frame
    (roll/pitch applied, yaw NOT applied) at capture time. cands = all map
    entries sharing the read number (aisle twins) as (3,) positions."""
    k: int
    ray_level: np.ndarray
    cands: tuple            # ((x,y,z), ...)
    sigma: float = 0.8


@dataclass
class LineCorridor:
    k: int
    a: np.ndarray           # (2,) segment start
    b: np.ndarray           # (2,) segment end
    sigma: float = 1.5


@dataclass
class SmootherResult:
    t: np.ndarray
    x: np.ndarray           # (N,4) [x,y,z,yaw]
    iters: int = 0
    cost: float = 0.0
    # health triad (T6c): per-step jump, anchor residual, factor counts
    step_jump_p95: float = 0.0
    anchor_resid_med: float = 0.0
    n_factors: dict = field(default_factory=dict)


def pillar_ray_level(u: float, v: float, roll: float, pitch: float) -> np.ndarray:
    """Pixel -> unit ray in the level frame (trusted roll/pitch, yaw=0)."""
    r_cam = np.array([(u - CX) / FX, (v - CY) / FX, 1.0])
    d = R_level_body(roll, pitch) @ (M_BODY_CAM @ r_cam)
    return d / np.linalg.norm(d)


def _pillar_residual(xk, f: PillarFactor):
    """min-mixture residual: for each twin candidate, implied position from
    bearing+elevation-range at the state's yaw; residual = state - implied
    (xy). Returns (best_res(2,), best_J_yaw(2,)) for the winning twin."""
    x, y, z, yaw = xk
    cy, sy = math.cos(yaw), math.sin(yaw)
    d = f.ray_level
    dw = np.array([cy * d[0] - sy * d[1], sy * d[0] + cy * d[1], d[2]])
    best = None
    for L in f.cands:
        if dw[2] > -0.05:
            continue
        s = (L[2] - z) / dw[2]
        if not (1.5 < s < 45.0):
            continue
        imp = np.array([L[0] - s * dw[0], L[1] - s * dw[1]])
        r = np.array([x, y]) - imp
        n = float(np.linalg.norm(r))
        if best is None or n < best[0]:
            # d(imp)/d(yaw): s * d(dw_xy)/dyaw (s is yaw-invariant)
            ddw = np.array([-sy * d[0] - cy * d[1], cy * d[0] - sy * d[1]])
            best = (n, r, s * ddw)
    return best


def solve(t: np.ndarray, x0: np.ndarray, odom: list, unaries: list,
          pillars: list, corridors: list, iters: int = 6,
          huber_delta: float = 1.0, prior_sigma: float = 0.5,
          damping: float = 1e-3) -> SmootherResult:
    """Damped Gauss-Newton over states (N,4). x0 seeds and priors state 0
    (window anchoring; the online slider passes the previous solution)."""
    N = len(t)
    x = x0.copy().astype(float)
    n_var = 4 * N
    last_cost = np.inf
    for it in range(iters):
        H = np.zeros((n_var, n_var))
        g = np.zeros(n_var)
        cost = 0.0

        def add(idx, J, r, w):
            nonlocal cost
            cost += w * float(r @ r)
            for a, ia in enumerate(idx):
                g[ia] += w * float(J[:, a] @ r)
                for b, ib in enumerate(idx):
                    H[ia, ib] += w * float(J[:, a] @ J[:, b])

        # prior on state 0 (window anchor)
        r0 = x[0] - x0[0]
        r0[3] = _wrap(r0[3])
        J0 = np.eye(4)
        add(range(4), J0, r0 / prior_sigma, 1.0)

        for f in odom:
            xi, xj = x[f.i], x[f.j]
            cyi, syi = math.cos(xi[3]), math.sin(xi[3])
            dp_w = xj[:3] - xi[:3]
            # predicted local delta = R(yaw_i)^T (p_j - p_i)
            pred = np.array([cyi * dp_w[0] + syi * dp_w[1],
                             -syi * dp_w[0] + cyi * dp_w[1], dp_w[2]])
            rp = (pred - f.dp_local) / f.sigma_p
            ry = _wrap((xj[3] - xi[3]) - f.dyaw) / f.sigma_yaw
            r = np.concatenate([rp, [ry]])
            w = _huber_w(float(np.linalg.norm(r)), huber_delta * 3.0)
            J = np.zeros((4, 8))     # d r / d [xi(4), xj(4)]
            Ri = np.array([[cyi, syi], [-syi, cyi]])
            J[0:2, 0:2] = -Ri / f.sigma_p
            J[0:2, 4:6] = Ri / f.sigma_p
            J[2, 2] = -1.0 / f.sigma_p
            J[2, 6] = 1.0 / f.sigma_p
            # d pred / d yaw_i
            J[0, 3] = (-syi * dp_w[0] + cyi * dp_w[1]) / f.sigma_p
            J[1, 3] = (-cyi * dp_w[0] - syi * dp_w[1]) / f.sigma_p
            J[3, 3] = -1.0 / f.sigma_yaw
            J[3, 7] = 1.0 / f.sigma_yaw
            add(list(range(4 * f.i, 4 * f.i + 4))
                + list(range(4 * f.j, 4 * f.j + 4)), J, r, w)

        for f in unaries:
            xk = x[f.k]
            if f.xy_only:
                r = (xk[:2] - f.p[:2]) / f.sigma
                J = np.zeros((2, 4))
                J[0, 0] = J[1, 1] = 1.0 / f.sigma
            else:
                r = (xk[:3] - f.p) / f.sigma
                J = np.zeros((3, 4))
                J[0, 0] = J[1, 1] = J[2, 2] = 1.0 / f.sigma
            w = _huber_w(float(np.linalg.norm(r)), huber_delta)
            add(range(4 * f.k, 4 * f.k + 4), J, r, w)

        for f in pillars:
            out = _pillar_residual(x[f.k], f)
            if out is None:
                continue
            n, r, dyaw = out
            r = r / f.sigma
            w = _huber_w(float(np.linalg.norm(r)), huber_delta)
            J = np.zeros((2, 4))
            J[0, 0] = J[1, 1] = 1.0 / f.sigma
            # r = p_state - imp(yaw); d imp/d yaw = -s*ddw  =>  dr/dyaw = +s*ddw
            J[:, 3] = dyaw / f.sigma
            add(range(4 * f.k, 4 * f.k + 4), J, r, w)

        for f in corridors:
            xk = x[f.k]
            ab = f.b - f.a
            L2 = float(ab @ ab)
            tt = float(np.clip(((xk[:2] - f.a) @ ab) / L2, 0.0, 1.0))
            cp = f.a + tt * ab
            d = xk[:2] - cp
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            r = np.array([n / f.sigma])
            w = _huber_w(n / f.sigma, huber_delta)
            J = np.zeros((1, 4))
            J[0, 0:2] = (d / n) / f.sigma
            add(range(4 * f.k, 4 * f.k + 4), J, r, w)

        H[np.diag_indices_from(H)] += damping
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        x = x + dx.reshape(N, 4)
        x[:, 3] = np.vectorize(_wrap)(x[:, 3])
        if abs(last_cost - cost) < 1e-6 * max(1.0, cost):
            last_cost = cost
            break
        last_cost = cost

    steps = np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1)
    anchor_res = [float(np.linalg.norm(x[f.k][:2] - f.p[:2]))
                  for f in unaries]
    return SmootherResult(
        t=t, x=x, iters=it + 1, cost=float(last_cost),
        step_jump_p95=float(np.percentile(steps, 95)) if len(steps) else 0.0,
        anchor_resid_med=float(np.median(anchor_res)) if anchor_res else 0.0,
        n_factors={'odom': len(odom), 'unary': len(unaries),
                   'pillar': len(pillars), 'corridor': len(corridors)})
