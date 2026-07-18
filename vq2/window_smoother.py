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
class YawUnary:
    """Absolute yaw measurement (e.g. gate-PnP R_cam_gate via the fitted
    gate world frame — Phase C). Wrapped residual."""
    k: int
    yaw: float
    sigma: float = 0.05


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
          damping: float = 1e-3, yaws: list = ()) -> SmootherResult:
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
            idx = np.asarray(idx)
            g[idx] += w * (J.T @ r)
            H[np.ix_(idx, idx)] += w * (J.T @ J)

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

        for f in yaws:
            r = np.array([_wrap(x[f.k][3] - f.yaw) / f.sigma])
            w = _huber_w(abs(float(r[0])), huber_delta)
            J = np.zeros((1, 4))
            J[0, 3] = 1.0 / f.sigma
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
        # scipy Cholesky, NOT np.linalg.solve: this box's LAPACK gesv path
        # is pathological (measured 72 ms vs 0.32 ms on a 124x124 system);
        # H is SPD by construction (damped normal equations)
        try:
            import scipy.linalg as _sla
            dx = _sla.cho_solve(_sla.cho_factor(H), -g)
        except Exception:
            H[np.diag_indices_from(H)] += 1e-2
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


class SlidingSmoother:
    """Online fixed-lag wrapper (T6c emit pattern): the HEAD pose is
    available immediately at odometry rate (dead-reckoned from the last
    solve); the window behind it is re-solved when absolute factors arrive
    (or on a cadence), refining the recent past retroactively. The oldest
    kept state carries a prior from the previous solution (naive
    marginalization).

    Measurements insert at CAPTURE time anywhere inside the window — a
    0.5-1 s stale anchor lands on the state that saw it, not the present.
    Callers encode measurement quality in per-push sigmas (information-
    driven regimes: flow-inlier-scaled odometry sigma, controller-intent
    corridor on/off).
    """

    def __init__(self, t0: float, x0, window_s: float = 3.0,
                 dt: float = 0.1, resolve_every: float = 0.3,
                 prior_sigma: float = 0.5):
        self.dt = dt
        self.window_s = window_s
        self.resolve_every = resolve_every
        self.prior_sigma = prior_sigma
        self.t = [float(t0)]
        self.x = [np.asarray(x0, float).copy()]
        self.odom: list = []          # OdomFactor with ABSOLUTE indices
        self.unaries: list = []
        self.pillars: list = []
        self.corridors: list = []
        self._base = 0                # absolute index of self.t[0]
        self._last_solve_t = float(t0)
        self._dirty = False
        self.last_result: SmootherResult | None = None

    # ---------------------------------------------------------- pushes --
    def _k_at(self, t: float) -> int | None:
        """Absolute state index nearest capture time t, if in window."""
        if t < self.t[0] - 0.5 * self.dt:
            return None               # older than the window: dropped
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        return self._base + i

    def push_odom(self, t: float, dp_local, dyaw: float,
                  sigma_p: float = 0.2, sigma_yaw: float = 0.03) -> None:
        """Advance the head to time t with a local-frame delta. Emits a
        dead-reckoned head state immediately."""
        xi = self.x[-1]
        cy, sy = math.cos(xi[3]), math.sin(xi[3])
        dl = np.asarray(dp_local, float)
        xj = np.array([xi[0] + cy * dl[0] - sy * dl[1],
                       xi[1] + sy * dl[0] + cy * dl[1],
                       xi[2] + dl[2], _wrap(xi[3] + dyaw)])
        i_abs = self._base + len(self.t) - 1
        self.t.append(float(t))
        self.x.append(xj)
        self.odom.append(OdomFactor(i_abs, i_abs + 1, dl, float(dyaw),
                                    sigma_p, sigma_yaw))
        self._slide()
        if t - self._last_solve_t >= self.resolve_every and self._dirty:
            self._solve(t)

    def push_anchor(self, t_capture: float, p, sigma: float = 1.0,
                    xy_only: bool = True) -> None:
        k = self._k_at(t_capture)
        if k is None:
            return
        self.unaries.append(PosUnary(k, np.asarray(p, float), sigma,
                                     xy_only))
        self._dirty = True

    def push_pillar(self, t_capture: float, ray_level, cands,
                    sigma: float = 1.0) -> None:
        k = self._k_at(t_capture)
        if k is None:
            return
        self.pillars.append(PillarFactor(k, np.asarray(ray_level, float),
                                         tuple(cands), sigma))
        self._dirty = True

    def push_corridor(self, t_capture: float, a, b,
                      sigma: float = 1.5) -> None:
        k = self._k_at(t_capture)
        if k is None:
            return
        self.corridors.append(LineCorridor(k, np.asarray(a, float),
                                           np.asarray(b, float), sigma))
        self._dirty = True

    # ----------------------------------------------------------- state --
    def pose(self):
        """Head pose (dead-reckoned since the last solve)."""
        return self.t[-1], self.x[-1].copy()

    def trajectory(self):
        return np.asarray(self.t), np.stack(self.x)

    # -------------------------------------------------------- internal --
    def _slide(self) -> None:
        cut = self.t[-1] - self.window_s
        n_drop = 0
        while len(self.t) > 2 and self.t[n_drop] < cut:
            n_drop += 1
        if n_drop == 0:
            return
        self._base += n_drop
        self.t = self.t[n_drop:]
        self.x = self.x[n_drop:]
        b = self._base
        self.odom = [f for f in self.odom if f.i >= b]
        self.unaries = [f for f in self.unaries if f.k >= b]
        self.pillars = [f for f in self.pillars if f.k >= b]
        self.corridors = [f for f in self.corridors if f.k >= b]

    def _solve(self, now: float) -> None:
        b = self._base
        t_arr = np.asarray(self.t)
        x0 = np.stack(self.x)
        odom = [OdomFactor(f.i - b, f.j - b, f.dp_local, f.dyaw,
                           f.sigma_p, f.sigma_yaw) for f in self.odom]
        una = [PosUnary(f.k - b, f.p, f.sigma, f.xy_only)
               for f in self.unaries]
        pil = [PillarFactor(f.k - b, f.ray_level, f.cands, f.sigma)
               for f in self.pillars]
        cor = [LineCorridor(f.k - b, f.a, f.b, f.sigma)
               for f in self.corridors]
        res = solve(t_arr, x0, odom, una, pil, cor, iters=4,
                    prior_sigma=self.prior_sigma)
        self.x = [res.x[i].copy() for i in range(len(self.t))]
        self.last_result = res
        self._last_solve_t = now
        self._dirty = False
