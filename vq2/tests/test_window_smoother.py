"""window_smoother: recovery of a known trajectory from drifted odometry
+ absolute factors, twin disambiguation without pre-gates, corridor prior,
and capture-time (retroactive) folding of a late anchor.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vq2.window_smoother import (LineCorridor, OdomFactor, PillarFactor,
                                 PosUnary, pillar_ray_level, solve)

DEG = math.pi / 180.0


def make_truth(n=30, dt=0.1, v=2.0):
    """Straight +x flight at height z=-1.5, yaw 0."""
    t = np.arange(n) * dt
    x = np.zeros((n, 4))
    x[:, 0] = v * t
    x[:, 2] = -1.5
    return t, x


def drifted_odom(truth, yaw_rate_err=3.0 * DEG, sigma_p=0.05):
    """Odometry factors with an injected yaw-rate error (the gyro-bank
    failure mode) — integrating them reproduces a curved, wrong track."""
    rng = np.random.default_rng(0)
    odom = []
    n = len(truth)
    for i in range(n - 1):
        dp_w = truth[i + 1, :3] - truth[i, :3]
        cy, sy = math.cos(truth[i, 3]), math.sin(truth[i, 3])
        dp_l = np.array([cy * dp_w[0] + sy * dp_w[1],
                         -sy * dp_w[0] + cy * dp_w[1], dp_w[2]])
        odom.append(OdomFactor(
            i=i, j=i + 1,
            dp_local=dp_l + rng.normal(0, sigma_p, 3),
            dyaw=(truth[i + 1, 3] - truth[i, 3]) + yaw_rate_err * 0.1,
            sigma_p=0.15, sigma_yaw=0.03))
    return odom


def integrate(x0, odom, n):
    x = np.zeros((n, 4))
    x[0] = x0
    for f in odom:
        cy, sy = math.cos(x[f.i, 3]), math.sin(x[f.i, 3])
        dl = f.dp_local
        x[f.j, :3] = x[f.i, :3] + np.array(
            [cy * dl[0] - sy * dl[1], sy * dl[0] + cy * dl[1], dl[2]])
        x[f.j, 3] = x[f.i, 3] + f.dyaw
    return x


def pillar_obs_from_truth(truth, k, L):
    """Synthesize the level-frame ray a camera at truth[k] would record
    for a panel at L (inverting the measurement model)."""
    x, y, z, yaw = truth[k]
    to_l = np.array(L) - np.array([x, y, z])
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    d = np.array([cy * to_l[0] - sy * to_l[1],
                  sy * to_l[0] + cy * to_l[1], to_l[2]])
    return d / np.linalg.norm(d)


def test_recovers_yaw_drift_with_pillar_factors():
    """Drifted odometry alone curves away; pillar factors (position+yaw
    coupled) must straighten the whole window retroactively."""
    t, truth = make_truth()
    odom = drifted_odom(truth)
    x_dr = integrate(truth[0], odom, len(truth))
    assert np.linalg.norm(x_dr[-1, :2] - truth[-1, :2]) > 0.5  # drift real

    L = (8.0, 6.0, -7.0)
    pillars = [PillarFactor(k=k, ray_level=pillar_obs_from_truth(truth, k, L),
                            cands=(L,)) for k in range(4, 30, 3)]
    res = solve(t, x_dr, odom, [], pillars, [])
    err = np.linalg.norm(res.x[:, :2] - truth[:, :2], axis=1)
    # 9 sparse pillar obs vs 3 deg/s yaw drift: DR alone exceeds 0.5 m;
    # smoother must hold the whole window under 0.45 m
    assert float(err.max()) < 0.45
    assert abs(_yaw_err(res, truth)) < 1.5 * DEG


def _yaw_err(res, truth):
    return float(np.median(res.x[:, 3] - truth[:, 3]))


def test_twin_disambiguation_min_mixture():
    """Duplicate-number twins ~10 m apart: the solver must settle on the
    true twin with NO pre-gating (the EKF failure mode)."""
    t, truth = make_truth()
    odom = drifted_odom(truth, yaw_rate_err=2.0 * DEG)
    x_dr = integrate(truth[0], odom, len(truth))
    L_true = (8.0, 6.0, -7.0)
    L_twin = (18.0, 8.0, -7.0)
    pillars = [PillarFactor(
        k=k, ray_level=pillar_obs_from_truth(truth, k, L_true),
        cands=(L_true, L_twin)) for k in range(4, 30, 3)]
    res = solve(t, x_dr, odom, [], pillars, [])
    err = np.linalg.norm(res.x[:, :2] - truth[:, :2], axis=1)
    assert float(err.max()) < 0.6


def test_late_anchor_folds_retroactively():
    """A single capture-time-stamped anchor mid-window must correct states
    BEFORE and AFTER it (re-explaining the past — the T6c property the
    EKF lacks)."""
    t, truth = make_truth()
    odom = drifted_odom(truth, yaw_rate_err=0.0)
    # bias every local dx: odometry systematically 10% short
    for f in odom:
        f.dp_local = f.dp_local * 0.9
    x_dr = integrate(truth[0], odom, len(truth))
    assert x_dr[-1, 0] < truth[-1, 0] - 0.4
    # one perfect anchor at k=15 (capture time), nothing after
    unaries = [PosUnary(k=15, p=truth[15, :3].copy(), sigma=0.1)]
    res = solve(t, x_dr, odom, unaries, [], [])
    # the state at the anchor is fixed AND earlier states share the blame
    assert abs(res.x[15, 0] - truth[15, 0]) < 0.15
    assert abs(res.x[8, 0] - truth[8, 0]) < abs(x_dr[8, 0] - truth[8, 0])


def test_corridor_prior_bounds_lateral():
    t, truth = make_truth()
    odom = drifted_odom(truth, yaw_rate_err=4.0 * DEG)
    x_dr = integrate(truth[0], odom, len(truth))
    assert float(np.abs(x_dr[:, 1]).max()) > 0.5
    cors = [LineCorridor(k=k, a=np.array([0.0, 0.0]),
                         b=np.array([10.0, 0.0]), sigma=0.5)
            for k in range(len(truth))]
    res = solve(t, x_dr, odom, [], [], cors)
    assert float(np.abs(res.x[:, 1]).max()) < 0.5


def _run_sliding(t, truth, odom, ks, L, window_s):
    from vq2.window_smoother import SlidingSmoother
    sl = SlidingSmoother(t[0], truth[0], window_s=window_s,
                         resolve_every=0.2)
    for i, f in enumerate(odom):
        sl.push_odom(t[i + 1], f.dp_local, f.dyaw,
                     f.sigma_p, f.sigma_yaw)
        if (i + 1) in ks:
            sl.push_pillar(t[i + 1],
                           pillar_obs_from_truth(truth, i + 1, L), (L,))
    sl._solve(t[-1])
    ts, xs = sl.trajectory()
    rows = np.round((ts - t[0]) / 0.1).astype(int)   # time-align to truth
    return np.linalg.norm(xs[:, :2] - truth[rows, :2], axis=1)


def test_sliding_machinery_matches_batch():
    """With an infinite window the incremental path must reproduce the
    batch solution — proves pushes/prior-carry/indexing add no error.
    (A SHORT window is legitimately worse: drift older than the window is
    hardened by marginalization — window-length physics, not machinery;
    the real sizing decision is made on corpus replays.)"""
    t, truth = make_truth(n=40)
    odom = drifted_odom(truth, yaw_rate_err=2.0 * DEG)
    L = (8.0, 6.0, -7.0)
    ks = set(range(4, 40, 3))
    x_dr = integrate(truth[0], odom, len(truth))
    pil = [PillarFactor(k=k, ray_level=pillar_obs_from_truth(truth, k, L),
                        cands=(L,)) for k in sorted(ks)]
    batch = solve(t, x_dr, odom, [], pil, [])
    e_batch = np.linalg.norm(batch.x[:, :2] - truth[:, :2], axis=1)
    e_inf = _run_sliding(t, truth, odom, ks, L, window_s=999.0)
    assert abs(float(np.median(e_inf)) - float(np.median(e_batch))) < 0.05
    e_short = _run_sliding(t, truth, odom, ks, L, window_s=2.0)
    assert float(np.median(e_short)) < 0.8   # documented short-window cost


def test_sliding_late_anchor_retro_corrects():
    """An anchor arriving 0.6 s after capture must correct the state that
    SAW it (capture-time folding) and improve the head."""
    from vq2.window_smoother import SlidingSmoother
    t, truth = make_truth(n=30)
    odom = drifted_odom(truth, yaw_rate_err=0.0)
    for f in odom:
        f.dp_local = f.dp_local * 0.85    # short odometry
    sl = SlidingSmoother(t[0], truth[0], window_s=3.0, resolve_every=0.2)
    for i, f in enumerate(odom):
        sl.push_odom(t[i + 1], f.dp_local, f.dyaw, 0.15, 0.03)
        if i + 1 == 20:                    # anchor captured at k=14,
            sl.push_anchor(t[14], truth[14, :3], sigma=0.1)   # arrives late
    _, head = sl.pose()
    dr_head_err = abs(0.85 * truth[-1, 0] - truth[-1, 0])
    assert abs(head[0] - truth[-1, 0]) < dr_head_err


def test_sliding_window_bounded():
    from vq2.window_smoother import SlidingSmoother
    t, truth = make_truth(n=200)
    odom = drifted_odom(truth)
    sl = SlidingSmoother(t[0], truth[0], window_s=2.0)
    for i, f in enumerate(odom):
        sl.push_odom(t[i + 1], f.dp_local, f.dyaw)
    assert len(sl.t) <= int(2.0 / 0.1) + 2
    assert len(sl.odom) <= len(sl.t)


def test_solve_time_budget():
    """A representative window (30 states, mixed factors) must solve well
    under the flight budget. Generous bound for CI noise; prints measured."""
    import time
    from vq2.window_smoother import SlidingSmoother
    t, truth = make_truth(n=30)
    odom = drifted_odom(truth)
    L = (8.0, 6.0, -7.0)
    sl = SlidingSmoother(t[0], truth[0], window_s=3.0,
                         resolve_every=999.0)   # manual solve
    for i, f in enumerate(odom):
        sl.push_odom(t[i + 1], f.dp_local, f.dyaw)
        if (i + 1) % 3 == 0:
            sl.push_pillar(t[i + 1],
                           pillar_obs_from_truth(truth, i + 1, L), (L,))
            sl.push_corridor(t[i + 1], np.zeros(2), np.array([10.0, 0.0]))
    t0 = time.perf_counter()
    sl._solve(t[-1])
    ms = (time.perf_counter() - t0) * 1000.0
    print(f'solve: {ms:.1f} ms for {len(sl.t)} states')
    assert ms < 100.0


def test_health_triad_populated():
    t, truth = make_truth()
    odom = drifted_odom(truth)
    x_dr = integrate(truth[0], odom, len(truth))
    unaries = [PosUnary(k=10, p=truth[10, :3].copy(), sigma=0.3)]
    res = solve(t, x_dr, odom, unaries, [], [])
    assert res.n_factors == {'odom': 29, 'unary': 1, 'pillar': 0,
                             'corridor': 0}
    assert res.step_jump_p95 > 0.0
    assert res.anchor_resid_med >= 0.0
