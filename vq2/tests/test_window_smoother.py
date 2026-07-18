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
