"""Tests for VO->smoother association (COR-147, Phase 3)."""
import numpy as np
import pytest

from vq2.live.vo_cv import VoStep
from vq2.live.vo_association import (
    integrate_leg, calibrate_scale, sigma_for, feed_smoother)
from vq2.window_smoother import SlidingSmoother


def _fwd_step(t0, t1, n_inl=120):
    """A pure-forward unit VO step (body +x)."""
    return VoStep(t0=t0, t1=t1, R_body=np.eye(3),
                  t_body_unit=np.array([1.0, 0.0, 0.0]), dyaw=0.0,
                  n_inliers=n_inl, n_tracked=300, median_flow_px=10.0)


def test_integrate_leg_and_scale():
    # 5 forward unit steps => unit leg length 5; known dist 10 => scale 2.0
    steps = [_fwd_step(i * 0.1, (i + 1) * 0.1) for i in range(5)]
    leg = integrate_leg(steps, t0_ns=0.0, t1_ns=0.5e9)
    assert leg[0] > 0 and abs(leg[1]) < 1e-9
    scale = calibrate_scale(steps, 0.0, 0.5e9, known_dist_m=10.0)
    assert scale == pytest.approx(2.0, rel=0.2)


def test_sigma_information_weighting():
    base = 0.15
    tight = sigma_for(_fwd_step(0, 0.1, n_inl=480), base)   # many inliers
    loose = sigma_for(_fwd_step(0, 0.1, n_inl=15), base)    # few inliers
    assert tight < base < loose
    assert loose <= base * 3.0 + 1e-9                       # clamped


def test_feed_smoother_advances_metric_trajectory():
    # forward unit VO, scale 2 => each step advances world +x by ~2 m
    sm = SlidingSmoother(t0=0.0, x0=[0.0, 0.0, 0.0, 0.0],
                         window_s=100.0, dt=0.1)
    steps = [_fwd_step(i * 0.1, (i + 1) * 0.1) for i in range(1, 11)]
    n = feed_smoother(sm, steps, scale=2.0, base_sigma_p=0.15)
    assert n == 10
    _, x = sm.pose()
    assert x[0] == pytest.approx(20.0, abs=1.0)   # 10 steps * 2 m forward
    assert abs(x[1]) < 0.5 and abs(x[3]) < 0.05   # straight, no yaw


def test_feed_smoother_yaw_rotates_world_frame():
    # a step with dyaw then forward should move in the rotated world dir
    sm = SlidingSmoother(t0=0.0, x0=[0.0, 0.0, 0.0, 0.0],
                         window_s=100.0, dt=0.1)
    turn = VoStep(t0=0.0, t1=0.1, R_body=np.eye(3),
                  t_body_unit=np.array([1.0, 0.0, 0.0]), dyaw=np.pi / 2,
                  n_inliers=120, n_tracked=300, median_flow_px=10.0)
    fwd = _fwd_step(0.1, 0.2)
    feed_smoother(sm, [turn, fwd], scale=1.0)
    _, x = sm.pose()
    # after +90deg yaw, body-forward maps to world +y
    assert x[3] == pytest.approx(np.pi / 2, abs=0.05)
    assert x[1] > 0.5    # moved in +y
