"""Tests for the COR-147 Phase 2 scale estimators (vq2/tools/vo_scale.py).

Pin that each estimator recovers a known metric scale from synthetic inputs and
fails closed when its window is absent. VO trajectory is unit-scale along +x;
the reference is that trajectory times a known scale.
"""
import numpy as np
import pytest

from vq2.tools.vo_scale import climb_cal_scale, gate_scale, pest_fit_scale

NS = 1_000_000_000  # 1 s in ns


def _line(n, step=1.0):
    ns = np.arange(n, dtype=float) * NS
    traj = np.zeros((n, 3))
    traj[:, 0] = np.arange(n) * step
    return ns, traj


def test_climb_cal_recovers_scale():
    kf_ns, traj = _line(5, step=1.0)                 # VO: 1 unit / s along +x
    ref_ns = np.arange(4, dtype=float) * NS
    ref_xyz = np.zeros((4, 3)); ref_xyz[:, 0] = np.arange(4) * 2.0  # metric = 2x
    out = climb_cal_scale(kf_ns, traj, ref_ns, ref_xyz, min_disp_m=0.8)
    assert out["scale"] == pytest.approx(2.0, rel=1e-6)


def test_climb_cal_none_outside_coverage():
    kf_ns, traj = _line(3, step=1.0)                 # VO covers ns 0..2
    ref_ns = np.array([10.0, 11.0]) * NS             # ref after VO ends
    ref_xyz = np.array([[0, 0, 0], [2, 0, 0]], float)
    assert climb_cal_scale(kf_ns, traj, ref_ns, ref_xyz)["scale"] is None


def test_gate_scale_recovers_scale():
    kf_ns, traj = _line(4, step=1.0)                 # VO 1 unit / s
    obs_ns = np.arange(4, dtype=float) * NS
    obs_range = np.array([10.0, 8.0, 6.0, 4.0])      # closes 6 m over 3 units
    out = gate_scale(kf_ns, traj, obs_ns, obs_range, min_baseline_m=2.0)
    assert out["scale"] == pytest.approx(2.0, rel=1e-6)
    assert out["baseline_m"] == pytest.approx(6.0)


def test_gate_scale_needs_baseline():
    kf_ns, traj = _line(4, step=1.0)
    obs_ns = np.arange(4, dtype=float) * NS
    obs_range = np.array([6.0, 5.8, 5.6, 5.4])       # only 0.6 m closing
    assert gate_scale(kf_ns, traj, obs_ns, obs_range, min_baseline_m=2.0)["scale"] is None


def test_gate_scale_picks_closing_run():
    # a rising prefix then a long closing run: only the closing run scales
    kf_ns, traj = _line(6, step=1.0)
    obs_ns = np.arange(6, dtype=float) * NS
    obs_range = np.array([3.0, 4.0, 12.0, 9.0, 6.0, 3.0])   # closing 12->3
    out = gate_scale(kf_ns, traj, obs_ns, obs_range, min_baseline_m=2.0)
    assert out["scale"] == pytest.approx(3.0, rel=1e-6)      # 9 m over 3 units
    assert out["n"] == 4


def test_pest_fit_recovers_scale():
    kf_ns, traj = _line(4, step=1.0)
    ref_ns = np.arange(4, dtype=float) * NS
    ref_xyz = np.zeros((4, 3)); ref_xyz[:, 0] = np.arange(4) * 2.0
    out = pest_fit_scale(kf_ns, traj, ref_ns, ref_xyz)
    assert out["scale"] == pytest.approx(2.0, rel=1e-3)
