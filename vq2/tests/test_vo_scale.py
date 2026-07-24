"""Tests for the COR-147 Phase 2 scale estimators (vq2/tools/vo_scale.py).

Pin that each estimator recovers a known metric scale from synthetic inputs and
fails closed when its window is absent. VO trajectory is unit-scale along +x;
the reference is that trajectory times a known scale.
"""
import numpy as np
import pytest

from vq2.tools.vo_scale import (
    climb_cal_scale, gate_scale, gate1_anchor_scale, pest_fit_scale,
)

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


def test_gate1_anchor_recovers_scale():
    # gate seen at 6 m; VO travels 3 units from pad view to the crossing -> scale 2
    kf_ns, traj = _line(5, step=1.0)
    pad_ns = 0.0
    tick_ns = 3.0 * NS                               # VO(3) - VO(0) = 3 units
    out = gate1_anchor_scale(kf_ns, traj, pad_range_m=6.0, pad_ns=pad_ns, tick_ns=tick_ns)
    assert out["scale"] == pytest.approx(2.0, rel=1e-6)
    assert out["baseline_m"] == pytest.approx(6.0)


def test_gate1_anchor_needs_pad_coverage():
    kf_ns, traj = _line(3, step=1.0)                 # VO covers ns 0..2
    out = gate1_anchor_scale(kf_ns, traj, pad_range_m=6.0,
                             pad_ns=10.0 * NS, tick_ns=11.0 * NS)  # pad after VO
    assert out["scale"] is None


def test_gate1_anchor_needs_tick():
    kf_ns, traj = _line(4, step=1.0)
    assert gate1_anchor_scale(kf_ns, traj, 6.0, 0.0, None)["scale"] is None


def test_pest_fit_recovers_scale():
    kf_ns, traj = _line(4, step=1.0)
    ref_ns = np.arange(4, dtype=float) * NS
    ref_xyz = np.zeros((4, 3)); ref_xyz[:, 0] = np.arange(4) * 2.0
    out = pest_fit_scale(kf_ns, traj, ref_ns, ref_xyz)
    assert out["scale"] == pytest.approx(2.0, rel=1e-3)


# --- gate1_anchor_bridged (DR bridge over the VO-uncovered climb) -----------

def _bridge_scene(scale=0.5):
    """Gate at [6,0,0]. Pad view at t=1s from the origin; VO coverage starts
    at t=3s with the drone at [0.5,0,-1] (climb happened off-VO); tick at
    t=9s at the gate. VO traj = metric truth / scale, covering 3..9s."""
    gate = np.array([6.0, 0.0, 0.0])
    p_t0 = np.array([0.5, 0.0, -1.0])
    # DR: dense 20 Hz from 0.5s to 4s covering [pad_ns, t0]
    dr_ns = (np.arange(0.5, 4.0, 0.05) * NS)
    dr_xyz = np.zeros((len(dr_ns), 3))
    m = dr_ns >= 1.0 * NS                            # static until the pad view
    f = (dr_ns[m] - 1.0 * NS) / (2.0 * NS)           # linear climb 1s -> 3s
    dr_xyz[m] = np.clip(f, 0, 1)[:, None] * p_t0
    # VO: t0=3s at truth [0.5,0,-1] -> tick 9s at the gate, in VO units
    kf_ns = np.linspace(3.0, 9.0, 13) * NS
    fvo = (kf_ns - 3.0 * NS) / (6.0 * NS)
    truth = p_t0 + fvo[:, None] * (gate - p_t0)
    traj = truth / scale
    pad_g_lvl = gate - np.zeros(3)                   # gate offset seen at the pad
    return kf_ns, traj, 1.0 * NS, 9.0 * NS, pad_g_lvl, dr_ns.astype(np.int64), dr_xyz


def test_gate1_bridged_recovers_scale():
    from vq2.tools.vo_scale import gate1_anchor_bridged
    kf_ns, traj, pad_ns, tick_ns, g_lvl, dr_ns, dr_xyz = _bridge_scene(scale=0.5)
    out = gate1_anchor_bridged(kf_ns, traj, 6.0, pad_ns, tick_ns, g_lvl,
                               dr_ns, dr_xyz)
    assert out["scale"] == pytest.approx(0.5, rel=1e-6)
    # r' = ||gate - p_t0|| = sqrt(5.5^2 + 1)
    assert out["baseline_m"] == pytest.approx(np.hypot(5.5, 1.0), rel=1e-6)
    assert out["bridge_s"] == pytest.approx(2.0, rel=1e-6)
    assert out["bridge_m"] == pytest.approx(np.linalg.norm([0.5, 0, -1]), rel=1e-3)


def test_gate1_bridged_falls_back_when_pad_covered():
    from vq2.tools.vo_scale import gate1_anchor_bridged
    kf_ns, traj = _line(10, step=1.0)                # coverage 0..9s
    out = gate1_anchor_bridged(kf_ns, traj, 6.0, pad_ns=2.0 * NS,
                               tick_ns=8.0 * NS, pad_g_lvl=np.array([6.0, 0, 0]),
                               dr_ns=None, dr_xyz=None)
    ref = gate1_anchor_scale(kf_ns, traj, 6.0, 2.0 * NS, 8.0 * NS)
    assert out["scale"] == pytest.approx(ref["scale"])


def test_gate1_bridged_fails_closed():
    from vq2.tools.vo_scale import gate1_anchor_bridged
    kf_ns, traj, pad_ns, tick_ns, g_lvl, dr_ns, dr_xyz = _bridge_scene()
    # no DR rows (pre-55dd6210 corpus)
    out = gate1_anchor_bridged(kf_ns, traj, 6.0, pad_ns, tick_ns, g_lvl, None, None)
    assert out["scale"] is None and "kf_pose" in out["reason"]
    # DR does not span the bridge (starts after the pad view)
    late = dr_ns > 1.5 * NS
    out = gate1_anchor_bridged(kf_ns, traj, 6.0, pad_ns, tick_ns, g_lvl,
                               dr_ns[late], dr_xyz[late])
    assert out["scale"] is None and "span" in out["reason"]
    # DR gap inside the bridge span
    keep = (dr_ns < 1.2 * NS) | (dr_ns > 2.6 * NS)
    out = gate1_anchor_bridged(kf_ns, traj, 6.0, pad_ns, tick_ns, g_lvl,
                               dr_ns[keep], dr_xyz[keep])
    assert out["scale"] is None and "gap" in out["reason"]
