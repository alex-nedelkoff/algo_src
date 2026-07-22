"""Tests for the livelog capture-time join (COR-147).

Pins the loop-iteration linkage: `obs*` rows carry both the loop wall clock `t`
and the frame capture clock `ns`; position rows (`kf_upd`/`mapfollow`/...) carry
only `t`. The join places each position row on the capture-time axis. The bug
being fixed is the old carry-forward stamping each position with the PREVIOUS
iteration's frame ns (off by ~one loop iteration).
"""
import json
import os

import numpy as np
import pytest

from vq2.tools.livelog_join import (
    build_capture_clock,
    map_t_to_capture_s,
    first_tick_capture_ns,
    load_estimator_reference,
    POS_KINDS,
)


def _obs(t, cap_s, kind="obs"):
    return {"kind": kind, "t": t, "ns": int(round(cap_s * 1e9))}


def _pos(t, p, kind="kf_upd"):
    return {"kind": kind, "t": t, "p": list(p)}


def test_build_clock_monotone_and_dedup():
    rows = [
        _obs(100.0, 99.5),
        _obs(100.0, 99.5, kind="obs_nofix"),   # duplicate frame, same t -> collapse
        _obs(100.5, 99.9, kind="obs_wronggate"),
        _obs(100.5, 99.9),                      # tie ns at new t -> dropped by filter
        _obs(101.0, 100.4),
    ]
    clock = build_capture_clock(rows)
    assert clock is not None
    t_a, ns_a = clock
    assert np.all(np.diff(t_a) > 0)
    assert np.all(np.diff(ns_a) > 0)            # strictly increasing both axes


def test_build_clock_none_when_sparse():
    assert build_capture_clock([_obs(1.0, 0.5)]) is None      # <2 anchors
    assert build_capture_clock([_pos(1.0, [0, 0, 0])]) is None  # no obs at all


def test_exact_pairing_at_anchor():
    # A position row sharing an obs row's t must get that frame's EXACT capture
    # time (same-iteration pairing), not an interpolated value.
    clock = build_capture_clock([_obs(10.0, 9.4), _obs(10.5, 9.95)])
    assert map_t_to_capture_s(10.0, clock) == pytest.approx(9.4)
    assert map_t_to_capture_s(10.5, clock) == pytest.approx(9.95)


def test_interp_between_anchors():
    clock = build_capture_clock([_obs(10.0, 9.0), _obs(12.0, 11.0)])
    assert map_t_to_capture_s(11.0, clock) == pytest.approx(10.0)  # linear midpoint


def test_no_extrapolation():
    clock = build_capture_clock([_obs(10.0, 9.0), _obs(12.0, 11.0)])
    assert map_t_to_capture_s(9.9, clock) is None
    assert map_t_to_capture_s(12.1, clock) is None


def test_join_beats_carry_forward(tmp_path):
    """The load-bearing behavior: a position row logged in the SAME iteration as
    an obs is stamped with THAT frame's capture time, not the previous frame's.
    Carry-forward (last-seen ns) would assign the earlier obs -> off by an
    iteration; the join assigns the co-iteration frame."""
    # iteration A: frame cap 9.0 processed at t=10.0
    # iteration B: kf_upd position + its obs (frame cap 9.5) both at t=10.5
    rows = [
        _obs(10.0, 9.0),
        _pos(10.5, [1.0, 2.0, 0.0]),   # position row, no ns
        _obs(10.5, 9.5),               # same-iteration obs
        _pos(11.0, [1.5, 2.5, 0.0]),
        _obs(11.0, 10.0),
        _pos(11.5, [2.0, 3.0, 0.0]),
        _obs(11.5, 10.5),
    ]
    lp = tmp_path / "livelog.jsonl"
    lp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ref = load_estimator_reference(str(tmp_path))
    assert ref is not None
    ns, xyz = ref
    # the first mapped position must sit at capture 9.5s (its own iteration),
    # NOT 9.0s (the previous obs that carry-forward would have used).
    assert ns[0] == pytest.approx(int(9.5e9), rel=0, abs=2)
    np.testing.assert_allclose(xyz[0], [1.0, 2.0, 0.0])


def test_att_scalar_pest_excluded():
    """att.p_est is a SCALAR (x projection), not a position -> att must not be a
    position kind and must not leak into the reference."""
    assert "att" not in POS_KINDS


def test_load_reference_none_without_log(tmp_path):
    assert load_estimator_reference(str(tmp_path)) is None


def _write(tmp_path, rows):
    (tmp_path / "livelog.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


def test_pre_tick_clip_excludes_post_tick(tmp_path):
    """pre_tick_only must drop position samples captured after the first tick
    (the gate-1 turn / mapfollow-reset region)."""
    rows = [
        _obs(10.0, 9.0),
        _pos(10.2, [1.0, 0.0, 0.0]),
        _obs(10.2, 9.2),
        _pos(10.4, [2.0, 0.0, 0.0]),
        _obs(10.4, 9.4),
        _pos(10.5, [2.5, 0.0, 0.0]),
        _obs(10.5, 9.5),
        {"kind": "tick_fix", "t": 10.6, "p": [3.0, 0.0, -1.0]},  # TICK at cap~9.6
        _obs(10.6, 9.6),
        _pos(10.8, [3.0, 5.0, 0.0]),   # post-tick (turn) -> must be excluded
        _obs(10.8, 9.8),
        _pos(11.0, [3.0, 8.0, 0.0]),
        _obs(11.0, 10.0),
    ]
    _write(tmp_path, rows)
    clock = build_capture_clock(read_rows_or(tmp_path))
    assert first_tick_capture_ns(read_rows_or(tmp_path), clock) == pytest.approx(
        int(9.6e9), abs=2)
    full = load_estimator_reference(str(tmp_path), pre_tick_only=False)
    clipped = load_estimator_reference(str(tmp_path), pre_tick_only=True)
    assert full is not None and clipped is not None
    assert len(clipped[0]) < len(full[0])            # post-tick samples dropped
    assert clipped[0].max() <= int(9.6e9) + 5        # nothing past the tick


def read_rows_or(tmp_path):
    from vq2.tools.livelog_join import read_rows
    return read_rows(str(tmp_path / "livelog.jsonl"))
