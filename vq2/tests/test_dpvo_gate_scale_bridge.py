from __future__ import annotations

import inspect

from vq2.live.dpvo_gate_scale import GateScaleCalibrator, GateScaleEstimate
from vq2.live.dpvo_odom_bridge import (
    DpvoOdom,
    read_dpvo_frame,
    publish_gate_scale,
    track_during_gatenet,
    update_gate_scale,
)


def test_publish_gate_scale_never_sets_control_lock_or_route_scale():
    state = {}
    estimate = GateScaleEstimate(
        gate_id=0,
        frame_ns=123,
        scale=5.5,
        relative_mad=0.04,
        baseline_m=1.4,
        pair_count=12,
        sample_count=8,
        ready=True,
    )

    publish_gate_scale(state, estimate, wall=10.0)

    assert state["dpvo_gate_scale_ready"] is True
    assert state["dpvo_gate_scale"] == 5.5
    assert state["dpvo_gate_scale_frame_ns"] == 123
    assert "dpvo_scale_locked" not in state
    assert "dpvo_route_scale" not in state


def test_gnscale_overlap_requires_observe_only():
    assert track_during_gatenet({"GNSCALE": "1", "DPVO_OBSERVE": "1"})
    assert not track_during_gatenet(
        {"GNSCALE": "1", "DPVO_OBSERVE": "0"}
    )
    assert not track_during_gatenet(
        {"GNSCALE": "0", "DPVO_OBSERVE": "1"}
    )


def test_bridge_ingests_gatenet_by_source_frame_timestamp():
    cal = GateScaleCalibrator(min_samples=3, max_bracket_ns=100)
    state = {}
    last_ns, pairs, estimates = update_gate_scale(
        cal, state, 100, (0.0, 0.0, 0.0), last_gatenet_ns=0
    )
    assert (last_ns, pairs, estimates) == (0, (), ())

    state["gatenet_sample"] = (150, 0, (1.0, 0.0, 0.0))
    last_ns, pairs, estimates = update_gate_scale(
        cal, state, 200, (1.0, 0.0, 0.0), last_gatenet_ns=last_ns
    )

    assert last_ns == 150
    assert pairs[0].frame_ns == 150
    assert pairs[0].raw_p == (0.5, 0.0, 0.0)
    assert estimates[0].frame_ns == 150


def test_bridge_reads_atomic_camera_tuple_and_ignores_split_legacy_fields():
    state = {"frame_ns": 200, "jpeg": b"old"}
    assert read_dpvo_frame(state, last_frame_ns=0) is None

    state["dpvo_frame"] = (150, b"matched")
    assert read_dpvo_frame(state, last_frame_ns=0) == (150, b"matched")
    assert read_dpvo_frame(state, last_frame_ns=150) is None


def test_bridge_tracking_wires_observe_only_calibrator_and_overlap():
    source = inspect.getsource(DpvoOdom)
    assert "GateScaleCalibrator(" in source
    assert "update_gate_scale(" in source
    assert "publish_gate_scale(" in source
    assert "if track_during_gatenet():" in source
    overlap = source.split("if track_during_gatenet():", 1)[1].split(
        "print(\"DPVO route thread waiting", 1
    )[0]
    assert "go_passed" not in overlap
    assert "pre_go=True" in overlap


def test_vq2wp_publishes_atomic_snapshots_and_observe_only_gpu_hold():
    from pathlib import Path

    source = (Path(__file__).parents[2] / "vq2/live/vq2wp.py").read_text()
    assert "state['dpvo_frame'] = (ns, data)" in source
    assert "state['gatenet_sample'] = (" in source
    assert "and os.environ.get('DPVO_OBSERVE') == '1'" in source
