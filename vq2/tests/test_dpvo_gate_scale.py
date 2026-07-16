from __future__ import annotations

import pytest

from vq2.live.dpvo_gate_scale import GateScaleCalibrator


def test_pairs_gatenet_at_source_time_by_interpolating_dpvo():
    cal = GateScaleCalibrator(min_samples=3, min_baseline_m=1.0)
    cal.add_dpvo(0, (0.0, 0.0, 0.0))
    cal.add_dpvo(200_000_000, (1.0, 0.0, 0.0))

    estimates = cal.add_gatenet(
        100_000_000, 0, (2.0, 0.0, 0.0)
    )

    assert estimates
    assert cal.pairs[-1].raw_p == (0.5, 0.0, 0.0)
    assert cal.pairs[-1].frame_ns == 100_000_000
    assert cal.pairs[-1].bracket_before_ns == 0
    assert cal.pairs[-1].bracket_after_ns == 200_000_000


def test_pending_gatenet_pairs_when_future_dpvo_bracket_arrives():
    cal = GateScaleCalibrator(min_samples=3, max_bracket_ns=600_000_000)
    cal.add_dpvo(1_000_000_000, (0.0, 0.0, 0.0))
    assert not cal.add_gatenet(1_500_000_000, 0, (0.0, 0.0, 0.0))

    estimates = cal.add_dpvo(2_000_000_000, (1.0, 0.0, 0.0))

    assert estimates
    assert cal.pairs[-1].raw_p == (0.5, 0.0, 0.0)


def test_rejects_pair_without_a_tight_timestamp_bracket():
    cal = GateScaleCalibrator(min_samples=3, max_bracket_ns=250_000_000)
    cal.add_dpvo(0, (0.0, 0.0, 0.0))
    cal.add_dpvo(1_000_000_000, (1.0, 0.0, 0.0))

    assert not cal.add_gatenet(500_000_000, 0, (1.0, 0.0, 0.0))
    assert not cal.pairs


def test_reports_stable_scale_only_after_sample_baseline_and_dispersion_gates():
    cal = GateScaleCalibrator(
        min_samples=8,
        min_baseline_m=1.0,
        min_pair_baseline_m=0.4,
        max_relative_mad=0.10,
    )
    latest = []
    for index in range(10):
        ns = index * 1_000_000_000
        cal.add_dpvo(ns, (index * 0.1, 0.0, 0.0))
        latest = cal.add_gatenet(ns, 0, (index * 0.2, 0.0, 0.0))

    assert latest
    estimate = latest[-1]
    assert estimate.ready
    assert estimate.scale == pytest.approx(2.0)
    assert estimate.relative_mad == pytest.approx(0.0)
    assert estimate.baseline_m == pytest.approx(1.8)
    assert estimate.sample_count == 10
    assert estimate.pair_count > 0


def test_outlier_and_other_gate_do_not_create_bad_lock():
    cal = GateScaleCalibrator(
        min_samples=8,
        min_baseline_m=1.0,
        min_pair_baseline_m=0.4,
        max_relative_mad=0.10,
    )
    latest = []
    for index in range(10):
        ns = index * 1_000_000_000
        cal.add_dpvo(ns, (index * 0.1, 0.0, 0.0))
        metric_x = index * 0.2
        if index == 5:
            metric_x += 3.0
        latest = cal.add_gatenet(ns, 0, (metric_x, 0.0, 0.0))

    # Add a second gate at the same raw poses but a wildly different origin.
    for index in range(4):
        ns = (10 + index) * 1_000_000_000
        cal.add_dpvo(ns, ((10 + index) * 0.1, 0.0, 0.0))
        cal.add_gatenet(ns, 1, (100.0 + index, 0.0, 0.0))

    gate0 = [item for item in latest if item.gate_id == 0][-1]
    assert gate0.ready
    assert gate0.scale == pytest.approx(2.0)
    assert gate0.sample_count == 10


def test_duplicate_source_timestamp_is_ingested_once():
    cal = GateScaleCalibrator(min_samples=3)
    cal.add_dpvo(1, (0.0, 0.0, 0.0))
    cal.add_gatenet(1, 0, (0.0, 0.0, 0.0))
    cal.add_gatenet(1, 0, (9.0, 9.0, 9.0))

    assert len(cal.pairs) == 1


def test_one_high_leverage_endpoint_cannot_create_a_false_ready_scale():
    cal = GateScaleCalibrator(
        min_samples=8,
        min_baseline_m=1.0,
        min_pair_baseline_m=0.4,
        max_relative_mad=0.10,
    )
    latest = []
    for index in range(8):
        raw_x = 0.01 * index if index < 7 else 0.5
        metric_x = 2.0 * raw_x if index < 7 else 2.0
        cal.add_dpvo(index, (raw_x, 0.0, 0.0))
        latest = cal.add_gatenet(index, 0, (metric_x, 0.0, 0.0))

    assert latest[-1].scale == pytest.approx(4.127659574468085)
    assert latest[-1].relative_mad < 0.10
    assert not latest[-1].ready
