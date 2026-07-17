"""E3 cyan-line height-hold + FLOWOBS observe-only wiring (07-17 re-apply;
the 07-16 DPVO rewrite wiped the original E3 code — HANDOFF_classical_lineflow).
vq2wp.py cannot be imported (module-level flight): source-text assertions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]
SOURCE = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")


def _fastgate_loop_body() -> str:
    start = SOURCE.index("def fastgate_loop():")
    end = SOURCE.index("\ndef ", start + 1)
    return SOURCE[start:end]


def test_linefollow_detector_lives_in_the_fastgate_loop():
    body = _fastgate_loop_body()
    assert "os.environ.get('LINEFOLLOW') == '1'" in body
    assert "state['line_row']" in body
    assert "state['line_wall']" in body


def test_linefollow_height_consumer_is_freshness_gated():
    consumer = SOURCE.index("LINE_ROW_REF")
    gate = SOURCE.rindex("_line_age < 0.4", 0, consumer)
    assert gate > 0  # stale line rows must not steer the row consumer


def test_line_lateral_is_opt_in():
    # fg75: lateral line-centering fought the map carrot -> default OFF
    assert "os.environ.get('LINE_LAT') == '1'" in SOURCE
    lat = SOURCE.index("os.environ.get('LINE_LAT') == '1'")
    height = SOURCE.index("LINE_ROW_REF")
    assert lat < height  # lateral is the guarded branch inside the consumer


def test_height_correction_is_clamped_to_mapfollow_authority():
    consumer_at = SOURCE.index("LINE_ROW_REF")
    window = SOURCE[consumer_at - 700:consumer_at + 400]
    assert "max(-0.05, min(0.05" in window


def test_line_lost_descend_is_the_stale_branch_and_bounded():
    # test3: the climb pushes the line out of view -> absence IS the signal
    lost = SOURCE.index("LINE_LOST_DESCEND")
    fresh = SOURCE.rindex("_line_age < 0.4", 0, lost)
    assert fresh > 0  # descend branch only runs when the row branch is stale
    window = SOURCE[lost:lost + 1600]
    assert "state.get('line_n', 0) > 0" in window   # never before first sight
    assert "_line_age < 10.0" in window             # runaway stop
    assert "LINE_LOST_DZ" in window                 # bias env-tunable
    assert "max(-0.055, min(0.055" in window        # bounded by MF authority


def test_flow_is_observe_only():
    body = _fastgate_loop_body()
    assert "os.environ.get('FLOWOBS') == '1'" in body
    assert "from flow_vel import FlowVelocity" in body
    assert "state['flow_v']" in body
    assert "jlog('flow', v=" in body
    # observe-only: the flow result must NOT touch the KF or any command
    # (match calls, not the comment that mentions the method by name)
    assert ".update_velocity(" not in body
    assert ".update_position(" not in body


def test_flow_logs_kf_velocity_for_divergence_analysis():
    body = _fastgate_loop_body()
    flow_log = body.index("jlog('flow', v=")
    assert "kfv=" in body[flow_log:flow_log + 400]


def test_mapfollow_log_carries_the_line_evidence():
    mf_log = SOURCE.index("jlog('mapfollow'")
    assert "lr=" in SOURCE[mf_log:mf_log + 500]
