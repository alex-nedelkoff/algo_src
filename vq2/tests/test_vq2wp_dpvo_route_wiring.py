from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_flight_records_receive_clock_for_exact_tick_rebase():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert "state['gate_tick_ns'] = time.time_ns()" in source
    assert "state['judge_tickstamp'] = rs[5]" in source


def test_route_mode_has_observe_only_and_explicit_unhealthy_abort():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert "from dpvo_route import select_route_position" in source
    assert "os.environ.get('DPVO_ROUTE') == '1'" in source
    assert "os.environ.get('DPVO_OBSERVE') == '1'" in source
    assert "DPVO route unhealthy" in source


def test_bridge_client_never_fuses_dpvo_into_kf():
    source = (ROOT / "vq2/live/dpvo_odom_bridge.py").read_text(encoding="utf-8")
    assert "update_position" not in source
    assert "self.kf" not in source


def test_route_dpvo_prewarm_starts_during_hard_reset_settle():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    reset = source.index("31000, 0, 1")
    started = source.index("_dpvo_route_started = True")
    reset_settle = source.index("time.sleep(6.0)", reset)
    assert reset < started < reset_settle


def test_gatenet_handoff_is_explicit_and_timestamped():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert "state['gatenet_unloaded'] = True" in source
    assert "state['gatenet_unloaded_wall'] = _handoff_wall" in source
    assert "jlog('gatenet_unloaded'" in source


def test_go_wall_is_published_before_go_flag():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert source.index("state['go_wall'] = time.time()") < source.index(
        "state['go_passed'] = True"
    )