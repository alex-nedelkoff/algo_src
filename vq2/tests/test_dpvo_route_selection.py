from __future__ import annotations

from vq2.live.dpvo_route import select_route_position


def _state():
    return {
        "dpvo_route_p": (7.0, 1.0, -1.35),
        "dpvo_route_wall": 10.0,
        "dpvo_route_healthy": True,
        "dpvo_route_reason": "ok",
    }


def test_control_selects_fresh_healthy_dpvo_only_after_tick_one():
    p, reason = select_route_position(_state(), ticks=1, now=10.1, observe_only=False)
    assert p == (7.0, 1.0, -1.35)
    assert reason == "ok"


def test_observe_only_never_returns_a_control_position():
    p, reason = select_route_position(_state(), ticks=1, now=10.1, observe_only=True)
    assert p is None
    assert reason == "observe_only"


def test_selection_rejects_pre_tick_unhealthy_and_stale_observations():
    p, reason = select_route_position(_state(), ticks=0, now=10.1, observe_only=False)
    assert p is None and reason == "await_tick"

    state = _state()
    state["dpvo_route_healthy"] = False
    state["dpvo_route_reason"] = "speed"
    p, reason = select_route_position(state, ticks=1, now=10.1, observe_only=False)
    assert p is None and reason == "speed"

    p, reason = select_route_position(_state(), ticks=1, now=10.6, observe_only=False)
    assert p is None and reason == "stale"
