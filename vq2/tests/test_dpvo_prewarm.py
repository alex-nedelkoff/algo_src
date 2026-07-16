from __future__ import annotations

from vq2.live.dpvo_odom_bridge import (
    prewarm_abort_reason,
    wait_for_gpu_handoff,
)


def test_prewarm_deadline_is_go():
    assert prewarm_abort_reason({}) is None
    assert prewarm_abort_reason({"stop": True}) == "stop"
    assert prewarm_abort_reason({"go_passed": True}) == "prewarm_late"


def test_gpu_handoff_waits_for_explicit_gatenet_release():
    state = {"gatenet_unloaded": False}
    polls = []

    def release_after_one_poll(_seconds):
        polls.append(1)
        state["gatenet_unloaded"] = True

    assert wait_for_gpu_handoff(state, sleep=release_after_one_poll) == "handoff"
    assert polls == [1]


def test_gpu_handoff_exits_when_flight_stops():
    assert wait_for_gpu_handoff({"stop": True}, sleep=lambda _: None) == "stop"


import numpy as np

from vq2.live.dpvo_odom_bridge import DpvoOdom


def bare_odom(state=None):
    return DpvoOdom({} if state is None else state, object(), object(), np.eye(3))


def test_run_orders_prepare_prewarm_handoff_track(monkeypatch):
    odom = bare_odom()
    calls = []
    monkeypatch.setattr(odom, "_prepare", lambda: calls.append("prepare"))
    monkeypatch.setattr(odom, "_prewarm", lambda: calls.append("prewarm") or True)
    monkeypatch.setattr(
        "vq2.live.dpvo_odom_bridge.wait_for_gpu_handoff",
        lambda state: calls.append("handoff") or "handoff",
    )
    monkeypatch.setattr(odom, "_track", lambda: calls.append("track"))
    odom._run()
    assert calls == ["prepare", "prewarm", "handoff", "track"]


def test_prewarm_late_latches_health_without_connect(monkeypatch):
    state = {"go_passed": True}
    odom = bare_odom(state)
    monkeypatch.setattr(
        odom,
        "_connect",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("late prewarm must not connect")
        ),
    )
    assert not odom._prewarm()
    assert state["dpvo_route_healthy"] is False
    assert state["dpvo_route_reason"] == "prewarm_late"


def test_run_never_tracks_when_handoff_stops(monkeypatch):
    odom = bare_odom({"stop": True})
    monkeypatch.setattr(odom, "_prepare", lambda: None)
    monkeypatch.setattr(odom, "_prewarm", lambda: True)
    monkeypatch.setattr(
        "vq2.live.dpvo_odom_bridge.wait_for_gpu_handoff",
        lambda state: "stop",
    )
    monkeypatch.setattr(
        odom,
        "_track",
        lambda: (_ for _ in ()).throw(AssertionError("tracking must not start")),
    )
    odom._run()