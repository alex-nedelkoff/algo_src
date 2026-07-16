from __future__ import annotations

import numpy as np

from vq2.live.dpvo_route import TickRebasedRoute


def test_route_rebases_on_first_tick_without_position_jump():
    route = TickRebasedRoute(scale=2.0, gate1_world=[6, 0, -1], max_speed=8.0)
    assert route.observe([1, 0, 0], 1_000_000_000, 0, 0).p is None
    at_tick = route.observe([2, 0, 0], 2_000_000_000, 1, 0)
    assert np.allclose(at_tick.p, [6, 0, -1])
    moved = route.observe([2.5, 0, 0], 2_500_000_000, 1, 0)
    assert np.allclose(moved.p, [7, 0, -1])


def test_route_rejects_impossible_temporal_jump():
    route = TickRebasedRoute(scale=1.0, gate1_world=[0, 0, 0], max_speed=3.0)
    route.observe([0, 0, 0], 1_000_000_000, 1, 0)
    bad = route.observe([10, 0, 0], 2_000_000_000, 1, 0)
    assert not bad.healthy
    assert bad.reason == "speed"


def test_route_rejects_non_monotonic_timestamp_without_advancing_state():
    route = TickRebasedRoute(scale=1.0, gate1_world=[0, 0, 0], max_speed=3.0)
    route.observe([0, 0, 0], 2_000_000_000, 1, 0)
    bad = route.observe([0.1, 0, 0], 2_000_000_000, 1, 0)
    assert not bad.healthy
    assert bad.reason == "timestamp"
    good = route.observe([0.2, 0, 0], 3_000_000_000, 1, 0)
    assert not good.healthy
    assert good.reason == "timestamp"


def test_route_rotates_relative_translation_by_tick_yaw():
    route = TickRebasedRoute(scale=2.0, gate1_world=[6, 0, -1], max_speed=8.0)
    route.observe([1, 1, 0], 1_000_000_000, 1, np.pi / 2)
    moved = route.observe([2, 1, 0], 2_000_000_000, 1, 0)
    assert np.allclose(moved.p, [6, 2, -1], atol=1e-9)
