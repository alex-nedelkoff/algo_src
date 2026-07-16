from __future__ import annotations

from vq2.live.dpvo_route import TickRebasedRoute


def test_temporal_discontinuity_latches_route_unhealthy():
    route = TickRebasedRoute(scale=1.0, gate1_world=(0, 0, 0), max_speed=3.0)
    route.observe((0, 0, 0), 1_000_000_000, 1, 0.0)
    assert route.observe((10, 0, 0), 2_000_000_000, 1, 0.0).reason == "speed"
    later = route.observe((0.1, 0, 0), 3_000_000_000, 1, 0.0)
    assert not later.healthy
    assert later.reason == "speed"
