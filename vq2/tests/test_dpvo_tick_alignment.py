from __future__ import annotations

from collections import deque

from vq2.live.dpvo_odom_bridge import observe_route_pose
from vq2.live.dpvo_route import TickRebasedRoute


def test_same_frame_tick_rebase_is_not_immediately_marked_duplicate():
    route = TickRebasedRoute(scale=1.0, gate1_world=(6, 0, -1), max_speed=8.0)
    history = deque([(2_000_000_000, (1.0, 0.0, 0.0), 0.0)])
    observation = observe_route_pose(
        route, history, (1.0, 0.0, 0.0), 2_000_000_000, 1, 0.0, 2_000_000_000
    )
    assert observation.healthy
    assert observation.reason == "rebase"


def test_tick_rebase_requires_timestamp_aligned_pose():
    route = TickRebasedRoute(scale=1.0, gate1_world=(6, 0, -1), max_speed=8.0)
    history = deque([(2_000_000_000, (1.0, 0.0, 0.0), 0.0)])
    observation = observe_route_pose(
        route, history, (1.0, 0.0, 0.0), 2_000_000_000, 1, 0.0, 0
    )
    assert not observation.healthy
    assert observation.reason == "tick_timestamp"
    assert route.raw_origin is None
