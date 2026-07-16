from __future__ import annotations

import numpy as np

from vq2.live.dpvo_odom_bridge import DpvoOdom


def test_thread_construction_does_not_raise_in_armed_main_thread(monkeypatch):
    monkeypatch.delenv("DPVO_CAL", raising=False)
    odom = DpvoOdom({}, object(), object(), np.eye(3))
    assert odom.config is None
    assert odom.route is None
