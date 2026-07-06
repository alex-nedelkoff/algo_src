import os

import pytest

from vq2 import corpus as corpus_mod

REC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_rec")


def test_bridge_on_synthetic_streams():
    seg = corpus_mod.Segment()
    # IMU boot clock starts at 50.0 s boot, wall = boot + 1000.0
    for i in range(200):
        t_us = int((50.0 + i * 0.007) * 1e6)
        seg.imu.append(
            corpus_mod.ImuSample(t_us=t_us, acc=(0, 0, -9.81), gyr=(0, 0, 0),
                                 rx_wall=1000.0 + t_us / 1e6)
        )
    # camera epoch clock: wall = sim_s - 700.0  => frame boot_s = sim_s - 1700.0
    frames = [
        corpus_mod.FrameRef(sim_ns=int((1750.0 + i / 30.0) * 1e9),
                            rx_wall=1050.0 + i / 30.0, path="x")
        for i in range(90)
    ]
    res = corpus_mod.clock_bridge(seg, frames)
    assert res is not None
    offset_s, spread_ms = res
    assert abs(offset_s - (-1700.0)) < 0.01
    assert spread_ms < 1.0


def test_bridge_none_without_stamped_frames():
    seg = corpus_mod.Segment()
    seg.imu.append(
        corpus_mod.ImuSample(t_us=1, acc=(0, 0, -9.81), gyr=(0, 0, 0), rx_wall=5.0)
    )
    frames = [corpus_mod.FrameRef(sim_ns=123, rx_wall=0.0, path="x")]
    assert corpus_mod.clock_bridge(seg, frames) is None


@pytest.mark.skipif(not os.path.isdir(REC), reason="vq2_rec corpus not present")
def test_bridge_on_real_corpus_is_stable():
    c = corpus_mod.load(REC)
    res = corpus_mod.clock_bridge(c.flight_segment, c.frames)
    assert res is not None
    offset_s, spread_ms = res
    assert spread_ms < 200.0  # UDP receive jitter bound, not precision sync
