from __future__ import annotations

import struct

import pytest

from vq2.dpvo_tick_calibrate import build_calibration
from vq2.live.dpvo_route import DpvoSessionConfig, calibration_identity


def _race(clock_ms, gate_idx, tickstamp, wall):
    data = struct.pack("<BQqqIq", 1, clock_ms, -1, -1, gate_idx, tickstamp)
    return {
        "mavpackettype": "ENCAPSULATED_DATA",
        "data": data.hex(),
        "_rx_wall": wall,
    }


def _fixture():
    config = DpvoSessionConfig()
    identity = calibration_identity(config, "model")
    mavlink = [
        _race(90_000, 2, 88, 90.0),
        _race(0, 0, -1, 100.0),
        _race(10_000, 0, -1, 110.0),
        _race(17_550, 1, 14_188_933_372, 117.55),
        _race(22_060, 2, 18_537_551_879, 122.06),
    ]
    poses = [
        {"frame_ns": 117_500_000_000, "p": [1, 0, 0], "identity": identity},
        {"frame_ns": 122_100_000_000, "p": [3, 0, 0], "identity": identity},
    ]
    return config, identity, mavlink, poses


def test_build_calibration_uses_first_fresh_two_tick_segment():
    config, identity, mavlink, poses = _fixture()
    result = build_calibration(
        mavlink,
        poses,
        config,
        "model",
        gate1_world=(6, 0, -1),
        gate2_world=(10, 0, -1),
    )
    assert result["scale"] == 2.0
    assert result["identity"] == identity
    assert result["model_sha256"] == "model"
    assert result["control_approved"] is False
    assert result["validation"]["status"] == "observe_only"
    assert result["ticks"][0]["race_ms"] == 17_550
    assert result["ticks"][1]["tickstamp"] == 18_537_551_879
    assert result["ticks"][0]["sample_offset_ms"] == -50.0
    assert result["ticks"][1]["sample_offset_ms"] == 40.0
    assert result["sample_separation_ms"] == 4600.0


def test_build_calibration_rejects_pose_identity_mismatch():
    config, _, mavlink, poses = _fixture()
    poses[1]["identity"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        build_calibration(mavlink, poses, config, "model")


def test_build_calibration_rejects_stale_or_one_tick_segments():
    config, identity, _, poses = _fixture()
    stale = [_race(80_000, 0, -1, 80.0), _race(90_000, 1, 1, 90.0)]
    with pytest.raises(ValueError, match="fresh.*two-tick"):
        build_calibration(stale, poses, config, "model")

    one_tick = [_race(0, 0, -1, 100.0), _race(10_000, 1, 1, 110.0)]
    with pytest.raises(ValueError, match="fresh.*two-tick"):
        build_calibration(one_tick, poses, config, "model")
