from __future__ import annotations

from vq2.dpvo_bridge_replay import pose_record, select_frame_rows


def test_replay_selects_last_contiguous_block_and_applies_stride():
    rows = [
        {"sim_ns": 1_000_000_000, "rx_wall": 1.0},
        {"sim_ns": 1_030_000_000, "rx_wall": 1.03},
        {"sim_ns": 10_000_000_000, "rx_wall": 10.0},
        {"sim_ns": 10_030_000_000, "rx_wall": 10.03},
        {"sim_ns": 10_060_000_000, "rx_wall": 10.06},
    ]
    selected = select_frame_rows(rows, stride=2, block_gap_s=5.0)
    assert [row["sim_ns"] for row in selected] == [10_000_000_000, 10_060_000_000]


def test_pose_record_preserves_identity_and_service_telemetry():
    reply = {
        "type": "pose",
        "ok": True,
        "frame_ns": 123,
        "p": [1, 2, 3],
        "dt_ms": 17.5,
        "vram_alloc_mb": 900.0,
        "cuda_free_mb": 2500.0,
    }
    record = pose_record(reply, "identity")
    assert record["identity"] == "identity"
    assert record["frame_ns"] == 123
    assert record["p"] == [1.0, 2.0, 3.0]
    assert record["dt_ms"] == 17.5
    assert record["cuda_free_mb"] == 2500.0
