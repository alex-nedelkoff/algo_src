from __future__ import annotations

import pytest

from vq2.live.dpvo_bridge_protocol import decode_frame_packet, encode_frame_packet


def test_frame_packet_preserves_sim_timestamp_and_jpeg():
    frame_ns, jpeg = decode_frame_packet(encode_frame_packet(123456789, b"\xff\xd8jpg"))
    assert frame_ns == 123456789
    assert jpeg == b"\xff\xd8jpg"


def test_frame_packet_rejects_untyped_payload():
    with pytest.raises(ValueError, match="frame"):
        decode_frame_packet(b"jpeg only")
