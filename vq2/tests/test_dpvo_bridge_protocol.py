from __future__ import annotations

import json
import socket

import pytest

from vq2.live.dpvo_bridge_protocol import (
    encode_packet,
    parse_session_message,
    recv_packet,
    session_message,
)
from vq2.live.dpvo_route import DpvoSessionConfig, calibration_identity


def test_session_message_round_trip_uses_32_patches_and_226_fx():
    cfg = DpvoSessionConfig(patches=32)
    parsed = parse_session_message(session_message(cfg, "abc"))
    assert parsed.config.patches == 32
    assert parsed.config.intrinsics[0] == 226.0
    assert parsed.model_sha256 == "abc"
    assert parsed.identity == calibration_identity(cfg, "abc")


def test_session_message_round_trips_bounded_keyframe_window():
    cfg = DpvoSessionConfig(patches=24, width=320, height=180,
                            removal_window=12, optimization_window=6)
    parsed = parse_session_message(session_message(cfg, "abc"))
    assert parsed.config.removal_window == 12
    assert parsed.config.optimization_window == 6
    assert parsed.identity == calibration_identity(cfg, "abc")


def test_length_prefixed_packet_round_trip():
    left, right = socket.socketpair()
    try:
        left.sendall(encode_packet(b"payload"))
        assert recv_packet(right) == b"payload"
    finally:
        left.close()
        right.close()


def test_protocol_rejects_jpeg_before_session():
    with pytest.raises(ValueError, match="session"):
        parse_session_message(b"not json")


def test_protocol_rejects_forged_identity():
    payload = json.loads(session_message(DpvoSessionConfig(), "abc"))
    payload["identity"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        parse_session_message(json.dumps(payload).encode("utf-8"))
