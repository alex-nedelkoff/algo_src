from __future__ import annotations

import pytest

from vq2.live.bridge_dpvo import validate_model_identity
from vq2.live.dpvo_bridge_protocol import parse_session_message, session_message
from vq2.live.dpvo_route import DpvoSessionConfig


def test_service_accepts_matching_model_hash_without_loading_cuda():
    request = parse_session_message(session_message(DpvoSessionConfig(), "model"))
    assert validate_model_identity(request, "model") == request.identity


def test_service_rejects_model_hash_mismatch():
    request = parse_session_message(session_message(DpvoSessionConfig(), "client"))
    with pytest.raises(ValueError, match="model"):
        validate_model_identity(request, "service")
