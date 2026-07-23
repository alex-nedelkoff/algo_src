from __future__ import annotations

import pytest

from vq2.live.dpvo_odom_bridge import (
    config_from_env,
    publish_route_observation,
    validate_ready,
)
from vq2.live.dpvo_route import RouteObservation, calibration_identity


def test_client_defaults_to_32_patches_and_calibrated_intrinsics():
    config = config_from_env({})
    assert config.patches == 32
    assert config.intrinsics == (320.0, 320.0, 319.5, 179.5)
    assert config.cuda_fraction == 0.48


def test_client_scales_intrinsics_for_half_resolution():
    config = config_from_env({"DPVO_WIDTH": "320", "DPVO_HEIGHT": "180"})
    assert config.intrinsics == (160.0, 160.0, 159.75, 89.75)


def test_client_defaults_to_stock_keyframe_graph_window():
    config = config_from_env({})
    assert config.removal_window == 22
    assert config.optimization_window == 10


def test_client_reads_bounded_keyframe_graph_window_from_env():
    config = config_from_env({"DPVO_REMOVAL_WINDOW": "12", "DPVO_OPT_WINDOW": "6"})
    assert config.removal_window == 12
    assert config.optimization_window == 6


def test_ready_identity_must_match_requested_session():
    config = config_from_env({})
    identity = calibration_identity(config, "model")
    validate_ready({"type": "ready", "identity": identity}, config, "model")
    with pytest.raises(ValueError, match="identity"):
        validate_ready({"type": "ready", "identity": "wrong"}, config, "model")


def test_publish_route_observation_sets_independent_state_fields():
    state = {}
    observation = RouteObservation((7.0, 1.0, -1.35), True, "ok", 123)
    publish_route_observation(state, observation, wall=10.5)
    assert state == {
        "dpvo_route_p": (7.0, 1.0, -1.35),
        "dpvo_route_wall": 10.5,
        "dpvo_route_healthy": True,
        "dpvo_route_reason": "ok",
        "dpvo_route_frame_ns": 123,
    }


def test_bridge_host_env_overrides_local_wsl_discovery():
    # Vagon has no WSL: DPVO_BRIDGE_HOST must win before any wsl.exe call
    import inspect
    from vq2.live import dpvo_odom_bridge as mod
    source = inspect.getsource(mod._wsl_ip)
    env_at = source.index("DPVO_BRIDGE_HOST")
    wsl_at = source.index('"wsl"')
    assert env_at < wsl_at
