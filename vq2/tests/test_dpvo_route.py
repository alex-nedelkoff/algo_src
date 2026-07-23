from __future__ import annotations

import pytest

from vq2.live.dpvo_route import (
    DpvoSessionConfig,
    calibration_identity,
    fit_tick_scale,
    scale_intrinsics,
)


def test_half_resolution_scales_calibrated_intrinsics():
    assert scale_intrinsics(
        (320, 320, 319.5, 179.5), (640, 360), (320, 180)
    ) == (160.0, 160.0, 159.75, 89.75)


def test_identity_changes_with_tracking_configuration():
    a = DpvoSessionConfig(patches=32, width=640, height=360, stride=2)
    b = DpvoSessionConfig(patches=24, width=640, height=360, stride=2)
    assert calibration_identity(a, "weights") != calibration_identity(b, "weights")


def test_default_window_matches_stock_dpvo_config():
    config = DpvoSessionConfig()
    assert config.removal_window == 22
    assert config.optimization_window == 10


def test_identity_changes_with_keyframe_graph_window():
    stock = DpvoSessionConfig()
    bounded = DpvoSessionConfig(removal_window=12, optimization_window=6)
    assert calibration_identity(stock, "w") != calibration_identity(bounded, "w")


def test_window_bounds_are_validated():
    with pytest.raises(ValueError, match="optimization_window"):
        DpvoSessionConfig(optimization_window=1)
    with pytest.raises(ValueError, match="removal_window"):
        DpvoSessionConfig(removal_window=5, optimization_window=6)


def test_tick_scale_uses_known_gate_displacement():
    scale = fit_tick_scale([0, 0, 0], [2, 0, 0], [6, 0, 0], [10, 0, 0])
    assert scale == 2.0
