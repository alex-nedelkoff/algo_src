"""Inbound handler table — refactor structural test."""
from __future__ import annotations

from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.shim import MavlinkShim


def test_shim_exposes_inbound_handler_table():
    """After __post_init__, the shim has a dict of inbound handlers covering
    the message types that COR-98 already supported, plus an unknown fallback."""
    params = VehicleParams()
    shim = MavlinkShim(backend=NumpyQuadBackend(params=params), params=params)
    handlers = shim._inbound_handlers  # internal API by design
    assert isinstance(handlers, dict)
    # COR-98 message types must be registered.
    assert "SET_ATTITUDE_TARGET" in handlers
    assert "SET_POSITION_TARGET_LOCAL_NED" in handlers
    assert "TIMESYNC" in handlers
    # All registered handlers must be callable.
    for k, fn in handlers.items():
        assert callable(fn), f"handler for {k} not callable"
    # A default unknown-message handler must exist as an attribute.
    assert hasattr(shim, "_on_unknown")
    assert callable(shim._on_unknown)
