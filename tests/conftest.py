"""Shared test fixtures for the algo_src test suite."""

from __future__ import annotations

# Windows DLL loading order workaround.
# On the monorace conda env, torch's fbgemm.dll has a libomp dependency that
# can't resolve once numpy has already pulled in Intel MKL's libiomp5md.dll.
# Pre-loading torch (in a try/except — torch is an optional dep for many
# tests) BEFORE numpy is imported makes the right libomp version stick.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch  # noqa: F401  -- pre-load before numpy
except ImportError:
    pass

import numpy as np
import pytest

from sim.tracks import Track
from sim.types import GateState, QuadState


@pytest.fixture
def default_quad_state() -> QuadState:
    """A QuadState at the origin with identity orientation and zero velocity."""
    return QuadState()


@pytest.fixture
def random_quad_state(rng: np.random.Generator | None = None) -> QuadState:
    """A QuadState with random but valid values."""
    rng = np.random.default_rng(42)
    return QuadState(
        pos=rng.uniform(-10, 10, size=3),
        vel=rng.uniform(-5, 5, size=3),
        quat=rng.normal(size=4),  # Will be normalized in __post_init__
        omega=rng.uniform(-3, 3, size=3),
        motor_speeds=rng.uniform(0, 1000, size=4),
    )


@pytest.fixture
def simple_track() -> Track:
    """A simple 3-gate track for testing."""
    gates = [
        GateState(position=np.array([5.0, 0.0, 2.0])),
        GateState(position=np.array([10.0, 5.0, 2.0])),
        GateState(position=np.array([5.0, 10.0, 2.0])),
    ]
    return Track(gates)
