"""Residual action math: applied = clip(anchor + tanh(delta)*scale, -1, 1).

Pins the contract the ResidualVecEnv wrapper and the laptop deploy mirror must both
implement identically (COR-127 residual learning). Numpy-only -> runs on the Mac.
"""
import numpy as np

SCALE = 0.15


def _apply(anchor, delta, scale=SCALE):
    return np.clip(anchor + np.tanh(delta) * scale, -1.0, 1.0)


def test_residual_identity_when_delta_zero():
    anchor = np.array([[0.1, -0.2, 0.3, -0.05]], dtype=np.float32)
    delta = np.zeros((1, 4), dtype=np.float32)
    np.testing.assert_allclose(_apply(anchor, delta), np.clip(anchor, -1, 1), atol=1e-7)


def test_residual_bounded():
    anchor = np.full((1, 4), 0.95, dtype=np.float32)
    delta = np.full((1, 4), 5.0, dtype=np.float32)   # saturates tanh -> +scale
    applied = _apply(anchor, delta)
    assert np.all(applied <= 1.0) and np.all(applied >= -1.0)


def test_residual_delta_bounded_by_scale():
    anchor = np.zeros((1, 4), dtype=np.float32)
    for d in (-100.0, -1.0, 0.3, 100.0):
        applied = _apply(anchor, np.full((1, 4), d, dtype=np.float32))
        assert np.all(np.abs(applied) <= SCALE + 1e-7)   # delta magnitude capped at scale
