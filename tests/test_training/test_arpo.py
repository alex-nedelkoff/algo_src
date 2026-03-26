"""Tests for α-RPO action fusion and attenuation."""
import numpy as np
import pytest

from training.arpo import AlphaSchedule, fuse_actions


class TestAlphaSchedule:
    def test_starts_at_zero(self):
        sched = AlphaSchedule(k_end_fraction=0.25, total_steps=1_000_000)
        assert sched.get_alpha(0) == 0.0

    def test_reaches_one_at_k_end(self):
        sched = AlphaSchedule(k_end_fraction=0.25, total_steps=1_000_000)
        assert sched.get_alpha(250_000) == pytest.approx(1.0)

    def test_stays_at_one_after_k_end(self):
        sched = AlphaSchedule(k_end_fraction=0.25, total_steps=1_000_000)
        assert sched.get_alpha(500_000) == pytest.approx(1.0)

    def test_midpoint(self):
        sched = AlphaSchedule(k_end_fraction=0.25, total_steps=1_000_000)
        assert sched.get_alpha(125_000) == pytest.approx(0.5)


class TestFuseActions:
    def test_alpha_zero_gives_base(self):
        base = np.array([[1.0, 2.0, 3.0, 4.0]])
        learned = np.array([[10.0, 20.0, 30.0, 40.0]])
        fused = fuse_actions(base, learned, alpha=0.0)
        np.testing.assert_allclose(fused, base)

    def test_alpha_one_gives_learned(self):
        base = np.array([[1.0, 2.0, 3.0, 4.0]])
        learned = np.array([[10.0, 20.0, 30.0, 40.0]])
        fused = fuse_actions(base, learned, alpha=1.0)
        np.testing.assert_allclose(fused, learned)

    def test_alpha_half_gives_blend(self):
        base = np.array([[0.0, 0.0, 0.0, 0.0]])
        learned = np.array([[2.0, 2.0, 2.0, 2.0]])
        fused = fuse_actions(base, learned, alpha=0.5)
        np.testing.assert_allclose(fused, [[1.0, 1.0, 1.0, 1.0]])

    def test_batch(self):
        base = np.zeros((5, 4))
        learned = np.ones((5, 4))
        fused = fuse_actions(base, learned, alpha=0.5)
        assert fused.shape == (5, 4)
        np.testing.assert_allclose(fused, 0.5)
