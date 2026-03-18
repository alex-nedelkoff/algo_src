"""Tests for gate-to-gate cubic spline."""
import numpy as np
import pytest

from sim.spline import GateSpline


class TestGateSpline:
    @pytest.fixture
    def square_gates(self):
        positions = np.array([
            [0.0, 0.0, 1.5],
            [3.0, 0.0, 1.5],
            [3.0, 3.0, 1.5],
            [0.0, 3.0, 1.5],
        ])
        return positions

    def test_spline_passes_through_gates(self, square_gates):
        spline = GateSpline(square_gates)
        for pos in square_gates:
            dist = spline.distance_to_nearest(pos)
            assert dist < 0.01, f"Spline should pass through gate at {pos}, dist={dist}"

    def test_midpoint_is_on_spline(self, square_gates):
        spline = GateSpline(square_gates)
        midpoint = (square_gates[0] + square_gates[1]) / 2.0
        dist = spline.distance_to_nearest(midpoint)
        assert dist < 1.0

    def test_far_point_has_large_distance(self, square_gates):
        spline = GateSpline(square_gates)
        far_point = np.array([50.0, 50.0, 1.5])
        dist = spline.distance_to_nearest(far_point)
        assert dist > 10.0

    def test_tangent_is_unit_vector(self, square_gates):
        spline = GateSpline(square_gates)
        _, tangent = spline.nearest_point_and_tangent(square_gates[0])
        np.testing.assert_allclose(np.linalg.norm(tangent), 1.0, atol=0.1)

    def test_closed_spline_wraps(self, square_gates):
        spline = GateSpline(square_gates)
        approach = square_gates[3] + 0.5 * (square_gates[0] - square_gates[3])
        dist = spline.distance_to_nearest(approach)
        assert dist < 1.0

    def test_batch_distance(self, square_gates):
        spline = GateSpline(square_gates)
        points = np.array([[0.0, 0.0, 1.5], [50.0, 50.0, 1.5]])
        dists = spline.distance_to_nearest_batch(points)
        assert dists.shape == (2,)
        assert dists[0] < 0.01
        assert dists[1] > 10.0
