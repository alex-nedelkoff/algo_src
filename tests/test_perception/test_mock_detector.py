"""Tests for MockDetector: noise injection, dropout, and ground-truth fidelity."""

from __future__ import annotations

import numpy as np
import pytest

from perception.detectors.base import GateDetection
from perception.detectors.mock_detector import MockDetector


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def gate_positions() -> list[np.ndarray]:
    return [
        np.array([5.0, 0.0, 2.0]),
        np.array([10.0, 5.0, 2.0]),
        np.array([5.0, 10.0, 2.0]),
    ]


@pytest.fixture
def dummy_image() -> np.ndarray:
    """A placeholder image — MockDetector ignores it."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


class TestMockDetectorBasic:
    """Correct number of detections and GateDetection structure."""

    def test_returns_correct_number_of_gates(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        det = MockDetector(gate_positions, noise_std=0.0, dropout_rate=0.0, seed=42)
        results = det.detect(dummy_image)
        assert len(results) == len(gate_positions)

    def test_returns_gate_detection_instances(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        det = MockDetector(gate_positions, noise_std=0.0, dropout_rate=0.0, seed=42)
        for d in det.detect(dummy_image):
            assert isinstance(d, GateDetection)
            assert d.corners_2d.shape == (4, 2)
            assert 0.0 <= d.confidence <= 1.0

    def test_gate_ids_are_sequential(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        det = MockDetector(gate_positions, noise_std=0.0, dropout_rate=0.0, seed=42)
        ids = [d.gate_id for d in det.detect(dummy_image)]
        assert ids == [0, 1, 2]


class TestMockDetectorZeroNoise:
    """Zero noise should produce exact ground truth."""

    def test_zero_noise_exact_ground_truth(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        det = MockDetector(gate_positions, noise_std=0.0, dropout_rate=0.0, seed=0)
        results = det.detect(dummy_image)

        for detection, pos in zip(results, gate_positions):
            expected = MockDetector._gate_to_corners_2d(pos)
            np.testing.assert_array_equal(detection.corners_2d, expected)


class TestMockDetectorNoise:
    """Gaussian noise is within expected bounds."""

    def test_noise_within_3_sigma(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        noise_std = 0.5
        det = MockDetector(gate_positions, noise_std=noise_std, dropout_rate=0.0, seed=99)
        n_calls = 500
        max_abs_error = 0.0

        gt_corners = [MockDetector._gate_to_corners_2d(p) for p in gate_positions]

        for _ in range(n_calls):
            results = det.detect(dummy_image)
            for d, gt in zip(results, gt_corners):
                diff = np.abs(d.corners_2d - gt)
                max_abs_error = max(max_abs_error, diff.max())

        # With 500 calls x 3 gates x 8 values each, we'd expect at
        # least some values near 3*sigma, but virtually none beyond it.
        assert max_abs_error < 5 * noise_std, (
            f"Max error {max_abs_error:.4f} exceeds 5 * noise_std = {5 * noise_std}"
        )

    def test_noise_is_nonzero(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        det = MockDetector(gate_positions, noise_std=1.0, dropout_rate=0.0, seed=7)
        gt = MockDetector._gate_to_corners_2d(gate_positions[0])
        result = det.detect(dummy_image)[0]
        assert not np.allclose(result.corners_2d, gt), "Noise should perturb corners"


class TestMockDetectorDropout:
    """Dropout rate matches config within tolerance."""

    def test_dropout_rate_matches_config(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        dropout_rate = 0.3
        n_calls = 2000
        det = MockDetector(
            gate_positions, noise_std=0.0, dropout_rate=dropout_rate, seed=123
        )

        total_possible = n_calls * len(gate_positions)
        total_detected = sum(len(det.detect(dummy_image)) for _ in range(n_calls))
        empirical_dropout = 1.0 - total_detected / total_possible

        assert abs(empirical_dropout - dropout_rate) < 0.05, (
            f"Empirical dropout {empirical_dropout:.3f} vs "
            f"expected {dropout_rate:.3f} (tolerance 0.05)"
        )

    def test_zero_dropout_returns_all(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        det = MockDetector(gate_positions, noise_std=0.0, dropout_rate=0.0, seed=0)
        for _ in range(100):
            assert len(det.detect(dummy_image)) == len(gate_positions)

    def test_full_dropout_returns_none(
        self, gate_positions: list[np.ndarray], dummy_image: np.ndarray
    ) -> None:
        det = MockDetector(gate_positions, noise_std=0.0, dropout_rate=1.0, seed=0)
        for _ in range(100):
            assert len(det.detect(dummy_image)) == 0
