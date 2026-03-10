"""Tests for QuadGate: geometry, projection, and PnP round-trip.

Tests that require OpenCV are skipped when cv2 is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.quadgate import QuadGate

cv2 = pytest.importorskip("cv2", reason="OpenCV required for QuadGate PnP tests")


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def camera_intrinsics() -> np.ndarray:
    """Synthetic pinhole camera: 640x480, fx=fy=500, principal point at center."""
    return np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@pytest.fixture
def default_gate() -> QuadGate:
    return QuadGate()


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


class TestQuadGateGeometry:
    """Default gate has correct dimensions and center."""

    def test_default_gate_shape(self, default_gate: QuadGate) -> None:
        assert default_gate.corners_3d.shape == (4, 3)

    def test_default_gate_size(self, default_gate: QuadGate) -> None:
        w, h = default_gate.size
        assert abs(w - 1.0) < 1e-10, f"Expected width 1.0, got {w}"
        assert abs(h - 1.0) < 1e-10, f"Expected height 1.0, got {h}"

    def test_default_gate_center(self, default_gate: QuadGate) -> None:
        np.testing.assert_allclose(default_gate.center, [0.0, 0.0, 0.0], atol=1e-10)


class TestQuadGateProjectPnPRoundtrip:
    """project -> pnp_pose should recover the original camera pose."""

    def test_roundtrip_recovers_pose(
        self, default_gate: QuadGate, camera_intrinsics: np.ndarray
    ) -> None:
        # Place camera 3 m in front of the gate (along +Z), looking at origin.
        # Rotation: identity (camera Z aligns with world Z towards the gate).
        R_true = np.eye(3, dtype=np.float64)
        t_true = np.array([0.0, 0.0, 3.0], dtype=np.float64)

        # Forward project
        corners_2d = default_gate.project(camera_intrinsics, R_true, t_true)
        assert corners_2d.shape == (4, 2)

        # PnP inverse
        R_est, t_est = default_gate.pnp_pose(corners_2d, camera_intrinsics)

        # Translation should match within 1 cm
        np.testing.assert_allclose(t_est, t_true, atol=0.01, err_msg="Translation mismatch")

        # Rotation should match within 1 degree
        R_diff = R_est @ R_true.T
        angle_err = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
        assert np.degrees(angle_err) < 1.0, (
            f"Rotation error {np.degrees(angle_err):.3f} deg exceeds 1 deg"
        )

    def test_roundtrip_offset_pose(
        self, default_gate: QuadGate, camera_intrinsics: np.ndarray
    ) -> None:
        """Camera offset from center — still recovers pose."""
        R_true = np.eye(3, dtype=np.float64)
        t_true = np.array([0.5, -0.3, 4.0], dtype=np.float64)

        corners_2d = default_gate.project(camera_intrinsics, R_true, t_true)
        R_est, t_est = default_gate.pnp_pose(corners_2d, camera_intrinsics)

        np.testing.assert_allclose(t_est, t_true, atol=0.01)

        R_diff = R_est @ R_true.T
        angle_err = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
        assert np.degrees(angle_err) < 1.0
