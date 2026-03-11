"""Tests for VehicleParams construction."""

from __future__ import annotations

import numpy as np
import pytest

from sim.dynamics.params import VehicleParams


class TestInertiaInput:
    """VehicleParams should accept both 3x3 matrix and 3-element diagonal."""

    def test_3x3_matrix_accepted(self) -> None:
        """Standard 3x3 inertia matrix works (existing behavior)."""
        J = np.diag([0.0025, 0.0025, 0.0045])
        p = VehicleParams(inertia=J)
        np.testing.assert_array_equal(p.inertia, J)

    def test_3elem_list_becomes_diag(self) -> None:
        """A 3-element list is converted to a diagonal 3x3 matrix."""
        p = VehicleParams(inertia=[0.0025, 0.0025, 0.0045])
        expected = np.diag([0.0025, 0.0025, 0.0045])
        np.testing.assert_array_equal(p.inertia, expected)
        assert p.inertia.shape == (3, 3)

    def test_3elem_array_becomes_diag(self) -> None:
        """A 3-element numpy array is converted to a diagonal 3x3 matrix."""
        p = VehicleParams(inertia=np.array([0.0025, 0.0025, 0.0045]))
        assert p.inertia.shape == (3, 3)
        np.testing.assert_allclose(np.diag(p.inertia), [0.0025, 0.0025, 0.0045])

    def test_invalid_shape_raises(self) -> None:
        """Non-3x3, non-3-elem inertia raises ValueError."""
        with pytest.raises(ValueError, match="inertia"):
            VehicleParams(inertia=[1.0, 2.0])

    def test_4elem_raises(self) -> None:
        """4-element inertia raises ValueError."""
        with pytest.raises(ValueError, match="inertia"):
            VehicleParams(inertia=[1.0, 2.0, 3.0, 4.0])


class TestRacingQuadParams:
    """Verify racing quad parameters construct correctly."""

    def test_racing_quad_construction(self) -> None:
        """Racing quad params from config construct a valid VehicleParams."""
        p = VehicleParams(
            mass=0.752,
            arm_length=0.170,
            k_thrust=8.55e-6,
            k_torque=3.02e-7,
            tau_motor=0.02,
            prop_radius=0.0635,
            inertia=[0.0025, 0.0025, 0.0045],
            drag_coeff=[0.0, 0.0, 0.0],
            max_rpm=21702.0,
        )
        assert p.mass == 0.752
        assert p.arm_length == 0.170
        assert p.inertia.shape == (3, 3)
        np.testing.assert_allclose(np.diag(p.inertia), [0.0025, 0.0025, 0.0045])
