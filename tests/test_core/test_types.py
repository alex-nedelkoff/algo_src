"""Tests for core.types module."""

from __future__ import annotations

import numpy as np
import pytest

from core.types import Action, ActionMode, GateState, QuadState


class TestQuadState:
    """Tests for QuadState dataclass."""

    def test_default_construction(self, default_quad_state: QuadState) -> None:
        """Default QuadState should be at origin with identity quaternion."""
        np.testing.assert_array_equal(default_quad_state.pos, [0, 0, 0])
        np.testing.assert_array_equal(default_quad_state.vel, [0, 0, 0])
        np.testing.assert_array_almost_equal(default_quad_state.quat, [1, 0, 0, 0])
        np.testing.assert_array_equal(default_quad_state.omega, [0, 0, 0])
        np.testing.assert_array_equal(default_quad_state.motor_speeds, [0, 0, 0, 0])

    def test_quaternion_normalization(self) -> None:
        """QuadState should normalize quaternion on construction."""
        state = QuadState(quat=np.array([2.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(np.linalg.norm(state.quat), 1.0)
        np.testing.assert_array_almost_equal(state.quat, [1.0, 0.0, 0.0, 0.0])

    def test_quaternion_normalization_arbitrary(self) -> None:
        """Arbitrary quaternion should be normalized to unit length."""
        q = np.array([1.0, 1.0, 1.0, 1.0])
        state = QuadState(quat=q)
        np.testing.assert_almost_equal(np.linalg.norm(state.quat), 1.0)
        expected = q / np.linalg.norm(q)
        np.testing.assert_array_almost_equal(state.quat, expected)

    def test_quaternion_zero_raises(self) -> None:
        """Zero quaternion should raise ValueError."""
        with pytest.raises(ValueError, match="near zero"):
            QuadState(quat=np.array([0.0, 0.0, 0.0, 0.0]))

    def test_serialization_round_trip(self) -> None:
        """to_vector/from_vector should round-trip correctly."""
        state = QuadState(
            pos=np.array([1.0, 2.0, 3.0]),
            vel=np.array([0.1, 0.2, 0.3]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.array([0.5, -0.5, 0.1]),
            motor_speeds=np.array([100.0, 200.0, 300.0, 400.0]),
        )
        vec = state.to_vector()
        assert vec.shape == (17,)

        reconstructed = QuadState.from_vector(vec)
        np.testing.assert_array_almost_equal(reconstructed.pos, state.pos)
        np.testing.assert_array_almost_equal(reconstructed.vel, state.vel)
        np.testing.assert_array_almost_equal(reconstructed.quat, state.quat)
        np.testing.assert_array_almost_equal(reconstructed.omega, state.omega)
        np.testing.assert_array_almost_equal(reconstructed.motor_speeds, state.motor_speeds)

    def test_from_vector_wrong_size(self) -> None:
        """from_vector with wrong size should raise ValueError."""
        with pytest.raises(ValueError, match="17-element"):
            QuadState.from_vector(np.zeros(10))

    def test_wrong_pos_shape(self) -> None:
        """Wrong position shape should raise ValueError."""
        with pytest.raises(ValueError, match="pos must have shape"):
            QuadState(pos=np.zeros(4))


class TestAction:
    """Tests for Action dataclass."""

    def test_default_action(self) -> None:
        """Default action should be zero motor RPMs."""
        action = Action()
        np.testing.assert_array_equal(action.values, [0, 0, 0, 0])
        assert action.mode == ActionMode.MOTOR_RPM

    def test_trpy_mode(self) -> None:
        """Action can be created in TRPY mode."""
        action = Action(values=np.array([9.81, 0.0, 0.0, 0.0]), mode=ActionMode.TRPY)
        assert action.mode == ActionMode.TRPY


class TestGateState:
    """Tests for GateState dataclass."""

    def test_gate_quaternion_normalization(self) -> None:
        """GateState should normalize orientation quaternion."""
        gate = GateState(orientation=np.array([2.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(np.linalg.norm(gate.orientation), 1.0)
