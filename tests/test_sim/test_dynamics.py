"""Tests for NumpyQuadDynamics.

Covers hover equilibrium, free-fall, vectorization consistency,
quaternion normalization, and motor lag convergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.dynamics.numpy_quad import GRAVITY, MOTOR, OMEGA, POS, QUAT, VEL, NumpyQuadDynamics
from sim.dynamics.params import VehicleParams


@pytest.fixture
def dynamics() -> NumpyQuadDynamics:
    """Default dynamics instance with CrazyFlie 2.1 params."""
    return NumpyQuadDynamics()


@pytest.fixture
def params() -> VehicleParams:
    """Default vehicle parameters."""
    return VehicleParams()


class TestHoverEquilibrium:
    """At analytical hover thrust, the drone should maintain altitude."""

    def test_hover_altitude_maintained(self, dynamics: NumpyQuadDynamics) -> None:
        """With hover thrust, altitude stays within 1cm over 5 seconds."""
        states = dynamics.reset(1)
        initial_z = states[0, 2]

        hover_w = dynamics.hover_omega()
        action = np.full((1, 4), hover_w)

        dt = 0.01
        n_steps = int(5.0 / dt)  # 5 seconds

        for _ in range(n_steps):
            states = dynamics.step(states, action, dt=dt)

        final_z = states[0, 2]
        assert abs(final_z - initial_z) < 0.01, (
            f"Altitude drifted by {abs(final_z - initial_z):.4f} m "
            f"(initial={initial_z:.4f}, final={final_z:.4f})"
        )

    def test_hover_velocity_near_zero(self, dynamics: NumpyQuadDynamics) -> None:
        """Velocities should remain near zero during hover."""
        states = dynamics.reset(1)
        hover_w = dynamics.hover_omega()
        action = np.full((1, 4), hover_w)

        for _ in range(500):
            states = dynamics.step(states, action)

        vel = states[0, VEL]
        assert np.all(np.abs(vel) < 0.05), f"Velocity during hover: {vel}"


class TestFreeFall:
    """With zero thrust, position should match 0.5*g*t^2 free-fall."""

    def test_free_fall_matches_analytical(self, dynamics: NumpyQuadDynamics) -> None:
        """Free-fall position matches 0.5*g*t^2 within 0.1% over 1s."""
        states = dynamics.reset(1)
        # Ensure motors are at zero so there's no residual thrust
        states[0, MOTOR] = 0.0
        initial_z = states[0, 2]

        # Zero motor command
        action = np.zeros((1, 4))

        dt = 0.0005  # Fine dt for forward Euler accuracy
        n_steps = int(1.0 / dt)

        for _ in range(n_steps):
            states = dynamics.step(states, action, dt=dt)

        t = 1.0
        expected_z = initial_z - 0.5 * GRAVITY * t * t
        actual_z = states[0, 2]

        # 0.1% of the displacement
        displacement = 0.5 * GRAVITY * t * t
        tolerance = 0.001 * displacement

        assert abs(actual_z - expected_z) < tolerance, (
            f"Free-fall z: expected={expected_z:.6f}, got={actual_z:.6f}, "
            f"error={abs(actual_z - expected_z):.6f}, tolerance={tolerance:.6f}"
        )

    def test_free_fall_velocity(self, dynamics: NumpyQuadDynamics) -> None:
        """Free-fall velocity should match g*t."""
        states = dynamics.reset(1)
        # Ensure motors are at zero so there's no residual thrust
        states[0, MOTOR] = 0.0
        action = np.zeros((1, 4))

        dt = 0.001
        n_steps = int(1.0 / dt)

        for _ in range(n_steps):
            states = dynamics.step(states, action, dt=dt)

        expected_vz = -GRAVITY * 1.0
        actual_vz = states[0, 5]  # vz

        tolerance = abs(expected_vz) * 0.001
        assert abs(actual_vz - expected_vz) < tolerance, (
            f"Free-fall vz: expected={expected_vz:.4f}, got={actual_vz:.4f}"
        )


class TestVectorization:
    """N=100 envs should produce the same result as N=1 iterated."""

    def test_batch_matches_single(self) -> None:
        """100 parallel envs match 1 env repeated 100 times."""
        dynamics = NumpyQuadDynamics()
        n_envs = 100

        # Reset all at once
        batch_states = dynamics.reset(n_envs)
        # All should be identical after reset
        single_state = dynamics.reset(1)

        # Apply same action to all
        hover_w = dynamics.hover_omega()
        batch_action = np.full((n_envs, 4), hover_w)
        single_action = np.full((1, 4), hover_w)

        # Step both
        n_steps = 50
        for _ in range(n_steps):
            batch_states = dynamics.step(batch_states, batch_action)
            single_state = dynamics.step(single_state, single_action)

        # All batch envs should match the single env
        for i in range(n_envs):
            np.testing.assert_allclose(
                batch_states[i],
                single_state[0],
                atol=1e-10,
                err_msg=f"Env {i} diverged from single env",
            )

    def test_different_actions_produce_different_states(self) -> None:
        """Environments with different actions diverge."""
        dynamics = NumpyQuadDynamics()
        states = dynamics.reset(2)
        hover_w = dynamics.hover_omega()

        actions = np.array([
            [hover_w, hover_w, hover_w, hover_w],
            [hover_w * 1.1, hover_w * 0.9, hover_w * 1.1, hover_w * 0.9],
        ])

        for _ in range(100):
            states = dynamics.step(states, actions)

        # States should differ
        assert not np.allclose(states[0], states[1], atol=1e-6)


class TestQuaternionNormalization:
    """Quaternion should stay normalized after many integration steps."""

    def test_quat_norm_after_1000_steps(self, dynamics: NumpyQuadDynamics) -> None:
        """Quaternion norm stays within [0.99, 1.01] after 1000 steps."""
        states = dynamics.reset(1)
        hover_w = dynamics.hover_omega()
        action = np.full((1, 4), hover_w)

        for _ in range(1000):
            states = dynamics.step(states, action)

        quat = states[0, QUAT]
        quat_norm = np.linalg.norm(quat)
        assert 0.99 < quat_norm < 1.01, f"Quaternion norm = {quat_norm}"

    def test_quat_norm_with_aggressive_maneuver(self) -> None:
        """Quaternion stays normalized even with asymmetric thrust."""
        dynamics = NumpyQuadDynamics()
        states = dynamics.reset(1)
        hover_w = dynamics.hover_omega()

        # Mildly asymmetric action to induce rotation without diverging
        action = np.array([[hover_w * 1.1, hover_w * 0.9, hover_w * 1.1, hover_w * 0.9]])

        for _ in range(200):
            states = dynamics.step(states, action)

        quat = states[0, QUAT]
        quat_norm = np.linalg.norm(quat)
        assert 0.99 < quat_norm < 1.01, f"Quaternion norm = {quat_norm}"


class TestMotorLag:
    """Motor speed should converge to command within 5*tau_motor."""

    def test_motor_step_response(self, dynamics: NumpyQuadDynamics) -> None:
        """Motor speed converges to command within 5*tau_motor."""
        states = dynamics.reset(1)
        # Start with zero motor speeds
        states[0, MOTOR] = 0.0

        target_w = dynamics.hover_omega()
        action = np.full((1, 4), target_w)

        tau = dynamics.params.tau_motor
        settle_time = 5.0 * tau
        dt = dynamics.dt
        n_steps = int(settle_time / dt)

        for _ in range(n_steps):
            states = dynamics.step(states, action)

        motor_speeds = states[0, MOTOR]
        # After 5*tau, first-order system reaches ~99.3% of target
        for j in range(4):
            error_frac = abs(motor_speeds[j] - target_w) / target_w
            assert error_frac < 0.02, (
                f"Motor {j}: speed={motor_speeds[j]:.2f}, target={target_w:.2f}, "
                f"error={error_frac*100:.1f}%"
            )

    def test_motor_lag_time_constant(self, dynamics: NumpyQuadDynamics) -> None:
        """After 1*tau, motor should reach ~63% of step command."""
        states = dynamics.reset(1)
        states[0, MOTOR] = 0.0

        target_w = dynamics.hover_omega()
        action = np.full((1, 4), target_w)

        tau = dynamics.params.tau_motor
        dt = 0.001  # Fine dt for accuracy
        n_steps = int(tau / dt)

        for _ in range(n_steps):
            states = dynamics.step(states, action, dt=dt)

        motor_speeds = states[0, MOTOR]
        expected_fraction = 1.0 - np.exp(-1.0)  # ~0.632
        for j in range(4):
            actual_fraction = motor_speeds[j] / target_w
            assert abs(actual_fraction - expected_fraction) < 0.05, (
                f"Motor {j}: fraction={actual_fraction:.3f}, "
                f"expected ~{expected_fraction:.3f}"
            )
