"""Tests for TRPY → motor speed mixer."""
import numpy as np
import pytest

from sim.dynamics.trpy_mixer import TRPYMixer
from sim.dynamics.params import VehicleParams


class TestTRPYMixer:
    """Test TRPY command → motor speed conversion."""

    @pytest.fixture
    def mixer(self):
        return TRPYMixer(VehicleParams())

    def test_hover_thrust_gives_equal_motors(self, mixer):
        params = VehicleParams()
        hover_thrust = params.mass * 9.81
        trpy = np.array([hover_thrust, 0.0, 0.0, 0.0])
        motor_speeds = mixer.mix(trpy)
        assert motor_speeds.shape == (4,)
        np.testing.assert_allclose(motor_speeds, motor_speeds[0], atol=1e-6)
        assert np.all(motor_speeds > 0)

    def test_zero_thrust_gives_zero_motors(self, mixer):
        trpy = np.array([0.0, 0.0, 0.0, 0.0])
        motor_speeds = mixer.mix(trpy)
        np.testing.assert_allclose(motor_speeds, 0.0, atol=1e-10)

    def test_positive_roll_rate_differential(self, mixer):
        params = VehicleParams()
        hover_thrust = params.mass * 9.81
        trpy_neutral = np.array([hover_thrust, 0.0, 0.0, 0.0])
        trpy_roll = np.array([hover_thrust, 5.0, 0.0, 0.0])
        motors_neutral = mixer.mix(trpy_neutral)
        motors_roll = mixer.mix(trpy_roll)
        assert not np.allclose(motors_roll, motors_neutral, atol=1e-3)

    def test_output_clipped_to_valid_range(self, mixer):
        params = VehicleParams()
        trpy = np.array([100.0, 50.0, 50.0, 50.0])
        motor_speeds = mixer.mix(trpy)
        assert np.all(motor_speeds >= 0)
        assert np.all(motor_speeds <= params.max_omega)

    def test_batch_mix(self, mixer):
        params = VehicleParams()
        hover_thrust = params.mass * 9.81
        trpy_batch = np.array([
            [hover_thrust, 0.0, 0.0, 0.0],
            [hover_thrust, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        motor_speeds = mixer.mix_batch(trpy_batch)
        assert motor_speeds.shape == (3, 4)
        np.testing.assert_allclose(motor_speeds[0], motor_speeds[0, 0], atol=1e-6)
        np.testing.assert_allclose(motor_speeds[2], 0.0, atol=1e-10)

    def test_yaw_torque_uses_motor_dirs(self, mixer):
        params = VehicleParams()
        hover_thrust = params.mass * 9.81
        trpy_yaw_pos = np.array([hover_thrust, 0.0, 0.0, 5.0])
        trpy_yaw_neg = np.array([hover_thrust, 0.0, 0.0, -5.0])
        motors_pos = mixer.mix(trpy_yaw_pos)
        motors_neg = mixer.mix(trpy_yaw_neg)
        diff_pos = motors_pos - mixer.mix(np.array([hover_thrust, 0.0, 0.0, 0.0]))
        diff_neg = motors_neg - mixer.mix(np.array([hover_thrust, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(diff_pos, -diff_neg, atol=1e-6)
