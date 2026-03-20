"""Tests for behavioral descriptor computation."""

import numpy as np
import pytest

from autoresearch.analysis.trajectory import load_trajectory
from autoresearch.descriptors.actuator import actuator_utilization
from autoresearch.descriptors.smoothness import control_smoothness
from autoresearch.descriptors.aero_regime import aero_regime_index
from autoresearch.descriptors.compute import compute_descriptors, DescriptorVector


class TestActuatorUtilization:
    def test_hovering_is_low(self, max_rpm):
        motor_rpms = np.full((100, 4), max_rpm * 0.25)
        result = actuator_utilization(motor_rpms, max_rpm)
        assert 0.2 < result < 0.3

    def test_full_throttle_is_near_one(self, max_rpm):
        motor_rpms = np.full((100, 4), max_rpm * 0.99)
        result = actuator_utilization(motor_rpms, max_rpm)
        assert result > 0.95

    def test_range_zero_to_one(self, max_rpm):
        motor_rpms = np.random.default_rng(42).uniform(0, max_rpm, (100, 4))
        result = actuator_utilization(motor_rpms, max_rpm)
        assert 0.0 <= result <= 1.0


class TestControlSmoothness:
    def test_constant_commands_are_smooth(self, max_rpm, control_freq):
        motor_rpms = np.full((100, 4), max_rpm * 0.5)
        result = control_smoothness(motor_rpms, max_rpm, control_freq)
        assert result < 0.01

    def test_oscillating_commands_are_rough(self, max_rpm, control_freq):
        motor_rpms = np.zeros((100, 4))
        motor_rpms[::2] = max_rpm
        result = control_smoothness(motor_rpms, max_rpm, control_freq)
        assert result > 0.5

    def test_range_zero_to_one(self, max_rpm, control_freq):
        motor_rpms = np.random.default_rng(42).uniform(0, max_rpm, (100, 4))
        result = control_smoothness(motor_rpms, max_rpm, control_freq)
        assert 0.0 <= result <= 1.0


class TestAeroRegimeIndex:
    def test_stationary_is_zero(self):
        velocities = np.zeros((100, 3))
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
        result = aero_regime_index(velocities, quaternions)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_forward_flight_level_is_low(self):
        velocities = np.tile([10.0, 0.0, 0.0], (100, 1))
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
        result = aero_regime_index(velocities, quaternions)
        # Identity quat = body z points up [0,0,1], velocity is [10,0,0]
        # alpha = 90 degrees, sin(90)=1, speed=10 → index ≈ 10
        assert result > 5.0

    def test_hovering_is_zero(self):
        velocities = np.tile([0.0, 0.0, 0.1], (100, 1))
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
        result = aero_regime_index(velocities, quaternions)
        assert result < 0.2


class TestComputeDescriptors:
    def test_returns_descriptor_vector(self, sample_npz, max_rpm, control_freq):
        traj = load_trajectory(sample_npz)
        desc = compute_descriptors(traj, max_rpm, control_freq)
        assert isinstance(desc, DescriptorVector)
        assert 0.0 <= desc.actuator_utilization <= 1.0
        assert 0.0 <= desc.control_smoothness <= 1.0
        assert desc.aero_regime >= 0.0

    def test_descriptor_vector_to_tuple(self, sample_npz, max_rpm, control_freq):
        traj = load_trajectory(sample_npz)
        desc = compute_descriptors(traj, max_rpm, control_freq)
        t = desc.to_tuple()
        assert len(t) == 3
        assert t == (desc.actuator_utilization, desc.control_smoothness, desc.aero_regime)
