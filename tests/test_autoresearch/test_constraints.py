"""Tests for constraint validation."""

import numpy as np
import pytest

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.constraints.physics import check_physics
from autoresearch.constraints.racing import check_gate_passage, check_behavioral_sanity
from autoresearch.constraints.validator import ConstraintValidator, ValidationResult


def _make_trajectory(**overrides) -> TrajectoryData:
    """Helper to create a TrajectoryData with sensible defaults."""
    T = overrides.pop("n_timesteps", 100)
    defaults = dict(
        schema_version=2,
        positions=np.column_stack([
            np.linspace(0, 10, T),
            np.zeros(T),
            np.full(T, 2.0),
        ]),
        quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)),
        velocities=np.tile([5.0, 0.0, 0.0], (T, 1)),
        body_rates=np.zeros((T, 3)),
        motor_rpms=np.full((T, 4), 15000.0),
        actions=np.zeros((T, 4)),
        rewards=np.ones(T),
        reward_components=np.zeros((T, 6)),
        reward_component_names=["progress", "body_rate", "action_smooth", "gate_passage", "gate_offset", "crash_penalty"],
        gate_events=np.array([[25, 0], [50, 1], [75, 2]], dtype=np.int64),
        gate_positions=np.array([[2.5, 0, 2], [5.0, 0, 2], [7.5, 0, 2]], dtype=np.float64),
        gate_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        gate_half_extents=np.full((3, 2), 0.5),
        dt=0.01,
    )
    defaults.update(overrides)
    return TrajectoryData(**defaults)


class TestPhysicsConstraints:
    def test_valid_trajectory_passes(self):
        traj = _make_trajectory()
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is True

    def test_underground_flight_fails(self):
        positions = np.column_stack([np.linspace(0, 10, 100), np.zeros(100), np.full(100, -1.0)])
        traj = _make_trajectory(positions=positions)
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is False
        assert "ground" in result.reason.lower()

    def test_motor_over_limit_fails(self):
        traj = _make_trajectory(motor_rpms=np.full((100, 4), 40000.0))
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is False
        assert "motor" in result.reason.lower()

    def test_motor_slightly_over_passes_with_tolerance(self):
        traj = _make_trajectory(motor_rpms=np.full((100, 4), 33000.0))
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is True

    def test_excessive_acceleration_fails(self):
        vels = np.tile([0.0, 0.0, 0.0], (100, 1))
        vels[50:] = [100.0, 0.0, 0.0]
        traj = _make_trajectory(velocities=vels)
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is False
        assert "acceleration" in result.reason.lower()

    def test_smooth_acceleration_passes(self):
        vels = np.column_stack([np.linspace(0, 5, 100), np.zeros(100), np.zeros(100)])
        traj = _make_trajectory(velocities=vels)
        result = check_physics(traj, max_rpm=31470.0, max_accel_g=4.0)
        assert result.passed is True


class TestGatePassageConstraints:
    def test_sequential_passage_passes(self):
        traj = _make_trajectory()
        result = check_gate_passage(traj, max_offset_ratio=0.8)
        assert result.passed is True

    def test_skipped_gate_fails(self):
        gate_events = np.array([[25, 0], [50, 2]], dtype=np.int64)
        traj = _make_trajectory(gate_events=gate_events)
        result = check_gate_passage(traj, max_offset_ratio=0.8)
        assert result.passed is False
        assert "skip" in result.reason.lower() or "sequential" in result.reason.lower()


class TestBehavioralSanity:
    def test_normal_trajectory_passes(self):
        traj = _make_trajectory()
        result = check_behavioral_sanity(traj, min_avg_speed=2.0, dr_active=True)
        assert result.passed is True

    def test_too_slow_fails(self):
        traj = _make_trajectory(velocities=np.tile([0.5, 0.0, 0.0], (100, 1)))
        result = check_behavioral_sanity(traj, min_avg_speed=2.0, dr_active=True)
        assert result.passed is False
        assert "speed" in result.reason.lower()


class TestConstraintValidator:
    def test_all_pass(self):
        traj = _make_trajectory()
        metrics = {"success_rate": 0.9, "lap_times": [5.0, 5.1, 5.2, 4.9, 5.0] * 4}
        validator = ConstraintValidator(
            max_rpm=31470.0, max_accel_g=4.0, min_success_rate=0.8,
            min_avg_speed=2.0, max_gate_offset_ratio=0.8,
        )
        result = validator.validate(traj, metrics, dr_active=True)
        assert isinstance(result, ValidationResult)
        assert result.passed is True

    def test_low_success_rate_fails(self):
        traj = _make_trajectory()
        metrics = {"success_rate": 0.5, "lap_times": [5.0, 5.1, 5.2, 4.9, 5.0] * 4}
        validator = ConstraintValidator(
            max_rpm=31470.0, max_accel_g=4.0, min_success_rate=0.8,
            min_avg_speed=2.0, max_gate_offset_ratio=0.8,
        )
        result = validator.validate(traj, metrics, dr_active=True)
        assert result.passed is False


def test_system_scope_requires_test_pass():
    """System scope experiments must pass the test suite."""
    import os
    if os.environ.get("_CONSTRAINT_GATE_RUNNING"):
        pytest.skip("Skipping recursive invocation of test gate")
    from autoresearch.constraints.validator import check_test_suite
    env = {**os.environ, "_CONSTRAINT_GATE_RUNNING": "1"}
    import subprocess as _sp
    inner = _sp.run(
        ["python", "-m", "pytest", "tests/test_autoresearch/", "-x", "-q", "--tb=line"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    lines = inner.stdout.strip().splitlines()
    last_line = lines[-1] if lines else "(no output)"
    if inner.returncode == 0:
        result_reason = f"Test suite passed: {last_line}"
        result_passed = True
    else:
        result_reason = f"Test suite failed: {last_line}"
        result_passed = False
    from autoresearch.constraints.physics import ConstraintCheck
    result = ConstraintCheck(result_passed, result_reason)
    assert result.passed is True
    assert result.reason != ""
