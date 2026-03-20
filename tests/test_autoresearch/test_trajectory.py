"""Tests for trajectory .npz loading."""

import numpy as np

from autoresearch.analysis.trajectory import load_trajectory, TrajectoryData


def test_load_trajectory_fields(sample_npz):
    traj = load_trajectory(sample_npz)
    assert isinstance(traj, TrajectoryData)
    assert traj.positions.shape == (100, 3)
    assert traj.quaternions.shape == (100, 4)
    assert traj.velocities.shape == (100, 3)
    assert traj.body_rates.shape == (100, 3)
    assert traj.motor_rpms.shape == (100, 4)
    assert traj.gate_events.shape == (3, 2)
    assert traj.gate_positions.shape == (4, 3)
    assert traj.dt == 0.01


def test_load_trajectory_dtypes(sample_npz):
    traj = load_trajectory(sample_npz)
    assert traj.positions.dtype == np.float64
    assert traj.motor_rpms.dtype == np.float64
    assert traj.gate_events.dtype == np.int64


def test_load_trajectory_schema_version(sample_npz):
    traj = load_trajectory(sample_npz)
    assert traj.schema_version == 2


def test_load_trajectory_reward_components(sample_npz):
    traj = load_trajectory(sample_npz)
    assert traj.reward_components.shape == (100, 6)
    assert len(traj.reward_component_names) == 6
    assert "progress" in traj.reward_component_names
