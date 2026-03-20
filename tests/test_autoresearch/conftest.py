"""Shared fixtures for autoresearch tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_npz(tmp_path: Path) -> Path:
    """Create a minimal .npz trajectory file matching schema_version=2."""
    rng = np.random.default_rng(42)
    T = 100  # timesteps
    n_gates = 4
    dt = 0.01

    positions = np.cumsum(rng.normal(0, 0.1, (T, 3)), axis=0)
    positions[:, 2] = np.abs(positions[:, 2]) + 1.0  # keep z > 0

    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))
    # Add small perturbations
    quaternions += rng.normal(0, 0.01, (T, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    # Physically realistic: smooth velocities with bounded acceleration
    # Start at 5 m/s forward, add small random walk (keeps accel < 4g)
    vel_base = np.tile([5.0, 0.0, 0.0], (T, 1))
    vel_noise = np.cumsum(rng.normal(0, 0.1, (T, 3)), axis=0)
    velocities = vel_base + vel_noise
    body_rates = rng.normal(0, 0.5, (T, 3))
    motor_rpms = rng.uniform(5000, 25000, (T, 4))
    actions = rng.uniform(-1, 1, (T, 4))
    rewards = rng.normal(1.0, 0.5, (T,))

    n_components = 6
    reward_components = rng.normal(0, 1, (T, n_components))
    reward_component_names = np.array([
        "progress", "body_rate", "action_smooth",
        "gate_passage", "gate_offset", "crash_penalty",
    ])

    gate_events = np.array([[25, 0], [50, 1], [75, 2]], dtype=np.int64)
    gate_positions = rng.uniform(-5, 5, (n_gates, 3))
    gate_orientations = np.tile([1.0, 0.0, 0.0, 0.0], (n_gates, 1))
    gate_half_extents = np.full((n_gates, 2), 0.5)

    path = tmp_path / "episode_0.npz"
    np.savez_compressed(
        path,
        schema_version=np.int64(2),
        positions=positions,
        quaternions=quaternions,
        velocities=velocities,
        body_rates=body_rates,
        motor_rpms=motor_rpms,
        actions=actions,
        rewards=rewards,
        reward_components=reward_components,
        reward_component_names=reward_component_names,
        gate_events=gate_events,
        gate_positions=gate_positions,
        gate_orientations=gate_orientations,
        gate_half_extents=gate_half_extents,
        dt=np.float64(dt),
    )
    return path


@pytest.fixture
def max_rpm() -> float:
    """Racing quad max RPM from configs/sim/numpy_quad.yaml."""
    return 31470.0


@pytest.fixture
def control_freq() -> float:
    """Control frequency = 1/dt."""
    return 100.0  # 1 / 0.01
