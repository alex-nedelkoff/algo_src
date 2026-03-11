"""Tests for random gate episode start and gate collision (COR-48 gap 7c/7d).

Covers:
- Random gate start positions are behind the selected gate
- Velocity and attitude perturbations are applied
- Deterministic seeding produces reproducible starts
- Gate collision terminates episode when crossing outside opening
- Gate collision can be disabled
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.dynamics.numpy_quad import POS, QUAT, VEL, OMEGA
from sim.envs.gate_race_env import (
    GateRaceEnv,
    OBS_DIM,
    _euler_to_quat,
    _gate_normal,
    _quat_to_euler,
)
from sim.tracks import Track
from sim.types import GateState


def _make_env(**kwargs) -> GateRaceEnv:
    """Create a GateRaceEnv with 3 gates along the x-axis."""
    gates = [
        GateState(position=np.array([5.0, 0.0, 2.0])),
        GateState(position=np.array([10.0, 5.0, 2.0])),
        GateState(position=np.array([5.0, 10.0, 2.0])),
    ]
    defaults = dict(
        track=Track(gates),
        n_envs=1,
        max_steps=10000,
        ceiling=50.0,
    )
    defaults.update(kwargs)
    return GateRaceEnv(**defaults)


# ---------------------------------------------------------------------------
# Euler-to-quaternion helper
# ---------------------------------------------------------------------------

class TestEulerToQuat:
    """_euler_to_quat should be the inverse of _quat_to_euler."""

    def test_identity(self) -> None:
        """Zero Euler angles produce identity quaternion."""
        q = _euler_to_quat(0.0, 0.0, 0.0)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_roundtrip(self) -> None:
        """euler -> quat -> euler roundtrip is identity."""
        roll, pitch, yaw = 0.3, -0.2, 1.1
        q = _euler_to_quat(roll, pitch, yaw)
        r2, p2, y2 = _quat_to_euler(q)
        np.testing.assert_allclose([r2, p2, y2], [roll, pitch, yaw], atol=1e-10)

    def test_unit_quaternion(self) -> None:
        """Output quaternion has unit norm."""
        q = _euler_to_quat(0.5, -0.3, 2.0)
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Random gate start (7c)
# ---------------------------------------------------------------------------

class TestRandomGateStart:
    """With random_gate_start=True, drones start near a randomly chosen gate."""

    def test_position_near_gate(self) -> None:
        """Start position is within ~2m of some gate (behind it)."""
        env = _make_env(n_envs=10, random_gate_start=True)
        env.reset(seed=42)

        for i in range(env.n_envs):
            pos = env._states[i, POS]
            gate_idx = int(env._gate_indices[i])
            gate = env.track.gates[gate_idx]

            dist = np.linalg.norm(pos - gate.position)
            assert dist < 3.0, f"Env {i} too far from gate {gate_idx}: {dist:.2f}m"

    def test_position_behind_gate(self) -> None:
        """Start position is on the negative-normal side (approaching side)."""
        env = _make_env(n_envs=10, random_gate_start=True)
        env.reset(seed=42)

        for i in range(env.n_envs):
            gate_idx = int(env._gate_indices[i])
            gate = env.track.gates[gate_idx]
            normal = _gate_normal(gate)
            rel_pos = env._states[i, POS] - gate.position
            along_normal = np.dot(rel_pos, normal)
            assert along_normal < 0, (
                f"Env {i} should be behind gate (along_normal={along_normal:.2f})"
            )

    def test_different_gates_selected(self) -> None:
        """With enough envs, not all start at the same gate."""
        env = _make_env(n_envs=20, random_gate_start=True)
        env.reset(seed=42)
        unique_gates = set(int(env._gate_indices[i]) for i in range(env.n_envs))
        assert len(unique_gates) > 1, "All envs started at the same gate"

    def test_velocity_perturbation(self) -> None:
        """Random start applies non-zero velocity perturbation."""
        env = _make_env(n_envs=10, random_gate_start=True)
        env.reset(seed=42)

        any_nonzero = False
        for i in range(env.n_envs):
            vel = env._states[i, VEL]
            if np.linalg.norm(vel) > 0.01:
                any_nonzero = True
                break
        assert any_nonzero, "No velocity perturbation applied"

    def test_attitude_perturbation(self) -> None:
        """Random start applies non-identity quaternion."""
        env = _make_env(n_envs=10, random_gate_start=True)
        env.reset(seed=42)

        any_perturbed = False
        for i in range(env.n_envs):
            q = env._states[i, QUAT]
            # Identity is [1,0,0,0]; perturbed should differ
            if np.linalg.norm(q - np.array([1, 0, 0, 0])) > 0.01:
                any_perturbed = True
                break
        assert any_perturbed, "No attitude perturbation applied"

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same start positions."""
        env1 = _make_env(n_envs=5, random_gate_start=True)
        env2 = _make_env(n_envs=5, random_gate_start=True)

        env1.reset(seed=123)
        env2.reset(seed=123)

        np.testing.assert_array_equal(env1._states, env2._states)
        np.testing.assert_array_equal(env1._gate_indices, env2._gate_indices)

    def test_obs_finite_after_random_reset(self) -> None:
        """Observations are finite after random gate start."""
        env = _make_env(n_envs=5, random_gate_start=True)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))

    def test_disabled_starts_at_origin(self) -> None:
        """With random_gate_start=False, all envs start at gate 0."""
        env = _make_env(n_envs=5, random_gate_start=False)
        env.reset(seed=42)

        for i in range(env.n_envs):
            assert env._gate_indices[i] == 0
            # Default start position: [0, 0, 1]
            np.testing.assert_allclose(env._states[i, POS], [0, 0, 1], atol=1e-6)


# ---------------------------------------------------------------------------
# Gate collision (7d)
# ---------------------------------------------------------------------------

class TestGateCollision:
    """Crossing gate plane outside opening should crash when enabled."""

    def _teleport_and_step(
        self, env: GateRaceEnv, before_pos: np.ndarray, after_pos: np.ndarray
    ) -> tuple:
        """Teleport drone to before_pos, step so it ends at after_pos."""
        env._states[0, POS] = before_pos
        env._states[0, QUAT] = [1, 0, 0, 0]
        env._states[0, VEL] = [0, 0, 0]
        env._states[0, OMEGA] = [0, 0, 0]

        gate = env.track.gates[int(env._gate_indices[0])]
        normal = _gate_normal(gate)
        rel_before = before_pos - gate.position
        env._prev_along_normal[0] = float(np.dot(rel_before, normal))
        env._prev_gate_dists[0] = float(np.linalg.norm(rel_before))

        # Override dynamics step to place drone at after_pos
        orig_step = env.dynamics.step

        def fake_step(states, actions):
            result = orig_step(states, actions)
            result[0, POS] = after_pos
            result[0, QUAT] = [1, 0, 0, 0]
            return result

        env.dynamics.step = fake_step
        action = np.zeros(4, dtype=np.float32)
        result = env.step(action)
        env.dynamics.step = orig_step
        return result

    def test_crossing_outside_radius_crashes(self) -> None:
        """Crossing gate plane far from center terminates episode."""
        env = _make_env(gate_collision=True, gate_passage_radius=1.0)
        env.reset(seed=42)

        # Gate 0 is at [5, 0, 2] facing +x. Cross from x=4.5 to x=5.5
        # but at y=3 (outside radius=1.0)
        _, _, terminated, _, _ = self._teleport_and_step(
            env,
            before_pos=np.array([4.5, 3.0, 2.0]),
            after_pos=np.array([5.5, 3.0, 2.0]),
        )
        assert terminated.item(), "Should crash when crossing outside gate opening"

    def test_crossing_inside_radius_passes(self) -> None:
        """Crossing gate plane near center is a gate passage, not collision."""
        env = _make_env(gate_collision=True, gate_passage_radius=1.0)
        env.reset(seed=42)

        # Cross at y=0.5 (inside radius=1.0)
        _, _, terminated, _, _ = self._teleport_and_step(
            env,
            before_pos=np.array([4.5, 0.5, 2.0]),
            after_pos=np.array([5.5, 0.5, 2.0]),
        )
        assert not terminated.item(), "Should pass through gate, not crash"

    def test_collision_disabled(self) -> None:
        """With gate_collision=False, crossing outside radius is ignored."""
        env = _make_env(gate_collision=False, gate_passage_radius=1.0)
        env.reset(seed=42)

        _, _, terminated, _, _ = self._teleport_and_step(
            env,
            before_pos=np.array([4.5, 3.0, 2.0]),
            after_pos=np.array([5.5, 3.0, 2.0]),
        )
        assert not terminated.item(), "Should not crash with gate_collision=False"

    def test_collision_gives_crash_penalty(self) -> None:
        """Gate collision applies the crash penalty reward."""
        env = _make_env(
            gate_collision=True,
            gate_passage_radius=1.0,
            reward_weights={"crash_penalty": 15.0},
        )
        env.reset(seed=42)

        _, reward, terminated, _, _ = self._teleport_and_step(
            env,
            before_pos=np.array([4.5, 3.0, 2.0]),
            after_pos=np.array([5.5, 3.0, 2.0]),
        )
        assert terminated.item()
        assert reward.item() == pytest.approx(-15.0, abs=1.0)
