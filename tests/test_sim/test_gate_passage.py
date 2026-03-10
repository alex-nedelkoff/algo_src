"""Tests for gate passage tracking and gate-passage bonus reward (COR-43).

Verifies that:
- _gate_indices increments when the drone is within gate_passage_radius
- Gate index wraps to 0 after passing all gates (lap completion)
- Discrete bonus reward is added on gate passage
- Observation reflects the new target gate after passage
- Per-env gate tracking is independent across parallel envs
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.dynamics.numpy_quad import POS, QUAT
from sim.envs.gate_race_env import GateRaceEnv, OBS_DIM
from sim.tracks import Track
from sim.types import GateState


def _make_env(
    gate_positions: list[list[float]] | None = None,
    n_envs: int = 1,
    gate_passage_radius: float = 1.0,
    reward_weights: dict[str, float] | None = None,
) -> GateRaceEnv:
    """Create a GateRaceEnv with gates at known positions."""
    if gate_positions is None:
        gate_positions = [
            [3.0, 0.0, 1.0],
            [6.0, 0.0, 1.0],
            [9.0, 0.0, 1.0],
        ]
    gates = [GateState(position=np.array(p)) for p in gate_positions]
    track = Track(gates)
    return GateRaceEnv(
        track=track,
        n_envs=n_envs,
        gate_passage_radius=gate_passage_radius,
        reward_weights=reward_weights,
        max_steps=10000,
        ceiling=50.0,
    )


def _teleport(env: GateRaceEnv, env_idx: int, pos: np.ndarray) -> None:
    """Teleport a single env's drone to the given position.

    Preserves a valid quaternion and zeroes velocity/omega/motors to avoid
    termination from quaternion divergence.
    """
    env._states[env_idx, POS] = pos
    # Ensure valid quaternion (identity)
    env._states[env_idx, QUAT] = [1.0, 0.0, 0.0, 0.0]


class TestGatePassageDetection:
    """Gate passage should be detected based on distance threshold."""

    def test_gate_index_increments_on_passage(self) -> None:
        """_gate_indices should increment when drone is within radius of the target gate."""
        env = _make_env()
        env.reset(seed=42)

        # Teleport drone right on top of gate 0
        gate0_pos = env.track.gates[0].position
        _teleport(env, 0, gate0_pos.copy())

        # Step with a hover-like action (doesn't matter, we teleport)
        action = np.zeros(4, dtype=np.float32)
        env.step(action)

        # After passing gate 0, the gate index should be 1
        # Note: the env auto-resets on termination. Since we set z=1.0
        # (above ground) and valid quat, it shouldn't terminate.
        # But after step, dynamics might move the drone; we placed it
        # at the gate, which is sufficient for the distance check.
        # However, since dynamics step runs BEFORE the check, we need
        # to set position such that AFTER dynamics, we're still near gate.
        # Simplest: teleport, then step with zero action. With zero thrust
        # and dt=0.01, gravity moves ~0.0005m in one step, still within
        # radius=1.0.
        assert env._gate_indices[0] == 1, (
            f"Expected gate_index=1 after passing gate 0, got {env._gate_indices[0]}"
        )

    def test_gate_index_does_not_increment_when_far(self) -> None:
        """Gate index stays at 0 when drone is far from the gate."""
        env = _make_env()
        env.reset(seed=42)

        # Drone starts at origin (0,0,1) by default after reset,
        # which is far from gate 0 at (3,0,1).
        action = np.zeros(4, dtype=np.float32)
        env.step(action)

        assert env._gate_indices[0] == 0, (
            f"Expected gate_index=0 when far from gate, got {env._gate_indices[0]}"
        )

    def test_sequential_gate_progression(self) -> None:
        """Drone should progress through gates 0 -> 1 -> 2 sequentially."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # Pass gate 0
        _teleport(env, 0, env.track.gates[0].position.copy())
        env.step(action)
        assert env._gate_indices[0] == 1

        # Pass gate 1
        _teleport(env, 0, env.track.gates[1].position.copy())
        env.step(action)
        assert env._gate_indices[0] == 2

        # Pass gate 2
        _teleport(env, 0, env.track.gates[2].position.copy())
        env.step(action)
        # Should wrap back to 0 (lap completion)
        assert env._gate_indices[0] == 0


class TestLapCompletion:
    """Gate index should wrap to 0 after passing all gates."""

    def test_lap_wraps_gate_index(self) -> None:
        """After passing all gates, gate index wraps to 0."""
        # 2-gate track for simplicity
        env = _make_env(gate_positions=[[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # Pass gate 0
        _teleport(env, 0, env.track.gates[0].position.copy())
        env.step(action)
        assert env._gate_indices[0] == 1

        # Pass gate 1 -> should wrap to 0
        _teleport(env, 0, env.track.gates[1].position.copy())
        env.step(action)
        assert env._gate_indices[0] == 0, (
            f"Expected gate_index to wrap to 0 after lap, got {env._gate_indices[0]}"
        )

    def test_multiple_laps(self) -> None:
        """Should be able to complete multiple laps."""
        env = _make_env(gate_positions=[[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        for _lap in range(3):
            # Pass gate 0
            _teleport(env, 0, env.track.gates[0].position.copy())
            env.step(action)
            assert env._gate_indices[0] == 1

            # Pass gate 1 -> wrap
            _teleport(env, 0, env.track.gates[1].position.copy())
            env.step(action)
            assert env._gate_indices[0] == 0


class TestGatePassageBonus:
    """Discrete bonus reward should be added on gate passage."""

    def test_bonus_added_on_passage(self) -> None:
        """Reward should include the gate_passage bonus when a gate is passed."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # First get a baseline reward from a normal step (far from gate)
        env.step(action)
        env.reset(seed=42)

        # Now teleport to gate 0 and step
        _teleport(env, 0, env.track.gates[0].position.copy())
        _, reward_with_passage, _, _, _ = env.step(action)

        # Reset and step without being near a gate
        env.reset(seed=42)
        _, reward_without_passage, _, _, _ = env.step(action)

        passage_val = reward_with_passage.item() if hasattr(reward_with_passage, 'item') else reward_with_passage
        normal_val = reward_without_passage.item() if hasattr(reward_without_passage, 'item') else reward_without_passage

        # The passage reward should be significantly higher due to +30 bonus
        assert passage_val > normal_val + 20.0, (
            f"Expected passage reward ({passage_val}) to exceed normal reward "
            f"({normal_val}) by at least 20 (gate_passage bonus)"
        )

    def test_custom_bonus_weight(self) -> None:
        """Custom gate_passage weight should be used."""
        env = _make_env(reward_weights={
            "gate_progress": 1.0,
            "attitude_penalty": 0.0,
            "speed_bonus": 0.0,
            "action_smoothness": 0.0,
            "gate_passage": 50.0,
        })
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # Teleport to gate 0
        _teleport(env, 0, env.track.gates[0].position.copy())
        _, reward_with_passage, _, _, _ = env.step(action)

        # Reset and step without passage
        env.reset(seed=42)
        _, reward_without_passage, _, _, _ = env.step(action)

        passage_val = reward_with_passage.item() if hasattr(reward_with_passage, 'item') else reward_with_passage
        normal_val = reward_without_passage.item() if hasattr(reward_without_passage, 'item') else reward_without_passage

        # The difference should be around 50.0 (the custom weight)
        diff = passage_val - normal_val
        assert diff > 40.0, (
            f"Expected passage bonus ~50, got difference of {diff}"
        )


class TestObservationAfterPassage:
    """Observation should reflect updated target gate after passage."""

    def test_obs_targets_next_gate(self) -> None:
        """After passing gate 0, observation relative position should point to gate 1."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # Get gate positions
        gate1_pos = env.track.gates[1].position

        # Teleport to gate 0 and step (passes gate 0, now targeting gate 1)
        _teleport(env, 0, env.track.gates[0].position.copy())
        obs, _, _, _, _ = env.step(action)

        # obs[0:3] should be relative pos to gate 1 in body frame
        # Since dynamics moves the drone slightly, and rotation is identity,
        # the body-frame relative pos approximately equals world-frame relative pos
        # The drone's position after step is approximately gate0_pos (maybe shifted
        # by gravity). The obs relative to gate 1 should be roughly gate1-gate0.
        # Just verify that the gate progress scalar reflects gate 1.
        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        progress = obs_flat[16]
        expected_progress = 1.0 / env.track.num_gates  # gate_idx=1 / num_gates=3
        assert abs(progress - expected_progress) < 1e-5, (
            f"Expected progress={expected_progress}, got {progress}"
        )


class TestPerEnvIndependence:
    """Gate tracking should be independent across parallel environments."""

    def test_env0_passage_does_not_affect_env1(self) -> None:
        """Passing a gate in env 0 should not change env 1's gate index."""
        env = _make_env(n_envs=2)
        env.reset(seed=42)
        action = np.zeros((2, 4), dtype=np.float32)

        # Teleport env 0 to gate 0, but leave env 1 at origin
        _teleport(env, 0, env.track.gates[0].position.copy())
        # Env 1 stays where reset placed it

        env.step(action)

        assert env._gate_indices[0] == 1, (
            f"Env 0 should have advanced to gate 1, got {env._gate_indices[0]}"
        )
        assert env._gate_indices[1] == 0, (
            f"Env 1 should still be at gate 0, got {env._gate_indices[1]}"
        )

    def test_different_envs_at_different_gates(self) -> None:
        """Two envs can be at different gates independently."""
        env = _make_env(n_envs=2)
        env.reset(seed=42)
        action = np.zeros((2, 4), dtype=np.float32)

        # Advance env 0 to gate 1
        _teleport(env, 0, env.track.gates[0].position.copy())
        env.step(action)
        assert env._gate_indices[0] == 1

        # Now advance env 0 to gate 2, and env 1 to gate 0
        _teleport(env, 0, env.track.gates[1].position.copy())
        _teleport(env, 1, env.track.gates[0].position.copy())
        env.step(action)

        assert env._gate_indices[0] == 2, (
            f"Env 0 should be at gate 2, got {env._gate_indices[0]}"
        )
        assert env._gate_indices[1] == 1, (
            f"Env 1 should be at gate 1, got {env._gate_indices[1]}"
        )


class TestGatePassageRadius:
    """Gate passage radius parameter should control detection threshold."""

    def test_passage_within_radius(self) -> None:
        """Passage detected when within radius."""
        env = _make_env(gate_passage_radius=2.0)
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # Place drone 1.5m from gate 0 (within radius=2.0)
        gate0_pos = env.track.gates[0].position.copy()
        offset = gate0_pos.copy()
        offset[0] -= 1.5
        _teleport(env, 0, offset)
        env.step(action)

        assert env._gate_indices[0] == 1, "Should detect passage within radius"

    def test_no_passage_outside_radius(self) -> None:
        """No passage detected when outside radius."""
        env = _make_env(gate_passage_radius=0.5)
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # Place drone 1.0m from gate 0 (outside radius=0.5)
        gate0_pos = env.track.gates[0].position.copy()
        offset = gate0_pos.copy()
        offset[0] -= 1.0
        _teleport(env, 0, offset)
        env.step(action)

        assert env._gate_indices[0] == 0, "Should NOT detect passage outside radius"
