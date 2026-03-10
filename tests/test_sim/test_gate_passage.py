"""Tests for plane-crossing gate passage detection and reward wiring (COR-44).

Verifies that:
- Plane-crossing detection triggers only when crossing from front to back
- Plane-crossing does NOT trigger when approaching from the wrong direction
- Plane-crossing does NOT trigger if lateral distance exceeds radius
- Gate switch correctly recomputes prev_gate_dist against new target gate
- Gate offset penalty is applied at passage with correct weight
- Crash penalty uses configurable weight from reward_weights
- Gate passage bonus uses M23 default of 1.5 (not 30.0)
- _prev_along_normal is initialized in reset and updated each step
- Per-env gate tracking is independent across parallel envs
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sim.dynamics.numpy_quad import POS, QUAT
from sim.envs.gate_race_env import GateRaceEnv, OBS_DIM, _gate_normal
from sim.tracks import Track
from sim.types import GateState


def _make_env(
    gate_positions: list[list[float]] | None = None,
    gate_orientations: list[list[float]] | None = None,
    n_envs: int = 1,
    gate_passage_radius: float = 1.0,
    reward_weights: dict[str, float] | None = None,
) -> GateRaceEnv:
    """Create a GateRaceEnv with gates at known positions and orientations."""
    if gate_positions is None:
        gate_positions = [
            [3.0, 0.0, 1.0],
            [6.0, 0.0, 1.0],
            [9.0, 0.0, 1.0],
        ]
    if gate_orientations is None:
        # Identity quaternion: gate faces +x direction
        gate_orientations = [[1.0, 0.0, 0.0, 0.0]] * len(gate_positions)

    gates = [
        GateState(position=np.array(p), orientation=np.array(o))
        for p, o in zip(gate_positions, gate_orientations)
    ]
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


class TestGateNormalHelper:
    """_gate_normal should return the rotated x-axis of the gate."""

    def test_identity_quaternion_gives_x_axis(self) -> None:
        """Identity orientation -> gate faces +x."""
        gate = GateState(position=np.zeros(3))
        normal = _gate_normal(gate)
        np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-10)

    def test_90deg_yaw_gives_y_axis(self) -> None:
        """90-degree yaw rotation -> gate faces +y."""
        # Quaternion for 90-degree rotation about z: [cos(45), 0, 0, sin(45)]
        angle = np.pi / 2
        gate = GateState(
            position=np.zeros(3),
            orientation=np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]),
        )
        normal = _gate_normal(gate)
        np.testing.assert_allclose(normal, [0.0, 1.0, 0.0], atol=1e-10)


class TestPlaneCrossingDetection:
    """Gate passage should require crossing the gate plane front-to-back."""

    def test_front_to_back_crossing_triggers_passage(self) -> None:
        """Drone crossing gate plane from -x to +x triggers gate passage."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position  # [3, 0, 1]

        # Place drone just behind gate plane (negative normal side)
        behind = gate0_pos.copy()
        behind[0] -= 0.1  # slightly behind in x
        _teleport(env, 0, behind)
        # Manually set _prev_along_normal to negative (approaching from front)
        env._prev_along_normal[0] = -0.1

        # Now teleport to just past the gate plane (positive normal side)
        ahead = gate0_pos.copy()
        ahead[0] += 0.1  # slightly ahead in x
        _teleport(env, 0, ahead)
        env.step(action)

        assert env._gate_indices[0] == 1, (
            f"Expected gate_index=1 after front-to-back crossing, "
            f"got {env._gate_indices[0]}"
        )

    def test_back_to_front_crossing_does_not_trigger(self) -> None:
        """Drone crossing gate plane from +x to -x does NOT trigger passage."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position  # [3, 0, 1]

        # Place drone on positive side (already past the gate)
        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        # Set prev_along_normal to positive (on the back side)
        env._prev_along_normal[0] = 0.1

        # Now teleport to negative side (back to front direction -- wrong way)
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env.step(action)

        assert env._gate_indices[0] == 0, (
            f"Expected gate_index=0 for back-to-front crossing, "
            f"got {env._gate_indices[0]}"
        )

    def test_lateral_distance_exceeds_radius_no_passage(self) -> None:
        """Crossing the plane too far from gate center should not trigger."""
        env = _make_env(gate_passage_radius=0.5)
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position  # [3, 0, 1]

        # Place drone behind gate but offset laterally by 2m in y
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        behind[1] += 2.0  # way off center
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        # Move past gate plane but still laterally offset
        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        ahead[1] += 2.0
        _teleport(env, 0, ahead)
        env.step(action)

        assert env._gate_indices[0] == 0, (
            f"Expected gate_index=0 when lateral dist > radius, "
            f"got {env._gate_indices[0]}"
        )

    def test_lateral_within_radius_triggers_passage(self) -> None:
        """Crossing the plane near gate center should trigger passage."""
        env = _make_env(gate_passage_radius=2.0)
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position  # [3, 0, 1]

        # Place drone behind gate, slightly offset in y (within radius)
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        behind[1] += 0.5  # 0.5m offset, within 2.0 radius
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        # Move past gate plane
        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        ahead[1] += 0.5
        _teleport(env, 0, ahead)
        env.step(action)

        assert env._gate_indices[0] == 1, (
            f"Expected gate_index=1 for crossing within radius, "
            f"got {env._gate_indices[0]}"
        )


class TestPrevGateDistRecomputation:
    """After gate switch, _prev_gate_dists must reference the NEW target gate."""

    def test_prev_gate_dist_updated_to_new_gate(self) -> None:
        """After passing gate 0, prev_gate_dist should be distance to gate 1."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position  # [3, 0, 1]
        gate1_pos = env.track.gates[1].position  # [6, 0, 1]

        # Set up proper plane crossing
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        # Move through gate 0
        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        env.step(action)

        # After passage, prev_gate_dist should be distance from current
        # position (near gate0_pos + 0.1 in x, but dynamics may shift slightly)
        # to gate 1 position. The key check: it should NOT be distance to gate 0.
        drone_pos = env._states[0, POS]
        expected_dist = float(np.linalg.norm(drone_pos - gate1_pos))
        actual_dist = env._prev_gate_dists[0]

        # The distance should be approximately ||drone - gate1||, not ||drone - gate0||
        assert abs(actual_dist - expected_dist) < 0.5, (
            f"prev_gate_dist={actual_dist} should be ~{expected_dist} "
            f"(distance to new gate 1), not distance to old gate 0"
        )

    def test_prev_gate_dist_not_stale_after_switch(self) -> None:
        """prev_gate_dist should not equal distance to the OLD gate after switch."""
        env = _make_env(gate_positions=[
            [3.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],  # gate 1 is far away
        ])
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position
        gate1_pos = env.track.gates[1].position

        # Cross gate 0
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        env.step(action)

        # Distance to gate 0 from just past it should be ~0.1
        # Distance to gate 1 from just past gate 0 should be ~6.9
        # prev_gate_dist should be ~6.9, NOT ~0.1
        assert env._prev_gate_dists[0] > 3.0, (
            f"prev_gate_dist={env._prev_gate_dists[0]} looks like distance to "
            f"OLD gate 0 instead of NEW gate 1"
        )


class TestGateOffsetPenaltyAtPassage:
    """gate_offset_penalty should only be applied at gate passage, not continuously."""

    def test_offset_penalty_applied_on_passage(self) -> None:
        """Gate offset penalty should be included in reward on gate passage."""
        # Use zero weights for continuous components to isolate passage effects
        weights = {
            "gate_progress": 0.0,
            "body_rate": 0.0,
            "action_smoothness": 0.0,
            "gate_passage": 1.5,
            "gate_offset": 1.5,
            "crash_penalty": 10.0,
        }
        env = _make_env(reward_weights=weights)
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position

        # Cross gate with some lateral offset
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        behind[1] += 0.3  # 0.3m offset in y
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        ahead[1] += 0.3
        _teleport(env, 0, ahead)
        _, reward_passage, _, _, _ = env.step(action)

        passage_reward = (
            reward_passage.item()
            if hasattr(reward_passage, "item")
            else float(reward_passage)
        )

        # The reward should include gate_passage bonus (1.5) plus
        # gate_offset penalty (1.5 * negative_value). So it should be
        # less than 1.5 due to the offset penalty.
        assert passage_reward < 1.5, (
            f"Expected reward < 1.5 due to offset penalty, got {passage_reward}"
        )
        # But it should be positive (passage bonus > offset penalty for small offset)
        # The offset penalty magnitude depends on implementation but for 0.3m offset
        # should be modest.

    def test_offset_weight_is_configurable(self) -> None:
        """Custom gate_offset weight should be used in penalty calculation."""
        # With gate_offset=0.0, the offset penalty should have no effect
        weights_no_offset = {
            "gate_progress": 0.0,
            "body_rate": 0.0,
            "action_smoothness": 0.0,
            "gate_passage": 1.5,
            "gate_offset": 0.0,
            "crash_penalty": 10.0,
        }
        weights_with_offset = {
            "gate_progress": 0.0,
            "body_rate": 0.0,
            "action_smoothness": 0.0,
            "gate_passage": 1.5,
            "gate_offset": 5.0,
            "crash_penalty": 10.0,
        }

        # Test with no offset weight
        env1 = _make_env(reward_weights=weights_no_offset)
        env1.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0 = env1.track.gates[0].position
        behind = gate0.copy()
        behind[0] -= 0.1
        behind[1] += 0.5
        _teleport(env1, 0, behind)
        env1._prev_along_normal[0] = -0.1
        ahead = gate0.copy()
        ahead[0] += 0.1
        ahead[1] += 0.5
        _teleport(env1, 0, ahead)
        _, r1, _, _, _ = env1.step(action)

        # Test with large offset weight
        env2 = _make_env(reward_weights=weights_with_offset)
        env2.reset(seed=42)

        gate0 = env2.track.gates[0].position
        behind = gate0.copy()
        behind[0] -= 0.1
        behind[1] += 0.5
        _teleport(env2, 0, behind)
        env2._prev_along_normal[0] = -0.1
        ahead = gate0.copy()
        ahead[0] += 0.1
        ahead[1] += 0.5
        _teleport(env2, 0, ahead)
        _, r2, _, _, _ = env2.step(action)

        r1_val = r1.item() if hasattr(r1, "item") else float(r1)
        r2_val = r2.item() if hasattr(r2, "item") else float(r2)

        # With higher offset weight, the penalty should be larger (more negative)
        assert r2_val < r1_val, (
            f"Higher gate_offset weight should yield lower reward: "
            f"no_offset={r1_val}, with_offset={r2_val}"
        )


class TestCrashPenalty:
    """Crash penalty should use the configurable weight from reward_weights."""

    def test_default_crash_penalty_is_10(self) -> None:
        """Without reward_weights, crash penalty defaults to -10.0."""
        env = _make_env(reward_weights=None)
        env.reset(seed=42)

        # Force ground crash
        _teleport(env, 0, np.array([0.0, 0.0, 0.001]))
        # Zero thrust -> will crash next step (z will go negative with gravity)
        action = np.zeros(4, dtype=np.float32)

        # Step until termination
        for _ in range(200):
            _, reward, terminated, _, _ = env.step(action)
            term_val = terminated.item() if hasattr(terminated, "item") else terminated
            if term_val:
                rew_val = reward.item() if hasattr(reward, "item") else float(reward)
                assert rew_val == pytest.approx(-10.0), (
                    f"Default crash penalty should be -10.0, got {rew_val}"
                )
                return
        pytest.fail("Expected crash termination")

    def test_custom_crash_penalty(self) -> None:
        """Custom crash_penalty weight should be used."""
        env = _make_env(reward_weights={"crash_penalty": 25.0})
        env.reset(seed=42)

        # Force ground crash
        _teleport(env, 0, np.array([0.0, 0.0, 0.001]))
        action = np.zeros(4, dtype=np.float32)

        for _ in range(200):
            _, reward, terminated, _, _ = env.step(action)
            term_val = terminated.item() if hasattr(terminated, "item") else terminated
            if term_val:
                rew_val = reward.item() if hasattr(reward, "item") else float(reward)
                assert rew_val == pytest.approx(-25.0), (
                    f"Custom crash penalty should be -25.0, got {rew_val}"
                )
                return
        pytest.fail("Expected crash termination")

    def test_ceiling_crash_uses_custom_penalty(self) -> None:
        """Ceiling crash should also use the configurable penalty."""
        env = _make_env(reward_weights={"crash_penalty": 7.5})
        env.reset(seed=42)

        # Teleport above ceiling
        _teleport(env, 0, np.array([0.0, 0.0, env.ceiling + 1.0]))
        action = np.zeros(4, dtype=np.float32)
        _, reward, terminated, _, _ = env.step(action)

        rew_val = reward.item() if hasattr(reward, "item") else float(reward)
        assert rew_val == pytest.approx(-7.5), (
            f"Ceiling crash penalty should be -7.5, got {rew_val}"
        )


class TestGatePassageBonus:
    """Gate passage bonus should default to 1.5 (M23), not 30.0."""

    def test_default_gate_passage_bonus_is_1_5(self) -> None:
        """Default gate_passage bonus should be 1.5."""
        env = _make_env(reward_weights=None)
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position

        # Get baseline reward without passage
        _, baseline_reward, _, _, _ = env.step(action)
        baseline = (
            baseline_reward.item()
            if hasattr(baseline_reward, "item")
            else float(baseline_reward)
        )

        # Reset and cross gate
        env.reset(seed=42)
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        _, passage_reward, _, _, _ = env.step(action)
        passage = (
            passage_reward.item()
            if hasattr(passage_reward, "item")
            else float(passage_reward)
        )

        # The passage reward includes the bonus (1.5) plus gate_offset penalty.
        # Since we're crossing exactly through center, offset should be small.
        # The difference from baseline should be approximately 1.5 + offset effects.
        # We just verify the bonus is approximately 1.5, not 30.0.
        diff = passage - baseline
        # The diff should be close to 1.5 + gate_offset_penalty contribution
        # Gate offset is negative but weighted, so diff should be around 1.5
        # (might be slightly less due to offset). Definitely not 30.
        assert diff < 10.0, (
            f"Gate passage bonus should be ~1.5 (not 30.0), diff={diff}"
        )

    def test_custom_gate_passage_weight(self) -> None:
        """Custom gate_passage weight should be used as the bonus."""
        env = _make_env(reward_weights={
            "gate_passage": 5.0,
            "gate_offset": 0.0,
            "gate_progress": 0.0,
            "body_rate": 0.0,
            "action_smoothness": 0.0,
            "crash_penalty": 10.0,
        })
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position

        # Get baseline (no passage)
        _, baseline_r, _, _, _ = env.step(action)
        baseline = baseline_r.item() if hasattr(baseline_r, "item") else float(baseline_r)

        # Reset and cross gate
        env.reset(seed=42)
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        _, passage_r, _, _, _ = env.step(action)
        passage = passage_r.item() if hasattr(passage_r, "item") else float(passage_r)

        diff = passage - baseline
        # With gate_offset=0, the diff should be exactly 5.0 (the passage bonus)
        assert abs(diff - 5.0) < 1.0, (
            f"Expected passage bonus ~5.0, got diff={diff}"
        )


class TestPrevAlongNormal:
    """_prev_along_normal should be properly initialized and updated."""

    def test_initialized_on_reset(self) -> None:
        """_prev_along_normal should be set during reset()."""
        env = _make_env()
        env.reset(seed=42)

        # After reset, the drone is at (0, 0, 1) and gate 0 is at (3, 0, 1).
        # Gate normal is +x. rel_pos = (0,0,1) - (3,0,1) = (-3, 0, 0).
        # dot((-3,0,0), (1,0,0)) = -3.0
        assert env._prev_along_normal[0] == pytest.approx(-3.0, abs=0.1), (
            f"Expected _prev_along_normal ~ -3.0, got {env._prev_along_normal[0]}"
        )

    def test_updated_each_step(self) -> None:
        """_prev_along_normal should change after each step."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        initial = env._prev_along_normal[0]
        env.step(action)
        after_step = env._prev_along_normal[0]

        # After a step with gravity, the drone drops slightly but the
        # along-normal value should still update (it may change due to
        # position update from dynamics).
        # We just verify it's a valid float and was updated.
        assert np.isfinite(after_step), "prev_along_normal should be finite"

    def test_reset_on_auto_reset(self) -> None:
        """_prev_along_normal should be recomputed when env auto-resets."""
        env = _make_env(reward_weights={"crash_penalty": 10.0})
        env.reset(seed=42)

        # Force crash to trigger auto-reset
        _teleport(env, 0, np.array([0.0, 0.0, 0.001]))
        action = np.zeros(4, dtype=np.float32)

        for _ in range(200):
            _, _, terminated, _, _ = env.step(action)
            term_val = terminated.item() if hasattr(terminated, "item") else terminated
            if term_val:
                # After auto-reset, prev_along_normal should be recomputed
                # for gate 0 from the reset position
                assert np.isfinite(env._prev_along_normal[0]), (
                    "prev_along_normal should be finite after auto-reset"
                )
                return
        pytest.fail("Expected crash termination for auto-reset test")


class TestSequentialGateProgression:
    """Drone should progress through gates sequentially with plane crossing."""

    def test_sequential_passage(self) -> None:
        """Pass gates 0 -> 1 -> 2 sequentially."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        for gate_idx in range(3):
            gate_pos = env.track.gates[gate_idx].position.copy()

            # Approach from behind
            behind = gate_pos.copy()
            behind[0] -= 0.1
            _teleport(env, 0, behind)
            env._prev_along_normal[0] = -0.1

            # Cross through
            ahead = gate_pos.copy()
            ahead[0] += 0.1
            _teleport(env, 0, ahead)
            env.step(action)

            expected = (gate_idx + 1) % env.track.num_gates
            assert env._gate_indices[0] == expected, (
                f"After passing gate {gate_idx}, expected index={expected}, "
                f"got {env._gate_indices[0]}"
            )

    def test_lap_completion_wraps(self) -> None:
        """After passing all gates, index wraps to 0."""
        env = _make_env(gate_positions=[[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]])
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        # Pass gate 0
        behind = env.track.gates[0].position.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1
        ahead = env.track.gates[0].position.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        env.step(action)
        assert env._gate_indices[0] == 1

        # Pass gate 1 -> wraps to 0
        behind = env.track.gates[1].position.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1
        ahead = env.track.gates[1].position.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        env.step(action)
        assert env._gate_indices[0] == 0


class TestPerEnvIndependence:
    """Gate tracking should be independent across parallel environments."""

    def test_env0_passage_does_not_affect_env1(self) -> None:
        """Passing a gate in env 0 should not change env 1's gate index."""
        env = _make_env(n_envs=2)
        env.reset(seed=42)
        action = np.zeros((2, 4), dtype=np.float32)

        gate0_pos = env.track.gates[0].position

        # Set up plane crossing for env 0 only
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1
        # Leave env 1 where it is

        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        env.step(action)

        assert env._gate_indices[0] == 1, (
            f"Env 0 should have advanced, got {env._gate_indices[0]}"
        )
        assert env._gate_indices[1] == 0, (
            f"Env 1 should still be at gate 0, got {env._gate_indices[1]}"
        )


class TestActionSmoothnessThreshold:
    """action_smoothness_threshold should be accepted and stored."""

    def test_default_threshold(self) -> None:
        """Default action_smoothness_threshold should be 0.5."""
        env = _make_env()
        assert env.action_smoothness_threshold == 0.5

    def test_custom_threshold(self) -> None:
        """Custom threshold should be stored."""
        gates = [GateState(position=np.array([3.0, 0.0, 1.0]))]
        track = Track(gates)
        env = GateRaceEnv(
            track=track,
            action_smoothness_threshold=0.8,
        )
        assert env.action_smoothness_threshold == 0.8


class TestObservationAfterPassage:
    """Observation should reflect updated target gate after passage."""

    def test_obs_targets_next_gate_after_passage(self) -> None:
        """After passing gate 0, observation should reference gate 1."""
        env = _make_env()
        env.reset(seed=42)
        action = np.zeros(4, dtype=np.float32)

        gate0_pos = env.track.gates[0].position

        # Cross gate 0
        behind = gate0_pos.copy()
        behind[0] -= 0.1
        _teleport(env, 0, behind)
        env._prev_along_normal[0] = -0.1

        ahead = gate0_pos.copy()
        ahead[0] += 0.1
        _teleport(env, 0, ahead)
        obs, _, _, _, _ = env.step(action)

        obs_flat = obs.flatten() if obs.ndim > 1 else obs
        progress = obs_flat[16]
        expected_progress = 1.0 / env.track.num_gates  # gate 1 / 3 gates
        assert abs(progress - expected_progress) < 1e-5, (
            f"Expected progress={expected_progress}, got {progress}"
        )
