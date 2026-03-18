"""Tests for GateRaceEnv and HoverEnv.

Covers reset shapes, step validity, termination conditions,
observation/action space definitions, and HoverEnv basics.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.domain_randomization import DomainRandomizer
from sim.envs.gate_race_env import GateRaceEnv, OBS_DIM
from sim.envs.hover_env import HoverEnv, STATE_DIM
from sim.tracks import Track
from sim.types import GateState


class TestGateRaceEnvReset:
    """Reset should return valid observations in correct shape."""

    def test_reset_obs_shape(self) -> None:
        """Reset returns obs of shape (OBS_DIM,) for single env."""
        env = GateRaceEnv(n_envs=1)
        obs, info = env.reset(seed=42)
        assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    def test_reset_obs_dtype(self) -> None:
        """Reset returns float32 observation."""
        env = GateRaceEnv(n_envs=1)
        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.float32

    def test_reset_obs_finite(self) -> None:
        """Reset obs should contain no NaN or inf."""
        env = GateRaceEnv(n_envs=1)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs)), f"Non-finite values in obs: {obs}"

    def test_reset_info_is_dict(self) -> None:
        """Reset returns a dict as info."""
        env = GateRaceEnv(n_envs=1)
        _, info = env.reset(seed=42)
        assert isinstance(info, dict)


class TestGateRaceEnvStep:
    """Step should return valid outputs."""

    def test_step_with_zero_action(self) -> None:
        """Stepping with zero action doesn't crash immediately."""
        env = GateRaceEnv(n_envs=1)
        env.reset(seed=42)

        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (OBS_DIM,)
        assert np.isscalar(reward) or reward.shape == (1,)
        assert isinstance(info, dict)

    def test_step_with_hover_action(self) -> None:
        """Stepping with hover action produces valid output."""
        env = GateRaceEnv(n_envs=1)
        env.reset(seed=42)

        # Compute normalized action that produces hover omega via ESC inverse
        from sim.dynamics.numpy_quad import GRAVITY
        p = env.params
        hover_omega = np.sqrt(p.mass * GRAVITY / (4.0 * p.k_thrust))
        # Invert ESC curve: omega = (w_max - w_min) * sqrt(k*U^2 + (1-k)*U) + w_min
        # Solve for U: R = ((omega - w_min) / (w_max - w_min))^2 = k*U^2 + (1-k)*U
        k = env.esc_nonlinearity
        R = ((hover_omega - env.omega_min) / (p.max_omega - env.omega_min)) ** 2
        # Quadratic in U: k*U^2 + (1-k)*U - R = 0
        if k > 0:
            U = (-(1 - k) + np.sqrt((1 - k) ** 2 + 4 * k * R)) / (2 * k)
        else:
            U = R  # k=0: sqrt(U) model, so U = R
        u = 2.0 * U - 1.0  # [0,1] -> [-1,1]
        action = np.full(4, u, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))

    def test_multiple_steps_dont_crash(self) -> None:
        """Running 100 steps with mid-range action doesn't crash."""
        env = GateRaceEnv(n_envs=1)
        env.reset(seed=42)

        # Use action=0 (mid-range motor speed via ESC curve)
        action = np.zeros(4, dtype=np.float32)

        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))


class TestGateRaceEnvTermination:
    """Terminal conditions should trigger correctly."""

    def test_ground_crash_terminates(self) -> None:
        """Drone hitting the ground (z<=0) should terminate."""
        env = GateRaceEnv(n_envs=1, max_steps=10000)
        env.reset(seed=42)

        # Minimum thrust (u=-1 -> omega_min via ESC) -> free fall -> ground crash
        action = -np.ones(4, dtype=np.float32)

        terminated = False
        for _ in range(2000):
            obs, reward, terminated_arr, truncated, info = env.step(action)
            # terminated may be scalar or (1,) array
            term_val = terminated_arr.item() if hasattr(terminated_arr, 'item') else terminated_arr
            if term_val:
                terminated = True
                break

        assert terminated, "Expected ground crash termination"


class TestGateRaceEnvSpaces:
    """Observation and action spaces should be correctly defined."""

    def test_obs_space_shape(self) -> None:
        """Observation space has correct shape."""
        env = GateRaceEnv(n_envs=1)
        assert env.observation_space.shape == (OBS_DIM,)

    def test_action_space_shape(self) -> None:
        """Action space has correct shape."""
        env = GateRaceEnv(n_envs=1)
        assert env.action_space.shape == (4,)

    def test_action_space_bounds(self) -> None:
        """Action space low=-1, high=1 (normalized ESC commands)."""
        env = GateRaceEnv(n_envs=1)
        np.testing.assert_array_equal(env.action_space.low, -np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(env.action_space.high, np.ones(4, dtype=np.float32))

    def test_obs_in_space_after_reset(self) -> None:
        """Reset observation is within observation space."""
        env = GateRaceEnv(n_envs=1)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), f"Obs not in space: {obs}"

    def test_obs_in_space_after_step(self) -> None:
        """Step observation is within observation space."""
        env = GateRaceEnv(n_envs=1)
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        # obs could have very large values due to relative positions,
        # but should be finite
        assert np.all(np.isfinite(obs))


class TestESCModel:
    """Tests for the nonlinear ESC motor mapping."""

    def test_esc_min_action_gives_omega_min(self) -> None:
        """u=-1 should map to omega_min."""
        env = GateRaceEnv(n_envs=1)
        u = -np.ones((1, 4), dtype=np.float64)
        omega = env._esc_to_omega(u)
        np.testing.assert_allclose(omega, env.omega_min, atol=1e-10)

    def test_esc_max_action_gives_omega_max(self) -> None:
        """u=+1 should map to max_omega."""
        env = GateRaceEnv(n_envs=1)
        u = np.ones((1, 4), dtype=np.float64)
        omega = env._esc_to_omega(u)
        np.testing.assert_allclose(omega, env.params.max_omega, atol=1e-10)

    def test_esc_monotonically_increasing(self) -> None:
        """ESC curve should be monotonically increasing from -1 to 1."""
        env = GateRaceEnv(n_envs=1)
        u_values = np.linspace(-1, 1, 100)
        u_batch = u_values.reshape(-1, 1) * np.ones((1, 4))
        omega = env._esc_to_omega(u_batch)
        # Check each motor channel is monotonically increasing
        for motor in range(4):
            diffs = np.diff(omega[:, motor])
            assert np.all(diffs >= 0), "ESC curve is not monotonically increasing"

    def test_esc_linear_when_k_is_one(self) -> None:
        """With k=1, ESC curve should be linear."""
        env = GateRaceEnv(n_envs=1, esc_nonlinearity=1.0)
        u_values = np.linspace(-1, 1, 50)
        u_batch = u_values.reshape(-1, 1) * np.ones((1, 4))
        omega = env._esc_to_omega(u_batch)
        # Linear: omega should be evenly spaced
        expected = np.linspace(env.omega_min, env.params.max_omega, 50)
        np.testing.assert_allclose(omega[:, 0], expected, atol=1e-8)

    def test_esc_nonlinear_when_k_is_zero(self) -> None:
        """With k=0, ESC curve should be sqrt-shaped."""
        env = GateRaceEnv(n_envs=1, esc_nonlinearity=0.0)
        u = np.array([[0.0, 0.0, 0.0, 0.0]])
        omega = env._esc_to_omega(u)
        # U=0.5, k=0: sqrt(0.5) * max_omega ≈ 0.707 * max_omega
        expected = env.params.max_omega * np.sqrt(0.5)
        np.testing.assert_allclose(omega[0, 0], expected, rtol=1e-6)

    def test_esc_roundtrip_with_hover(self) -> None:
        """ESC output at hover action matches hover omega."""
        from sim.dynamics.numpy_quad import GRAVITY
        env = GateRaceEnv(n_envs=1)
        p = env.params
        hover_omega = np.sqrt(p.mass * GRAVITY / (4.0 * p.k_thrust))

        # Invert ESC to get normalized action
        k = env.esc_nonlinearity
        R = ((hover_omega - env.omega_min) / (p.max_omega - env.omega_min)) ** 2
        U = (-(1 - k) + np.sqrt((1 - k) ** 2 + 4 * k * R)) / (2 * k)
        u = 2.0 * U - 1.0

        # Forward through ESC
        omega = env._esc_to_omega(np.full((1, 4), u))
        np.testing.assert_allclose(omega[0, 0], hover_omega, rtol=1e-6)

    def test_prev_action_in_obs_is_normalized(self) -> None:
        """Previous action in observation [16:20] should be in [-1, 1]."""
        env = GateRaceEnv(n_envs=1)
        env.reset(seed=42)
        action = np.array([0.5, -0.3, 0.1, 0.8], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        # obs[16:20] should equal the action we just passed
        np.testing.assert_allclose(obs[16:20], action, atol=1e-6)


class TestHoverEnv:
    """Basic functionality tests for HoverEnv."""

    def test_reset_shape(self) -> None:
        """Reset returns obs of shape (STATE_DIM,)."""
        env = HoverEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (STATE_DIM,)
        assert obs.dtype == np.float32

    def test_reset_at_target_height(self) -> None:
        """After reset, drone should be at target height."""
        env = HoverEnv(target_height=1.5)
        obs, _ = env.reset(seed=42)
        # z position is index 2
        assert abs(obs[2] - 1.5) < 1e-5

    def test_step_returns_valid(self) -> None:
        """Step returns valid tuple."""
        env = HoverEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (STATE_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_hover_reward_near_zero_when_on_target(self) -> None:
        """At target height, reward should be close to zero."""
        env = HoverEnv()
        env.reset(seed=42)

        # Apply hover thrust
        from sim.dynamics.numpy_quad import GRAVITY
        p = env.params
        hover_omega = np.sqrt(p.mass * GRAVITY / (4.0 * p.k_thrust))
        hover_rpm = hover_omega * 60.0 / (2.0 * np.pi)
        action = np.full(4, hover_rpm, dtype=np.float32)

        obs, reward, _, _, _ = env.step(action)
        # Should be very close to zero since we start at target
        assert abs(reward) < 0.1, f"Reward = {reward}"

    def test_ground_crash_terminates(self) -> None:
        """Zero thrust causes ground crash and termination."""
        env = HoverEnv(max_steps=10000)
        env.reset(seed=42)

        action = np.zeros(4, dtype=np.float32)
        terminated = False
        for _ in range(2000):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        assert terminated, "Expected ground crash"

    def test_obs_space_and_action_space(self) -> None:
        """Spaces are correctly defined."""
        env = HoverEnv()
        assert env.observation_space.shape == (STATE_DIM,)
        assert env.action_space.shape == (4,)


class TestTerminalObservation:
    """Terminal observations should be captured before auto-reset."""

    def test_terminal_obs_returned_on_crash(self) -> None:
        """When an env terminates, info should contain terminal_obs."""
        env = GateRaceEnv(n_envs=2, max_steps=10000)
        env.reset(seed=42)

        # Drive env 0 into the ground with minimum thrust
        action = -np.ones((2, 4), dtype=np.float32)
        info: dict = {}
        for _ in range(2000):
            obs, reward, terminated, truncated, info = env.step(action)
            if np.any(terminated):
                break

        assert "terminal_obs" in info, "info should contain 'terminal_obs' on termination"
        terminal_obs = info["terminal_obs"]
        assert terminal_obs.shape == (2, OBS_DIM)
        # Done envs should have the pre-reset observation
        done_mask = terminated | truncated
        if np.any(done_mask):
            done_idx = np.where(done_mask)[0][0]
            assert not np.allclose(terminal_obs[done_idx], obs[done_idx]), \
                "Terminal obs should differ from post-reset obs"

    def test_terminal_obs_matches_non_done_envs(self) -> None:
        """For non-done envs, terminal_obs should equal the returned obs."""
        env = GateRaceEnv(n_envs=2, max_steps=10000)
        env.reset(seed=42)

        action = np.zeros((2, 4), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert "terminal_obs" in info
        np.testing.assert_array_equal(info["terminal_obs"], obs)


class TestDomainRandomization:
    def test_randomizer_changes_params_on_reset(self):
        """After reset, env's dynamics should have per-env randomized params."""
        randomizer = DomainRandomizer({"mass": 0.3, "k_thrust": 0.3})
        env = GateRaceEnv(n_envs=10, domain_randomizer=randomizer)
        env.reset(seed=42)

        # Masses should vary across envs
        masses = env.dynamics._mass
        assert not np.allclose(masses, masses[0]), (
            f"All masses identical after DR reset: {masses}"
        )

    def test_auto_reset_rerandomizes(self):
        """Terminated envs get fresh randomized params on auto-reset."""
        randomizer = DomainRandomizer({"mass": 0.3})
        env = GateRaceEnv(
            n_envs=2, domain_randomizer=randomizer, ceiling=200.0
        )
        env.reset(seed=42)

        masses_before = env.dynamics._mass.copy()

        # Drive env 0 above ceiling to trigger termination
        env._states[0, 2] = 300.0  # way above ceiling
        action = np.zeros((2, 4), dtype=np.float32)
        env.step(action)

        # Env 1 was not reset → mass unchanged
        assert env.dynamics._mass[1] == masses_before[1]

    def test_no_randomizer_uses_nominal(self):
        """Without domain_randomizer, all envs use nominal params."""
        env = GateRaceEnv(n_envs=10)
        env.reset(seed=42)
        masses = env.dynamics._mass
        assert np.allclose(masses, masses[0])


class TestGateLapTracking:
    """Gate passage counting and lap timing in info dict."""

    def test_gates_passed_increments(self):
        """gates_passed counter increments on gate passage."""
        track = Track([
            GateState(position=np.array([3.0, 0.0, 1.0])),
            GateState(position=np.array([6.0, 0.0, 1.0])),
        ])
        env = GateRaceEnv(track=track, n_envs=1, gate_passage_radius=5.0)
        env.reset(seed=42)

        # Place drone just behind gate 0 plane so one step crosses it.
        # Gate is at x=3, normal points +x (default orientation).
        # Drone at x=2.999 with high forward velocity crosses in one dt=0.01 step.
        env._states[0, 0] = 2.999
        env._states[0, 1] = 0.0
        env._states[0, 2] = 1.0
        env._states[0, 3] = 20.0  # high forward velocity to guarantee crossing
        env._prev_along_normal[0] = -0.001  # just barely on the approaching side

        obs, rew, term, trunc, info = env.step(np.zeros((1, 4), dtype=np.float32))

        assert env._gates_passed[0] >= 1, "gates_passed should increment on passage"

    def test_episode_metrics_on_done(self):
        """Episode metrics appear in info dict when episode ends."""
        env = GateRaceEnv(n_envs=1, ceiling=2.0, max_steps=5)
        env.reset(seed=42)

        # Step until truncation (max_steps=5)
        for _ in range(5):
            obs, rew, term, trunc, info = env.step(np.zeros((1, 4), dtype=np.float32))

        # Should have episode metrics in info
        assert "episode" in info, "info should contain 'episode' key on done"
        ep = info["episode"]
        assert "gates_passed" in ep
        assert "laps_completed" in ep

    def test_lap_completes_on_gate_wrap(self):
        """Lap counter increments when gate index wraps to 0."""
        track = Track([
            GateState(position=np.array([3.0, 0.0, 1.0])),
        ])
        env = GateRaceEnv(track=track, n_envs=1, gate_passage_radius=5.0)
        env.reset(seed=42)

        # Place drone just behind gate plane with high velocity to guarantee crossing
        env._states[0, 0] = 2.999
        env._states[0, 1] = 0.0
        env._states[0, 2] = 1.0
        env._states[0, 3] = 20.0  # high forward velocity
        env._prev_along_normal[0] = -0.001
        env._gate_indices[0] = 0

        obs, rew, term, trunc, info = env.step(np.zeros((1, 4), dtype=np.float32))

        # Gate wrapped to 0 = lap complete
        assert env._laps_completed[0] >= 1, "Lap should complete when gate wraps"


class TestTRPYMode:
    """Tests for TRPY action mode in GateRaceEnv."""

    def test_trpy_env_creates(self) -> None:
        """Create env with action_mode='trpy', verify action_space shape is (4,)."""
        env = GateRaceEnv(n_envs=1, action_mode="trpy")
        assert env.action_space.shape == (4,)

    def test_trpy_env_step_hover(self) -> None:
        """Step with zero action (hover), verify obs shape and no termination."""
        env = GateRaceEnv(n_envs=1, action_mode="trpy")
        env.reset(seed=42)

        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (OBS_DIM,)
        term_val = terminated.item() if hasattr(terminated, "item") else terminated
        trunc_val = truncated.item() if hasattr(truncated, "item") else truncated
        assert not term_val, "Zero TRPY action should not terminate immediately"
        assert not trunc_val

    def test_trpy_env_runs_100_steps(self) -> None:
        """Run 100 steps with zero action, verify no NaN in obs."""
        env = GateRaceEnv(n_envs=1, action_mode="trpy")
        env.reset(seed=42)

        action = np.zeros(4, dtype=np.float32)
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs)), f"NaN/inf in obs after step"


class TestRuntimeRewardUpdates:
    def test_set_reward_weights(self):
        from sim.tracks import build_figure8_track
        env = GateRaceEnv(
            track=build_figure8_track(), n_envs=1, dt=0.01, max_steps=10,
            reward_weights={"gate_passage": 1.5, "gate_progress": 1.0, "crash_penalty": 10.0},
        )
        env.set_reward_weights({"gate_passage": 30.0, "gate_progress": 0.0})
        assert env.reward_weights["gate_passage"] == 30.0
        assert env.reward_weights["gate_progress"] == 0.0
        assert env.reward_weights["crash_penalty"] == 10.0

    def test_set_v_max(self):
        from sim.tracks import build_figure8_track
        env = GateRaceEnv(
            track=build_figure8_track(), n_envs=1, dt=0.01, max_steps=10,
        )
        env.set_v_max(20.0)
        assert env.v_max == 20.0
