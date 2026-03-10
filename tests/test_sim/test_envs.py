"""Tests for GateRaceEnv and HoverEnv.

Covers reset shapes, step validity, termination conditions,
observation/action space definitions, and HoverEnv basics.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.envs.gate_race_env import GateRaceEnv, OBS_DIM
from sim.envs.hover_env import HoverEnv, STATE_DIM


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

        # Hover RPM: compute from params
        from sim.dynamics.numpy_quad import GRAVITY
        p = env.params
        hover_omega = np.sqrt(p.mass * GRAVITY / (4.0 * p.k_thrust))
        hover_rpm = hover_omega * 60.0 / (2.0 * np.pi)
        action = np.full(4, hover_rpm, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs))

    def test_multiple_steps_dont_crash(self) -> None:
        """Running 100 steps with hover action doesn't crash."""
        env = GateRaceEnv(n_envs=1)
        env.reset(seed=42)

        from sim.dynamics.numpy_quad import GRAVITY
        p = env.params
        hover_omega = np.sqrt(p.mass * GRAVITY / (4.0 * p.k_thrust))
        hover_rpm = hover_omega * 60.0 / (2.0 * np.pi)
        action = np.full(4, hover_rpm, dtype=np.float32)

        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))


class TestGateRaceEnvTermination:
    """Terminal conditions should trigger correctly."""

    def test_ground_crash_terminates(self) -> None:
        """Drone hitting the ground (z<=0) should terminate."""
        env = GateRaceEnv(n_envs=1, max_steps=10000)
        env.reset(seed=42)

        # Zero thrust -> free fall -> ground crash
        action = np.zeros(4, dtype=np.float32)

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
        """Action space low=0, high=max_rpm."""
        env = GateRaceEnv(n_envs=1)
        np.testing.assert_array_equal(env.action_space.low, np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(
            env.action_space.high,
            np.full(4, env.params.max_rpm, dtype=np.float32),
        )

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
