"""Tests for PerceptionNoiseWrapper and PerceptionPassthroughWrapper.

Uses a minimal mock Gymnasium environment — no sim dependency required.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from perception.noise_injection import (
    PerceptionNoiseWrapper,
    PerceptionPassthroughWrapper,
)


# --------------------------------------------------------------------- #
# Minimal mock environment
# --------------------------------------------------------------------- #


class _ConstantEnv(gym.Env):
    """Trivial env that returns a constant observation vector.

    The observation is ``np.arange(obs_dim, dtype=float64) + 1`` on every
    step, making it easy to verify noise, dropout, and latency effects.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 6, max_steps: int = 200) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64
        )
        self._step_count = 0
        self._obs = np.arange(1, obs_dim + 1, dtype=np.float64)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        return self._obs.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        return self._obs.copy(), 0.0, terminated, False, {}


class _CountingEnv(gym.Env):
    """Env whose observation is the step count — useful for latency tests."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 4, max_steps: int = 200) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.max_steps = max_steps
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64
        )
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        return np.full(self.obs_dim, 0.0, dtype=np.float64), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs = np.full(self.obs_dim, float(self._step_count), dtype=np.float64)
        terminated = self._step_count >= self.max_steps
        return obs, 0.0, terminated, False, {}


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

ACTION = np.zeros(2, dtype=np.float64)


@pytest.fixture
def constant_env() -> _ConstantEnv:
    return _ConstantEnv(obs_dim=6)


@pytest.fixture
def counting_env() -> _CountingEnv:
    return _CountingEnv(obs_dim=4)


# --------------------------------------------------------------------- #
# HMM dropout tests
# --------------------------------------------------------------------- #


class TestHMMDropout:
    """HMM dropout should converge to its stationary distribution."""

    def test_stationary_distribution(self, constant_env: _ConstantEnv) -> None:
        """Run many steps and check that the empirical missing rate matches
        the stationary distribution of the 2-state Markov chain."""
        p_dm = 0.1  # P(detect -> miss)
        p_md = 0.3  # P(miss -> detect)
        # Stationary P(missing) = p_dm / (p_dm + p_md)
        expected_missing = p_dm / (p_dm + p_md)

        env = PerceptionNoiseWrapper(
            constant_env,
            dropout_hmm_config={
                "p_detect_to_miss": p_dm,
                "p_miss_to_detect": p_md,
            },
            noise_std=0.0,
            latency_frames=0,
            seed=42,
        )
        obs, _ = env.reset()

        n_steps = 5000
        n_missing = 0
        for _ in range(n_steps):
            obs, _, terminated, truncated, _ = env.step(ACTION)
            if np.allclose(obs, 0.0):
                n_missing += 1
            if terminated or truncated:
                obs, _ = env.reset()

        empirical_missing = n_missing / n_steps
        assert abs(empirical_missing - expected_missing) < 0.05, (
            f"Empirical missing rate {empirical_missing:.3f} vs "
            f"expected {expected_missing:.3f}"
        )

    def test_no_dropout_when_disabled(self, constant_env: _ConstantEnv) -> None:
        env = PerceptionNoiseWrapper(
            constant_env,
            dropout_hmm_config={
                "p_detect_to_miss": 0.0,
                "p_miss_to_detect": 1.0,
            },
            noise_std=0.0,
            latency_frames=0,
            seed=0,
        )
        obs, _ = env.reset()
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(ACTION)
            assert not np.allclose(obs, 0.0), "No dropout expected"
            if terminated or truncated:
                break


# --------------------------------------------------------------------- #
# Latency tests
# --------------------------------------------------------------------- #


class TestLatency:
    """Observation latency should delay observations by exactly N steps."""

    def test_latency_delay(self, counting_env: _CountingEnv) -> None:
        latency = 3
        env = PerceptionNoiseWrapper(
            counting_env,
            dropout_hmm_config={
                "p_detect_to_miss": 0.0,
                "p_miss_to_detect": 1.0,
            },
            noise_std=0.0,
            latency_frames=latency,
            seed=0,
        )
        obs, _ = env.reset()
        # After reset, latency buffer is pre-filled with the reset observation
        # (all zeros), so the first `latency` steps should return the reset obs.
        # Step 1 -> env returns [1,1,1,1], but latency returns reset obs [0,0,0,0]
        # ...
        # Step latency -> env returns [latency,...], returns step 0 obs [0,0,0,0]
        # Step latency+1 -> env returns [latency+1,...], returns step 1 obs [1,1,1,1]

        observations: list[np.ndarray] = []
        for _ in range(latency + 5):
            obs, _, terminated, truncated, _ = env.step(ACTION)
            observations.append(obs.copy())
            if terminated or truncated:
                break

        # The first `latency` observations should be the reset obs (zeros)
        for i in range(latency):
            np.testing.assert_allclose(
                observations[i],
                np.zeros(4),
                err_msg=f"Step {i}: expected reset obs (zeros) due to latency",
            )

        # After the latency period, observations should be delayed by `latency` steps
        for i in range(latency, len(observations)):
            expected_step = i - latency + 1
            expected = np.full(4, float(expected_step))
            np.testing.assert_allclose(
                observations[i],
                expected,
                err_msg=f"Step {i}: expected delayed obs from step {expected_step}",
            )

    def test_zero_latency_no_delay(self, counting_env: _CountingEnv) -> None:
        env = PerceptionNoiseWrapper(
            counting_env,
            dropout_hmm_config={
                "p_detect_to_miss": 0.0,
                "p_miss_to_detect": 1.0,
            },
            noise_std=0.0,
            latency_frames=0,
            seed=0,
        )
        env.reset()
        for step in range(1, 10):
            obs, _, _, _, _ = env.step(ACTION)
            expected = np.full(4, float(step))
            np.testing.assert_allclose(obs, expected)


# --------------------------------------------------------------------- #
# Passthrough tests
# --------------------------------------------------------------------- #


class TestPassthrough:
    """PerceptionPassthroughWrapper should leave observations unchanged."""

    def test_passthrough_unchanged(self, constant_env: _ConstantEnv) -> None:
        env = PerceptionPassthroughWrapper(constant_env)
        obs, _ = env.reset()
        expected = np.arange(1, 7, dtype=np.float64)
        np.testing.assert_array_equal(obs, expected)

        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(ACTION)
            np.testing.assert_array_equal(obs, expected)
            if terminated or truncated:
                break


# --------------------------------------------------------------------- #
# Reset tests
# --------------------------------------------------------------------- #


class TestResetBehavior:
    """Wrapper should properly reset internal state on env.reset()."""

    def test_hmm_resets_to_detecting(self, constant_env: _ConstantEnv) -> None:
        """After reset, the HMM should be in the detecting state (not missing)."""
        env = PerceptionNoiseWrapper(
            constant_env,
            dropout_hmm_config={
                "p_detect_to_miss": 1.0,  # always transition to missing
                "p_miss_to_detect": 0.0,
            },
            noise_std=0.0,
            latency_frames=0,
            seed=0,
        )
        # First episode: after a step the HMM will go to missing
        env.reset()
        obs, _, _, _, _ = env.step(ACTION)
        assert np.allclose(obs, 0.0), "Should be missing after step with p_dm=1.0"

        # Reset should bring HMM back to detecting
        obs, _ = env.reset()
        # The reset observation itself should not be modified (HMM only steps in step())
        expected = np.arange(1, 7, dtype=np.float64)
        np.testing.assert_array_equal(obs, expected)

    def test_latency_buffer_clears_on_reset(
        self, counting_env: _CountingEnv
    ) -> None:
        """After reset, old latency buffer contents should be gone."""
        latency = 2
        env = PerceptionNoiseWrapper(
            counting_env,
            dropout_hmm_config={
                "p_detect_to_miss": 0.0,
                "p_miss_to_detect": 1.0,
            },
            noise_std=0.0,
            latency_frames=latency,
            seed=0,
        )

        # Run a few steps
        env.reset()
        for _ in range(10):
            env.step(ACTION)

        # Reset and verify buffer is cleared
        env.reset()
        # After reset, buffer is pre-filled with reset obs.
        # First two steps should return reset obs (zeros), not stale data.
        for i in range(latency):
            obs, _, _, _, _ = env.step(ACTION)
            np.testing.assert_allclose(
                obs,
                np.zeros(4),
                err_msg=f"Step {i} after reset: should see reset obs, not stale data",
            )
