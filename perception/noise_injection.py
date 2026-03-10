"""Gymnasium wrappers for perception noise injection during RL training.

Implements the noise models described in the perception-in-the-loop research:
  - HMM dropout: 2-state Markov chain (detecting / missing) with correlated
    burst failures (MAVLab/MonoRace model).
  - Gaussian noise: additive noise on gate-related observation dimensions.
  - Latency: observation delay buffer simulating real perception pipeline lag.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class PerceptionNoiseWrapper(gym.Wrapper):
    """Wrap a Gymnasium env to inject perception-realistic noise.

    The wrapper corrupts gate-related dimensions of the observation with:

    1. **HMM dropout** — a 2-state Markov chain toggles between *detecting*
       and *missing*.  In the *missing* state all gate-related observation
       dims are zeroed out, simulating a detection dropout burst.
    2. **Gaussian noise** — additive noise on gate dims.
    3. **Latency** — a FIFO buffer delays observations by ``latency_frames``
       steps, simulating CNN inference latency.

    Observation space must be ``gymnasium.spaces.Box``.

    Args:
        env: The environment to wrap.
        dropout_hmm_config: Dict with keys:
            - ``p_detect_to_miss``: transition prob detecting -> missing.
            - ``p_miss_to_detect``: transition prob missing -> detecting.
        noise_std: Std-dev of Gaussian noise added to gate dims.
        latency_frames: Number of frames to delay observations.
        gate_obs_indices: Indices into the flat observation that correspond
            to gate-related dims.  If ``None``, noise/dropout is applied to
            the full observation vector.
        seed: Random seed.
    """

    # HMM states
    _STATE_DETECTING = 0
    _STATE_MISSING = 1

    def __init__(
        self,
        env: gym.Env,
        dropout_hmm_config: dict[str, float] | None = None,
        noise_std: float = 0.0,
        latency_frames: int = 0,
        gate_obs_indices: list[int] | NDArray[np.intp] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(env)

        # HMM config
        if dropout_hmm_config is None:
            dropout_hmm_config = {
                "p_detect_to_miss": 0.0,
                "p_miss_to_detect": 1.0,
            }
        self.p_detect_to_miss = dropout_hmm_config["p_detect_to_miss"]
        self.p_miss_to_detect = dropout_hmm_config["p_miss_to_detect"]

        self.noise_std = noise_std
        self.latency_frames = latency_frames
        self.gate_obs_indices = (
            np.asarray(gate_obs_indices, dtype=np.intp)
            if gate_obs_indices is not None
            else None
        )
        self._rng = np.random.default_rng(seed)

        # Mutable state — reset in _reset_state()
        self._hmm_state: int = self._STATE_DETECTING
        self._latency_buffer: deque[NDArray[np.float64]] = deque()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _reset_state(self) -> None:
        """Reset HMM and latency buffer (called on env reset)."""
        self._hmm_state = self._STATE_DETECTING
        self._latency_buffer.clear()

    def _step_hmm(self) -> None:
        """Advance the HMM by one step."""
        r = self._rng.random()
        if self._hmm_state == self._STATE_DETECTING:
            if r < self.p_detect_to_miss:
                self._hmm_state = self._STATE_MISSING
        else:  # MISSING
            if r < self.p_miss_to_detect:
                self._hmm_state = self._STATE_DETECTING

    def _apply_noise(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply dropout and Gaussian noise to observation."""
        obs = obs.copy().astype(np.float64)
        idx = self.gate_obs_indices

        # HMM dropout
        if self._hmm_state == self._STATE_MISSING:
            if idx is not None:
                obs[idx] = 0.0
            else:
                obs[:] = 0.0

        # Gaussian noise (only when detecting)
        if self._hmm_state == self._STATE_DETECTING and self.noise_std > 0:
            noise = self._rng.normal(scale=self.noise_std, size=obs.shape)
            if idx is not None:
                obs[idx] += noise[idx]
            else:
                obs += noise

        return obs

    def _apply_latency(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Buffer observation and return the delayed one."""
        self._latency_buffer.append(obs.copy())

        if len(self._latency_buffer) > self.latency_frames:
            return self._latency_buffer.popleft()
        else:
            # Not enough history yet — return a zero observation
            return np.zeros_like(obs)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        self._reset_state()
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float64)

        # Pre-fill latency buffer with the initial observation
        for _ in range(self.latency_frames):
            self._latency_buffer.append(obs.copy())

        return obs, info

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float64)

        self._step_hmm()
        obs = self._apply_noise(obs)

        if self.latency_frames > 0:
            obs = self._apply_latency(obs)

        return obs, reward, terminated, truncated, info


class PerceptionPassthroughWrapper(gym.Wrapper):
    """No-op wrapper that passes observations through unchanged.

    Used when ``perception: none`` is set in the config, so that the
    wrapper slot in the training pipeline is always occupied (simplifies
    code paths).
    """

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float64), info

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return np.asarray(obs, dtype=np.float64), reward, terminated, truncated, info
