"""Vectorized perception noise injection wrapper for SB3 VecEnv.

Port of ``perception.noise_injection.PerceptionNoiseWrapper`` (single-env
gym.Wrapper) to the SB3 ``VecEnvWrapper`` interface, with fully vectorized
HMM state, Gaussian noise, and latency buffer across all sub-environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


# --------------------------------------------------------------------------- #
# VecEnvWrapper (the actual wrapper applied to the env)
# --------------------------------------------------------------------------- #


class VecNoiseInjection(VecEnvWrapper):
    """Vectorized perception noise wrapper operating on SB3 VecEnv.

    Per-env HMM dropout, Gaussian noise, and observation latency — all
    computed in vectorized numpy without Python loops over envs.

    Args:
        venv: The VecEnv to wrap.
        p_detect_to_miss: HMM transition probability detecting -> missing.
        p_miss_to_detect: HMM transition probability missing -> detecting.
        noise_std: Std-dev of additive Gaussian noise on gate dims.
        latency_frames: Number of frames to delay observations.
        gate_obs_indices: Indices into the flat observation for gate-related
            dims.  If ``None``, noise/dropout applies to the full vector.
        seed: Random seed.
    """

    # HMM states
    _STATE_DETECTING = 0
    _STATE_MISSING = 1

    def __init__(
        self,
        venv: VecEnv,
        p_detect_to_miss: float = 0.0,
        p_miss_to_detect: float = 1.0,
        noise_std: float = 0.0,
        latency_frames: int = 0,
        gate_obs_indices: list[int] | NDArray[np.intp] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(venv)

        self.p_detect_to_miss = p_detect_to_miss
        self.p_miss_to_detect = p_miss_to_detect
        self.noise_std = noise_std
        self.latency_frames = latency_frames
        self.gate_obs_indices = (
            np.asarray(gate_obs_indices, dtype=np.intp)
            if gate_obs_indices is not None
            else None
        )
        self._rng = np.random.default_rng(seed)

        n = self.num_envs
        obs_dim = int(np.prod(self.observation_space.shape))

        # Per-env HMM state: (n_envs,) int array
        self._hmm_state = np.full(n, self._STATE_DETECTING, dtype=np.int8)

        # Latency ring buffer: (latency_frames, n_envs, obs_dim)
        if self.latency_frames > 0:
            self._latency_buf = np.zeros(
                (self.latency_frames, n, obs_dim), dtype=np.float64
            )
        else:
            self._latency_buf = None
        self._latency_idx = 0  # write cursor into ring buffer

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _reset_env_state(self, env_mask: NDArray[np.bool_]) -> None:
        """Reset HMM and latency buffer for envs indicated by *env_mask*."""
        self._hmm_state[env_mask] = self._STATE_DETECTING
        if self._latency_buf is not None:
            self._latency_buf[:, env_mask, :] = 0.0

    def _step_hmm(self) -> None:
        """Advance the per-env HMM by one step (vectorized)."""
        r = self._rng.random(self.num_envs)

        detecting = self._hmm_state == self._STATE_DETECTING
        missing = ~detecting

        # detecting -> missing
        flip_to_miss = detecting & (r < self.p_detect_to_miss)
        # missing -> detecting
        flip_to_detect = missing & (r < self.p_miss_to_detect)

        self._hmm_state[flip_to_miss] = self._STATE_MISSING
        self._hmm_state[flip_to_detect] = self._STATE_DETECTING

    def _apply_noise(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply per-env dropout and Gaussian noise (vectorized).

        Args:
            obs: (n_envs, obs_dim) float64 array (modified in-place and returned).
        """
        idx = self.gate_obs_indices
        missing = self._hmm_state == self._STATE_MISSING  # (n_envs,)
        detecting = ~missing

        # --- HMM dropout: zero out gate dims for missing envs ---
        if missing.any():
            if idx is not None:
                obs[np.ix_(missing, idx)] = 0.0
            else:
                obs[missing] = 0.0

        # --- Gaussian noise on detecting envs ---
        if self.noise_std > 0 and detecting.any():
            noise = self._rng.normal(
                scale=self.noise_std, size=obs.shape
            )
            if idx is not None:
                # Only add noise to gate dims of detecting envs
                det_idx = np.where(detecting)[0]
                obs[np.ix_(det_idx, idx)] += noise[np.ix_(det_idx, idx)]
            else:
                obs[detecting] += noise[detecting]

        return obs

    def _apply_latency(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Swap current obs into ring buffer and return delayed obs.

        Args:
            obs: (n_envs, obs_dim) float64 array.

        Returns:
            Delayed observation from ``latency_frames`` steps ago.
        """
        assert self._latency_buf is not None
        # Read the oldest entry (what was written latency_frames ago)
        delayed = self._latency_buf[self._latency_idx].copy()
        # Write the new obs into the ring buffer slot
        self._latency_buf[self._latency_idx] = obs
        # Advance cursor
        self._latency_idx = (self._latency_idx + 1) % self.latency_frames
        return delayed

    # ------------------------------------------------------------------ #
    # VecEnv API overrides
    # ------------------------------------------------------------------ #

    def reset(self) -> NDArray[np.float64]:
        """Reset all envs, initialize HMM and latency buffer."""
        obs = self.venv.reset()
        obs = np.asarray(obs, dtype=np.float64)

        # Reset all HMM states
        self._hmm_state[:] = self._STATE_DETECTING

        # Pre-fill latency buffer with the reset observation
        if self._latency_buf is not None:
            for i in range(self.latency_frames):
                self._latency_buf[i] = obs
            self._latency_idx = 0

        return obs

    def step_wait(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        """Step, then apply perception noise and latency."""
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = np.asarray(obs, dtype=np.float64).copy()

        # Advance HMM
        self._step_hmm()

        # Apply noise / dropout
        obs = self._apply_noise(obs)

        # Apply latency
        if self._latency_buf is not None:
            obs = self._apply_latency(obs)

        # Reset state for auto-reset envs
        if dones.any():
            self._reset_env_state(dones)

        return obs, rewards, dones, infos


# --------------------------------------------------------------------------- #
# Factory wrapper (what Hydra instantiates)
# --------------------------------------------------------------------------- #


class NoiseInjectionWrapper:
    """Factory that creates a :class:`VecNoiseInjection` around a VecEnv.

    This is the object Hydra instantiates from config.  Its ``wrap`` method
    conforms to the :class:`~perception.wrappers.base.ObservationWrapper`
    protocol.

    Args:
        dropout_hmm: Dict with ``p_detect_to_miss`` and ``p_miss_to_detect``.
        noise_std: Std-dev of additive Gaussian noise.
        latency_frames: Number of frames of observation delay.
        gate_obs_indices: Observation indices for gate-related dims (optional).
        seed: Random seed for the noise RNG.
    """

    def __init__(
        self,
        dropout_hmm: dict[str, float] | None = None,
        noise_std: float = 0.0,
        latency_frames: int = 0,
        gate_obs_indices: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        if dropout_hmm is None:
            dropout_hmm = {
                "p_detect_to_miss": 0.0,
                "p_miss_to_detect": 1.0,
            }
        self.dropout_hmm = dropout_hmm
        self.noise_std = noise_std
        self.latency_frames = latency_frames
        self.gate_obs_indices = gate_obs_indices
        self.seed = seed

    def wrap(self, env: VecEnv) -> VecEnv:
        """Wrap *env* with vectorized perception noise injection."""
        return VecNoiseInjection(
            venv=env,
            p_detect_to_miss=self.dropout_hmm["p_detect_to_miss"],
            p_miss_to_detect=self.dropout_hmm["p_miss_to_detect"],
            noise_std=self.noise_std,
            latency_frames=self.latency_frames,
            gate_obs_indices=self.gate_obs_indices,
            seed=self.seed,
        )
