"""VecEnv perception wrapper for playground corner noise + HMM dropout.

Implements the ObservationWrapper protocol so Hydra can instantiate it
via ``_target_: perception.wrappers.corner_noise_wrapper.CornerNoisePerceptionWrapper``.

Wraps a VecEnv (backed by RateCtrlEnv) with the PerceptionNoise model
from :mod:`perception.wrappers.corner_noise`.  In non-asymmetric mode,
returns corrupted 24D obs.  In asymmetric mode, returns 50D obs
(noisy_24D + clean_24D + metadata_2D).
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium
import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from perception.wrappers.corner_noise import PerceptionNoise


class VecCornerNoise(VecEnvWrapper):
    """Vectorized corner noise wrapper operating on SB3 VecEnv.

    Applies distance-scaled Gaussian noise + HMM dropout to observations.
    In asymmetric mode, produces 50D obs: [noisy_24D, clean_24D, metadata_2D].

    Args:
        venv: The VecEnv to wrap (must be backed by a RateCtrlEnv).
        noise_scale: Base noise scale multiplier.
        dropout_onset: Fixed dropout onset probability (None = random per-env).
        dropout_continuation: HMM continuation probability.
        asymmetric: If True, return 50D obs for asymmetric actor-critic.
        noise_scale_range: Optional (lo, hi) for domain-randomized noise scale.
    """

    def __init__(
        self,
        venv: VecEnv,
        noise_scale: float = 1.0,
        dropout_onset: Optional[float] = None,
        dropout_continuation: float = 0.7,
        asymmetric: bool = False,
        noise_scale_range: Optional[tuple[float, float]] = None,
    ) -> None:
        self.asymmetric = asymmetric
        self.noise_scale_range = noise_scale_range

        n_envs = venv.num_envs
        self._noise = PerceptionNoise(
            n_envs=n_envs,
            seed=42,
            noise_scale=noise_scale,
            dropout_onset=dropout_onset,
            dropout_continuation=dropout_continuation,
        )

        # Build new observation space for asymmetric mode
        if asymmetric:
            obs_dim = 50
        else:
            obs_dim = venv.observation_space.shape[0]

        observation_space = gymnasium.spaces.Box(
            low=-np.inf * np.ones(obs_dim, dtype=np.float32),
            high=np.inf * np.ones(obs_dim, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        super().__init__(venv, observation_space=observation_space)

    def reset(self) -> NDArray[np.float32]:
        obs = self.venv.reset()
        mask = np.ones(self.num_envs, dtype=bool)
        self._noise.reset(mask)
        if self.noise_scale_range is not None:
            self._noise.randomize_noise_scale(
                mask, self.noise_scale_range[0], self.noise_scale_range[1]
            )
        return self._apply_noise(obs)

    def step_wait(self) -> tuple[
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        obs, rewards, dones, infos = self.venv.step_wait()

        # Reset noise state for done envs
        if np.any(dones):
            self._noise.reset(dones)
            if self.noise_scale_range is not None:
                self._noise.randomize_noise_scale(
                    dones, self.noise_scale_range[0], self.noise_scale_range[1]
                )

        noisy_obs = self._apply_noise(obs)
        return noisy_obs, rewards, dones, infos

    def _apply_noise(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply perception noise to clean observations."""
        clean_obs = obs.astype(np.float64)

        # Compute distance-to-gate from obs (position in gate frame is obs[:, 0:3])
        d2g = np.linalg.norm(clean_obs[:, 0:3], axis=1)

        # Corrupt state-level features (position, euler)
        noisy_obs = self._noise.corrupt_state(clean_obs, d2g)

        # Apply HMM dropout
        noisy_obs = self._noise.apply_dropout(noisy_obs)

        if not self.asymmetric:
            return noisy_obs.astype(np.float32)

        # Asymmetric: [noisy_24D, clean_24D, metadata_2D]
        dt_det = self._noise.steps_since_detection / 100.0  # (n_envs,)
        det_conf = self._noise.det_confidence(d2g)            # (n_envs,)
        metadata = np.column_stack([dt_det, det_conf])        # (n_envs, 2)

        asymmetric_obs = np.concatenate(
            [noisy_obs, clean_obs, metadata], axis=1
        )
        return asymmetric_obs.astype(np.float32)


class CornerNoisePerceptionWrapper:
    """Hydra-instantiable perception wrapper that wraps a VecEnv with corner noise.

    Implements the ``ObservationWrapper`` protocol: ``wrap(env) -> env``.
    """

    def __init__(
        self,
        noise_scale: float = 1.0,
        dropout_onset: Optional[float] = None,
        dropout_continuation: float = 0.7,
        asymmetric: bool = False,
        noise_scale_range: Optional[list[float]] = None,
    ) -> None:
        self.noise_scale = noise_scale
        self.dropout_onset = dropout_onset
        self.dropout_continuation = dropout_continuation
        self.asymmetric = asymmetric
        self.noise_scale_range = (
            tuple(noise_scale_range) if noise_scale_range else None
        )

    def wrap(self, env: VecEnv) -> VecEnv:
        """Wrap *env* with corner noise perception augmentation."""
        return VecCornerNoise(
            env,
            noise_scale=self.noise_scale,
            dropout_onset=self.dropout_onset,
            dropout_continuation=self.dropout_continuation,
            asymmetric=self.asymmetric,
            noise_scale_range=self.noise_scale_range,
        )
