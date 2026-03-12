"""Perception noise model simulating GateNet+EKF errors with HMM dropout."""

from __future__ import annotations

import numpy as np


class PerceptionNoise:
    """Simulates realistic GateNet+EKF perception errors.

    Adds distance-scaled Gaussian noise to position/euler dims and applies
    a 2-state HMM dropout model that holds stale observations on dropout
    while keeping IMU dims fresh.
    """

    # Observation indices that are always fresh (IMU / motor speeds).
    _IMU_SLICES: list[slice] = [slice(9, 15), slice(19, 23)]
    # Observation indices subject to dropout (perception dims).
    _PERCEPTION_SLICES: list[slice] = [slice(0, 9), slice(15, 19)]

    def __init__(
        self,
        n_envs: int,
        seed: int,
        noise_scale: float = 1.0,
        dropout_onset: float | None = None,
        dropout_continuation: float = 0.7,
        n_corners: int = 4,
        n_gates_visible: int = 1,
    ) -> None:
        self.n_envs = n_envs
        self.noise_scale = noise_scale
        self.dropout_onset_fixed = dropout_onset
        self.dropout_continuation = dropout_continuation
        self.n_corners = n_corners
        self.n_gates_visible = n_gates_visible

        self._rng = np.random.default_rng(seed)

        # Per-env noise scale (domain randomization support).
        self._per_env_scale = np.full(n_envs, noise_scale)

        # HMM state: True = currently in dropout.
        self._hmm_state = np.zeros(n_envs, dtype=bool)
        # Per-env onset probability.
        self._p_onset = self._sample_onset(n_envs)
        # Last valid observation (initialized on first apply_dropout call).
        self._last_valid_obs: np.ndarray | None = None

        # Metadata: steps since last valid detection per env.
        self._steps_since_detection = np.zeros(n_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def steps_since_detection(self) -> np.ndarray:
        """Per-env counter: steps since last valid (non-dropout) observation."""
        return self._steps_since_detection

    def det_confidence(self, d2g: np.ndarray) -> np.ndarray:
        """Detection confidence as a function of distance-to-gate.

        Returns:
            (n_envs,) array with values in (0, 1]: ``1 / (1 + d2g)``.
        """
        return 1.0 / (1.0 + d2g)

    def randomize_noise_scale(self, env_mask: np.ndarray, lo: float, hi: float) -> None:
        """Randomize per-env noise scale on reset."""
        n = int(env_mask.sum())
        if n > 0:
            self._per_env_scale[env_mask] = self._rng.uniform(lo, hi, size=n)

    def reset(self, env_mask: np.ndarray) -> None:
        """Clear HMM state and last_valid_obs for masked envs."""
        self._hmm_state[env_mask] = False
        self._steps_since_detection[env_mask] = 0
        if self._last_valid_obs is not None:
            self._last_valid_obs[env_mask] = 0.0
        # Re-randomize onset for reset envs if not fixed.
        n_reset = int(np.sum(env_mask))
        if n_reset > 0:
            self._p_onset[env_mask] = self._sample_onset(n_reset)

    def corrupt_state(self, state: np.ndarray, d2g: np.ndarray) -> np.ndarray:
        """Add distance-scaled Gaussian noise to position and euler angles.

        Args:
            state: (n_envs, state_dim) array.
            d2g: (n_envs,) distance-to-gate per env.

        Returns:
            A copy of state with noise on [0:3] (position) and [6:9] (euler).
        """
        if self.noise_scale == 0.0:
            return state

        out = state.copy()
        d2g_sq = d2g ** 2
        denom = self.n_corners ** 2 * self.n_gates_visible

        # Position noise.
        sigma_pos = np.sqrt(0.02 * d2g_sq / denom) * self._per_env_scale
        out[:, 0:3] += self._rng.normal(scale=sigma_pos[:, None], size=(self.n_envs, 3))

        # Euler noise.
        sigma_euler = np.sqrt(0.01 * d2g_sq / denom) * self._per_env_scale
        out[:, 6:9] += self._rng.normal(scale=sigma_euler[:, None], size=(self.n_envs, 3))

        return out

    def apply_dropout(self, obs: np.ndarray) -> np.ndarray:
        """Apply HMM dropout to observation.

        First call initialises last_valid_obs. On dropout, perception dims
        are held from last valid while IMU dims stay fresh.

        Args:
            obs: (n_envs, obs_dim) array.

        Returns:
            A (possibly modified) copy of obs.
        """
        if self._last_valid_obs is None:
            # First call: everything passes through, store as last valid.
            self._last_valid_obs = obs.copy()
            self._steps_since_detection[:] = 0
            return obs.copy()

        # Advance HMM.
        drop = self._step_hmm()

        out = obs.copy()

        if np.any(drop):
            # For dropped envs, hold perception dims from last valid.
            for s in self._PERCEPTION_SLICES:
                if s.start < obs.shape[1]:
                    actual = slice(s.start, min(s.stop, obs.shape[1]))
                    out[drop, actual] = self._last_valid_obs[drop, actual]

        # Update steps_since_detection counter.
        self._steps_since_detection[drop] += 1
        self._steps_since_detection[~drop] = 0

        # Update last_valid for non-dropped envs.
        not_drop = ~drop
        self._last_valid_obs[not_drop] = obs[not_drop]

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_onset(self, n: int) -> np.ndarray:
        """Sample dropout onset probabilities."""
        if self.dropout_onset_fixed is not None:
            return np.full(n, self.dropout_onset_fixed)
        return self._rng.uniform(0.05, 0.50, size=n)

    def _step_hmm(self) -> np.ndarray:
        """Advance the 2-state Markov chain and return dropout mask."""
        u = self._rng.uniform(size=self.n_envs)

        # Transition probabilities depend on current state.
        # If currently dropped: P(stay dropped) = dropout_continuation
        # If currently valid:   P(become dropped) = p_onset
        threshold = np.where(
            self._hmm_state,
            self.dropout_continuation,
            self._p_onset,
        )
        self._hmm_state = u < threshold
        return self._hmm_state
