"""Identity (no-op) perception wrapper.

Used when ``perception: none`` is configured — returns the env unchanged
so the training pipeline always has a uniform wrapper interface.
"""

from __future__ import annotations

from stable_baselines3.common.vec_env import VecEnv


class IdentityWrapper:
    """No-op observation wrapper that returns the environment unchanged."""

    def wrap(self, env: VecEnv) -> VecEnv:
        """Return *env* as-is (no perception augmentation)."""
        return env
