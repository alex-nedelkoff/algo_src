"""Protocol for perception observation wrappers.

All perception wrappers expose a ``wrap(env)`` method that takes an SB3
``VecEnv`` and returns a (possibly wrapped) ``VecEnv``.  This lets Hydra
instantiate arbitrary wrappers via ``_target_`` without the training loop
knowing the concrete type.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from stable_baselines3.common.vec_env import VecEnv


@runtime_checkable
class ObservationWrapper(Protocol):
    """Protocol for perception wrappers used in the training pipeline."""

    def wrap(self, env: VecEnv) -> VecEnv:
        """Wrap *env* with perception augmentation and return the result."""
        ...
