"""Abstract interface for sim environment factories.

Defines the ``EnvFactory`` protocol so that training orchestration can
construct training and evaluation ``VecEnv`` instances without coupling
to a specific physics backend.  Concrete implementations (e.g.
``NumpyQuadEnvFactory``) are selected via Hydra ``_target_`` in the sim
config group.
"""

from __future__ import annotations

from typing import Protocol

from omegaconf import DictConfig
from stable_baselines3.common.vec_env import VecEnv


class EnvFactory(Protocol):
    """Protocol for sim-backend environment factories.

    Implementations receive sim-specific constructor args at ``__init__``
    time (via Hydra recursive instantiation) and expose two methods that
    accept only the *training-level* knobs: domain-randomization config
    and reward config.
    """

    def make_vec_env(
        self,
        domain_rand_cfg: DictConfig,
        reward_cfg: DictConfig,
    ) -> VecEnv:
        """Build a training ``VecEnv``.

        Args:
            domain_rand_cfg: Domain-randomization settings (``domain_rand``
                config group).  Implementations should respect the
                ``enabled`` flag.
            reward_cfg: Reward settings (``reward`` config group), including
                ``weights``, ``v_max``, ``action_smoothness_threshold``, etc.

        Returns:
            A Stable-Baselines3-compatible ``VecEnv``.
        """
        ...

    def make_eval_env(
        self,
        domain_rand_cfg: DictConfig,
        reward_cfg: DictConfig,
        n_envs: int,
    ) -> VecEnv:
        """Build an evaluation ``VecEnv``.

        Evaluation envs typically disable domain randomization and use
        a smaller ``n_envs`` count.

        Args:
            domain_rand_cfg: Domain-randomization settings.  Implementations
                should force ``enabled=False`` regardless of what is passed.
            reward_cfg: Reward settings.
            n_envs: Number of parallel evaluation environments.

        Returns:
            A Stable-Baselines3-compatible ``VecEnv``.
        """
        ...
