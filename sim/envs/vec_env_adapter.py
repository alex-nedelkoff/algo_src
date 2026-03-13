"""SB3 VecEnv adapter for internally-vectorized racing environments.

Wraps a single vectorized env (n_envs=N) as an SB3-compatible VecEnv,
preserving fast numpy vectorization without subprocess overhead.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from stable_baselines3.common.vec_env import VecEnv

from metrics.contract import validate_episode_metrics, ContractViolation


class VecEnvAdapter(VecEnv):
    """Adapt an internally-vectorized racing env to the SB3 VecEnv interface.

    SB3 expects:
      - step() returns (obs, rewards, dones, infos) where infos is a list of dicts
      - dones = terminated | truncated
      - Done envs have info["terminal_observation"] with the pre-reset obs
      - reset() returns just obs

    On first episode completion, validates the episode info dict against the
    metrics contract (one-shot). Raises ContractViolation on mismatch.

    Args:
        env: Any vectorized racing env with n_envs, observation_space,
            action_space, reset(), step(), and close() attributes.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        super().__init__(
            num_envs=env.n_envs,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        self._actions: NDArray[np.float32] | None = None
        self._pending_seed: int | None = None
        self._contract_validated = False

    def reset(self) -> NDArray[np.float32]:
        """Reset all environments and return observations."""
        obs, _ = self.env.reset(seed=self._pending_seed)
        self._pending_seed = None
        if obs.ndim == 1:
            obs = obs[None, :]
        return obs

    def step_async(self, actions: NDArray[np.float32]) -> None:
        """Store actions for step_wait()."""
        self._actions = actions

    def step_wait(self) -> tuple[
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        """Execute stored actions and return SB3-format results."""
        assert self._actions is not None, "Call step_async() before step_wait()"

        obs, rewards, terminated, truncated, info = self.env.step(self._actions)

        if obs.ndim == 1:
            obs = obs[None, :]
        if rewards.ndim == 0:
            rewards = rewards[None]
        if terminated.ndim == 0:
            terminated = terminated[None]
        if truncated.ndim == 0:
            truncated = truncated[None]

        dones = terminated | truncated

        # terminal_obs is always (n_envs, OBS_DIM) from _compute_obs_batched()
        terminal_obs = info.get("terminal_obs")
        episode_metrics = info.get("episode")
        infos: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            env_info: dict[str, Any] = {}
            if dones[i] and terminal_obs is not None:
                env_info["terminal_observation"] = terminal_obs[i]
                env_info["TimeLimit.truncated"] = bool(truncated[i]) and not bool(terminated[i])
                if episode_metrics is not None:
                    # Extract per-env scalars from batched arrays
                    ep = {
                        "r": float(episode_metrics["r"][i]),
                        "l": int(episode_metrics["l"][i]),
                        "effective_dt": float(episode_metrics["effective_dt"])
                            if np.ndim(episode_metrics["effective_dt"]) == 0
                            else float(episode_metrics["effective_dt"][i]),
                        "gates_passed": int(episode_metrics["gates_passed"][i]),
                        "laps_completed": int(episode_metrics["laps_completed"][i]),
                        "termination": str(episode_metrics["termination"][i]),
                        "success": bool(episode_metrics["success"][i]),
                        "success_criterion": str(episode_metrics["success_criterion"])
                            if isinstance(episode_metrics["success_criterion"], str)
                            else str(episode_metrics["success_criterion"][i]),
                        "avg_speed": float(episode_metrics["avg_speed"][i]),
                        "first_gate_step": int(episode_metrics["first_gate_step"][i]),
                        "reward_components": episode_metrics["reward_components"][i]
                            if isinstance(episode_metrics["reward_components"][i], dict)
                            else dict(episode_metrics["reward_components"][i]),
                    }
                    if not self._contract_validated:
                        env_name = type(self.env).__name__
                        validate_episode_metrics(ep, env_name)
                        self._contract_validated = True
                    env_info["episode"] = ep
            infos.append(env_info)

        return obs, rewards, dones, infos

    def close(self) -> None:
        pass

    def render(self, mode: str | None = None) -> None:
        pass

    def env_is_wrapped(
        self, wrapper_class: type, indices: Sequence[int] | None = None
    ) -> list[bool]:
        return [False] * self.num_envs

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Sequence[int] | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        result = getattr(self.env, method_name)(*method_args, **method_kwargs)
        return [result] * (len(indices) if indices else self.num_envs)

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None) -> list[Any]:
        val = getattr(self.env, attr_name)
        return [val] * (len(indices) if indices else self.num_envs)

    def set_attr(
        self, attr_name: str, value: Any, indices: Sequence[int] | None = None
    ) -> None:
        setattr(self.env, attr_name, value)

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._pending_seed = seed
        return [seed] * self.num_envs
