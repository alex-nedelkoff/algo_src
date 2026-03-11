"""SB3 VecEnv adapter for internally-vectorized GateRaceEnv.

Wraps a single GateRaceEnv(n_envs=N) as an SB3-compatible VecEnv,
preserving our fast numpy vectorization without subprocess overhead.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from stable_baselines3.common.vec_env import VecEnv

from sim.envs.gate_race_env import GateRaceEnv


class VecEnvAdapter(VecEnv):
    """Adapt an internally-vectorized GateRaceEnv to the SB3 VecEnv interface.

    SB3 expects:
      - step() returns (obs, rewards, dones, infos) where infos is a list of dicts
      - dones = terminated | truncated
      - Done envs have info["terminal_observation"] with the pre-reset obs
      - reset() returns just obs

    Args:
        env: A GateRaceEnv instance (with n_envs >= 1).
    """

    def __init__(self, env: GateRaceEnv) -> None:
        self.env = env
        super().__init__(
            num_envs=env.n_envs,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        self._actions: NDArray[np.float32] | None = None
        self._pending_seed: int | None = None

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
                # Episode metrics for SB3 logging
                if episode_metrics is not None:
                    env_info["episode"] = {
                        "gates_passed": int(episode_metrics["gates_passed"][i]),
                        "laps_completed": int(episode_metrics["laps_completed"][i]),
                        "l": int(episode_metrics["episode_length"][i]),
                    }
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
