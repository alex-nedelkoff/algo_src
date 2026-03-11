"""Tests for VecEnvAdapter wrapping GateRaceEnv as SB3 VecEnv."""

from __future__ import annotations

import numpy as np
import pytest

from stable_baselines3.common.vec_env import VecEnv

from sim.envs.gate_race_env import GateRaceEnv, OBS_DIM
from sim.envs.vec_env_adapter import VecEnvAdapter


@pytest.fixture
def adapter() -> VecEnvAdapter:
    """VecEnvAdapter wrapping a small GateRaceEnv."""
    env = GateRaceEnv(n_envs=4)
    return VecEnvAdapter(env)


class TestVecEnvAdapterInterface:
    """Adapter should satisfy the SB3 VecEnv interface."""

    def test_is_vec_env(self, adapter: VecEnvAdapter) -> None:
        assert isinstance(adapter, VecEnv)

    def test_num_envs(self, adapter: VecEnvAdapter) -> None:
        assert adapter.num_envs == 4

    def test_observation_space(self, adapter: VecEnvAdapter) -> None:
        assert adapter.observation_space.shape == (OBS_DIM,)

    def test_action_space(self, adapter: VecEnvAdapter) -> None:
        assert adapter.action_space.shape == (4,)


class TestVecEnvAdapterReset:
    """Reset should return obs in the shape SB3 expects."""

    def test_reset_shape(self, adapter: VecEnvAdapter) -> None:
        obs = adapter.reset()
        assert obs.shape == (4, OBS_DIM)

    def test_reset_dtype(self, adapter: VecEnvAdapter) -> None:
        obs = adapter.reset()
        assert obs.dtype == np.float32

    def test_reset_finite(self, adapter: VecEnvAdapter) -> None:
        obs = adapter.reset()
        assert np.all(np.isfinite(obs))


class TestVecEnvAdapterStep:
    """Step should return (obs, rewards, dones, infos) per SB3 convention."""

    def test_step_shapes(self, adapter: VecEnvAdapter) -> None:
        adapter.reset()
        actions = np.zeros((4, 4), dtype=np.float32)
        obs, rewards, dones, infos = adapter.step(actions)
        assert obs.shape == (4, OBS_DIM)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4

    def test_step_infos_are_dicts(self, adapter: VecEnvAdapter) -> None:
        adapter.reset()
        actions = np.zeros((4, 4), dtype=np.float32)
        _, _, _, infos = adapter.step(actions)
        for info in infos:
            assert isinstance(info, dict)

    def test_terminal_observation_in_info(self, adapter: VecEnvAdapter) -> None:
        """When an env is done, info should have 'terminal_observation'."""
        adapter.reset()
        action = -np.ones((4, 4), dtype=np.float32)
        found_terminal = False
        for _ in range(2000):
            obs, rewards, dones, infos = adapter.step(action)
            if np.any(dones):
                for i, done in enumerate(dones):
                    if done:
                        assert "terminal_observation" in infos[i]
                        assert infos[i]["terminal_observation"].shape == (OBS_DIM,)
                found_terminal = True
                break

        assert found_terminal, "Expected at least one termination"

    def test_step_100_steps(self, adapter: VecEnvAdapter) -> None:
        """100 steps without crash (adapter-level)."""
        adapter.reset()
        for _ in range(100):
            actions = np.zeros((4, 4), dtype=np.float32)
            obs, rewards, dones, infos = adapter.step(actions)
            assert np.all(np.isfinite(obs))
