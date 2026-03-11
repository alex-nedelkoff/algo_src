"""Tests for the training entrypoint (control/__main__.py)."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from control.__main__ import build_env, build_ppo


# Use absolute config path for Hydra
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@pytest.fixture(autouse=True)
def clear_hydra():
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


class TestBuildEnv:
    """Test environment construction from Hydra config."""

    def test_build_env_returns_vec_env(self) -> None:
        from sim.envs.vec_env_adapter import VecEnvAdapter

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["sim.n_envs=2"])
            env = build_env(cfg)
            assert isinstance(env, VecEnvAdapter)
            assert env.num_envs == 2
            env.close()

    def test_build_env_uses_config_params(self) -> None:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["sim.n_envs=2"])
            env = build_env(cfg)
            assert env.env.params.mass > 0.5, (
                f"Expected racing quad mass, got {env.env.params.mass}"
            )
            env.close()

    def test_build_env_obs_shape(self) -> None:
        from sim.envs.gate_race_env import OBS_DIM

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["sim.n_envs=2"])
            env = build_env(cfg)
            obs = env.reset()
            assert obs.shape == (2, OBS_DIM)
            env.close()


class TestBuildPPO:
    """Test PPO construction from Hydra config."""

    def test_build_ppo_returns_ppo(self) -> None:
        from control.algorithms.ppo import PPO

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train")
            ppo = build_ppo(cfg)
            assert isinstance(ppo, PPO)

    def test_build_ppo_uses_config_hyperparams(self) -> None:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train")
            ppo = build_ppo(cfg)
            assert ppo.gamma == 0.999
            assert ppo.batch_size == 5000
            assert ppo.n_steps == 1000
            assert ppo.ent_coef == 0.005


class TestTrainSmoke:
    """Smoke test: build env + PPO and train for a few steps."""

    def test_train_64_steps(self) -> None:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "sim.n_envs=2",
                    "total_timesteps=64",
                    "control.n_steps=32",
                    "control.batch_size=32",
                ],
            )
            env = build_env(cfg)
            ppo = build_ppo(cfg)
            ppo.train(env, total_timesteps=cfg.total_timesteps)

            obs = env.reset()
            action, _ = ppo.predict(obs[0])
            assert action.shape == (4,)
            env.close()
