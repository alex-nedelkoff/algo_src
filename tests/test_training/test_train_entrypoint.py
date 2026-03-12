"""Tests for the training entrypoint (training/ module).

Exercises the NumpyQuadEnvFactory + build_ppo path that replaces the
old ``control.__main__`` monolith.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

import hydra.utils
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from training.loops.rl import build_ppo


# Use absolute config path for Hydra
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _make_env(cfg):
    """Build a VecEnvAdapter from Hydra config via factory pattern."""
    factory = hydra.utils.instantiate(cfg.sim)
    return factory.make_vec_env(cfg.domain_rand, cfg.reward)


@pytest.fixture(autouse=True)
def clear_hydra():
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


class TestBuildEnv:
    """Test environment construction via NumpyQuadEnvFactory."""

    def test_build_env_returns_vec_env(self) -> None:
        from sim.envs.vec_env_adapter import VecEnvAdapter

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["sim.n_envs=2"])
            env = _make_env(cfg)
            assert isinstance(env, VecEnvAdapter)
            assert env.num_envs == 2
            env.close()

    def test_build_env_uses_config_params(self) -> None:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["sim.n_envs=2"])
            env = _make_env(cfg)
            assert env.env.params.mass > 0.5, (
                f"Expected racing quad mass, got {env.env.params.mass}"
            )
            env.close()

    def test_build_env_obs_shape(self) -> None:
        from sim.envs.gate_race_env import OBS_DIM

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="train", overrides=["sim.n_envs=2"])
            env = _make_env(cfg)
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


class TestDomainRandWiring:
    """Test domain randomization wiring via NumpyQuadEnvFactory."""

    def test_build_env_wires_domain_randomizer(self) -> None:
        """Factory wires domain_randomizer to GateRaceEnv when enabled."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "sim.n_envs=2",
                    "domain_rand.enabled=true",
                ],
            )
            # Verify the config has params (from uniform_30pct default)
            assert "params" in cfg.domain_rand
            env = _make_env(cfg)
            assert env.env._domain_randomizer is not None
            assert len(env.env._domain_randomizer.config) > 0
            env.close()

    def test_build_env_no_domain_rand_when_disabled(self) -> None:
        """Factory passes None when domain_rand is disabled."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "sim.n_envs=2",
                    "domain_rand.enabled=false",
                ],
            )
            env = _make_env(cfg)
            assert env.env._domain_randomizer is None
            env.close()


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
            env = _make_env(cfg)
            ppo = build_ppo(cfg)
            ppo.train(env, total_timesteps=cfg.total_timesteps)

            obs = env.reset()
            action, _ = ppo.predict(obs[0])
            assert action.shape == (4,)
            env.close()
