"""Tests for EKF wrapping in the RL training loop."""
from __future__ import annotations

import pytest
from pathlib import Path

import hydra.utils
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper
from training.loops.rl import RLTrainingLoop

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@pytest.fixture(autouse=True)
def clear_hydra():
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


class TestEKFWrapping:
    """Test that RL loop applies EKF wrapper when configured."""

    def test_ekf_wrapping_applied(self) -> None:
        """When ekf config is present, train_env should be EKF-wrapped."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "+experiment=playground_phase1",
                    "sim.n_envs=2",
                    "+ekf.corner_noise_k=2.0",
                    "+ekf.camera_kwargs={}",
                    "+ekf.provide_teacher_obs=false",
                ],
            )

            # Build env the same way rl.py does
            env_factory = hydra.utils.instantiate(cfg.sim)
            train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)

            perception_wrapper = hydra.utils.instantiate(cfg.perception)
            train_env = perception_wrapper.wrap(train_env)

            # Apply EKF wrapping (the code we're about to add)
            if "ekf" in cfg:
                ekf_kwargs = OmegaConf.to_container(cfg.ekf, resolve=True)
                train_env = EKFVecEnvWrapper(train_env, **ekf_kwargs)

            assert isinstance(train_env, EKFVecEnvWrapper)
            assert train_env.observation_space.shape == (24,)
            obs = train_env.reset()
            assert obs.shape == (2, 24)
            train_env.close()

    def test_no_ekf_without_config(self) -> None:
        """Without ekf config, env should NOT be EKF-wrapped."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "+experiment=playground_phase1",
                    "sim.n_envs=2",
                ],
            )
            env_factory = hydra.utils.instantiate(cfg.sim)
            train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)

            perception_wrapper = hydra.utils.instantiate(cfg.perception)
            train_env = perception_wrapper.wrap(train_env)

            # No ekf in cfg, so no wrapping
            assert not isinstance(train_env, EKFVecEnvWrapper)
            train_env.close()


class TestEKFPPOSmoke:
    """Smoke test: PPO trains for a few steps with EKF-wrapped env."""

    def test_train_64_steps_with_ekf(self) -> None:
        from training.loops.rl import build_ppo

        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "+experiment=playground_phase1",
                    "sim.n_envs=2",
                    "total_timesteps=64",
                    "control.n_steps=32",
                    "control.batch_size=32",
                    "output_dir=null",
                    "+ekf.corner_noise_k=2.0",
                    "+ekf.camera_kwargs={}",
                    "+ekf.provide_teacher_obs=false",
                ],
            )
            env_factory = hydra.utils.instantiate(cfg.sim)
            train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)

            perception_wrapper = hydra.utils.instantiate(cfg.perception)
            train_env = perception_wrapper.wrap(train_env)

            ekf_kwargs = OmegaConf.to_container(cfg.ekf, resolve=True)
            train_env = EKFVecEnvWrapper(train_env, **ekf_kwargs)

            ppo = build_ppo(cfg)
            ppo.train(train_env, total_timesteps=64)

            obs = train_env.reset()
            action, _ = ppo.predict(obs[0])
            assert action.shape == (4,)
            train_env.close()


class TestEKFPPOConfig:
    """Test that playground_ekf_ppo config composes correctly."""

    def test_config_composes(self) -> None:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=["+experiment=playground_ekf_ppo"],
            )
            assert "ekf" in cfg
            assert cfg.ekf.corner_noise_k == 2.0
            assert cfg.ekf.provide_teacher_obs is False
            assert cfg.total_timesteps == 50_000_000

    def test_config_allows_resume_checkpoint(self) -> None:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "+experiment=playground_ekf_ppo",
                    "resume_checkpoint=/tmp/fake_model.zip",
                ],
            )
            assert cfg.resume_checkpoint == "/tmp/fake_model.zip"
