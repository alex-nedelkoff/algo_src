"""Tests for recurrent GCNet policy components."""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from control.policies.recurrent_gcnet import RecurrentGCNetExtractor


class TestRecurrentGCNetExtractor:
    @pytest.fixture
    def obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

    def test_output_shape(self, obs_space):
        ext = RecurrentGCNetExtractor(obs_space)
        obs = torch.randn(4, 32)
        out = ext(obs)
        assert out.shape == (4, 64)

    def test_features_dim(self, obs_space):
        ext = RecurrentGCNetExtractor(obs_space)
        assert ext.features_dim == 64

    def test_custom_hidden_dims(self, obs_space):
        ext = RecurrentGCNetExtractor(obs_space, hidden_dims=(128, 128))
        obs = torch.randn(2, 32)
        out = ext(obs)
        assert out.shape == (2, 128)
        assert ext.features_dim == 128

    def test_different_obs_dim(self):
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        ext = RecurrentGCNetExtractor(obs_space, hidden_dims=(64, 64))
        obs = torch.randn(3, 28)
        out = ext(obs)
        assert out.shape == (3, 64)


class TestRecurrentPPOWrapper:
    def test_recurrent_flag_default_false(self):
        from control.algorithms.ppo import PPO
        ppo = PPO()
        assert ppo.recurrent is False

    def test_recurrent_creates_recurrent_model(self):
        from control.algorithms.ppo import PPO
        from sim.tracks import build_figure8_track
        from sim.envs.gate_race_env import GateRaceEnv
        from sim.envs.vec_env_adapter import VecEnvAdapter

        env = VecEnvAdapter(GateRaceEnv(
            track=build_figure8_track(), n_envs=2, dt=0.01, max_steps=50,
            action_mode="trpy", n_lookahead_gates=3,
        ))
        ppo = PPO(
            recurrent=True, lstm_hidden_size=64, n_lstm_layers=1,
            net_arch={"pi": [32], "vf": [32]},
        )
        model = ppo._create_model(env)
        from sb3_contrib import RecurrentPPO as SB3RecurrentPPO
        assert isinstance(model, SB3RecurrentPPO)
        env.close()

    def test_recurrent_train_100_steps(self):
        from control.algorithms.ppo import PPO
        from sim.tracks import build_figure8_track
        from sim.envs.gate_race_env import GateRaceEnv
        from sim.envs.vec_env_adapter import VecEnvAdapter

        env = VecEnvAdapter(GateRaceEnv(
            track=build_figure8_track(), n_envs=2, dt=0.01, max_steps=50,
            action_mode="trpy", n_lookahead_gates=3,
        ))
        ppo = PPO(
            recurrent=True, lstm_hidden_size=64, n_lstm_layers=1,
            n_steps=50, batch_size=50, n_epochs=1,
            net_arch={"pi": [32], "vf": [32]},
        )
        ppo.train(env, total_timesteps=100)
        assert ppo._model is not None
        env.close()
