"""Smoke test: RecurrentPPO with LSTM trains without crashing."""
import pytest


class TestRecurrentSmoke:
    def test_recurrent_ppo_trains_200_steps(self):
        """RecurrentPPO + LSTM should complete a short training run."""
        from sim.tracks import build_figure8_track
        from sim.envs.gate_race_env import GateRaceEnv
        from sim.envs.vec_env_adapter import VecEnvAdapter
        from control.algorithms.ppo import PPO

        env = VecEnvAdapter(GateRaceEnv(
            track=build_figure8_track(),
            n_envs=4,
            dt=0.01,
            max_steps=100,
            action_mode="trpy",
            n_lookahead_gates=3,
            reward_weights={
                "gate_passage": 1.5, "gate_progress": 1.0,
                "crash_penalty": 10.0, "gate_centering": 3.0,
            },
        ))

        ppo = PPO(
            recurrent=True,
            lstm_hidden_size=64,
            n_lstm_layers=1,
            n_steps=100,
            batch_size=100,
            n_epochs=2,
            net_arch={"pi": [32], "vf": [32]},
        )
        ppo.train(env, total_timesteps=200)
        assert ppo._model is not None

        # Verify predict works
        import numpy as np
        obs = env.reset()
        action, state = ppo.predict(obs, deterministic=True)
        assert action.shape == (4, 4)  # 4 envs, 4 actions

        env.close()
