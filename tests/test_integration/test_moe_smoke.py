"""Smoke test: MoE policy trains without crashing."""
import pytest


class TestMoESmoke:
    def test_moe_trains_200_steps(self):
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
            n_lookahead_gates=2,
        ))

        ppo = PPO(
            moe=True,
            n_experts=4,
            expert_hidden_dim=64,
            top_k=2,
            balance_coef=0.01,
            n_steps=100,
            batch_size=100,
            n_epochs=2,
        )
        ppo.train(env, total_timesteps=200)
        assert ppo._model is not None

        import numpy as np
        obs = env.reset()
        action, _ = ppo.predict(obs, deterministic=True)
        assert action.shape == (4, 4)

        env.close()
