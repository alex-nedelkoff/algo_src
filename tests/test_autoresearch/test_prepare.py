"""Tests for autoresearch prepare module."""
from __future__ import annotations

import pytest

from autoresearch.prepare import register_reward_preset
from sim.rewards_mavlab import PRESETS, RewardPreset


class TestRegisterRewardPreset:
    """Test dynamic reward preset registration."""

    def test_registers_new_preset(self) -> None:
        weights = {
            "lambda_gate": 15.0,
            "lambda_prog": 2.0,
            "lambda_rate": 0.005,
            "lambda_offset": 1.0,
            "lambda_perc": 0.5,
            "lambda_delta_u": 0.01,
            "lambda_crash": 20.0,
            "lambda_alive": 0.1,
            "v_max": 10.0,
        }
        register_reward_preset("test_preset", weights)
        assert "test_preset" in PRESETS
        assert isinstance(PRESETS["test_preset"], RewardPreset)
        assert PRESETS["test_preset"].lambda_gate == 15.0
        # Cleanup
        del PRESETS["test_preset"]

    def test_rejects_missing_keys(self) -> None:
        with pytest.raises(ValueError, match="Missing"):
            register_reward_preset("bad", {"lambda_gate": 1.0})

    def test_rejects_extra_keys(self) -> None:
        weights = {
            "lambda_gate": 1.0, "lambda_prog": 1.0, "lambda_rate": 0.0,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.0,
            "lambda_crash": 1.0, "lambda_alive": 0.0, "v_max": 0.0,
            "extra_key": 99.0,
        }
        with pytest.raises(ValueError, match="Extra"):
            register_reward_preset("bad", weights)


class TestMakeTrainingEnv:
    """Test environment construction."""

    def test_creates_vec_env(self) -> None:
        from autoresearch.prepare import make_training_env
        weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        register_reward_preset("test_env", weights)
        env = make_training_env(n_envs=2, dr_percentage=0.0, preset_name="test_env", seed=42)
        assert env.num_envs == 2
        assert env.observation_space.shape == (24,)
        assert env.action_space.shape == (4,)
        env.close()
        del PRESETS["test_env"]

    def test_dr_percentage_applied(self) -> None:
        from autoresearch.prepare import make_training_env
        weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        register_reward_preset("test_dr", weights)
        env = make_training_env(n_envs=2, dr_percentage=0.3, preset_name="test_dr", seed=42)
        assert env.num_envs == 2
        env.close()
        del PRESETS["test_dr"]


class TestWrapEkf:
    """Test EKF wrapping."""

    def test_wraps_env_with_ekf(self) -> None:
        from autoresearch.prepare import make_training_env, wrap_ekf
        from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper
        weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        register_reward_preset("test_ekf", weights)
        env = make_training_env(n_envs=2, dr_percentage=0.0, preset_name="test_ekf", seed=42)
        wrapped = wrap_ekf(env, corner_noise_k=2.0, corner_dropout_onset=None)
        assert isinstance(wrapped, EKFVecEnvWrapper)
        assert wrapped.observation_space.shape == (24,)
        obs = wrapped.reset()
        assert obs.shape == (2, 24)
        wrapped.close()
        del PRESETS["test_ekf"]


class TestLoadAndConfigureModel:
    """Test model loading with hyperparameter overrides."""

    def test_overrides_hyperparameters(self, tmp_path) -> None:
        """Load a baseline checkpoint, override params, verify they stick."""
        from stable_baselines3 import PPO as SB3_PPO
        from autoresearch.prepare import make_training_env, load_and_configure_model

        weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        register_reward_preset("test_load", weights)
        env = make_training_env(n_envs=2, dr_percentage=0.0, preset_name="test_load", seed=42)

        # Create and save a dummy model
        dummy = SB3_PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=32, batch_size=32)
        ckpt = tmp_path / "dummy.zip"
        dummy.save(str(ckpt))

        # Load with overrides
        training_params = {
            "learning_rate": 1e-4,
            "ent_coef": 0.01,
            "clip_range": 0.3,
            "gae_lambda": 0.98,
            "gamma": 0.995,
        }
        model = load_and_configure_model(env, str(ckpt), training_params)
        assert model.learning_rate == 1e-4
        assert model.ent_coef == 0.01
        assert model.clip_range(0) == 0.3  # SB3 clip_range is a callable
        assert model.gae_lambda == 0.98
        assert model.gamma == 0.995
        env.close()
        del PRESETS["test_load"]


class TestRunEval:
    """Test evaluation harness."""

    def test_returns_expected_metrics(self, tmp_path) -> None:
        from stable_baselines3 import PPO as SB3_PPO
        from autoresearch.prepare import make_training_env, wrap_ekf, run_eval

        weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        register_reward_preset("test_eval", weights)
        env = make_training_env(n_envs=2, dr_percentage=0.0, preset_name="test_eval", seed=42)
        ekf_env = wrap_ekf(env, corner_noise_k=2.0)

        dummy = SB3_PPO("MlpPolicy", ekf_env, n_steps=32, batch_size=32)
        results = run_eval(dummy, ekf_env, n_episodes=5)

        assert "avg_gates" in results
        assert "crash_rate" in results
        assert "alt_std" in results
        assert "avg_steps" in results
        assert "max_gates" in results
        assert "score" in results
        assert isinstance(results["score"], float)
        # score = avg_gates - 2 * crash_rate
        expected_score = results["avg_gates"] - 2 * results["crash_rate"]
        assert abs(results["score"] - expected_score) < 5e-4
        ekf_env.close()
        del PRESETS["test_eval"]


class TestLogExperiment:
    """Test experiment logging to results.tsv."""

    def test_creates_tsv_with_header(self, tmp_path) -> None:
        from autoresearch.prepare import log_experiment

        tsv = tmp_path / "results.tsv"
        results = {
            "score": 5.0, "avg_gates": 6.0, "crash_rate": 0.5,
            "alt_std": 0.1, "avg_steps": 800.0, "max_gates": 10,
        }
        reward_weights = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        ekf_params = {"corner_noise_k": 2.0, "corner_dropout_onset": None}
        training_params = {
            "learning_rate": 1e-4, "ent_coef": 0.005, "clip_range": 0.2,
            "gae_lambda": 0.98, "gamma": 0.999,
        }
        domain_rand = {"percentage": 0.3}

        log_experiment(
            exp_id=0, seed=42, results=results,
            reward_weights=reward_weights, ekf_params=ekf_params,
            training_params=training_params, domain_rand=domain_rand,
            results_file=str(tsv),
        )

        lines = tsv.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert lines[0].startswith("exp_id\t")
        assert "lambda_gate" in lines[0]
        row = lines[1].split("\t")
        assert row[0] == "0"  # exp_id
        assert row[2] == "5.0"  # score

    def test_appends_to_existing_tsv(self, tmp_path) -> None:
        from autoresearch.prepare import log_experiment

        tsv = tmp_path / "results.tsv"
        results = {
            "score": 5.0, "avg_gates": 6.0, "crash_rate": 0.5,
            "alt_std": 0.1, "avg_steps": 800.0, "max_gates": 10,
        }
        rw = {
            "lambda_gate": 10.0, "lambda_prog": 1.0, "lambda_rate": 0.001,
            "lambda_offset": 0.0, "lambda_perc": 0.0, "lambda_delta_u": 0.001,
            "lambda_crash": 10.0, "lambda_alive": 0.0, "v_max": 0.0,
        }
        ekf = {"corner_noise_k": 2.0, "corner_dropout_onset": None}
        tp = {"learning_rate": 1e-4, "ent_coef": 0.005, "clip_range": 0.2,
              "gae_lambda": 0.98, "gamma": 0.999}
        dr = {"percentage": 0.3}

        log_experiment(0, 42, results, rw, ekf, tp, dr, str(tsv))
        log_experiment(1, 43, results, rw, ekf, tp, dr, str(tsv))

        lines = tsv.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert lines[2].split("\t")[0] == "1"
