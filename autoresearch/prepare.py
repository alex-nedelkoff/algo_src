"""Autoresearch eval harness — immutable.

Provides env construction, model loading, EKF wrapping, evaluation,
and reward preset registration for the autoresearch tuning loop.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim.rewards_mavlab import PRESETS, RewardPreset

EXPECTED_KEYS = frozenset(RewardPreset.__dataclass_fields__.keys())


def register_reward_preset(name: str, weights: dict[str, float]) -> None:
    """Inject a new RewardPreset into the global PRESETS dict.

    Validates that exactly the 9 required keys are present.
    """
    provided = set(weights.keys())
    missing = EXPECTED_KEYS - provided
    if missing:
        raise ValueError(f"Missing reward weight keys: {missing}")
    extra = provided - EXPECTED_KEYS
    if extra:
        raise ValueError(f"Extra reward weight keys: {extra}")
    PRESETS[name] = RewardPreset(**weights)


def make_training_env(
    n_envs: int,
    dr_percentage: float,
    preset_name: str,
    seed: int,
) -> Any:
    """Build a vectorized training environment.

    Constructs RateCtrlEnv directly (bypasses PlaygroundEnvFactory
    to avoid DictConfig overhead).
    """
    from sim.envs.rate_ctrl_env import RateCtrlEnv
    from sim.envs.vec_env_adapter import VecEnvAdapter

    inner = RateCtrlEnv(
        n_envs=n_envs,
        seed=seed,
        dr_percentage=dr_percentage,
        reward_preset=preset_name,
    )
    return VecEnvAdapter(inner)


def wrap_ekf(
    env: Any,
    corner_noise_k: float = 2.0,
    corner_dropout_onset: float | None = None,
) -> Any:
    """Apply EKF filtering wrapper to a VecEnv."""
    from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper

    return EKFVecEnvWrapper(
        env,
        corner_noise_k=corner_noise_k,
        corner_dropout_onset=corner_dropout_onset,
        provide_teacher_obs=False,
    )


def load_and_configure_model(
    env: Any,
    checkpoint_path: str,
    training_params: dict[str, float],
) -> Any:
    """Load an SB3 PPO checkpoint and override training hyperparameters."""
    from stable_baselines3 import PPO as SB3_PPO

    model = SB3_PPO.load(checkpoint_path, env=env, device="auto")
    model.learning_rate = training_params["learning_rate"]
    model.ent_coef = training_params["ent_coef"]
    clip_val = training_params["clip_range"]
    model.clip_range = lambda _, v=clip_val: v
    model.gae_lambda = training_params["gae_lambda"]
    model.gamma = training_params["gamma"]
    # Locked hyperparameters — not agent-editable but critical for wall-clock time.
    # Without these, SB3 defaults (n_steps=2048, batch_size=64) make training ~80x slower.
    model.n_steps = 1000
    model.batch_size = 5000
    return model


def save_checkpoint(model: Any, exp_id: int) -> str:
    """Save model checkpoint to autoresearch/checkpoints/exp_NNN.zip."""
    from pathlib import Path

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    path = ckpt_dir / f"exp_{exp_id:03d}.zip"
    model.save(str(path))
    return str(path)


def run_eval(
    model: Any,
    env: Any,
    n_episodes: int = 50,
) -> dict[str, float]:
    """Deterministic rollout evaluation.

    Uses the provided env (which should have the same EKF config as training).
    Returns avg_gates, crash_rate, alt_std, avg_steps, max_gates, and
    score = avg_gates - 2 * crash_rate.
    """
    gates_list: list[int] = []
    steps_list: list[int] = []
    crashes: list[float] = []
    alt_stds: list[float] = []

    episodes_done = 0
    obs = env.reset()
    ep_steps = np.zeros(env.num_envs, dtype=int)

    while episodes_done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        ep_steps += 1

        for i, info in enumerate(infos):
            ep = info.get("episode")
            if ep is not None:
                gates_list.append(ep.get("gates_passed", 0))
                steps_list.append(int(ep.get("l", ep_steps[i])))
                crashes.append(1.0 if info.get("crash", False) else 0.0)
                alt_stds.append(float(ep.get("alt_std", 0.0)))
                ep_steps[i] = 0
                episodes_done += 1

    avg_gates = float(np.mean(gates_list)) if gates_list else 0.0
    crash_rate = float(np.mean(crashes)) if crashes else 0.0
    alt_std = float(np.mean(alt_stds)) if alt_stds else 0.0
    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
    max_gates = int(np.max(gates_list)) if gates_list else 0
    score = avg_gates - 2.0 * crash_rate

    if not np.isfinite(score):
        score = -999.0

    return {
        "avg_gates": avg_gates,
        "crash_rate": crash_rate,
        "alt_std": alt_std,
        "avg_steps": avg_steps,
        "max_gates": max_gates,
        "score": round(score, 4),
    }


def log_experiment(
    exp_id: int,
    seed: int,
    results: dict[str, float],
    reward_weights: dict[str, float],
    ekf_params: dict[str, Any],
    training_params: dict[str, float],
    domain_rand: dict[str, float],
    results_file: str = "autoresearch/results.tsv",
) -> None:
    """Append one experiment row to results.tsv."""
    from datetime import datetime
    from pathlib import Path

    header_cols = [
        "exp_id", "timestamp", "score", "avg_gates", "crash_rate",
        "alt_std", "avg_steps", "max_gates", "seed",
        "lambda_gate", "lambda_prog", "lambda_rate", "lambda_offset",
        "lambda_perc", "lambda_delta_u", "lambda_crash", "lambda_alive",
        "v_max", "corner_noise_k", "corner_dropout_onset",
        "learning_rate", "ent_coef", "clip_range", "gae_lambda", "gamma",
        "dr_percentage",
    ]

    path = Path(results_file)
    write_header = not path.exists()

    row_vals = [
        str(exp_id),
        datetime.now().isoformat(timespec="seconds"),
        str(results["score"]),
        str(results["avg_gates"]),
        str(results["crash_rate"]),
        str(results["alt_std"]),
        str(results["avg_steps"]),
        str(results["max_gates"]),
        str(seed),
        str(reward_weights["lambda_gate"]),
        str(reward_weights["lambda_prog"]),
        str(reward_weights["lambda_rate"]),
        str(reward_weights["lambda_offset"]),
        str(reward_weights["lambda_perc"]),
        str(reward_weights["lambda_delta_u"]),
        str(reward_weights["lambda_crash"]),
        str(reward_weights["lambda_alive"]),
        str(reward_weights["v_max"]),
        str(ekf_params["corner_noise_k"]),
        str(ekf_params.get("corner_dropout_onset", "None")),
        str(training_params["learning_rate"]),
        str(training_params["ent_coef"]),
        str(training_params["clip_range"]),
        str(training_params["gae_lambda"]),
        str(training_params["gamma"]),
        str(domain_rand["percentage"]),
    ]

    with open(path, "a") as f:
        if write_header:
            f.write("\t".join(header_cols) + "\n")
        f.write("\t".join(row_vals) + "\n")
