"""Autoresearch train.py — the ONLY file the agent edits.

The agent modifies the 4 parameter dicts below. Everything below
the FIXED line must not be touched.
"""

# ---- EDITABLE BELOW ---- #

REWARD_WEIGHTS = {
    "lambda_gate": 10.0,      # 1.0–50.0
    "lambda_prog": 1.0,       # 0.1–5.0
    "lambda_rate": 0.001,     # 0.0–0.01
    "lambda_offset": 0.0,     # 0.0–5.0
    "lambda_perc": 0.0,       # 0.0–1.0
    "lambda_delta_u": 0.001,  # 0.0–1.0
    "lambda_crash": 10.0,     # 1.0–50.0
    "lambda_alive": 0.0,      # 0.0–1.0
    "v_max": 0.0,             # 0.0–30.0
}

EKF_PARAMS = {
    "corner_noise_k": 2.0,          # 0.5–5.0
    "corner_dropout_onset": None,   # None or 0.0–1.0
}

TRAINING_PARAMS = {
    "learning_rate": 3e-4,    # 1e-5–1e-3
    "ent_coef": 0.005,        # 0.0–0.05
    "clip_range": 0.2,        # 0.1–0.4
    "gae_lambda": 0.95,       # 0.9–0.99
    "gamma": 0.999,           # 0.99–0.9999
}

DOMAIN_RAND = {
    "percentage": 0.3,        # 0.0–0.5
}

# ---- FIXED BELOW ---- #

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.prepare import (
    register_reward_preset,
    make_training_env,
    wrap_ekf,
    load_and_configure_model,
    run_eval,
)

CHECKPOINT = "../playground/runs/phase4_ekf_ppo_v2/final_model.zip"
N_STEPS = 10_000_000
N_ENVS = 500


def main(exp_id: int) -> None:
    seed = 42 + exp_id
    preset_name = f"autoresearch_exp{exp_id}"
    register_reward_preset(preset_name, REWARD_WEIGHTS)

    env = make_training_env(N_ENVS, DOMAIN_RAND["percentage"], preset_name, seed)
    env = wrap_ekf(env, **EKF_PARAMS)
    model = load_and_configure_model(env, CHECKPOINT, TRAINING_PARAMS)
    model.learn(total_timesteps=N_STEPS)

    eval_env = make_training_env(10, 0.0, preset_name, seed + 1000)
    eval_env = wrap_ekf(eval_env, **EKF_PARAMS)
    results = run_eval(model, eval_env)
    print(json.dumps(results))

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))
