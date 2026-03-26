# Autoresearch for algo_src PPO+EKF Training

## Overview

An autonomous experiment loop where a Claude agent iteratively tunes 18 parameters across reward shaping, EKF filtering, PPO training, and domain randomization. Each experiment fine-tunes from a fixed checkpoint for 10M steps with EKF-filtered observations, evaluated on gate-passing performance with crash penalty.

Based on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern and the existing `playground/autoresearch` implementation.

## Architecture

```
autoresearch/
├── run.sh          # Orchestrator loop (bash)
├── train.py        # Agent-editable: parameter dicts above FIXED line
├── train.py.orig   # Baseline snapshot for diff tracking
├── prepare.py      # Immutable: env construction, eval harness, scoring
├── program.md      # Human-written directives for the agent
├── results.tsv     # Append-only experiment log
├── diffs/          # Per-experiment diffs of train.py
└── .gitignore      # Exclude results.tsv, diffs/, viz/, train.py.orig
```

## Editable Surface (18 parameters)

The agent modifies ONLY the config dicts in `train.py` above the `# ---- FIXED BELOW ---- #` line.

### Reward Weights (9 params)

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `lambda_gate` | 10.0 | 1.0–50.0 | Gate passage reward |
| `lambda_prog` | 1.0 | 0.1–5.0 | Progress toward gate |
| `lambda_rate` | 0.001 | 0.0–0.01 | Angular rate penalty |
| `lambda_offset` | 0.0 | 0.0–5.0 | Gate center offset penalty |
| `lambda_perc` | 0.0 | 0.0–1.0 | Perception-aware reward |
| `lambda_delta_u` | 0.001 | 0.0–1.0 | Action smoothness penalty |
| `lambda_crash` | 10.0 | 1.0–50.0 | Crash penalty |
| `lambda_alive` | 0.0 | 0.0–1.0 | Survival bonus |
| `v_max` | 0.0 | 0.0–30.0 | Speed penalty threshold |

### EKF Parameters (2 params)

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `corner_noise_k` | 2.0 | 0.5–5.0 | Corner detector noise scale |
| `corner_dropout_onset` | None | None or 0.0–1.0 | HMM dropout probability |

### Training Hyperparameters (5 params)

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `learning_rate` | 3e-4 | 1e-5–1e-3 | PPO learning rate |
| `ent_coef` | 0.005 | 0.0–0.05 | Entropy coefficient |
| `clip_range` | 0.2 | 0.1–0.4 | PPO clipping range |
| `gae_lambda` | 0.95 | 0.9–0.99 | GAE lambda |
| `gamma` | 0.999 | 0.99–0.9999 | Discount factor |

### Domain Randomization (1 param)

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `percentage` | 0.3 | 0.0–0.5 | Uniform DR percentage (all physics params) |

### Locked

- **Network architecture**: `[64, 64, 64]` (MonoRace M23, 10,180 params) — not editable
- **n_steps**: 1000 — not editable (affects wall-clock fairness)
- **batch_size**: 5000 — not editable (affects wall-clock fairness)

## Scoring

```
score = avg_gates - 2 * crash_rate
```

Primary metric only. Diagnostics tracked in `results.tsv` but not scored:
- `avg_gates`: mean gates passed per episode
- `crash_rate`: fraction of episodes ending in crash
- `alt_std`: altitude standard deviation (stability proxy)
- `avg_steps`: mean episode length
- `max_gates`: best single-episode gate count

## Experiment Budget

- **Steps per experiment**: 10M (~11 min wall-clock with 500 parallel envs)
- **Parallel envs**: 500 (benchmarked: 100 envs = ~31min stepping alone; 500 envs brings total including PPO overhead to ~11min)
- **Throughput**: ~5 experiments/hour, ~40-50 overnight
- **Seed**: `42 + exp_id` for reproducibility (note: EKF wrapper IMU noise uses unseeded RNG — full determinism requires seeding it)
- **Starting checkpoint**: `../playground/runs/phase4_ekf_ppo_v2/final_model.zip` (relative path, fresh fork each experiment)

## Orchestration Loop (`run.sh`)

```
loop forever:
  1. exp_id = next sequential ID
  2. Save diff: train.py vs train.py.orig → diffs/exp_NNN.diff
  3. Extract all 4 parameter dicts (REWARD_WEIGHTS, EKF_PARAMS, TRAINING_PARAMS,
     DOMAIN_RAND) via Python AST parsing → populate TSV columns
  4. Run: PYTHONPATH=. python autoresearch/train.py <exp_id>
  5. Parse JSON output → score, avg_gates, crash_rate, diagnostics
  6. Append row to results.tsv
  7. Safety checks:
     - DIVERGENCE: if 3+ consecutive crash_rate >= 0.9 → pause
     - PLATEAU: if no improvement in last 10 experiments → pause
  8. Call claude -p with:
     - program.md (directives + parameter bounds for all 18 params)
     - results.tsv (full history)
     - current train.py
     - instruction: edit ONLY the 4 parameter dicts above the FIXED line
  9. Loop
```

## Fixed Infrastructure (`prepare.py`)

Provides immutable functions imported by `train.py`:

- `register_reward_preset(name, weights_dict)` — creates a `RewardPreset` from the agent's weight dict and injects it into `sim.rewards_mavlab.PRESETS`. Required because `PRESETS` is hardcoded with only "baseline", "M16", "M23". The function validates all 9 keys are present.
- `make_training_env(n_envs, dr_percentage, preset_name)` — constructs `RateCtrlEnv` + `VecEnvAdapter` directly (bypasses `PlaygroundEnvFactory` to avoid DictConfig overhead). Takes the preset name registered above.
- `wrap_ekf(env, corner_noise_k, corner_dropout_onset)` — applies `EKFVecEnvWrapper`
- `load_and_configure_model(env, checkpoint_path, training_params)` — loads SB3 PPO checkpoint via `SB3_PPO.load()`, then overrides hyperparameters on the loaded model:
  - `model.learning_rate = training_params["learning_rate"]`
  - `model.ent_coef = training_params["ent_coef"]`
  - `model.clip_range = lambda _: training_params["clip_range"]` (SB3 stores clip_range as a callable schedule)
  - `model.gamma = training_params["gamma"]`
  - `model.gae_lambda = training_params["gae_lambda"]`
- `run_eval(model, env, n_episodes=50)` — deterministic rollout, returns `{avg_gates, crash_rate, alt_std, avg_steps, max_gates, score}`. Eval uses the same EKF configuration as training so that EKF parameter changes are reflected in the score.

## `train.py` Structure

```python
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
from prepare import (
    register_reward_preset, make_training_env, wrap_ekf,
    load_and_configure_model, run_eval,
)

CHECKPOINT = "../playground/runs/phase4_ekf_ppo_v2/final_model.zip"
N_STEPS = 10_000_000
N_ENVS = 500

def main(exp_id):
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
```

## `results.tsv` Schema

```
exp_id  timestamp  score  avg_gates  crash_rate  alt_std  avg_steps  max_gates  seed  lambda_gate  lambda_prog  lambda_rate  lambda_offset  lambda_perc  lambda_delta_u  lambda_crash  lambda_alive  v_max  corner_noise_k  corner_dropout_onset  learning_rate  ent_coef  clip_range  gae_lambda  gamma  dr_percentage
```

One row per experiment. Tab-separated. Append-only.

## `program.md` Directives

The agent receives these instructions with every invocation:

- Goal: maximize `score = avg_gates - 2 * crash_rate`
- Editable: ONLY the 4 parameter dicts above the FIXED line
- Parameter bounds (copied from tables above)
- Strategy guidance: change 1-3 params per experiment, look for trends in results.tsv
- Baseline values and expected baseline score

## Safety

| Check | Trigger | Action |
|-------|---------|--------|
| Divergence | 3 consecutive `crash_rate >= 0.9` | Pause loop, print warning |
| Plateau | No improvement in last 10 experiments | Pause loop, print summary |
| Bounds | Parameter outside documented range | Agent is instructed to stay in bounds; no hard enforcement |
| Diffs | Every experiment | Saved to `diffs/exp_NNN.diff` for review |
| Reproducibility | Every experiment | Deterministic seed `42 + exp_id` |

## Dependencies

- Python 3.11+, PyTorch, stable-baselines3, gymnasium, omegaconf
- Claude CLI (`claude` command available on PATH)
- All algo_src modules on PYTHONPATH

## Success Criteria

- Loop runs unattended overnight (~40+ experiments)
- Score improves over baseline
- No divergence or repeated crashes
- Results reproducible via saved diffs + seeds
