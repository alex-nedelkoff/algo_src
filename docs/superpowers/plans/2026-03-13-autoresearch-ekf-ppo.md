# Autoresearch EKF+PPO Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous experiment loop that iteratively tunes 18 parameters across reward shaping, EKF filtering, PPO training, and domain randomization — running overnight to improve gate-passing performance.

**Architecture:** A bash orchestrator (`run.sh`) drives a loop: run `train.py` → parse results → append to `results.tsv` → call Claude CLI to edit `train.py` → repeat. The agent edits 4 parameter dicts above a FIXED line. An immutable `prepare.py` provides env construction, model loading with hyperparameter overrides, and evaluation. Based on the existing `playground/autoresearch` implementation, adapted for algo_src's expanded parameter surface (18 params vs 9) and EKF wrapping.

**Tech Stack:** Python 3.11+, stable-baselines3, PyTorch, bash, Claude CLI

---

## File Structure

| File | Responsibility |
|------|---------------|
| `autoresearch/prepare.py` | Immutable harness: env construction, preset registration, EKF wrapping, model loading with hyperparameter overrides, evaluation |
| `autoresearch/train.py` | Agent-editable: 4 parameter dicts above FIXED line, fixed training/eval logic below |
| `autoresearch/run.sh` | Orchestrator: loop, diff tracking, AST parameter extraction, safety checks, Claude CLI invocation |
| `autoresearch/program.md` | Agent directives: goal, parameter bounds, strategy tips |
| `autoresearch/.gitignore` | Exclude results.tsv, diffs/, train.py.orig |
| `autoresearch/__init__.py` | Package marker |
| `tests/test_autoresearch/test_prepare.py` | Tests for prepare.py functions |
| `tests/test_autoresearch/__init__.py` | Package marker |

---

## Chunk 1: Core Infrastructure

### Task 1: `prepare.py` — Reward Preset Registration

**Files:**
- Create: `autoresearch/__init__.py`
- Create: `autoresearch/prepare.py`
- Create: `tests/test_autoresearch/__init__.py`
- Create: `tests/test_autoresearch/test_prepare.py`

- [ ] **Step 1: Write the failing test for `register_reward_preset`**

```python
# tests/test_autoresearch/test_prepare.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestRegisterRewardPreset -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autoresearch'`

- [ ] **Step 3: Write `__init__.py` files and `register_reward_preset` in `prepare.py`**

```python
# autoresearch/__init__.py
# (empty)
```

```python
# autoresearch/prepare.py
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
```

```python
# tests/test_autoresearch/__init__.py
# (empty)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestRegisterRewardPreset -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add autoresearch/__init__.py autoresearch/prepare.py tests/test_autoresearch/__init__.py tests/test_autoresearch/test_prepare.py
git commit -m "feat(autoresearch): add prepare.py with reward preset registration"
```

---

### Task 2: `prepare.py` — Environment Construction and EKF Wrapping

**Files:**
- Modify: `autoresearch/prepare.py`
- Modify: `tests/test_autoresearch/test_prepare.py`

- [ ] **Step 1: Write the failing test for `make_training_env`**

```python
# Append to tests/test_autoresearch/test_prepare.py

class TestMakeTrainingEnv:
    """Test environment construction."""

    def test_creates_vec_env(self) -> None:
        from autoresearch.prepare import make_training_env, register_reward_preset
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
        from autoresearch.prepare import make_training_env, register_reward_preset
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestMakeTrainingEnv -v`
Expected: FAIL — `ImportError: cannot import name 'make_training_env'`

- [ ] **Step 3: Implement `make_training_env` in `prepare.py`**

Add to `autoresearch/prepare.py`:

```python
from sim.envs.rate_ctrl_env import RateCtrlEnv
from sim.envs.vec_env_adapter import VecEnvAdapter


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
    inner = RateCtrlEnv(
        n_envs=n_envs,
        seed=seed,
        dr_percentage=dr_percentage,
        reward_preset=preset_name,
    )
    return VecEnvAdapter(inner)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestMakeTrainingEnv -v`
Expected: 2 PASSED

- [ ] **Step 5: Write the failing test for `wrap_ekf`**

```python
# Append to tests/test_autoresearch/test_prepare.py

from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper


class TestWrapEkf:
    """Test EKF wrapping."""

    def test_wraps_env_with_ekf(self) -> None:
        from autoresearch.prepare import make_training_env, wrap_ekf, register_reward_preset
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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestWrapEkf -v`
Expected: FAIL — `ImportError: cannot import name 'wrap_ekf'`

- [ ] **Step 7: Implement `wrap_ekf` in `prepare.py`**

Add to `autoresearch/prepare.py`:

```python
from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper


def wrap_ekf(
    env: Any,
    corner_noise_k: float = 2.0,
    corner_dropout_onset: float | None = None,
) -> EKFVecEnvWrapper:
    """Apply EKF filtering wrapper to a VecEnv."""
    return EKFVecEnvWrapper(
        env,
        corner_noise_k=corner_noise_k,
        corner_dropout_onset=corner_dropout_onset,
        provide_teacher_obs=False,
    )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestWrapEkf -v`
Expected: 1 PASSED

- [ ] **Step 9: Commit**

```bash
git add autoresearch/prepare.py tests/test_autoresearch/test_prepare.py
git commit -m "feat(autoresearch): add make_training_env and wrap_ekf to prepare.py"
```

---

### Task 3: `prepare.py` — Model Loading with Hyperparameter Overrides

**Files:**
- Modify: `autoresearch/prepare.py`
- Modify: `tests/test_autoresearch/test_prepare.py`

- [ ] **Step 1: Write the failing test for `load_and_configure_model`**

```python
# Append to tests/test_autoresearch/test_prepare.py

class TestLoadAndConfigureModel:
    """Test model loading with hyperparameter overrides."""

    def test_overrides_hyperparameters(self, tmp_path) -> None:
        """Load a baseline checkpoint, override params, verify they stick."""
        from stable_baselines3 import PPO as SB3_PPO
        from autoresearch.prepare import (
            make_training_env, register_reward_preset, load_and_configure_model,
        )

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestLoadAndConfigureModel -v`
Expected: FAIL — `ImportError: cannot import name 'load_and_configure_model'`

- [ ] **Step 3: Implement `load_and_configure_model`**

Add to `autoresearch/prepare.py`:

```python
from stable_baselines3 import PPO as SB3_PPO


def load_and_configure_model(
    env: Any,
    checkpoint_path: str,
    training_params: dict[str, float],
) -> SB3_PPO:
    """Load an SB3 PPO checkpoint and override training hyperparameters."""
    model = SB3_PPO.load(checkpoint_path, env=env, device="auto")
    model.learning_rate = training_params["learning_rate"]
    model.ent_coef = training_params["ent_coef"]
    clip_val = training_params["clip_range"]
    model.clip_range = lambda _, v=clip_val: v
    model.gae_lambda = training_params["gae_lambda"]
    model.gamma = training_params["gamma"]
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestLoadAndConfigureModel -v`
Expected: 1 PASSED

- [ ] **Step 5: Commit**

```bash
git add autoresearch/prepare.py tests/test_autoresearch/test_prepare.py
git commit -m "feat(autoresearch): add load_and_configure_model with hyperparameter overrides"
```

---

### Task 4: `prepare.py` — Evaluation Harness

**Files:**
- Modify: `autoresearch/prepare.py`
- Modify: `tests/test_autoresearch/test_prepare.py`

- [ ] **Step 1: Write the failing test for `run_eval`**

```python
# Append to tests/test_autoresearch/test_prepare.py

class TestRunEval:
    """Test evaluation harness."""

    def test_returns_expected_metrics(self, tmp_path) -> None:
        from stable_baselines3 import PPO as SB3_PPO
        from autoresearch.prepare import (
            make_training_env, wrap_ekf, register_reward_preset, run_eval,
        )

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestRunEval -v`
Expected: FAIL — `ImportError: cannot import name 'run_eval'`

- [ ] **Step 3: Implement `run_eval`**

Add to `autoresearch/prepare.py`:

```python
def run_eval(
    model: SB3_PPO,
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
                # alt_std from episode info if available
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/test_prepare.py::TestRunEval -v`
Expected: 1 PASSED

- [ ] **Step 5: Commit**

```bash
git add autoresearch/prepare.py tests/test_autoresearch/test_prepare.py
git commit -m "feat(autoresearch): add run_eval with score = avg_gates - 2 * crash_rate"
```

---

## Chunk 2: Train Script, Orchestrator, and Program

### Task 5: `train.py` — Agent-Editable Training Script

**Files:**
- Create: `autoresearch/train.py`

- [ ] **Step 1: Write `train.py`**

```python
# autoresearch/train.py
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
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('autoresearch/train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add autoresearch/train.py
git commit -m "feat(autoresearch): add agent-editable train.py with 18-parameter surface"
```

---

### Task 6: `program.md` — Agent Directives

**Files:**
- Create: `autoresearch/program.md`

- [ ] **Step 1: Write `program.md`**

```markdown
# EKF+PPO Drone Racing Autoresearch

## Goal
Maximize `score = avg_gates - 2 * crash_rate` by tuning 18 parameters
across reward shaping, EKF filtering, PPO hyperparameters, and domain
randomization. The policy is fine-tuned from a Phase 4 PPO+EKF checkpoint
that already flies with filtered observations. You are refining, not
teaching from scratch.

## What You Can Change
ONLY the 4 parameter dicts above the FIXED line in train.py:
- REWARD_WEIGHTS (9 params)
- EKF_PARAMS (2 params)
- TRAINING_PARAMS (5 params)
- DOMAIN_RAND (1 param)

Do NOT modify anything below the FIXED line.

## Parameter Bounds

### Reward Weights
| Parameter       | Min   | Max   | Baseline | Purpose |
|----------------|-------|-------|----------|---------|
| lambda_gate    | 1.0   | 50.0  | 10.0     | Gate passage reward |
| lambda_prog    | 0.1   | 5.0   | 1.0      | Progress toward gate |
| lambda_rate    | 0.0   | 0.01  | 0.001    | Angular rate penalty |
| lambda_offset  | 0.0   | 5.0   | 0.0      | Gate center offset penalty |
| lambda_perc    | 0.0   | 1.0   | 0.0      | Perception-aware reward |
| lambda_delta_u | 0.0   | 1.0   | 0.001    | Action smoothness penalty |
| lambda_crash   | 1.0   | 50.0  | 10.0     | Crash penalty |
| lambda_alive   | 0.0   | 1.0   | 0.0      | Survival bonus |
| v_max          | 0.0   | 30.0  | 0.0      | Speed penalty threshold (m/s) |

### EKF Parameters
| Parameter             | Min   | Max   | Baseline | Purpose |
|----------------------|-------|-------|----------|---------|
| corner_noise_k       | 0.5   | 5.0   | 2.0      | Corner detector noise scale |
| corner_dropout_onset | None  | 0.0–1.0 | None   | HMM dropout probability |

### Training Hyperparameters
| Parameter      | Min     | Max      | Baseline | Purpose |
|---------------|---------|----------|----------|---------|
| learning_rate | 1e-5    | 1e-3     | 3e-4     | PPO learning rate |
| ent_coef      | 0.0     | 0.05     | 0.005    | Entropy coefficient |
| clip_range    | 0.1     | 0.4      | 0.2      | PPO clipping range |
| gae_lambda    | 0.9     | 0.99     | 0.95     | GAE lambda |
| gamma         | 0.99    | 0.9999   | 0.999    | Discount factor |

### Domain Randomization
| Parameter  | Min  | Max  | Baseline | Purpose |
|-----------|------|------|----------|---------|
| percentage | 0.0  | 0.5  | 0.3      | Uniform DR on all physics params |

## Scoring
```
score = avg_gates - 2 * crash_rate
```
Higher is better. Gate completion is the primary objective; crashing is
heavily penalized.

## Diagnostics (in results.tsv, not scored)
- avg_gates: mean gates passed per episode
- crash_rate: fraction of episodes ending in crash
- alt_std: altitude standard deviation (stability proxy)
- avg_steps: mean episode length
- max_gates: best single-episode gate count

## Budget
Each experiment trains for 10M steps (~11 min with 500 envs).

## Strategy Tips
- Change 1-3 parameters per experiment to isolate effects
- Look for trends in results.tsv before making changes
- Reward weights interact: large lambda_gate with low lambda_crash
  tends to produce reckless crashing policies
- v_max clamps progress reward — prevents reward hacking from
  high-speed straight-line flying
- Lower learning_rate for fine-tuning is generally safer
- Higher ent_coef encourages exploration but can destabilize
- corner_noise_k affects EKF trust in measurements vs predictions
- DR percentage trades sim-to-real transfer against training difficulty
- If crash_rate is high, try increasing lambda_crash or decreasing
  lambda_gate before changing training hyperparameters
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/program.md
git commit -m "feat(autoresearch): add program.md agent directives"
```

---

### Task 7: `run.sh` — Orchestrator Loop

**Files:**
- Create: `autoresearch/run.sh`

- [ ] **Step 1: Write `run.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # project root

RESULTS_FILE="autoresearch/results.tsv"
DIFFS_DIR="autoresearch/diffs"
TRAIN_PY="autoresearch/train.py"
TRAIN_PY_ORIG="autoresearch/train.py.orig"

mkdir -p "$DIFFS_DIR"

# Save original train.py for diffing
if [ ! -f "$TRAIN_PY_ORIG" ]; then
    cp "$TRAIN_PY" "$TRAIN_PY_ORIG"
fi

# Initialize results.tsv with header if it doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    printf "exp_id\ttimestamp\tscore\tavg_gates\tcrash_rate\talt_std\tavg_steps\tmax_gates\tseed\tlambda_gate\tlambda_prog\tlambda_rate\tlambda_offset\tlambda_perc\tlambda_delta_u\tlambda_crash\tlambda_alive\tv_max\tcorner_noise_k\tcorner_dropout_onset\tlearning_rate\tent_coef\tclip_range\tgae_lambda\tgamma\tdr_percentage\n" > "$RESULTS_FILE"
fi

# Determine next exp_id
get_next_exp_id() {
    local last_id
    last_id=$(tail -n 1 "$RESULTS_FILE" 2>/dev/null | cut -f1)
    if [ "$last_id" = "exp_id" ] || [ -z "$last_id" ]; then
        echo 0
    else
        echo $((last_id + 1))
    fi
}

# Extract all 4 parameter dicts from train.py via AST parsing
extract_params() {
    python -c "
import ast

with open('$TRAIN_PY') as f:
    tree = ast.parse(f.read())

dicts = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                'REWARD_WEIGHTS', 'EKF_PARAMS', 'TRAINING_PARAMS', 'DOMAIN_RAND'
            ):
                dicts[target.id] = ast.literal_eval(node.value)

rw = dicts['REWARD_WEIGHTS']
ekf = dicts['EKF_PARAMS']
tp = dicts['TRAINING_PARAMS']
dr = dicts['DOMAIN_RAND']

vals = [
    rw['lambda_gate'], rw['lambda_prog'], rw['lambda_rate'],
    rw['lambda_offset'], rw['lambda_perc'], rw['lambda_delta_u'],
    rw['lambda_crash'], rw['lambda_alive'], rw['v_max'],
    ekf['corner_noise_k'], ekf.get('corner_dropout_onset', 'None'),
    tp['learning_rate'], tp['ent_coef'], tp['clip_range'],
    tp['gae_lambda'], tp['gamma'],
    dr['percentage'],
]
print('\t'.join(str(v) for v in vals))
"
}

# Count consecutive diverged experiments (crash_rate >= 0.9)
count_consecutive_diverged() {
    python -c "
import csv
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f, delimiter='\t')
    count = 0
    for row in reader:
        if float(row['crash_rate']) >= 0.9:
            count += 1
        else:
            count = 0
    print(count)
"
}

# Check for score plateau (no improvement in last 10 experiments)
check_plateau() {
    local n_exps
    n_exps=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    if [ "$n_exps" -lt 10 ]; then
        echo "no"
        return
    fi
    python -c "
import csv
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f, delimiter='\t')
    scores = [float(row['score']) for row in reader]
if len(scores) < 10:
    print('no')
else:
    best_before = max(scores[:-10]) if len(scores) > 10 else -999
    best_recent = max(scores[-10:])
    print('yes' if best_recent <= best_before else 'no')
"
}

MAX_CONSECUTIVE_DIVERGED=3

# Verify dependencies
python --version >/dev/null 2>&1 || { echo "python not found — activate conda env first"; exit 1; }
claude --version >/dev/null 2>&1 || { echo "Claude CLI not found. Install from https://claude.ai/claude-code"; exit 1; }

echo "=== Autoresearch EKF+PPO Tuning Loop ==="
echo "18 parameters: 9 reward + 2 EKF + 5 training + 1 DR"
echo "Score = avg_gates - 2 * crash_rate"
echo "Press Ctrl+C to stop."
echo ""

while true; do
    EXP_ID=$(get_next_exp_id)
    SEED=$((42 + EXP_ID))
    TIMESTAMP=$(date -Iseconds)

    echo "--- Experiment $EXP_ID (seed=$SEED) ---"

    # Save diff
    diff -u "$TRAIN_PY_ORIG" "$TRAIN_PY" > "$DIFFS_DIR/exp_$(printf '%03d' "$EXP_ID").diff" || true

    # Extract current parameters
    PARAMS=$(extract_params)

    # Run training + eval
    echo "Running training ($EXP_ID)..."
    RESULTS_JSON=$(PYTHONPATH=. python autoresearch/train.py "$EXP_ID" 2>&1 | tail -n 1)

    # Parse JSON results
    SCORE=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['score'])")
    AVG_GATES=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['avg_gates'])")
    CRASH_RATE=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['crash_rate'])")
    ALT_STD=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['alt_std'])")
    AVG_STEPS=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['avg_steps'])")
    MAX_GATES=$(echo "$RESULTS_JSON" | python -c "import sys,json; d=json.load(sys.stdin); print(d['max_gates'])")

    # Append to results.tsv
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$EXP_ID" "$TIMESTAMP" "$SCORE" "$AVG_GATES" "$CRASH_RATE" "$ALT_STD" "$AVG_STEPS" "$MAX_GATES" "$SEED" "$PARAMS" \
        >> "$RESULTS_FILE"

    echo "Score: $SCORE | Gates: $AVG_GATES | Crash: $CRASH_RATE | Alt: $ALT_STD"

    # Check divergence
    DIVERGED=$(count_consecutive_diverged)
    if [ "$DIVERGED" -ge "$MAX_CONSECUTIVE_DIVERGED" ]; then
        echo ""
        echo "!!! $DIVERGED consecutive diverged experiments (crash_rate >= 0.9). Pausing."
        echo "Review results.tsv and adjust bounds or strategy before restarting."
        exit 1
    fi

    # Check plateau
    PLATEAU=$(check_plateau)
    if [ "$PLATEAU" = "yes" ]; then
        echo ""
        echo "Score plateau detected: no improvement in last 10 experiments."
        echo "Best score: $(python -c "
import csv
with open('$RESULTS_FILE') as f:
    scores = [float(row['score']) for row in csv.DictReader(f, delimiter='\t')]
print(f'{max(scores):.4f}')
")"
        echo "Pausing. Review results.tsv for insights."
        exit 0
    fi

    # Invoke LLM agent to edit train.py
    echo ""
    echo "Invoking LLM agent to propose next experiment..."
    claude -p "$(cat <<PROMPT
You are an autonomous parameter tuning agent for drone racing RL with EKF.

Read the research directives:
$(cat autoresearch/program.md)

Here are all experiment results so far:
$(cat "$RESULTS_FILE")

Your task: edit the 4 parameter dicts (REWARD_WEIGHTS, EKF_PARAMS,
TRAINING_PARAMS, DOMAIN_RAND) in autoresearch/train.py.
Stay within the parameter bounds. Change 1-3 params per experiment
to isolate effects. Look for trends in the results before choosing
what to change.

$(python -c "print('WARNING: The last experiment DIVERGED (crash_rate=$CRASH_RATE). Try a less aggressive change.' if float('$CRASH_RATE') >= 0.9 else '')")

Current train.py:
$(cat "$TRAIN_PY")

Edit the parameter dict values only. Do not touch anything below the FIXED line.
PROMPT
)" --allowedTools Edit

    echo ""
done
```

- [ ] **Step 2: Make executable**

Run: `chmod +x autoresearch/run.sh`

- [ ] **Step 3: Verify syntax**

Run: `bash -n autoresearch/run.sh && echo "OK"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add autoresearch/run.sh
git commit -m "feat(autoresearch): add run.sh orchestrator loop"
```

---

### Task 8: `.gitignore`

**Files:**
- Create: `autoresearch/.gitignore`

- [ ] **Step 1: Write `.gitignore`**

```
# Experiment artifacts (tracked in results.tsv, not git)
results.tsv
diffs/
train.py.orig
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/.gitignore
git commit -m "chore(autoresearch): add .gitignore for experiment artifacts"
```

---

## Chunk 3: Integration Verification

### Task 9: Run All Tests

- [ ] **Step 1: Run the full autoresearch test suite**

Run: `PYTHONPATH=. pytest tests/test_autoresearch/ -v`
Expected: All tests pass

- [ ] **Step 2: Run existing test suite to verify no regressions**

Run: `PYTHONPATH=. pytest tests/ -v --ignore=tests/test_autoresearch/ -x`
Expected: All existing tests still pass

- [ ] **Step 3: Verify AST extraction works on train.py**

Run: `cd autoresearch && python -c "
import ast
with open('train.py') as f:
    tree = ast.parse(f.read())
dicts = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('REWARD_WEIGHTS', 'EKF_PARAMS', 'TRAINING_PARAMS', 'DOMAIN_RAND'):
                dicts[target.id] = ast.literal_eval(node.value)
assert len(dicts) == 4, f'Expected 4 dicts, got {len(dicts)}'
assert len(dicts['REWARD_WEIGHTS']) == 9
assert len(dicts['EKF_PARAMS']) == 2
assert len(dicts['TRAINING_PARAMS']) == 5
assert len(dicts['DOMAIN_RAND']) == 1
print('AST extraction: OK (18 params)')
"`
Expected: `AST extraction: OK (18 params)`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(autoresearch): complete autoresearch EKF+PPO tuning loop"
```
