# Recurrent G&CNet (LSTM) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the memoryless 3×64 MLP policy with MLP encoder + LSTM + Actor/Critic heads using sb3-contrib RecurrentPPO, combined with n_lookahead_gates=3 for figure-8 crossing-point disambiguation.

**Architecture:** sb3-contrib's RecurrentPPO handles LSTM hidden state management internally via RecurrentRolloutBuffer. We create a custom `RecurrentGCNetExtractor` (MLP encoder, features_dim=64) and configure RecurrentPPO with lstm_hidden_size=128. No env-side changes needed — VecEnvAdapter, ARPOActionWrapper, and all reward infrastructure work unchanged.

**Tech Stack:** sb3-contrib (RecurrentPPO), PyTorch (LSTM), SB3, Hydra/OmegaConf

**Spec:** `docs/superpowers/specs/2026-03-20-recurrent-gcnet-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add sb3-contrib dependency |
| Create | `control/policies/recurrent_gcnet.py` | MLP encoder feature extractor for RecurrentPPO |
| Modify | `control/algorithms/ppo.py` | Support `recurrent: true` → use RecurrentPPO |
| Create | `configs/control/ppo_recurrent.yaml` | Hydra config for recurrent PPO |
| Create | `configs/experiment/recurrent_gcnet.yaml` | Full experiment config (40M, from scratch) |
| Create | `tests/test_control/test_recurrent_gcnet.py` | Tests for recurrent policy + PPO wrapper |

### Task 1: Add sb3-contrib Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add sb3-contrib to control extras**

In `pyproject.toml`, add `"sb3-contrib"` to the `control` extras list (after `"stable-baselines3"`):

```toml
control = [
    "torch",
    "stable-baselines3",
    "sb3-contrib",
    "tensorboard",
    "gymnasium",
    "onnx",
    "onnxruntime",
    "onnxscript",
]
```

- [ ] **Step 2: Install**

Run: `pip install sb3-contrib`

- [ ] **Step 3: Verify import**

Run: `python -c "from sb3_contrib import RecurrentPPO; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add sb3-contrib for RecurrentPPO (LSTM support)"
```

### Task 2: Recurrent GCNet Feature Extractor

The feature extractor is a simple 2-layer MLP that feeds into sb3-contrib's LSTM. It replaces `GCNetExtractor` for recurrent mode.

**Files:**
- Create: `control/policies/recurrent_gcnet.py`
- Create: `tests/test_control/test_recurrent_gcnet.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_control/test_recurrent_gcnet.py
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
        assert out.shape == (4, 64)  # features_dim=64

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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_control/test_recurrent_gcnet.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement**

```python
# control/policies/recurrent_gcnet.py
"""Recurrent GCNet — MLP encoder for sb3-contrib RecurrentPPO.

Provides a feature extractor that feeds into sb3-contrib's LSTM layer.
The LSTM hidden state management is handled entirely by RecurrentPPO —
this module only needs to encode observations into a feature vector.

Architecture:
    obs → MLP [hidden_dims, ReLU] → features (features_dim)
    features → LSTM (managed by RecurrentPPO) → Actor/Critic heads
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


if _AVAILABLE:

    class RecurrentGCNetExtractor(BaseFeaturesExtractor):
        """MLP encoder for RecurrentPPO.

        Two-layer MLP that encodes observations into a fixed-size feature
        vector. The LSTM layer sits after this extractor (managed by
        sb3-contrib's RecurrentPPO, not by this class).

        Args:
            observation_space: Gymnasium Box observation space.
            hidden_dims: MLP hidden layer sizes. Default (64, 64).
        """

        def __init__(
            self,
            observation_space: spaces.Box,
            hidden_dims: tuple[int, ...] = (64, 64),
        ) -> None:
            features_dim = hidden_dims[-1]
            super().__init__(observation_space, features_dim=features_dim)

            obs_dim = observation_space.shape[0]
            layers: list[nn.Module] = []
            in_dim = obs_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self._encoder = nn.Sequential(*layers)

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self._encoder(observations)

else:

    class RecurrentGCNetExtractor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RecurrentGCNetExtractor requires torch + stable-baselines3. "
                "Install with: pip install 'algo-src[control]'"
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_control/test_recurrent_gcnet.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add control/policies/recurrent_gcnet.py tests/test_control/test_recurrent_gcnet.py
git commit -m "feat(control): add RecurrentGCNetExtractor for LSTM-based policy"
```

### Task 3: Extend PPO Wrapper for Recurrent Mode

**Files:**
- Modify: `control/algorithms/ppo.py`
- Modify: `tests/test_control/test_recurrent_gcnet.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_control/test_recurrent_gcnet.py`:

```python
from control.algorithms.ppo import PPO


class TestRecurrentPPOWrapper:
    def test_recurrent_flag_default_false(self):
        ppo = PPO()
        assert ppo.recurrent is False

    def test_recurrent_creates_recurrent_model(self):
        """When recurrent=True, _create_model should use RecurrentPPO."""
        from sim.tracks import build_figure8_track
        from sim.envs.gate_race_env import GateRaceEnv
        from sim.envs.vec_env_adapter import VecEnvAdapter

        env = VecEnvAdapter(GateRaceEnv(
            track=build_figure8_track(), n_envs=2, dt=0.01, max_steps=50,
            action_mode="trpy", n_lookahead_gates=3,
        ))

        ppo = PPO(
            recurrent=True,
            lstm_hidden_size=64,
            n_lstm_layers=1,
            net_arch={"pi": [32], "vf": [32]},
        )
        model = ppo._create_model(env)

        from sb3_contrib import RecurrentPPO as SB3RecurrentPPO
        assert isinstance(model, SB3RecurrentPPO)

    def test_recurrent_train_10_steps(self):
        """Recurrent PPO should complete a short training run."""
        from sim.tracks import build_figure8_track
        from sim.envs.gate_race_env import GateRaceEnv
        from sim.envs.vec_env_adapter import VecEnvAdapter

        env = VecEnvAdapter(GateRaceEnv(
            track=build_figure8_track(), n_envs=2, dt=0.01, max_steps=50,
            action_mode="trpy", n_lookahead_gates=3,
        ))

        ppo = PPO(
            recurrent=True,
            lstm_hidden_size=64,
            n_lstm_layers=1,
            n_steps=50,
            batch_size=50,
            n_epochs=1,
            net_arch={"pi": [32], "vf": [32]},
        )
        ppo.train(env, total_timesteps=100)
        assert ppo._model is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_control/test_recurrent_gcnet.py::TestRecurrentPPOWrapper -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'recurrent'`

- [ ] **Step 3: Modify PPO wrapper**

In `control/algorithms/ppo.py`:

**A. Add recurrent params to `__init__`** (after `tensorboard_log`):

```python
        # Recurrent (LSTM) support via sb3-contrib
        recurrent: bool = False,
        lstm_hidden_size: int = 128,
        n_lstm_layers: int = 1,
```

Store them:
```python
        self.recurrent = recurrent
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
```

**B. Modify `_build_policy_kwargs`** — when recurrent, inject RecurrentGCNetExtractor:

After the existing `if self.hidden_dims:` block, add:

```python
        # Recurrent mode: inject MLP encoder for LSTM
        if self.recurrent:
            from control.policies.recurrent_gcnet import RecurrentGCNetExtractor
            encoder_dims = self.hidden_dims if self.hidden_dims else (64, 64)
            kwargs["features_extractor_class"] = RecurrentGCNetExtractor
            kwargs["features_extractor_kwargs"] = {"hidden_dims": encoder_dims}
            kwargs["lstm_hidden_size"] = self.lstm_hidden_size
            kwargs["n_lstm_layers"] = self.n_lstm_layers
```

**C. Modify `_create_model`** — use RecurrentPPO when recurrent:

```python
    def _create_model(self, env: gym.Env):
        """Instantiate the SB3 PPO or RecurrentPPO model."""
        policy_kwargs = self._build_policy_kwargs()

        if self.recurrent:
            from sb3_contrib import RecurrentPPO as SB3RecurrentPPO
            return SB3RecurrentPPO(
                policy="MlpLstmPolicy",
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                use_sde=self.use_sde,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.tensorboard_log,
                verbose=1,
            )

        return SB3_PPO(
            policy=self.policy_type,
            env=env,
            # ... existing params unchanged ...
        )
```

**D. Modify `load`** — detect recurrent checkpoint:

```python
    def load(self, path: str | Path, env: gym.Env | None = None) -> None:
        _check_deps()
        if self.recurrent:
            from sb3_contrib import RecurrentPPO as SB3RecurrentPPO
            self._model = SB3RecurrentPPO.load(str(path), env=env)
        else:
            self._model = SB3_PPO.load(str(path), env=env)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_control/test_recurrent_gcnet.py -v --timeout=30`
Expected: All PASS

- [ ] **Step 5: Run existing PPO tests for regression**

Run: `pytest tests/test_control/test_ppo.py -v --timeout=60`
Expected: All existing tests PASS (recurrent=False is default)

- [ ] **Step 6: Commit**

```bash
git add control/algorithms/ppo.py tests/test_control/test_recurrent_gcnet.py
git commit -m "feat(control): extend PPO wrapper to support RecurrentPPO (LSTM)"
```

### Task 4: Hydra Config for Recurrent PPO

**Files:**
- Create: `configs/control/ppo_recurrent.yaml`

- [ ] **Step 1: Create config**

```yaml
# configs/control/ppo_recurrent.yaml
# RecurrentPPO with LSTM for temporal context.
_target_: control.algorithms.ppo.PPO
learning_rate: 0.0003
n_steps: 1000
batch_size: 5000
n_epochs: 10
gamma: 0.999
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.005
vf_coef: 0.5
max_grad_norm: 0.5
policy_type: MlpLstmPolicy
activation_fn: relu
use_sde: false

# Recurrent settings
recurrent: true
lstm_hidden_size: 128
n_lstm_layers: 1

# Post-LSTM heads (smaller than MLP-only since LSTM does heavy lifting)
net_arch:
  pi: [64]
  vf: [64]
```

- [ ] **Step 2: Commit**

```bash
git add configs/control/ppo_recurrent.yaml
git commit -m "config: add ppo_recurrent.yaml for LSTM-based training"
```

### Task 5: Experiment Config (40M from scratch)

**Files:**
- Create: `configs/experiment/recurrent_gcnet.yaml`

- [ ] **Step 1: Create experiment config**

```yaml
# @package _global_
# Recurrent G&CNet: LSTM policy with 3-gate lookahead.
# Training from scratch — cannot resume from MLP checkpoints.

defaults:
  - override /sim: numpy_quad
  - override /control: ppo_recurrent
  - override /perception: corner_noise
  - override /domain_rand: uniform_30pct
  - override /logging: wandb
  - _self_

sim:
  action_mode: trpy
  n_lookahead_gates: 3        # 32-dim obs (was 2/28-dim)
  max_steps: 3000
  arena_bounds: 10
  gate_passage_radius: 1.5
  gate_collision: false

total_timesteps: 40_000_000
# No resume — training from scratch with new architecture

track_gen:
  n_gates_min: 4
  n_gates_max: 8
  gate_spacing_min: 1.5
  gate_spacing_max: 4.0
  turn_angle_min: -120
  turn_angle_max: 120
  elevation_min: 1.0
  elevation_max: 3.5
  elevation_delta_max: 0.6
  closure_max_angle: 90
  closure_max_retries: 10
  figure8_ratio: 0.4
  figure8:
    loop_radius_min: 1.5
    loop_radius_max: 2.5
    gates_per_loop_min: 3
    gates_per_loop_max: 4
    crossing_offset_min: 0.3
    crossing_offset_max: 0.6
    elevation_min: 1.5
    elevation_max: 2.5
    elevation_delta_max: 0.6

perception:
  noise_scale: 1.0
  dropout_onset: null
  dropout_continuation: 0.7

reward:
  weights:
    gate_passage: 1.5
    gate_progress: 1.0
    gate_offset: 1.5
    body_rate: 0.001
    action_smoothness: 0.0
    crash_penalty: 10.0
    spline_proximity: 0.5
    heading_alignment: 0.05
    speed_bonus: 0.0
    boundary_penalty: 3.0
    gate_approach: 0.2
    gate_centering: 3.0
  v_max: 10.0
  action_smoothness_threshold: 0.5

# aRPO bootstrap — helps LSTM learn patterns from consistent trajectories
arpo:
  enabled: true
  k_end_fraction: 0.15        # Remove base policy by 6M steps (15% of 40M)

curriculum:
  enabled: true
  stages:
    - timestep: 0
      reward_weights:
        gate_passage: 1.5
        gate_progress: 1.0
        gate_offset: 1.5
        crash_penalty: 10.0
        spline_proximity: 0.5
        heading_alignment: 0.05
        speed_bonus: 0.0
        boundary_penalty: 3.0
        gate_approach: 0.2
        gate_centering: 3.0
      v_max: 10.0
      ent_coef: 0.005
      trigger: null
    - timestep: 10_000_000
      reward_weights:
        gate_passage: 10.0
        gate_progress: 0.5
        gate_offset: 2.0
        crash_penalty: 5.0
        spline_proximity: 0.2
        heading_alignment: 0.03
        speed_bonus: 0.3
        boundary_penalty: 3.0
        gate_approach: 0.3
        gate_centering: 2.0
      v_max: 15.0
      ent_coef: 0.003
      trigger:
        metric: racing/gate_passage_rate
        threshold: 0.80
        window: 200
    - timestep: 25_000_000
      reward_weights:
        gate_passage: 50.0
        gate_progress: 0.0
        gate_offset: 2.0
        crash_penalty: 3.0
        spline_proximity: 0.0
        heading_alignment: 0.0
        speed_bonus: 0.8
        boundary_penalty: 2.0
        gate_approach: 0.3
        gate_centering: 1.0
      v_max: 30.0
      ent_coef: 0.001
      trigger:
        metric: racing/lap_completion_rate
        threshold: 0.5
        window: 200

multi_scene:
  enabled: true
  n_scenes: 10

lr_schedule:
  initial_lr: 0.0003
  final_lr: 0.00005

logging:
  tags: ["trpy", "recurrent", "lstm-128", "lookahead-3", "figure8", "arpo", "40M"]
  group: recurrent_gcnet
  notes: "Recurrent G&CNet: LSTM(128) + MLP encoder(64,64), n_lookahead=3, 40% figure-8, aRPO bootstrap, from scratch"
```

- [ ] **Step 2: Commit**

```bash
git add configs/experiment/recurrent_gcnet.yaml
git commit -m "config: add recurrent_gcnet experiment (LSTM, 40M from scratch)"
```

### Task 6: Smoke Test

**Files:**
- Create: `tests/test_integration/test_recurrent_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/test_integration/test_recurrent_smoke.py
"""Smoke test: RecurrentPPO with LSTM trains without crashing."""
import pytest


class TestRecurrentSmoke:
    def test_recurrent_ppo_trains_200_steps(self):
        """RecurrentPPO + LSTM should train 200 steps without error."""
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
        assert action.shape == (4, 4)

        env.close()
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_integration/test_recurrent_smoke.py -v --timeout=60`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration/test_recurrent_smoke.py
git commit -m "test(integration): add recurrent PPO smoke test"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | sb3-contrib dependency | `pyproject.toml` |
| 2 | RecurrentGCNetExtractor | `control/policies/recurrent_gcnet.py` |
| 3 | PPO wrapper recurrent mode | `control/algorithms/ppo.py` |
| 4 | Hydra config | `configs/control/ppo_recurrent.yaml` |
| 5 | Experiment config | `configs/experiment/recurrent_gcnet.yaml` |
| 6 | Smoke test | `tests/test_integration/test_recurrent_smoke.py` |

**To launch after implementation:**
```bash
python -m training +experiment=recurrent_gcnet
```

**Expected training time:** ~3.5 hours at ~3000 FPS for 40M steps.

**What to watch on W&B:**
- First 6M: α-RPO bootstrap phase — gate passage should climb quickly
- 6-10M: Standalone LSTM learning — should see memory benefits on figure-8 tracks
- 10-25M: Speed ramp via curriculum — eval speed should climb
- 25-40M: Full speed push — target 8+ m/s with >90% success

**Figure-8 eval after training:** Run the same 20-episode eval script to compare against the 75% baseline.
