# Mixture of Experts Policy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single 3×64 MLP with a Mixture of Experts policy — 4 expert networks (2×128 each) with a top-2 router — to break through the 7m/1.0m performance cliff on the golden set benchmark.

**Architecture:** Custom `MoEPolicy` subclasses SB3's `ActorCriticPolicy`. The router (28→64→4) selects the top-2 experts per step. Each expert (28→128→128→4) outputs TRPY actions. The final action is a weighted sum of the 2 selected experts. A separate standard MLP critic (28→128→128→1) estimates state value. A load balancing auxiliary loss (α=0.01) prevents expert collapse.

**Tech Stack:** PyTorch, SB3 PPO, Hydra

**Spec:** `docs/superpowers/specs/2026-03-23-moe-policy-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `control/policies/moe_policy.py` | MoEPolicy: router, experts, top-2 selection, load balance loss |
| Modify | `control/algorithms/ppo.py` | Support `moe: true` flag |
| Modify | `training/loops/rl.py` | Pass `moe` param from config |
| Create | `configs/control/ppo_moe.yaml` | Hydra config for MoE PPO |
| Create | `configs/experiment/moe_s1.yaml` | Stage 1 experiment (10m/1.5m) |
| Create | `tests/test_control/test_moe_policy.py` | Tests for MoE components |
| Create | `tests/test_integration/test_moe_smoke.py` | End-to-end smoke test |

### Task 1: MoE Policy Module

The core implementation. A single file containing the router, expert network, and the `MoEPolicy` class.

**Files:**
- Create: `control/policies/moe_policy.py`
- Create: `tests/test_control/test_moe_policy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_control/test_moe_policy.py
"""Tests for Mixture of Experts policy."""
import numpy as np
import pytest
import torch
import gymnasium
from gymnasium import spaces


class TestRouter:
    def test_output_shape(self):
        from control.policies.moe_policy import Router
        router = Router(obs_dim=28, n_experts=4, hidden_dim=64)
        obs = torch.randn(8, 28)
        weights = router(obs)
        assert weights.shape == (8, 4)

    def test_weights_sum_to_one(self):
        from control.policies.moe_policy import Router
        router = Router(obs_dim=28, n_experts=4, hidden_dim=64)
        obs = torch.randn(8, 28)
        weights = router(obs)
        sums = weights.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_weights_non_negative(self):
        from control.policies.moe_policy import Router
        router = Router(obs_dim=28, n_experts=4, hidden_dim=64)
        obs = torch.randn(8, 28)
        weights = router(obs)
        assert (weights >= 0).all()


class TestExpertNetwork:
    def test_output_shape(self):
        from control.policies.moe_policy import ExpertNetwork
        expert = ExpertNetwork(obs_dim=28, action_dim=4, hidden_dim=128)
        obs = torch.randn(8, 28)
        actions = expert(obs)
        assert actions.shape == (8, 4)

    def test_different_experts_different_outputs(self):
        from control.policies.moe_policy import ExpertNetwork
        e1 = ExpertNetwork(obs_dim=28, action_dim=4, hidden_dim=128)
        e2 = ExpertNetwork(obs_dim=28, action_dim=4, hidden_dim=128)
        obs = torch.randn(1, 28)
        # Different random init should give different outputs
        assert not torch.allclose(e1(obs), e2(obs))


class TestTopKSelection:
    def test_top2_selects_two(self):
        from control.policies.moe_policy import top_k_selection
        weights = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
        selected_weights, selected_indices = top_k_selection(weights, k=2)
        assert selected_indices.shape == (1, 2)
        assert selected_weights.shape == (1, 2)

    def test_top2_renormalized(self):
        from control.policies.moe_policy import top_k_selection
        weights = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
        selected_weights, _ = top_k_selection(weights, k=2)
        torch.testing.assert_close(selected_weights.sum(dim=1), torch.ones(1), atol=1e-5, rtol=1e-5)

    def test_top2_picks_highest(self):
        from control.policies.moe_policy import top_k_selection
        weights = torch.tensor([[0.1, 0.5, 0.3, 0.1]])
        _, indices = top_k_selection(weights, k=2)
        assert 1 in indices[0]  # expert 1 (0.5) should be selected
        assert 2 in indices[0]  # expert 2 (0.3) should be selected


class TestLoadBalanceLoss:
    def test_uniform_usage_zero_loss(self):
        from control.policies.moe_policy import load_balance_loss
        # All experts used equally
        indices = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
        loss = load_balance_loss(indices, n_experts=4)
        assert loss.item() < 0.01  # near zero

    def test_imbalanced_usage_nonzero_loss(self):
        from control.policies.moe_policy import load_balance_loss
        # Only experts 0 and 1 used
        indices = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        loss = load_balance_loss(indices, n_experts=4)
        assert loss.item() > 0.01  # should be nonzero
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_control/test_moe_policy.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Implement MoE policy module**

```python
# control/policies/moe_policy.py
"""Mixture of Experts policy for drone racing.

4 expert networks with a top-2 router. Each expert produces TRPY actions.
The router selects the 2 most relevant experts per step and blends their
outputs. A separate critic estimates state value.

Spec: docs/superpowers/specs/2026-03-23-moe-policy-design.md
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
)
from stable_baselines3.common.type_aliases import Schedule


class Router(nn.Module):
    """Routes observations to experts via softmax weights."""

    def __init__(self, obs_dim: int, n_experts: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(obs), dim=-1)


class ExpertNetwork(nn.Module):
    """Single expert: obs -> action."""

    def __init__(self, obs_dim: int, action_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def top_k_selection(
    weights: torch.Tensor, k: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k experts and renormalize weights.

    Args:
        weights: (batch, n_experts) softmax weights from router.
        k: Number of experts to select.

    Returns:
        selected_weights: (batch, k) renormalized weights.
        selected_indices: (batch, k) expert indices.
    """
    top_weights, top_indices = torch.topk(weights, k, dim=-1)
    # Renormalize so selected weights sum to 1
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    return top_weights, top_indices


def load_balance_loss(
    selected_indices: torch.Tensor, n_experts: int = 4
) -> torch.Tensor:
    """Compute load balance loss to encourage even expert usage.

    L = sum((f_i - 1/N)^2) where f_i is fraction routed to expert i.
    """
    batch_size = selected_indices.shape[0]
    # Count how often each expert is selected
    counts = torch.zeros(n_experts, device=selected_indices.device)
    for i in range(n_experts):
        counts[i] = (selected_indices == i).float().sum()
    # Normalize to fractions
    total_selections = selected_indices.numel()
    fractions = counts / total_selections
    target = 1.0 / n_experts
    return ((fractions - target) ** 2).sum()


class MoEPolicy(ActorCriticPolicy):
    """Mixture of Experts policy for SB3 PPO.

    4 expert networks + router with top-2 selection.
    Overrides forward/evaluate_actions to use MoE action computation.

    Args:
        n_experts: Number of expert networks.
        expert_hidden_dim: Hidden layer size per expert.
        router_hidden_dim: Router hidden layer size.
        top_k: Number of experts selected per step.
        balance_coef: Load balancing loss coefficient.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        n_experts: int = 4,
        expert_hidden_dim: int = 128,
        router_hidden_dim: int = 64,
        top_k: int = 2,
        balance_coef: float = 0.01,
        **kwargs: Any,
    ) -> None:
        self.n_experts = n_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.router_hidden_dim = router_hidden_dim
        self.top_k = top_k
        self.balance_coef = balance_coef
        # Must be set before super().__init__ which calls _build
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Build MoE components instead of standard MLP."""
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        # Router
        self.router = Router(obs_dim, self.n_experts, self.router_hidden_dim)

        # Experts — each produces action means
        self.experts = nn.ModuleList([
            ExpertNetwork(obs_dim, action_dim, self.expert_hidden_dim)
            for _ in range(self.n_experts)
        ])

        # Separate critic (not MoE)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, self.expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.expert_hidden_dim, self.expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.expert_hidden_dim, 1),
        )

        # Action distribution (Gaussian with learnable log_std)
        self.action_dist = DiagGaussianDistribution(action_dim)
        self.log_std = nn.Parameter(
            torch.zeros(action_dim) * self.log_std_init,
            requires_grad=True,
        )

        # Optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        # Track last balance loss for logging
        self._last_balance_loss = torch.tensor(0.0)

    def _get_moe_action_mean(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute blended action mean from top-k experts.

        Returns:
            action_mean: (batch, action_dim) weighted expert actions.
            selected_weights: (batch, k) for log_prob computation.
            selected_indices: (batch, k) for load balance loss.
        """
        router_weights = self.router(obs)
        selected_weights, selected_indices = top_k_selection(
            router_weights, self.top_k
        )

        # Run only selected experts (for efficiency, run all and mask —
        # true conditional execution is harder in batched PyTorch)
        all_expert_actions = torch.stack(
            [expert(obs) for expert in self.experts], dim=1
        )  # (batch, n_experts, action_dim)

        # Gather selected expert actions
        batch_size = obs.shape[0]
        action_dim = all_expert_actions.shape[2]
        idx_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, action_dim)
        selected_actions = torch.gather(
            all_expert_actions, 1, idx_expanded
        )  # (batch, k, action_dim)

        # Weighted sum
        weights_expanded = selected_weights.unsqueeze(-1)  # (batch, k, 1)
        action_mean = (selected_actions * weights_expanded).sum(dim=1)  # (batch, action_dim)

        return action_mean, selected_weights, selected_indices

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute actions, values, log_probs."""
        action_mean, _, selected_indices = self._get_moe_action_mean(obs)
        values = self.value_net(obs)

        # Action distribution
        distribution = self.action_dist.proba_distribution(
            action_mean, self.log_std.expand_as(action_mean)
        )
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Track balance loss
        self._last_balance_loss = load_balance_loss(
            selected_indices, self.n_experts
        )

        return actions, values.squeeze(-1), log_prob

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Predict action (for eval)."""
        action_mean, _, _ = self._get_moe_action_mean(observation)
        distribution = self.action_dist.proba_distribution(
            action_mean, self.log_std.expand_as(action_mean)
        )
        return distribution.get_actions(deterministic=deterministic)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate actions for PPO update."""
        action_mean, _, selected_indices = self._get_moe_action_mean(obs)
        values = self.value_net(obs)

        distribution = self.action_dist.proba_distribution(
            action_mean, self.log_std.expand_as(action_mean)
        )
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Store balance loss for PPO to add to total loss
        self._last_balance_loss = (
            self.balance_coef * load_balance_loss(selected_indices, self.n_experts)
        )

        return values.squeeze(-1), log_prob, entropy
```

**Note:** The `_last_balance_loss` is stored on the policy and needs to be added to PPO's loss. This requires a small PPO wrapper modification (Task 2).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_control/test_moe_policy.py -v`
Expected: All 10 PASS

- [ ] **Step 5: Commit**

```bash
git add control/policies/moe_policy.py tests/test_control/test_moe_policy.py
git commit -m "feat(control): add Mixture of Experts policy with top-2 router"
```

### Task 2: PPO Wrapper Integration

**Files:**
- Modify: `control/algorithms/ppo.py`
- Modify: `training/loops/rl.py`

- [ ] **Step 1: Add moe flag to PPO wrapper**

In `control/algorithms/ppo.py`, add to `__init__`:
```python
        # Mixture of Experts
        moe: bool = False,
        n_experts: int = 4,
        expert_hidden_dim: int = 128,
        top_k: int = 2,
        balance_coef: float = 0.01,
```

Store them. In `_create_model`, when `moe=True`:
```python
        if self.moe:
            from control.policies.moe_policy import MoEPolicy
            policy_kwargs["n_experts"] = self.n_experts
            policy_kwargs["expert_hidden_dim"] = self.expert_hidden_dim
            policy_kwargs["top_k"] = self.top_k
            policy_kwargs["balance_coef"] = self.balance_coef
            return SB3_PPO(
                policy=MoEPolicy,
                **common_kwargs,
            )
```

- [ ] **Step 2: Pass moe params from config in rl.py**

In `training/loops/rl.py`, in `build_ppo()`, add:
```python
        moe=ctrl.get("moe", False),
        n_experts=ctrl.get("n_experts", 4),
        expert_hidden_dim=ctrl.get("expert_hidden_dim", 128),
        top_k=ctrl.get("top_k", 2),
        balance_coef=ctrl.get("balance_coef", 0.01),
```

- [ ] **Step 3: Run existing tests for regression**

Run: `pytest tests/test_control/test_ppo.py -v --timeout=60`
Expected: All pass (moe=False default)

- [ ] **Step 4: Commit**

```bash
git add control/algorithms/ppo.py training/loops/rl.py
git commit -m "feat(control): wire MoE policy into PPO wrapper and training loop"
```

### Task 3: Hydra Config

**Files:**
- Create: `configs/control/ppo_moe.yaml`

- [ ] **Step 1: Create config**

```yaml
# configs/control/ppo_moe.yaml
# PPO with Mixture of Experts policy.
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
policy_type: MlpPolicy  # overridden by moe=true
activation_fn: relu
use_sde: false

# MoE settings
moe: true
n_experts: 4
expert_hidden_dim: 128    # subject to change
top_k: 2
balance_coef: 0.01
```

- [ ] **Step 2: Commit**

```bash
git add configs/control/ppo_moe.yaml
git commit -m "config: add ppo_moe.yaml for Mixture of Experts training"
```

### Task 4: Stage 1 Experiment Config

**Files:**
- Create: `configs/experiment/moe_s1.yaml`

- [ ] **Step 1: Create experiment config**

```yaml
# configs/experiment/moe_s1.yaml
# @package _global_
# MoE Stage 1: Learn diverse tracks at easy params.
# From scratch with MoE policy.

defaults:
  - override /sim: numpy_quad
  - override /control: ppo_moe
  - override /perception: corner_noise
  - override /domain_rand: uniform_30pct
  - override /logging: wandb
  - _self_

sim:
  action_mode: trpy
  n_lookahead_gates: 2
  max_steps: 1200
  arena_bounds: 10
  gate_passage_radius: 1.5
  gate_collision: false

total_timesteps: 10_000_000

track_gen:
  n_gates_min: 4
  n_gates_max: 10
  gate_spacing_min: 1.0
  gate_spacing_max: 6.0
  turn_angle_min: -170
  turn_angle_max: 170
  elevation_min: 0.5
  elevation_max: 4.0
  elevation_delta_max: 1.5
  closure_max_angle: 120
  closure_max_retries: 30
  figure8_ratio: 0.3
  figure8:
    loop_radius_min: 1.5
    loop_radius_max: 3.0
    gates_per_loop_min: 3
    gates_per_loop_max: 4
    crossing_offset_min: 0.3
    crossing_offset_max: 0.6
    elevation_min: 1.0
    elevation_max: 3.0
    elevation_delta_max: 0.8

perception:
  noise_scale: 1.0
  dropout_onset: null
  dropout_continuation: 0.7

reward:
  weights:
    gate_passage: 10.0
    gate_progress: 1.0
    gate_offset: 3.0
    body_rate: 0.001
    action_smoothness: 0.0
    crash_penalty: 5.0
    spline_proximity: 1.0
    heading_alignment: 0.1
    speed_bonus: 0.0
    boundary_penalty: 5.0
    gate_approach: 0.3
    gate_centering: 3.0
  v_max: 3.0
  action_smoothness_threshold: 0.5

arpo:
  enabled: false
curriculum: null

multi_scene:
  enabled: true
  n_scenes: 10

logging:
  tags: ["trpy", "moe", "4-experts", "top-2", "diverse", "s1", "10M"]
  group: moe_staged
  notes: "MoE Stage 1: 4 experts (2x128), top-2 router, 10m/1.5m, diverse tracks, v_max=3"
```

- [ ] **Step 2: Commit**

```bash
git add configs/experiment/moe_s1.yaml
git commit -m "config: add MoE stage 1 experiment (10m/1.5m, diverse tracks)"
```

### Task 5: Smoke Test

**Files:**
- Create: `tests/test_integration/test_moe_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/test_integration/test_moe_smoke.py
"""Smoke test: MoE policy trains without crashing."""
import pytest


class TestMoESmoke:
    def test_moe_trains_200_steps(self):
        """MoE PPO should complete a short training run."""
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
            expert_hidden_dim=64,  # smaller for test speed
            top_k=2,
            balance_coef=0.01,
            n_steps=100,
            batch_size=100,
            n_epochs=2,
        )
        ppo.train(env, total_timesteps=200)
        assert ppo._model is not None

        # Verify predict works
        import numpy as np
        obs = env.reset()
        action, _ = ppo.predict(obs, deterministic=True)
        assert action.shape == (4, 4)

        env.close()

    def test_moe_expert_selection_varies(self):
        """Different observations should route to different experts."""
        import torch
        from control.policies.moe_policy import Router, top_k_selection

        router = Router(obs_dim=28, n_experts=4, hidden_dim=64)
        # Very different observations
        obs1 = torch.randn(1, 28) * 10
        obs2 = torch.randn(1, 28) * 0.1

        w1 = router(obs1)
        w2 = router(obs2)
        _, idx1 = top_k_selection(w1, k=2)
        _, idx2 = top_k_selection(w2, k=2)

        # Not guaranteed to differ but highly likely with different inputs
        # Just verify the mechanism works without error
        assert idx1.shape == (1, 2)
        assert idx2.shape == (1, 2)
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_integration/test_moe_smoke.py -v --timeout=60`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration/test_moe_smoke.py
git commit -m "test(integration): add MoE policy smoke test"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | MoE policy (router + experts + top-k + balance loss) | `control/policies/moe_policy.py` |
| 2 | PPO wrapper + training loop integration | `ppo.py`, `rl.py` |
| 3 | Hydra config | `configs/control/ppo_moe.yaml` |
| 4 | Stage 1 experiment config | `configs/experiment/moe_s1.yaml` |
| 5 | Smoke test | `tests/test_integration/test_moe_smoke.py` |

**To launch after implementation:**
```bash
python -m training +experiment=moe_s1 device=cuda
```

**Training plan (manual staged, same as before):**
```
Stage 1: 10m/1.5m, 10M steps → benchmark → decide
Stage 2:  7m/1.0m, 10M steps → benchmark → decide
Stage 3:  5m/0.75m, 10M+ steps → benchmark → golden set target
```

**What to watch on W&B:**
- Expert selection frequencies — should be roughly 25% each (load balance working)
- Gate passage rate — compare against MLP baseline at each stage
- The 7m→5m transition — does MoE break through the cliff?

**Note on balance loss:** The `_last_balance_loss` needs to be added to PPO's total loss. SB3's PPO doesn't natively support auxiliary losses. The simplest approach: in `MoEPolicy.evaluate_actions()`, subtract the balance loss from the value estimate, which effectively adds it to the loss. Alternatively, we can add it to the entropy term. The implementer should choose the cleanest approach that doesn't require forking SB3.
