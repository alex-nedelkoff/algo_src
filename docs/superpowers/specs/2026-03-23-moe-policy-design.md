# Mixture of Experts Policy Design Spec

**Date:** 2026-03-23
**Status:** Approved
**Related:** COR-70 (generalist training), golden set benchmark

## Problem

The single 3×64 MLP (~10K params) hits a performance cliff at 7m arena / 1.0m gate radius. Any tightening toward the golden set's 5m/0.75m causes catastrophic collapse. The hypothesis: the single network can't encode both "fly fast on wide tracks" and "fly precisely in tight spaces" simultaneously. Different flight regimes need different control strategies that interfere when packed into one small network.

## Solution

Replace the single MLP with a Mixture of Experts (MoE) policy: 4 expert networks that each learn specialized strategies, with a router that selects the top-2 most relevant experts per timestep. Each expert is larger than the current entire policy (2×128 vs 3×64), giving substantially more capacity while keeping per-step inference efficient (only 2 of 4 experts run).

## Architecture

```
Observation (28-dim)
    │
    ├──→ Router (28 → 64 → 4)  → softmax → top-2 selection
    │
    ├──→ Expert 1 (28 → 128 → 128 → 4)  TRPY output
    ├──→ Expert 2 (28 → 128 → 128 → 4)  TRPY output
    ├──→ Expert 3 (28 → 128 → 128 → 4)  TRPY output
    └──→ Expert 4 (28 → 128 → 128 → 4)  TRPY output
                    │
                    ▼
    Weighted sum of top-2 expert actions → final TRPY (4-dim)
```

### Components

**Router:** Small network (28→64 ReLU→4 softmax). Takes the full observation and outputs 4 weights. The top-2 weights are renormalized to sum to 1.0; the other 2 experts don't execute. ~2K params.

**Experts:** 4 identical-architecture networks, each 2×128 ReLU→4 linear. Each is a complete obs→action pipeline. Only the top-2 selected by the router execute per step. ~33K params each.

**Expert roles:** Not hardcoded — they emerge from training. Conceptually we expect specialization into precision/speed/turning/recovery, but the router learns the optimal partitioning.

**Critic:** Separate standard MLP (28→128→128→1), NOT MoE. The critic estimates state value — it doesn't need specialized action strategies. ~17K params.

### Parameter Budget

| Component | Params | Runs per step |
|-----------|--------|:---:|
| Router | ~2K | Always |
| Expert 1 | ~33K | If selected |
| Expert 2 | ~33K | If selected |
| Expert 3 | ~33K | If selected |
| Expert 4 | ~33K | If selected |
| Critic | ~17K | Always |
| **Total** | **~151K** | **~85K per step** |

Per-step inference: router (2K) + 2 experts (66K) + critic (17K) = ~85K params. Swift ran 2×128 (~20K) at 500Hz on Jetson. Our per-step cost is ~4× Swift but still well within budget for 500Hz.

**Note:** Expert size (2×128) is the starting point and subject to change based on training results. Could scale down to 2×64 (~35K total) or up to 3×128 (~200K total).

### Top-2 Selection

Each step:
1. Router outputs 4 logits, apply softmax → 4 weights
2. Select indices of top-2 weights
3. Renormalize top-2 weights to sum to 1.0
4. Execute only the 2 selected experts
5. Final action = w1 * expert_a(obs) + w2 * expert_b(obs)

Gradients flow through both the router (which experts to pick) and the selected experts (what action to produce). Non-selected experts receive no gradient on that step.

### Load Balancing Loss

Without encouragement, the router may always select the same 2 experts. An auxiliary loss penalizes uneven usage:

```
L_balance = α * Σ (f_i - 1/N)²
```

Where `f_i` is the fraction of batch samples routed to expert `i`, N=4, α=0.01. This is added to PPO's loss. It's small enough not to dominate RL training but sufficient to prevent expert collapse.

## SB3 Integration

This requires a **custom policy class** (not just a feature extractor) because the MoE operates at the action level, not the feature level.

**`MoEPolicy(ActorCriticPolicy)`** overrides:
- `forward()` — runs router + top-2 experts for action, standard critic for value
- `_predict()` — deterministic action selection (for eval)
- `evaluate_actions()` — computes log_prob and entropy for PPO update

Follows the same pattern as the existing `AsymmetricPolicy` in `control/policies/asymmetric_policy.py`.

## Training Plan

From scratch using the manual staged approach that worked:
- Stage 1: 10m/1.5m diverse tracks, v_max=3 (10M steps)
- Stage 2: 7m/1.0m (10M steps)
- Stage 3: 5m/0.75m (10M+ steps)
- Benchmark at each stage

The MoE's additional capacity should allow the policy to handle tighter constraints without the catastrophic forgetting that the 10K MLP suffered.

## What Does NOT Change

- GateRaceEnv — no modifications
- VecEnvAdapter — no modifications
- Reward functions — all 12 components work as-is
- Track generation — same diverse config
- Benchmark — same golden set evaluation
- Observation space — same 28-dim

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Expert collapse (all load on 2 experts) | Load balancing loss (α=0.01) |
| Slower training (more params) | GPU acceleration (3500+ FPS), top-2 reduces per-step compute |
| Router instability (oscillating selections) | Softmax temperature, gradient clipping |
| ONNX export complexity | Top-k is supported in ONNX opset 11+ |
| Harder to debug | Log expert selection frequencies to W&B for monitoring |
