# 5th MoE Expert — Capacity Expansion

**Date:** 2026-03-29
**Linear:** COR-78 (competition readiness)
**Status:** Approved

## Motivation

The generalist v2 experiment (40M steps with descent training + start_vel_std=2.0) regressed vs v1 — gen_seed_123 went from 5/5 lapping to 1/5 with 4 crashes. The 4-expert MoE is being stretched too thin: adding descent and fast-start capabilities diluted existing specializations.

COR-80 proved the architecture *can* handle all layouts when trained on them (golden set finetune: 9.8 gates, 6% crash). The bottleneck is capacity — 4 experts can't simultaneously specialize in enough flight regimes to be a strong generalist.

## Design

### Checkpoint Surgery Script

New script `control/expand_moe.py`:

1. **Load** the best 4-expert checkpoint (`outputs/2026-03-29/00-24-04/final_model.zip` — generalist v1, 7.3 gates, 10% crash)
2. **Profile** router selections by running 1000 steps across 100 envs. Count how often each expert is selected. Identify the most-used expert.
3. **Clone** the most-used expert's weights as expert 5
4. **Perturb** the clone with 1% Gaussian noise (`clone_weights += 0.01 * torch.randn_like(clone_weights)`) so it diverges during training
5. **Expand router** output layer from 4→5 neurons. Copy existing weights for columns 0-3. Initialize column 5 by copying the cloned expert's column with small perturbation.
6. **Update** `n_experts` metadata in the checkpoint so SB3 loads it correctly with the new config
7. **Save** expanded checkpoint

### Config Changes

`configs/control/ppo_moe.yaml`:
```yaml
n_experts: 5       # was 4
top_k: 2           # unchanged
balance_coef: 0.01 # unchanged — target shifts from 25% to 20% per expert automatically
expert_hidden_dim: 128  # unchanged
```

No changes to the MoE policy code — `n_experts` is already parameterized throughout (`Router`, `ExpertNetwork` list, `load_balance_loss`, `top_k_selection`).

### Training

- Resume from expanded 5-expert checkpoint
- Use generalist **v1** config (`moe_generalist.yaml` with elevation_bias random flip but `start_vel_std: 1.5`, NOT 2.0)
  - v1 scored 7.3 gates / 10% crash; v2 regressed to 7.2 / 16%
- Run 40M steps
- Benchmark on golden set

### What to monitor

- **Router usage distribution** — are all 5 experts getting ~20% usage? If the 5th expert stays at <5%, the load balance loss isn't enough and we need approach B (freeze+warmup) or C (auxiliary loss)
- **Per-layout results** — does the 5th expert help on previously failing tracks (elevation_climb, gen_seed_256, perturbed_fast)?
- **No regression** — oval, hairpin, figure8, gen_seed_77 should maintain current performance

### Param count impact

| Component | 4 experts | 5 experts | Delta |
|-----------|----------|----------|-------|
| Experts | 4 × (28×128 + 128 + 128×128 + 128 + 128×4 + 4) = 4 × 20,612 = 82,448 | 5 × 20,612 = 103,060 | +20,612 |
| Router | 28×64 + 64 + 64×4 + 4 = 2,116 | 28×64 + 64 + 64×5 + 5 = 2,181 | +65 |
| Value net | 28×128 + 128 + 128×128 + 128 + 128×1 + 1 = 20,609 | unchanged | 0 |
| log_std | 4 | unchanged | 0 |
| **Total** | ~105K | ~126K | **+21K (~20%)** |

Still very lightweight for Jetson Orin NX deployment.

### Fallback plan

If the 5th expert doesn't differentiate after 20M steps (usage stays <5% or no benchmark improvement):
- **Option B**: Freeze experts 0-4 + value net, train only expert 5 + router for 5M steps, then unfreeze all
- **Option C**: Add auxiliary loss that activates expert 5 on high-dynamic states (body_rate > threshold, vertical velocity > threshold)

## Files

| File | Action |
|------|--------|
| `control/expand_moe.py` | Create — checkpoint surgery script |
| `configs/control/ppo_moe.yaml` | Modify — n_experts: 4→5 |
| `configs/experiment/moe_generalist.yaml` | Modify — resume from expanded checkpoint |
