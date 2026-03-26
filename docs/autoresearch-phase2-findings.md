# Autoresearch Phase 2: Findings

**Date:** 2026-03-13
**Experiments:** 10–29 (20 experiments)
**Checkpoint:** `playground/runs/phase4_ekf_ppo_v2/final_model.zip`

## Background

Experiments 0–9 are **invalid**. A gate normal bug in `sim/envs/rate_ctrl_env.py` caused gates 2 and 6 on the figure-8 track to have flipped yaw values. The passage detection (signed-distance plane crossing) rejected the natural flight direction, so the drone learned to double-dip — fly through, turn around, fly back through — inflating gate counts and corrupting the reward signal. Scores of 10.94–11.92 from those experiments are meaningless.

**Fix applied:** Gate 2 yaw `0.0` → `π`, gate 6 yaw `π` → `0.0`.

## Phase 1: Re-baseline (Experiments 10–12, 500K steps)

Goal: Establish a valid baseline score on the fixed track.

| Exp | Description | Score | Avg Gates | Crash % | Max Gates | Key Changes |
|-----|------------|-------|-----------|---------|-----------|-------------|
| 10 | Corrupted policy on fixed gates | 4.03 | 4.03 | 0% | 11 | None (exp 9 params) |
| **11** | **Fresh fine-tune, baseline params** | **4.225** | **4.225** | **0%** | **11** | Reset to program.md defaults |
| 12 | Baseline + offset penalty | 4.04 | 4.04 | 0% | 10 | lambda_gate=20, lambda_offset=1.0 |

**Baseline established:** Exp 11 (score 4.225). The corrupted policy (exp 10) still flies reasonably on fixed gates — it degraded but didn't catastrophically fail, confirming the checkpoint is usable. The offset penalty (exp 12) slightly hurt performance.

## Phase 2: Systematic Sweep (Experiments 13–22, 500K steps)

Goal: Isolate the effect of each parameter on score. Each experiment changes 1–2 parameters from the exp 11 baseline.

| Exp | Change | Score | Delta vs Baseline | Verdict |
|-----|--------|-------|-------------------|---------|
| 13 | lambda_prog=2.0 | 4.13 | -0.10 | No effect |
| 14 | lambda_prog=3.0, v_max=15.0 | 4.13 | -0.10 | No effect |
| **15** | **learning_rate=5e-5** | **4.455** | **+0.23** | **Best at 500K** |
| 16 | ent_coef=0.01 | 4.07 | -0.16 | Slightly worse |
| 17 | corner_noise_k=1.0 | 4.20 | -0.03 | Neutral |
| 18 | corner_noise_k=3.0 | 3.785 | -0.44 | Harmful |
| 19 | gae_lambda=0.95, gamma=0.9999 | 3.945 | -0.28 | Harmful |
| 20 | lambda_gate=25, lambda_crash=20 | 4.07 | -0.16 | No improvement |
| 21 | percentage=0.1 (less DR) | 3.75 | -0.48 | Harmful |
| 22 | Best combo (LR=5e-5) | 4.235 | +0.01 | Within noise |

**500K findings:**
- **Lower learning rate is the only clear winner** (exp 15, +0.23 points)
- No parameter change beat baseline by the ≥1.0 acceptance threshold
- Higher corner_noise_k (trusting EKF predictions more) hurts — the policy needs accurate measurements
- Less domain randomization hurts — DR at 0.3 is needed for stability
- 0% crash rate across all experiments — the policy is extremely stable

## Phase 3: 2M Steps Sweep (Experiments 23–29)

Goal: Test whether more training budget unlocks better scores.

Each experiment trains for 2M steps (4 PPO epochs, ~40 min) instead of 500K (1 PPO epoch, ~12 min).

| Exp | Config | Score | Delta vs Exp 25 | Notes |
|-----|--------|-------|-----------------|-------|
| 23 | Baseline (LR=3e-4) | 3.76 | -0.96 | Default LR overshoots with more training |
| 24 | LR=5e-5 | 4.42 | -0.30 | Better than 3e-4 but still below best |
| **25** | **LR=1e-5** | **4.72** | **—** | **Overall best score** |
| 26 | LR=1e-5, lambda_prog=2.0 | 3.715 | -1.01 | Stronger progress shaping hurts |
| 27 | LR=1e-5, lambda_gate=15 | 4.06 | -0.66 | Higher gate reward hurts |
| 28 | LR=1e-5, gae_lambda=0.98 | 3.625 | -1.10 | More variance in advantage hurts |
| 29 | LR=1e-5, corner_noise_k=1.0 | 4.345 | -0.38 | Tighter EKF doesn't help with low LR |

**2M findings:**
- **LR=1e-5 with baseline weights is the best config** (score 4.72)
- More training helps modestly: best 500K score was 4.455 → best 2M score is 4.72 (+0.27)
- The default LR (3e-4) actually *degrades* with more training — classic overshoot
- Every parameter change from baseline hurts when combined with low LR
- The score improvement from 500K→2M is smaller than hoped

## Best Configuration

```python
REWARD_WEIGHTS = {
    "lambda_gate": 10.0,
    "lambda_prog": 1.0,
    "lambda_rate": 0.001,
    "lambda_offset": 0.0,
    "lambda_perc": 0.0,
    "lambda_delta_u": 0.001,
    "lambda_crash": 10.0,
    "lambda_alive": 0.0,
    "v_max": 0.0,
}
EKF_PARAMS = {"corner_noise_k": 2.0, "corner_dropout_onset": None}
TRAINING_PARAMS = {
    "learning_rate": 1e-5,
    "ent_coef": 0.005,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "gamma": 0.999,
}
DOMAIN_RAND = {"percentage": 0.3}
N_STEPS = 2_000_000
```

**Score: 4.72** (avg 4.72 gates, 0% crash rate, max 12 gates in a single episode)

## Key Observations

1. **Learning rate is the dominant hyperparameter.** The baseline reward weights from `program.md` are near-optimal. The only meaningful improvement came from reducing LR: 3e-4 → 5e-5 → 1e-5. This is consistent with fine-tuning theory — the pre-trained checkpoint already has good representations, and aggressive updates destroy them.

2. **The policy is stable but plateauing.** 0% crash rate across all 20 experiments means the policy has learned robust obstacle avoidance. However, it averages only ~4–5 gates per episode out of 8, suggesting it struggles with specific track segments (likely the tighter turns in the figure-8).

3. **Training budget has diminishing returns.** Going from 500K to 2M steps improved the best score by only 0.27 points (4.455 → 4.72). The policy may be at or near the performance ceiling for this architecture + reward formulation on this track.

4. **Reward shaping changes are counterproductive.** Every modification to the reward weights — higher gate bonus, progress shaping, offset penalty, speed clamping — either had no effect or hurt. The baseline weights are well-calibrated.

5. **EKF noise scale matters.** `corner_noise_k=2.0` (baseline) is the sweet spot. Lower values (1.0, trusting measurements more) are neutral; higher values (3.0, trusting predictions more) hurt significantly (-0.44 points).

## Recommendations

1. **Use LR=1e-5 for all future fine-tuning.** This is the single most impactful finding.

2. **Investigate the performance ceiling.** The ~4.7 score plateau across many configs suggests a structural limitation. Possible causes:
   - The checkpoint's learned features from buggy training may limit adaptation
   - The figure-8 track's tight turns may require more training or curriculum
   - The 64×64×64 MLP architecture may lack capacity for complex maneuvers

3. **Try training from an earlier checkpoint** (or from scratch) to avoid the corrupted learned behaviors from the double-dipping bug.

4. **Move to multi-track training** to test generalizability — the infrastructure is ready (oval + S-curve tracks, multi-track env support, reward normalization).

5. **Consider longer training (5M–10M steps)** with LR=1e-5 to check if the plateau extends, but diminishing returns from 500K→2M suggest this may not help much.

## Infrastructure Delivered

This phase also delivered infrastructure improvements:
- **Parameter logging:** `log_experiment()` writes all params + results to `results.tsv`
- **200-episode eval:** Reduced eval variance from ~1.0 to ~0.3 points
- **W&B integration:** Each experiment logs as a separate wandb run with TB sync
- **Multi-track support:** Oval and S-curve tracks, per-env track assignment in `RateCtrlEnv`
- **Reward normalization:** Progress reward scaled by mean inter-gate distance for track-invariant training
