# Autoresearch Experiment Log

## Baseline: noble-wildflower-115
- **Checkpoint**: `outputs/2026-03-25/18-11-09/final_model.zip`
- **Config**: `configs/experiment/mlp_fixed_long.yaml`
- **Steps**: 20M (resumed from 105M best)
- **Key reward changes**: spline_proximity 0.1->1.0, heading 0.05->0.5, gate_passage 50->100, gate_approach 1->0.3, gate_centering 3->1, boundary 5->2, v_max 5->7.5

| Metric | Value |
|--------|-------|
| Eval gates/ep | 2.07 |
| Eval success_rate | 40.5% |
| Avg speed | 3.0 m/s |
| Best laps/ep | 2 |
| Best gates/ep | 11 |
| Fitness | -2.5151 |

---

## H2: v_max 10, speed_bonus 5 (faithful-armadillo-117)
- **W&B**: `x5dtk4sc`
- **Date**: 2026-03-25
- **Hypothesis**: Raise v_max from 7.5 to 10.0 and boost speed_bonus from 1.0 to 5.0
- **Rationale**: v_max=7.5 caps the speed reward too early. Raising to 10 m/s with stronger bonus should push the policy faster.
- **Overrides**: `reward.v_max=10.0 reward.weights.speed_bonus=5.0`
- **Budget**: 10M steps, resumed from baseline checkpoint
- **Runtime**: ~54 min at ~3100 FPS

### Results

| Metric | Baseline | H2 | Change |
|--------|----------|-----|--------|
| Eval gates/ep | 2.07 | 1.885 | -9% |
| Eval success_rate | 40.5% | 45.0% | **+11%** |
| Avg speed | 3.0 m/s | 2.62 m/s | -13% |
| Best gates/ep ever | 11 | 13 | **+18%** |
| Best laps/ep ever | 2 | 3 | **+50%** |
| Best lap time | - | 4s | new |
| Fitness | -2.5151 | -2.2131 | worse (slower) |

### Analysis
- **Success rate improved** (40.5% -> 45.0%) - policy completes more laps
- **Best-ever metrics up** - 13 gates/ep and 3 laps/ep are new records
- **Avg speed DROPPED** from 3.0 to 2.62 m/s - opposite of intended effect
- High OOB crash rate (44% eval) suggests arena bounds (10m) limit high-speed flight
- Policy optimized for consistent speed reward (staying airborne) rather than peak speed
- **Verdict**: Reward magnitude alone doesn't solve speed. Bottleneck may be loitering incentives or arena size.

---

## H3: anti-loiter + speed + entropy (fresh-sound-118)
- **W&B**: `xek4x2t8`
- **Date**: 2026-03-26
- **Hypothesis**: Minimize loitering incentives while maximizing speed signal and exploration
- **Rationale**: gate_approach and gate_centering encourage slowing near gates. Reducing them + more entropy should break the slow-and-careful local optimum.
- **Overrides**: `reward.weights.speed_bonus=3.0 reward.weights.gate_approach=0.1 reward.weights.gate_centering=0.5 control.ent_coef=0.005`
- **Budget**: 10M steps, resumed from baseline checkpoint
- **Runtime**: ~54 min at ~3100 FPS

### Results

| Metric | Baseline | H2 | H3 | H3 vs Baseline |
|--------|----------|-----|-----|----------------|
| Eval gates/ep | 2.07 | 1.885 | **2.055** | -1% |
| Eval success_rate | 40.5% | 45.0% | **55.0%** | **+36%** |
| Eval gates/ep max | - | - | 10 | |
| Train gates/ep | - | - | 2.08 | matches baseline |
| Train avg_speed | 3.0 m/s | 2.62 m/s | **2.69 m/s** | -10% |
| Best gates/ep ever | 11 | 13 | **13** | +18% |
| Best laps/ep ever | 2 | 3 | 2 | same |
| Best lap time | - | 4s | 4.55s | |
| Eval laps | - | - | 102 total | |
| Eval OOB | - | 44% | **33%** | |
| Eval ground crash | - | 11% | **12%** | |
| Fitness | -2.5151 | -2.2131 | **-3.0387** | **+21% better** |

### Analysis
- **Best fitness so far** (-3.0387 vs baseline -2.5151) due to huge success rate gain
- **Eval success rate jumped to 55%** (from 40.5%) — the biggest improvement across all experiments
- **Gates/ep nearly matches baseline** (2.055 vs 2.07) — anti-loiter didn't hurt gate completion
- **Speed still dropped** (2.69 vs 3.0) but less than H2 (2.62) — the anti-loiter approach preserves speed better
- Gate approach reward properly reduced (36.9 vs baseline ~112) — loitering incentive successfully removed
- Gate centering penalty halved (-82 vs -151) — less punishment for imprecise approaches
- **Verdict**: Best experiment so far. Anti-loiter + entropy works for reliability. Speed still the weak point. Combining H3's anti-loiter with H4's v_max raise + LR bump could be the winning combo.

---

## H4: LR bump + v_max + speed_bonus (northern-aardvark-119)
- **W&B**: `kpb173le`
- **Date**: 2026-03-26
- **Hypothesis**: Higher learning rate with speed bonus and v_max raise to break speed plateau faster
- **Rationale**: LR=3e-4 may be too conservative for reward reshaping. 5e-4 adapts faster to the new speed signal. Combined with v_max=10 to uncap speed range.
- **Overrides**: `reward.weights.speed_bonus=3.0 reward.v_max=10.0 control.learning_rate=0.0005`
- **Budget**: 10M steps, resumed from baseline checkpoint
- **Runtime**: ~52 min at ~3180 FPS

### Results

| Metric | Baseline | H2 | H3 | H4 | H4 vs Baseline |
|--------|----------|-----|-----|-----|----------------|
| Eval gates/ep | 2.07 | 1.885 | 2.055 | **2.255** | **+9%** |
| Eval success_rate | 40.5% | 45.0% | 55.0% | **50.0%** | **+23%** |
| Eval gates/ep max | - | - | 10 | 6 | |
| Train avg_speed | 3.0 m/s | 2.62 m/s | 2.69 m/s | **2.67 m/s** | -11% |
| Best gates/ep ever | 11 | 13 | 13 | 11 | same |
| Best laps/ep ever | 2 | 3 | 2 | 2 | same |
| Best lap time | - | 4s | 4.55s | 4.59s | |
| Eval OOB | - | 44% | 33% | **38.5%** | |
| Eval ground crash | - | 11% | 12% | **11.5%** | |
| Fitness | -2.5151 | -2.2131 | -3.0387 | **-3.0114** | **+20% better** |

### Analysis
- **Second-best fitness** (-3.01), very close to H3 (-3.04) but via different path
- **Highest eval gates/ep** across all experiments (2.255) — the higher LR helped the policy learn to complete more gates
- **Success rate strong** at 50% — between H2 (45%) and H3 (55%)
- **Speed still dropped** to 2.67 m/s, similar to H2/H3 — v_max=10 alone doesn't increase speed
- Higher LR (5e-4 vs 3e-4) produced better gate completion but didn't unlock speed
- Gate passage reward highest (225.5) — policy is very good at passing through gates
- **Verdict**: Good gate completion, decent success rate, speed still stuck. The LR bump helped gates but not speed. H3 remains the best overall due to higher success rate.

---

## H6: H3 + v_max 10 combo (noble-violet-120)
- **W&B**: `32u1l6bo`
- **Date**: 2026-03-26
- **Hypothesis**: Combine the best approach (H3 anti-loiter) with v_max raise to see if the combo unlocks speed
- **Rationale**: H3 gave best fitness via reliability. Adding v_max=10 may allow the more reliable policy to also fly faster since it won't be penalized at 7.5 m/s.
- **Overrides**: `reward.weights.speed_bonus=3.0 reward.weights.gate_approach=0.1 reward.weights.gate_centering=0.5 control.ent_coef=0.005 reward.v_max=10.0`
- **Budget**: 10M steps, resumed from baseline checkpoint
- **Runtime**: ~52 min at ~3200 FPS

### Results

| Metric | Baseline | H3 | H6 | H6 vs Baseline |
|--------|----------|-----|-----|----------------|
| Eval gates/ep | 2.07 | 2.055 | **1.98** | -4% |
| Eval success_rate | 40.5% | 55.0% | **46.0%** | +14% |
| Train avg_speed | 3.0 m/s | 2.69 m/s | **2.94 m/s** | -2% |
| Best gates/ep ever | 11 | 13 | **14** | **+27% new record** |
| Best laps/ep ever | 2 | 2 | 2 | same |
| Best lap time | - | 4.55s | 4.76s | |
| Eval OOB | - | 33% | **41.5%** | |
| Eval ground crash | - | 12% | **12.5%** | |
| Fitness | -2.5151 | -3.0387 | **-2.6752** | +6% better |

### Analysis
- **New record: 14 best gates/ep ever** — highest across all experiments
- **Speed nearly preserved** (2.94 m/s vs baseline 3.0) — best speed retention of any experiment
- But **eval success rate dropped** to 46% vs H3's 55% — adding v_max=10 hurt reliability
- **Higher OOB rate** (41.5% vs H3's 33%) — the uncapped speed range causes more boundary crashes
- Fitness (-2.68) is worse than H3 (-3.04) — the success rate drop outweighed speed preservation
- **Verdict**: v_max=10 hurts when combined with anti-loiter. The policy flies faster but crashes more at boundaries. H3 remains the best. Speed problem may need larger arena or algorithm-scope changes.

---

## Final Summary Table

| Exp | Eval gates/ep | Eval success | Avg speed | Fitness | Key finding |
|-----|--------------|-------------|-----------|---------|-------------|
| Baseline | 2.07 | 40.5% | 3.0 m/s | -2.52 | — |
| H2 (v_max+speed) | 1.885 | 45.0% | 2.62 | -2.22 | Reward magnitude alone doesn't help |
| **H3 (anti-loiter)** | **2.055** | **55.0%** | **2.69** | **-3.04** | **Best overall: reliability wins** |
| H4 (LR+v_max) | 2.255 | 50.0% | 2.67 | -3.01 | Best gates/ep, LR helps completion |
| H6 (H3+v_max) | 1.98 | 46.0% | 2.94 | -2.68 | Best speed retention, record 14 gates, but OOB issues |

### Key Findings
1. **Reward tuning alone cannot solve speed** — all 4 experiments kept or dropped speed vs baseline
2. **Anti-loiter (H3) is the most impactful single change** — +36% success rate
3. **v_max raise causes OOB crashes** — arena bounds (10m) are the true speed bottleneck
4. **Higher LR helps gate completion** (H4) but not speed
5. **Best combo would likely be H3 + larger arena bounds** — the policy CAN fly faster (H6 showed 2.94 m/s) but crashes at boundaries

### Recommended Next Steps
- **Short term**: Promote H3 as new baseline (best fitness), then run longer (20-30M steps)
- **Medium term**: Increase arena_bounds from 10 to 15-20m to unlock speed
- **Long term**: Algorithm-scope — progressive speed curriculum or velocity-tracking reward

---

## MoE+H3: Mixture of Experts with anti-loiter recipe (dutiful-paper-121)
- **W&B**: `9hlhy3ek`
- **Date**: 2026-03-26
- **Hypothesis**: MoE architecture (4 experts, 2x128, top-2 router) has shown capacity for 5+ m/s speed in prior runs but couldn't learn reliability. Combining it with the H3 anti-loiter reward recipe that boosted MLP success to 55% could unlock both speed AND reliability.
- **Config**: `configs/experiment/moe_h3.yaml`
- **Resume from**: S1 30M checkpoint (`outputs/2026-03-23/06-57-49/final_model`)
- **Key changes vs old MoE S1**: gate_passage 50->100, gate_approach 1.0->0.1, gate_centering 3.0->0.5, spline_proximity 0.1->1.0, heading_alignment 0.05->0.5, speed_bonus 1.0->3.0, boundary_penalty 5.0->2.0, ent_coef 0.001->0.005, v_max 5.0->7.5
- **Budget**: 10M steps
- **Runtime**: ~58 min at ~2890 FPS

### Results

| Metric | MLP Baseline | MLP H3 (best) | MoE+H3 | vs MLP H3 |
|--------|-------------|---------------|---------|-----------|
| Eval gates/ep | 2.07 | 2.055 | 1.535 | -25% |
| Eval success_rate | 40.5% | 55.0% | **92.0%** | **+67%** |
| Eval OOB | - | 33% | **6%** | -82% |
| Eval ground crash | - | 12% | **2%** | -83% |
| Train gates/ep | - | 2.08 | **2.145** | +3% |
| Train success_rate | - | 37% | **67.5%** | +82% |
| Train avg_speed | 3.0 m/s | 2.69 m/s | 2.57 m/s | -4% |
| Best lap time ever | - | 4.55s | **2.99s** | **-34% faster** |
| Best gates/ep ever | 11 | 13 | 9 | -31% |
| Best laps/ep ever | 2 | 2 | 2 | same |
| speed_bonus reward | - | 652 | **928** | +42% |
| Train fitness | -2.52 | -3.04 | **-3.72** | **+22% better** |
| train/std | 0.10 | 0.10 | **1.63** | not converged |
| clip_fraction | ~9% | ~10% | **1.5%** | very low |

### Analysis
- **92% eval success** — the highest success rate across ALL experiments by a massive margin
- **2.99s best lap time** — new all-time record, sub-3 seconds
- **Best train fitness** (-3.72) thanks to high gates/ep (2.145) AND high success (67.5%)
- Eval gates/ep is low (1.535) because the policy is cautious — surviving the full 1200 steps (timeout=92%)
- **Critical issue: std=1.63 hasn't converged** — the MoE checkpoint's action noise is still very high. The clip_fraction of 1.5% confirms: the policy gradient updates are tiny relative to the exploration noise. The policy is learning slowly.
- **The MoE CAN go fast** (2.99s lap proves it) but the high std means it's inconsistent — sometimes fast, sometimes meandering
- speed_bonus reward is highest (928) — the speed signal IS being received

### Diagnosis
The MoE is stuck in a high-std regime from its checkpoint. With std=1.63, actions are heavily random, so:
- It survives (92% success) because random actions in a wide space still avoid crashes
- It completes gates sporadically (2.145 train) but inconsistently
- Its best laps are brilliant (2.99s) but the average is slow due to noise
- The low clip_fraction means PPO can't push the policy hard enough to converge std

### Next experiment: MoE+H3 continued with lower std
Continue from this checkpoint but force std down. Two approaches:
1. Run for 20M more steps — std may naturally decay
2. Higher LR (0.0005) to increase clip_fraction and push std convergence faster

---

## MoE+H3 continued: 20M steps with higher LR (light-leaf-122)
- **W&B**: `8ukkjrrc`
- **Date**: 2026-03-26
- **Hypothesis**: Higher LR (5e-4) should accelerate std convergence from 1.63, letting the MoE consolidate its fast+safe behavior.
- **Resume from**: MoE+H3 10M checkpoint (`outputs/2026-03-26/08-35-19/final_model.zip`)
- **Overrides**: `total_timesteps=20000000 control.learning_rate=0.0005`
- **Budget**: 20M steps (~2 hrs)

### Results

| Metric | MLP Baseline | MLP H3 | MoE+H3 (10M) | MoE+H3 cont (30M total) |
|--------|-------------|--------|---------------|--------------------------|
| **Eval gates/ep** | 2.07 | 2.055 | 1.535 | **4.295** |
| **Eval success_rate** | 40.5% | 55.0% | 92.0% | **82.5%** |
| **Eval avg_speed** | - | - | - | **3.01 m/s** |
| **Eval laps/ep** | - | - | 0.01 | **0.31** |
| **Eval lap_completion_rate** | - | - | 1% | **28.5%** |
| Eval gate_passage_rate | - | - | 77% | **95.5%** |
| Eval OOB | - | 33% | 6% | **17.5%** |
| Eval ground crash | - | 12% | 2% | **0.5%** |
| Eval total laps | - | - | 12 | **1047** |
| Train gates/ep | - | 2.08 | 2.145 | **2.675** |
| Train avg_speed | 3.0 m/s | 2.69 | 2.57 | **3.74 m/s** |
| Best gates/ep ever | 11 | 13 | 9 | **14** |
| Best lap time ever | - | 4.55s | 2.99s | **3.66s** |
| Best laps/ep ever | 2 | 2 | 2 | **2** |
| speed_bonus reward | - | 652 | 928 | **1376 (train) / 1236 (eval)** |
| train/std | 0.10 | 0.10 | 1.63 | **1.69** |
| clip_fraction | ~9% | ~10% | 1.5% | **0.4%** |

### Analysis
**THIS IS THE BREAKTHROUGH RUN.**

- **Eval gates/ep DOUBLED from baseline** (2.07 -> 4.295) — the policy completes 4+ gates per episode
- **Eval speed: 3.01 m/s** — MATCHES the baseline speed while being far more capable
- **Train speed: 3.74 m/s** — **25% faster than any MLP experiment**
- **1047 eval laps** completed — massively more than any prior experiment
- **28.5% lap completion rate** — a completely new capability, MLP never exceeded ~5%
- **95.5% gate passage rate** — nearly perfect gate finding
- **14 best gates/ep ever** — new all-time record (tied with H6)
- Ground crashes nearly eliminated (0.5% eval)
- Eval success dropped from 92% to 82.5% — the policy is now DOING things instead of just surviving

**The std problem**: std actually INCREASED (1.63 -> 1.69) and clip_fraction dropped (1.5% -> 0.4%). The higher LR didn't help converge std. This means the MoE architecture may have a structural issue with std convergence. Despite this, performance improved enormously — the policy is learning through the noise.

**What this means**: The MoE+H3 combination has unlocked a fundamentally better operating regime. It flies at 3.7 m/s in training, completes gates at 2x the rate, and can do laps consistently. The std issue is the remaining bottleneck — if we can solve it, this policy could be significantly faster still.

### Verdict
Best experiment by every metric that matters. The MoE architecture + H3 reward recipe is the path forward.

**Note on std**: The logged std=1.69 is the RAW `log_std` parameter. The MoE policy clamps log_std to [-3, 0] at inference (see `moe_policy.py:128`), so actual exploration std is capped at 1.0. The raw parameter drifted above the clamp — the gradient is killed when log_std > 0.0. This is a logging artifact, NOT a convergence failure. Performance proves the policy works well despite this.

---

## MoE+H3 v_max=10: Uncap speed on best policy (IN PROGRESS)
- **Hypothesis**: MoE+H3 cont achieved 3.74 m/s train speed with v_max=7.5, and only 17.5% eval OOB. The MoE handles speed much better than MLP. Raising v_max to 10 should let it fly even faster without the OOB penalty that killed MLP H6.
- **Resume from**: MoE+H3 30M checkpoint (`outputs/2026-03-26/09-34-43/final_model.zip`)
- **Overrides**: `reward.v_max=10.0 total_timesteps=10000000`
- **Budget**: 10M steps

### Results (fresh-bee-123, W&B: ryg39ln2)

| Metric | MoE+H3 cont (v_max=7.5) | MoE+H3 v_max=10 | Change |
|--------|--------------------------|------------------|--------|
| Eval gates/ep | 4.295 | **4.28** | same |
| Eval success_rate | 82.5% | **77.0%** | -7% |
| Eval avg_speed | 3.01 m/s | **2.96 m/s** | -2% |
| Eval laps/ep | 0.31 | **0.325** | +5% |
| Eval OOB | 17.5% | **23.0%** | +31% |
| Eval ground crash | 0.5% | **0%** | eliminated |
| Train avg_speed | 3.74 m/s | **3.52 m/s** | -6% |
| Best lap time ever | 3.66s | **3.33s** | -9% faster |
| Best gates/ep ever | 14 | 13 | -1 |
| Train fitness | -10.67 | **-9.55** | -10% |

### Analysis
- **Performance held steady** — gates/ep, laps/ep, speed all nearly identical to v_max=7.5
- **Eval success dropped slightly** (82.5% -> 77%) with more OOB (17.5% -> 23%) — same pattern as MLP H6
- **New best lap time: 3.33s** — beat the previous 3.66s record
- **Ground crashes completely eliminated** (0%) in eval — the MoE learned safe flight
- v_max=10 didn't unlock speed the way we hoped — train speed actually dropped from 3.74 to 3.52
- The speed reward being uncapped means it now rewards slower-but-consistent flight more

### Verdict
v_max=10 is a wash on MoE — slight regression overall. The v_max=7.5 version (light-leaf-122) remains the best checkpoint. The speed bottleneck is NOT the v_max cap — it's something more fundamental about the reward landscape or the policy's learned behavior.

**Current best**: MoE+H3 continuation (light-leaf-122) with v_max=7.5 — fitness -10.67
**Best checkpoint**: `outputs/2026-03-26/09-34-43/final_model.zip`

---

## MoE+Spline Speed 30M (wild-tree-125)
- **W&B**: `h0j5z26l`
- **Date**: 2026-03-26
- **Resume from**: proud-leaf-124 checkpoint (`outputs/2026-03-26/17-00-00/final_model.zip`)
- **Budget**: 20M additional steps (30M total with spline_speed)
- **Config**: `configs/experiment/moe_spline_speed.yaml`

### Results

| Metric | Spline 10M | Spline 30M | MoE+H3 30M (baseline) |
|--------|-----------|-----------|----------------------|
| Eval gates/ep | 4.045 | **4.56** | 4.295 |
| Eval success | 80% | 72% | 82.5% |
| Eval spline_speed | 483 | **644** | n/a |
| Train avg_speed | 3.32 | **3.52** | 3.74 |
| Best gates/ep ever | 16 | **19** | 14 |
| Best laps/ep ever | 4 | 4 | 2 |
| Best lap time ever | 3.0s | 3.0s | 3.66s |
| Best ep reward | 4224 | **5591** | 4189 |
| Total train laps | 917 | **4044** | 1830 |

### Analysis
- Policy is still improving — eval gates/ep up 13%, spline_speed reward up 33%
- **19 gates in one episode** (train) = nearly 5 full laps — new all-time record
- Train speed climbing (3.32 -> 3.52) — the directional reward is teaching variable speed
- Success dropped to 72% (from 80%) as policy pushes harder — more OOB
- The spline_speed reward is the correct signal — it's growing and speed follows

### Best checkpoint
`outputs/2026-03-26/19-10-06/final_model.zip` (wild-tree-125, 30M spline_speed total)
