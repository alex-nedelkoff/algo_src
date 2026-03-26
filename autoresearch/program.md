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
