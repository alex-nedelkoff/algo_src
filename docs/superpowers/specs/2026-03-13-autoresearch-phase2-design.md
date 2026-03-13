# Autoresearch Phase 2: Re-baseline + Generalizability

## Context

The first 10 autoresearch experiments (exp 0-9) are **invalid**. A gate normal
bug in `sim/envs/rate_ctrl_env.py` caused gates 2 and 6 on the figure-8 track
to have flipped yaw values. The passage detection (signed-distance plane
crossing, negative-to-positive) rejected the natural flight direction and only
registered backwards traversals. The drone learned to double-dip — fly through,
turn around, fly back through — inflating gate counts and corrupting the reward
signal.

**Fix applied:** Gate 2 yaw `0.0` → `π`, gate 6 yaw `π` → `0.0`. All 8 gates
now have correct signed distances (negative on approach, positive on exit).

The prior experiments showed scores of 10.94–11.92 with 0% crash rate and max
13 gates, but these numbers are not meaningful. We need a clean re-baseline
before continuing the parameter search.

## Goals

1. Establish a valid baseline score on the fixed track
2. Run a systematic parameter sweep to find the best configuration
3. Make the autoresearch pipeline produce results that generalize across tracks
4. Fix infrastructure gaps (param logging, eval sample size)

## Design

### 1. Infrastructure Fixes

#### 1a. Parameter Logging in results.tsv

**Problem:** When experiments run manually (not via `run.sh`), the 17 parameter
columns in `results.tsv` are empty. The TSV header defines columns for all
params but the manual flow never writes them.

**Fix:** Add a `log_experiment()` function to `prepare.py` that:
- Accepts `exp_id`, `results` dict, and the 4 param dicts
- Appends a complete row to `results.tsv` (creating the file with header if needed)
- Uses AST-free extraction (params passed directly, not parsed from `train.py`)

Update `train.py` (fixed section) to call `log_experiment()` after `run_eval()`.
This also requires adding `log_experiment` to the import list in the fixed section.

**Mutability note:** `prepare.py` is labeled "immutable" meaning the LLM tuning
agent must not edit it during the autoresearch loop. Human/engineer edits to
`prepare.py` are expected and necessary for infrastructure improvements. The
same applies to the "FIXED" section of `train.py` — the agent only edits above
the FIXED line; engineers may modify below it.

**Files:**
- Modify: `autoresearch/prepare.py` — add `log_experiment()`
- Modify: `autoresearch/train.py` (fixed section) — add import, call `log_experiment()`

#### 1b. Longer Evaluation

**Problem:** 50-episode eval has high variance. Scores fluctuate by ~1 point
between runs with identical params.

**Fix:** Increase default `n_episodes` in `run_eval()` from 50 to 200. This
gives tighter confidence intervals (~0.3 points instead of ~1.0) at the cost
of ~30 seconds additional eval time per experiment.

**Files:**
- Modify: `autoresearch/prepare.py` — change `run_eval()` default
- Modify: `autoresearch/train.py` — pass `n_episodes=200` explicitly

### 2. Phase 1: Re-baseline (Experiments 10-12)

Three experiments to establish the clean baseline:

| Exp | Description | Params Changed from Exp 9 |
|-----|------------|---------------------------|
| 10 | Corrupted policy on fixed gates | None (measures degradation) |
| 11 | Fresh fine-tune, baseline params | Reset all to `program.md` defaults |
| 12 | Baseline + offset penalty | `lambda_offset=1.0`, `lambda_gate=20.0` |

**Exp 10** tells us how badly the double-dipping policy breaks on correct gates.
**Exp 11** gives us a clean starting point without corrupted learned behavior.
**Exp 12** tests whether penalizing off-center passes improves gate traversal
quality (centered passes naturally continue forward rather than doubling back).

The best Phase 1 score becomes the baseline for Phase 2.

### 3. Phase 2: Systematic Sweep (Experiments 13-22)

Each experiment changes 1-2 parameters from the Phase 1 baseline. The goal is
to isolate the effect of each parameter on score.

| Exp | Change | Hypothesis |
|-----|--------|-----------|
| 13 | `lambda_prog=2.0` | Stronger progress shaping pulls drone forward |
| 14 | `lambda_prog=3.0`, `v_max=15.0` | Reward velocity, penalize reckless speed |
| 15 | `learning_rate=5e-5` | Slower LR for more stable fine-tuning |
| 16 | `ent_coef=0.01` | More exploration for new paths |
| 17 | `corner_noise_k=1.0` | Trust EKF measurements more (tighter approaches) |
| 18 | `corner_noise_k=3.0` | Trust EKF predictions more (smoother flight) |
| 19 | `gae_lambda=0.95`, `gamma=0.9999` | Longer planning horizon |
| 20 | `lambda_crash=20`, `lambda_gate=25` | Aggressive gate reward + safety net |
| 21 | `percentage=0.1` (in DOMAIN_RAND) | Less DR for faster convergence |
| 22 | Best combo from 13-21 | Combine top findings |

### 4. Multi-Track Training

The highest-value generalizability improvement. Parameters tuned on the
figure-8 track are track-specific. Training across multiple tracks produces
parameters that generalize.

#### 4a. Track Library

Add 2-3 track definitions alongside `make_figure8_track` in `rate_ctrl_env.py`:

**`make_oval_track(n_gates=4)`** — Simple oval with 4 gates at cardinal
positions. Tests basic gate-following without complex turns.

**`make_s_curve_track(n_gates=6)`** — S-shaped course with alternating left/right
turns. Tests handling of varied turn directions.

All tracks use the same dict format:
```python
{
    "gates": [{"pos": np.array([x, y, z]), "yaw": float}, ...],
    "gate_width": 0.55,
    "gate_height": 0.55,
    "n_gates": int,
}
```

Gate yaws must be set so the normal faces the approach direction (the bug we
just fixed). For each gate, verify:
`signed_dist(prev_gate_pos - gate_pos, normal) < 0` (approach is negative)
`signed_dist(next_gate_pos - gate_pos, normal) > 0` (exit is positive)

**Files:**
- Modify: `sim/envs/rate_ctrl_env.py` — add `make_oval_track()`, `make_s_curve_track()`

#### 4b. Track Sampling in RateCtrlEnv

Add an optional `tracks: list[dict]` parameter to `RateCtrlEnv.__init__()`.
The existing `track: Optional[dict]` parameter is kept for backward
compatibility. When `tracks` is provided, it takes precedence. When neither is
provided, the default figure-8 track is used. When only `track` is provided,
single-track behavior is unchanged.

**Reset behavior:**
1. Sample track index uniformly from `self._tracks`
2. Update `self._gate_positions`, `self._gate_yaws`, `self._n_gates`
3. Reset `self._gate_idx` and `self._prev_signed_dist` for the new track
4. Place drone at a starting position appropriate for the sampled track
   (existing spawn logic — 1m behind a random gate using gate yaw — already
   works generically for any track dict)

**Per-env independence:** Each parallel env can be on a different track
simultaneously. Track index is stored per-env.

**Info dict compatibility:** `VecEnvAdapter` reshapes the batched info dict from
`RateCtrlEnv` into per-env dicts for SB3 compatibility. Multi-track changes
must preserve this contract — the `info["episode"]` structure must remain
unchanged regardless of which track an env is on.

**Files:**
- Modify: `sim/envs/rate_ctrl_env.py` — extend `__init__()`, `_reset_env()`

#### 4c. Autoresearch Multi-Track Eval

Extend the eval pipeline to support multi-track evaluation:

- `run_eval()` accepts an optional `tracks` list
- Runs eval episodes on each track separately
- Returns per-track scores + aggregate (mean across tracks)
- Aggregate score is what the autoresearch loop optimizes

Extend `prepare.py:make_training_env()` to accept a `tracks` parameter and
pass it through to `RateCtrlEnv`.

**Files:**
- Modify: `autoresearch/prepare.py` — extend `make_training_env()`, `run_eval()`

### 5. Reward Normalization

Normalize reward weights by track geometry so they transfer across tracks
without retuning:

- **Progress:** `effective_lambda_prog = lambda_prog / mean_inter_gate_distance`
  (progress reward magnitude is independent of gate spacing)
- **Gate passage:** Keep `lambda_gate` unnormalized. The scoring metric
  (`avg_gates - 2 * crash_rate`) counts raw gates, so per-gate reward should
  remain constant regardless of track size. Normalizing by `n_gates` would
  create a mismatch where the score rewards more gates but the training reward
  discounts them on longer tracks.

**Implementation:** Normalization must happen in `RateCtrlEnv` at reward
computation time, not in `register_reward_preset()`. Reason: with multi-track
training, different envs may be on different tracks simultaneously, so
normalization factors vary per-env. The env knows its current track geometry
and can apply the normalization before calling the reward function.

**Files:**
- Modify: `sim/envs/rate_ctrl_env.py` — apply per-track progress normalization in reward computation

### 6. Acceptance Criteria

**Phase 1 (re-baseline):**
- At least one of experiments 10-12 produces a score > 0 with < 50% crash rate
  on the fixed gates. This confirms the pipeline works correctly post-bugfix.
- If all three experiments crash > 90%, the Phase 4 checkpoint is too corrupted
  by the double-dipping bug and we need to retrain from an earlier checkpoint.

**Phase 2 (sweep):**
- At least one experiment beats the Phase 1 baseline by >= 1.0 score points.
- If no experiment improves after all 10, the parameter space may be saturated
  at this training budget (500K steps). Consider increasing N_STEPS or expanding
  the editable surface before continuing.

**Multi-track:**
- Training on 3 tracks simultaneously produces a policy that scores > 50% of
  the single-track best on each individual track (no catastrophic forgetting).
- Per-track eval scores are recorded and comparable.

### 7. Scope

**In scope:**
- Infrastructure fixes (param logging, 200-episode eval)
- Phase 1 re-baseline (3 experiments)
- Phase 2 systematic sweep (10 experiments)
- Multi-track training (track library, track sampling, multi-track eval)
- Reward normalization

**Deferred (future work):**
- Bayesian optimization (Optuna/BOHB) for parameter search
- Architecture search (MLP layer sizes, activations)
- Procedural track generation (random gate placement with constraints)
- Meta-learning across tracks (held-out track generalization)

## Execution Order

### Exit conditions
- **Stop Phase 2 early** if 3 consecutive experiments show no score change
  (within eval noise of ~0.3 points). Move to multi-track work instead.
- **Stop multi-track experiments** if aggregate score plateaus for 5 experiments.


1. Infrastructure fixes (param logging + longer eval)
2. Phase 1 experiments 10-12 (re-baseline on fixed gates)
3. Phase 2 experiments 13-22 (systematic sweep)
4. Multi-track infrastructure (track library + sampling + eval)
5. Reward normalization
6. Multi-track autoresearch experiments (new batch)

Steps 1-3 can proceed immediately. Steps 4-5 are code changes that enable
step 6. Step 3 results inform whether the current parameter space is
sufficient or needs expansion.
