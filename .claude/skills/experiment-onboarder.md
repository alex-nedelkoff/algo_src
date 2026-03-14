---
name: experiment-onboarder
description: Guide sessions through onboarding new components (environments, perception models, control algorithms, renderers) into the algo_src framework, ensuring metrics contract compliance and W&B/Rerun visualization integration. Proactively detects framework gaps and drives evolution. Triggers on "add a new env", "onboard", "create a new experiment", "port this simulator", "new perception model", "new control algo".
---

# Experiment Onboarder

Guides you through onboarding a new component into the algo_src framework. Proactively detects when a component doesn't fit the current framework and drives or escalates framework evolution.

## Framework Assumptions

These are the invariants that current code depends on. **Check each one against the new component before proceeding to checklists.** If any assumption is violated, follow the Framework Evolution Protocol below.

| # | Assumption | Enforced by |
|---|-----------|-------------|
| A1 | Perception and control are separate modules (`perception/`, `control/`) | Training entrypoint, Hydra config structure |
| A2 | All environments are quadrotor-based with 4 motors | `TrajectoryProvider.get_state()` returns `motor_rpms` (4,) |
| A3 | State includes quaternion orientation (w,x,y,z) | `TrajectoryProvider.get_state()` requires `quaternion` (4,) |
| A4 | `EpisodeMetrics` fields are mandatory for all envs; racing fields default to zero for non-racing | `validate_episode_metrics()` in `metrics/contract.py` |
| A5 | Envs are internally vectorized (n_envs parameter, batched arrays) | `VecEnvAdapter`, all env constructors |
| A6 | Actions are motor-level (RPM or rate commands, 4-dimensional) | `action_space` in existing envs |
| A7 | Hydra config composition: `configs/sim/`, `configs/experiment/` | Training entrypoint |
| A8 | `TrajectoryProvider.get_state()` returns exactly: position (3,), quaternion (4,), velocity (3,), body_rates (3,), motor_rpms (4,) | `validate_trajectory_state()`, `TrajectoryRecorderCallback` |
| A9 | Gate-based racing is the primary task (gates, laps, gate geometry) | `EpisodeMetrics`, `TrajectoryProvider.get_gate_geometry()` |

**When to update this table:** After any framework evolution, invoke `/superpowers:skill-creator` to update these assumptions and the checklists below.

## Step 0: Gap Detection

Before jumping to checklists, profile the new component:

1. **Ask:** "Describe what this component does and what makes it different from existing ones."
2. **Compare each assumption** (A1-A9) against the component. Surface mismatches as named gaps.
3. **Classify each gap:**
   - **Back-compatible** — solvable by adding optional fields, new protocol methods, or config options without changing existing code
   - **Breaking** — requires changes to existing contracts, protocols, or consumer code

**If no gaps:** Proceed to Component Detection and checklists.
**If gaps found:** Follow the Framework Evolution Protocol.

## Framework Evolution Protocol

### Back-compatible gaps (drive end-to-end)

1. Name the gap and the assumption it violates
2. Propose the minimal additive change (e.g., "add optional `acceleration` key to `get_state()`")
3. Invoke `/superpowers:brainstorming` to validate the proposal and design the change
4. Implement the framework change via the normal plan → execute cycle
5. Invoke `/superpowers:skill-creator` to update this skill's assumptions and checklists
6. Continue onboarding with the updated framework

### Breaking gaps (escalate to team)

1. Document the gap: what assumption is violated, why it can't be solved additively
2. Propose 2-3 options with trade-offs
3. Flag for team discussion — output a summary suitable for a Linear issue
4. **Pause onboarding** until the breaking change is resolved and landed
5. After resolution, invoke `/superpowers:skill-creator` to update this skill

**Encourage framework evolution** — gaps mean we're pushing the envelope. An organized, evolving framework enables the whole team to collaborate effectively.

## Component Detection

Ask: **"What type of component are you onboarding?"**

1. **Environment** — a new sim/training environment
2. **Perception model** — a new gate detection or vision model
3. **Control algorithm** — a new RL policy or controller
4. **Renderer / visualizer** — a new trajectory viewer or visualization

If the user's request makes the type obvious, skip asking and proceed directly.

---

## Checklist: New Environment

### 1. Metrics Contract Compliance

Read `metrics/contract.py` for the authoritative schema.

- [ ] **EpisodeMetrics**: Does `step()` populate all required keys in `info["episode"]`?
  - `r` (float): cumulative episode reward
  - `l` (int): episode length in RL steps
  - `effective_dt` (float): seconds per RL step (`physics_dt * action_repeat`)
  - `gates_passed` (int): gate passages this episode (0 for non-racing)
  - `laps_completed` (int): full laps (0 for non-racing)
  - `termination` (str): canonical name — "timeout", "crash", "ground", etc.
  - `success` (bool): env-defined success criterion
  - `success_criterion` (str): human-readable description
  - `avg_speed` (float): mean velocity magnitude (m/s)
  - `first_gate_step` (int): step of first gate passage (-1 if none)
  - `reward_components` (dict[str, float]): env-native component breakdown

- [ ] **Termination mapping**: List all termination conditions and map each to a canonical string name. Create a `TERM_NAMES` dict at module level.

- [ ] **Success criterion**: Define what "success" means for this env and document it in `success_criterion`.

- [ ] **effective_dt**: Set to `physics_dt * action_repeat` (or just `dt` if no action repeat).

- [ ] **Non-racing fields**: If the env has no gates/laps, set `gates_passed=0`, `laps_completed=0`, `first_gate_step=-1`.

### 2. TrajectoryProvider Protocol

- [ ] **`get_state(env_idx)`**: Returns dict with `position` (3,), `quaternion` (4, w-first), `velocity` (3,), `body_rates` (3,), `motor_rpms` (4,). If the env uses Euler angles internally, convert to quaternion here.

- [ ] **`get_gate_geometry()`**: Returns dict with `positions` (n_gates, 3), `orientations` (n_gates, 4, quaternion), `half_extents` (n_gates, 2). Shared across all envs.

- [ ] **`get_step_reward_components(env_idx)`**: Returns `(names: list[str], values: ndarray)`. Must track per-step components (not just episode-level accumulators).

### 3. Env Factory

- [ ] Create factory class following `NumpyQuadEnvFactory` or `PlaygroundEnvFactory` pattern in `sim/envs/`.
- [ ] Factory implements `EnvFactory` protocol from `sim/envs/base.py`.
- [ ] `make_vec_env()` wraps env in `VecEnvAdapter`.
- [ ] `make_eval_env()` creates single-env with no domain randomization.

### 4. Hydra Config

- [ ] Add config group entry under `configs/sim/`.
- [ ] Add experiment config under `configs/experiment/` that composes the sim config.

### 5. Smoke Test

Run a 1000-step training with the new experiment config:

```bash
docker run --rm algo-src-control-cpu python -m training \
  experiment=<your_experiment> training.total_timesteps=1000
```

Verify:
- [ ] W&B metrics appear with correct names under `racing/` and `termination/` prefixes (training episodes)
- [ ] If eval is enabled: `eval_racing/` and `eval_termination/` prefixes also appear (eval episodes)
- [ ] `racing/success_rate` is not always 0%
- [ ] Reward components show named labels (`reward_progress`, `reward_gate_passage`, etc.) not `reward_component_0`
- [ ] Termination breakdown shows string names (not int codes or mismatched labels)
- [ ] Trajectory .npz files are generated with non-empty `reward_components`
- [ ] Rerun `.rrd` generates from the `.npz` without errors

### 6. Contract Validation

Run the programmatic validators:

```python
from metrics.contract import validate_episode_metrics, validate_trajectory_state

# These are called automatically by VecEnvAdapter and TrajectoryRecorderCallback
# on first episode. If they pass, the contract is satisfied.
```

If validation fails, the error message points to the exact missing/mistyped field.

---

## Checklist: New Perception Model

- [ ] **Hydra config**: Add config group entry under `configs/perception/`.
- [ ] **Observation pipeline**: Verify obs dim matches env's `observation_space`.
- [ ] **W&B logging**: Ensure model-specific metrics (detection AP, inference time) logged under `perception/` prefix.
- [ ] **Smoke test**: Run with existing env, verify metrics appear in W&B.

---

## Checklist: New Control Algorithm

- [ ] **Hydra config**: Add config group entry under `configs/control/`.
- [ ] **Action space**: Verify action dim matches env's `action_space`.
- [ ] **Training loop**: Verify callback integration (GateMetricsCallback, TrajectoryRecorderCallback).
- [ ] **Smoke test**: Run 1000 steps with existing env, verify W&B metrics.

---

## Checklist: New Renderer / Visualizer

- [ ] **Trajectory schema**: Verify `.npz` fields are consumed correctly (see `SCHEMA_VERSION` in `training/trajectory_recorder.py`).
- [ ] **Rerun integration**: Verify `.rrd` generation from `.npz`.
- [ ] **W&B panel**: Verify viewer URLs appear in Rerun HTML panel.

---

## Reference

- **Contract source of truth**: `metrics/contract.py`
- **Env factories**: `sim/envs/numpy_quad_factory.py`, `sim/envs/playground_factory.py`
- **Existing envs**: `sim/envs/gate_race_env.py`, `sim/envs/rate_ctrl_env.py`
- **Callback**: `training/callbacks.py`
- **Trajectory recorder**: `training/trajectory_recorder.py`
- **VecEnv adapter**: `sim/envs/vec_env_adapter.py`
