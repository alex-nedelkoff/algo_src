# Playground Port to algo_src

**Date:** 2026-03-12
**Status:** Draft
**Source:** https://github.com/corvidx-drone-grand-prix/playground (main branch)

## Summary

Port the playground repository (`aigp/`) into algo_src as an independent experiment configuration. The playground represents a complete MAVLab-style drone racing pipeline with 4 training phases, SymPy-compiled dynamics, a PD rate controller action space, camera-based perception noise, EKF state estimation, and DAgger distillation. It is restructured into algo_src's module boundaries (sim, control, perception, state_estimation, training) and wired through Hydra config so that `python -m training +experiment=playground_phase{1-4}` runs each phase.

## Goals

- Validate algo_src's framework extensibility by onboarding a second, fully independent e2e system
- Preserve playground's exact algorithmic flow — no behavioral changes during the port
- Achieve clear experiment separation through config and code (Approach A: flat co-location)
- Share framework infrastructure (entrypoint, training loop, PPO wrapper, callbacks, artifacts) while keeping all domain-specific implementations independent

## Non-Goals

- Combining or merging playground logic with existing MonoRace implementations
- Ensuring all config combinations are valid across experiments
- Porting playground's `scripts/plot_trajectory.py` or evaluation scripts (can follow later)

## Source Repository Analysis

The playground repo (`aigp/`) implements a 4-phase MAVLab replication pipeline:

| Phase | Description | Training Loop | Key Additions |
|-------|-------------|---------------|---------------|
| Phase 1 | Baseline PPO | RL (SB3 PPO) | SymPy dynamics, PD rate controller, MAVLab reward |
| Phase 2 | Perception noise | RL (SB3 PPO) | Camera model, corner detector, HMM dropout |
| Phase 3 | Asymmetric actor-critic | RL (SB3 PPO) | 50D obs (noisy actor / clean critic) |
| Phase 4 | DAgger distillation | DAgger | EKF16 filtering, teacher-student, StudentNetwork |

### Key Differences from algo_src MonoRace

| Aspect | algo_src (MonoRace) | playground (aigp) |
|--------|---------------------|-------------------|
| Action space | Direct 4 motor RPM | Thrust + 3 body rates -> PD controller -> motor RPM |
| Dynamics | Hand-coded NumPy | SymPy-compiled lambdified NumPy |
| Physics params | Racing quad (752g) | MAVLab system-ID (different k_thrust, drag, moments) |
| Inner loop | Single step per action | action_repeat=5 (500Hz PD controller, 100Hz RL) |
| Perception | Wrapper protocol (identity/noise) | Full camera + corner detector + HMM dropout |
| State estimation | Module exists but unused | EKF16 (16-state Kalman filter, IMU + vision fusion) |
| Distillation | Not implemented | DAgger with decaying beta teacher forcing |
| Reward | Single monorace function | Presets (M23 safe, M16 fast, baseline) |
| Obs normalization | Motor RPM mapped to [-1, 1] | Motor speed mapped to [0, 1] via [238.49, 3295.50] |

## Architecture

### Module Mapping

| playground (`aigp/`) | algo_src destination | New file |
|----------------------|---------------------|----------|
| `sim/dynamics.py` | `sim/dynamics/` | `sympy_quad.py` |
| `sim/motor_model.py` | `sim/` | `motor_model.py` |
| `sim/quad_env.py` + `control/obs.py` | `sim/envs/` | `rate_ctrl_env.py` |
| `sim/sb3_wrapper.py` | `sim/envs/` | `playground_factory.py` |
| `sim/ekf_wrapper.py` + `ekf_sb3_wrapper.py` | `sim/envs/` | `ekf_env_wrapper.py` |
| `control/reward.py` | `sim/` | `rewards_mavlab.py` |
| `control/feature_extractor.py` | `control/policies/` | `asymmetric_policy.py` |
| `control/distill.py` | `control/algorithms/` | `dagger.py` |
| `perception/camera.py` | `perception/` | `camera.py` |
| `perception/noise.py` | `perception/wrappers/` | `corner_noise.py` |
| `ekf/ekf16.py` | `state_estimation/` | `ekf16.py` |
| (new) | `training/loops/` | `dagger.py` |

### Data Flow

```
python -m training +experiment=playground_phase1

training/__main__.py (unchanged)
    |
    +-- hydra.utils.instantiate(cfg.loop) --> RLTrainingLoop (existing, shared)
    |
    +-- RLTrainingLoop.run(cfg)
        |-- instantiate(cfg.sim) --> PlaygroundEnvFactory (NEW)
        |   |-- make_vec_env(domain_rand_cfg, reward_cfg)
        |   |   +-- RateCtrlEnv (NEW)
        |   |       |-- SympyQuadDynamics (NEW, lambdified NumPy)
        |   |       |-- MotorModel (NEW, first-order lag + ESC)
        |   |       |-- PD rate controller (500Hz inner loop, action_repeat=5)
        |   |       |-- Track (existing or playground gate coords via config)
        |   |       |-- DomainRandomizer (existing, shared)
        |   |       +-- mavlab_reward() (NEW, M23/M16 presets)
        |   +-- VecEnvAdapter (existing, shared)
        |
        |-- instantiate(cfg.perception)
        |   |-- Phase 1: IdentityWrapper (existing)
        |   |-- Phase 2: CornerNoiseWrapper (NEW, camera + HMM dropout)
        |   +-- Phase 3: CornerNoiseWrapper + asymmetric 50D obs
        |
        |-- build_ppo(cfg) --> PPO (existing wrapper, different config)
        |   |-- Phase 1-2: MlpPolicy [64,64,64] + custom init
        |   +-- Phase 3: AsymmetricPolicy (NEW, noisy actor / clean critic)
        |
        |-- setup_callbacks (existing, factory-aware trajectory recorder)
        +-- ppo.train(...)


python -m training +experiment=playground_phase4

training/__main__.py (unchanged)
    |
    +-- instantiate(cfg.loop) --> DAggerTrainingLoop (NEW)
    |
    +-- DAggerTrainingLoop.run(cfg)
        |-- instantiate(cfg.sim) --> PlaygroundEnvFactory
        |   +-- RateCtrlEnv (same as phases 1-3)
        |
        |-- EKFEnvWrapper (NEW, wraps VecEnv with EKF16 filtering)
        |   +-- EKF16 (NEW, in state_estimation/)
        |       |-- predict: kinematic propagation
        |       |-- update_imu: velocity/attitude/rates/motors
        |       +-- update: pixel observations from CornerDetector
        |
        |-- Load teacher from cfg.teacher_checkpoint (phase 3 output)
        |
        |-- StudentNetwork (NEW, 24D -> [64,64,64] -> 4D)
        |   +-- DAgger rounds with decaying beta
        |       |-- Collect (EKF_obs, teacher_action) pairs
        |       |-- Train student on MSE loss
        |       +-- Save checkpoints
        |
        +-- Save student_final.pt
```

### Config Layer

New Hydra config files:

```
configs/
  sim/
    playground_quad.yaml             # PlaygroundEnvFactory + SymPy dynamics + MAVLab params
  control/
    ppo_playground.yaml              # PPO with playground-specific kwargs (log_std_init, bias init)
    ppo_playground_asymmetric.yaml   # AsymmetricPolicy with 50D obs
  perception/
    corner_noise.yaml                # Camera + corner detector + HMM dropout
  reward/
    mavlab_m23.yaml                  # M23 preset (safe: lambda_gate=1.5, lambda_crash=10.0)
    mavlab_m16.yaml                  # M16 preset (fast: lambda_gate=30.0, lambda_crash=1.0)
  loop/
    dagger.yaml                      # DAggerTrainingLoop
  experiment/
    playground_phase1.yaml           # Baseline PPO, no perception noise
    playground_phase2.yaml           # + corner_noise perception
    playground_phase3.yaml           # + asymmetric actor-critic (50D obs)
    playground_phase4.yaml           # DAgger distillation (loop: dagger)
```

Example experiment config (`playground_phase1.yaml`):
```yaml
# @package _global_
defaults:
  - override /loop: rl
  - override /sim: playground_quad
  - override /control: ppo_playground
  - override /perception: none
  - override /reward: mavlab_m23
  - override /domain_rand: uniform_30pct
  - override /logging: tensorboard

total_timesteps: 100_000_000
seed: 42
```

Phase progression:
- Phase 2: overrides `perception: corner_noise`
- Phase 3: overrides `control: ppo_playground_asymmetric`, perception wrapper constructs 50D obs (noisy 24D + clean 24D + 2D metadata) — the env always outputs 24D, the asymmetric obs is assembled by `CornerNoiseWrapper` when `asymmetric: true` is set in the perception config
- Phase 4: overrides `loop: dagger`, adds `teacher_checkpoint` path

### Interface Contracts

**EnvFactory protocol** — `PlaygroundEnvFactory` implements:
- `make_vec_env(domain_rand_cfg, reward_cfg) -> VecEnvAdapter`
- `make_eval_env(domain_rand_cfg, reward_cfg, n_envs) -> VecEnvAdapter`

**PerceptionWrapper protocol** — `CornerNoiseWrapper` implements:
- `wrap(env: VecEnv) -> VecEnv`

**TrainingLoop protocol** — `DAggerTrainingLoop` implements:
- `run(cfg: DictConfig, uploader: ArtifactUploader | None = None) -> None`

**Info dict contract** — `RateCtrlEnv` must populate the same info keys as `GateRaceEnv` for callback compatibility:
- `gates_passed`, `laps_completed`, `termination_reason`, `reward_components`

## New Files (16 source + 8 config)

| Module | File | Lines (est.) | Ported from |
|--------|------|-------------|-------------|
| `sim/dynamics/` | `sympy_quad.py` | ~160 | `aigp/sim/dynamics.py` |
| `sim/` | `motor_model.py` | ~20 | `aigp/sim/motor_model.py` |
| `sim/envs/` | `rate_ctrl_env.py` | ~470 | `aigp/sim/quad_env.py` + `aigp/control/obs.py` |
| `sim/envs/` | `playground_factory.py` | ~120 | New (factory pattern wrapping RateCtrlEnv) |
| `sim/envs/` | `ekf_env_wrapper.py` | ~130 | `aigp/sim/ekf_wrapper.py` + `ekf_sb3_wrapper.py` |
| `sim/` | `rewards_mavlab.py` | ~100 | `aigp/control/reward.py` |
| `control/policies/` | `asymmetric_policy.py` | ~130 | `aigp/control/feature_extractor.py` |
| `control/algorithms/` | `dagger.py` | ~90 | `aigp/control/distill.py` |
| `perception/` | `camera.py` | ~200 | `aigp/perception/camera.py` |
| `perception/wrappers/` | `corner_noise.py` | ~230 | `aigp/perception/noise.py` |
| `state_estimation/` | `ekf16.py` | ~420 | `aigp/ekf/ekf16.py` |
| `training/loops/` | `dagger.py` | ~100 | New |
| `configs/sim/` | `playground_quad.yaml` | ~25 | New |
| `configs/control/` | `ppo_playground.yaml` | ~20 | New |
| `configs/control/` | `ppo_playground_asymmetric.yaml` | ~20 | New |
| `configs/perception/` | `corner_noise.yaml` | ~15 | New |
| `configs/reward/` | `mavlab_m23.yaml` | ~15 | New |
| `configs/reward/` | `mavlab_m16.yaml` | ~15 | New |
| `configs/loop/` | `dagger.yaml` | ~10 | New |
| `configs/experiment/` | `playground_phase{1-4}.yaml` | ~60 | New |

**Estimated total: ~2,300 lines of new code + ~180 lines of config**

## Framework Generalization (prerequisite refactoring)

The existing framework infrastructure is more tightly coupled to `GateRaceEnv` than initially apparent. Before porting playground-specific code, these shared components must be generalized:

### 1. `sim/envs/vec_env_adapter.py` — decouple from GateRaceEnv

Currently type-annotated and hardcoded to accept only `GateRaceEnv` (imports `GateRaceEnv`, `REWARD_COMPONENT_NAMES`). Must be generalized to accept any env conforming to a common racing env protocol. Options:
- **(Recommended)** Introduce a `RacingEnv` protocol in `sim/envs/base.py` that both `GateRaceEnv` and `RateCtrlEnv` satisfy (step/reset/info dict schema). Retype `VecEnvAdapter.__init__` to accept `RacingEnv`.
- Alternative: playground provides its own `PlaygroundVecEnvAdapter` (duplicates code).

### 2. `training/callbacks.py` — generalize reward component handling

`GateMetricsCallback` hardcodes MonoRace reward component names (`progress`, `body_rate`, `action_smooth`, `gate_passage`, `gate_offset`, `crash_penalty`). The playground's MAVLab reward has different component names (`prog`, `gate`, `offset`, `rate`, `delta_u`, `crash`, `alive`, `perc`). Fix: read component names dynamically from the info dict or from a config-provided list, rather than hardcoding.

### 3. `training/trajectory_recorder.py` — remove GateRaceEnv dependency

Deep coupling: directly imports and constructs `GateRaceEnv` in `_make_eval_env()`, accesses internal attributes (`env._states`, `env._gates_passed`, `env._gate_indices`, `env._step_reward_components`), imports `REWARD_COMPONENT_NAMES`. Fix: use `cfg.sim` factory to build recording env, access state through the public info dict protocol rather than internal attributes. The factory reference must be threaded through `setup_callbacks()` in `training/loops/rl.py` so the recorder can construct experiment-appropriate envs.

### 4. `control/algorithms/ppo.py` + `training/loops/rl.py` — configurable policy_kwargs

Two layers of coupling:
- `ppo.py`: `_build_policy_kwargs()` constructs kwargs internally with hardcoded `log_std_init=0.0`
- `training/loops/rl.py`: `build_ppo()` manually extracts specific fields from `cfg.control` and does not pass through arbitrary policy kwargs

For playground phase 3, both must support:
- Injecting custom `features_extractor_class` (for `AsymmetricPolicy`) from config
- Configurable `log_std_init` (playground uses `-2.0`)
- Arbitrary `policy_kwargs` passthrough from Hydra config through `build_ppo()` into the PPO constructor

### 5. `pyproject.toml` — add `state_estimation` to packages

Currently lists `packages = ["control", "perception", "sim", "training", "utils"]`. Must add `state_estimation` since `ekf16.py` will live there.

## Shared vs. Independent

| Shared (framework — requires generalization) | Independent (playground-specific) |
|----------------------------------------------|----------------------------------|
| `training/__main__.py` | `sim/dynamics/sympy_quad.py` |
| `training/loops/rl.py` (phases 1-3) | `sim/envs/rate_ctrl_env.py` |
| `control/algorithms/ppo.py` | `sim/motor_model.py` |
| `sim/domain_randomization.py` | `sim/rewards_mavlab.py` |
| `sim/envs/vec_env_adapter.py` | `perception/camera.py` |
| `sim/envs/base.py` (new RacingEnv protocol) | `perception/wrappers/corner_noise.py` |
| `training/callbacks.py` | `state_estimation/ekf16.py` |
| `training/trajectory_recorder.py` | `control/policies/asymmetric_policy.py` |
| `artifacts/` | `control/algorithms/dagger.py` |
| | `training/loops/dagger.py` |
| | `sim/envs/ekf_env_wrapper.py` |
| | `sim/envs/playground_factory.py` |

## Testing Strategy

Port relevant tests from playground's `tests/` directory, restructured to match algo_src's test layout. All tests run in Docker per project convention. Key test categories:
- Unit tests for each new module (dynamics, motor model, env, reward, EKF, camera, perception noise)
- Integration test: full phase 1 smoke test (short training run)
- Integration test: DAgger loop with mock teacher

## Dependencies

playground requires `sympy >= 1.13` for dynamics compilation. SymPy is used at import time to symbolically derive and `lambdify()` the dynamics equations into optimized NumPy callables. This is a runtime dependency (not pre-compiled). Add to `pyproject.toml` optional dependencies under a `[playground]` extra.

## DAgger Training Loop Details

`DAggerTrainingLoop` manages its own callback/logging strategy since SB3 callbacks are not applicable:

- **Metrics logged per round:** MSE loss, student vs. teacher action error, beta schedule value, mean gates passed by student policy
- **Checkpoints:** `student_round{N}.pt` saved after each DAgger round
- **Trajectory recording:** Optional — uses the env factory to build a single-env for rollout recording, same as RL loop but driven by the student policy
- **Artifact uploader:** Receives the same `uploader` from `__main__.py`, uploads student checkpoints and optional trajectory recordings
