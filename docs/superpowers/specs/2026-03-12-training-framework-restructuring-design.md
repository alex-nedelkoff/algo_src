# Training Framework Restructuring

## Problem

Training orchestration lives in `control/__main__.py` — inside one of the modules it orchestrates. `control/` mixes policy/algorithm code with env construction, callback setup, W&B init, and artifact uploading. This makes it hard to add new training loop types (e.g., supervised perception training) and new sim backends.

## Goals

- Training as its own top-level module with a single Hydra entrypoint
- Support RL and supervised training loops, composable via config
- Swappable sim backends (NumpyQuad now, DCL later) via `_target_`
- Swappable perception wrappers via `_target_`
- `control/` becomes a pure policy/algorithm library
- Existing W&B + R2 integration unchanged

## Non-Goals

- Backward compatibility shims
- Custom experiment comparison tooling (W&B is sufficient)
- Alternative task environments (only alternative physics backends)
- Overhauling Hydra config structure beyond what's needed

## Design

### 1. `training/` module

New top-level module that owns the training entrypoint and loop orchestration.

```
training/
├── __init__.py
├── __main__.py              # Hydra entrypoint (replaces control/__main__.py)
├── loops/
│   ├── __init__.py
│   ├── base.py              # TrainingLoop protocol
│   ├── rl.py                # RLTrainingLoop (env + policy + SB3)
│   └── supervised.py        # SupervisedTrainingLoop (stub)
├── callbacks.py             # Moved from control/callbacks.py
└── trajectory_recorder.py   # Moved from control/trajectory_recorder.py
```

**`__main__.py`** is the single Hydra entrypoint. It handles only cross-cutting concerns:
1. Loads config via `@hydra.main(config_path="../configs", config_name="train")`
2. Calls `_load_dotenv()` to load `.env` from project root (moved from `control/__main__.py`)
3. Seeds `np.random`
4. Initializes W&B (if `cfg.logging.backend == "wandb"`) and creates `ArtifactUploader`
5. Instantiates the training loop via `hydra.utils.instantiate(cfg.loop)`
6. Calls `loop.run(cfg, uploader=uploader)` — uploader is passed explicitly, not via config
7. Cleans up (uploader.close, wandb.finish)

**`loops/base.py`** defines the protocol:
```python
class TrainingLoop(Protocol):
    def run(self, cfg: DictConfig, uploader: ArtifactUploader | None = None) -> None: ...
```

The `uploader` parameter is passed explicitly because it is created by `__main__.py` (which owns the W&B run lifecycle) and consumed by callbacks inside the loop.

**`loops/rl.py`** (`RLTrainingLoop`) owns the full RL training lifecycle:
```python
class RLTrainingLoop:
    def run(self, cfg: DictConfig, uploader=None) -> None:
        # 1. Build train env via EnvFactory
        env_factory = hydra.utils.instantiate(cfg.sim)
        train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)

        # 2. Wrap with perception (no-op if identity)
        perception_wrapper = hydra.utils.instantiate(cfg.perception)
        train_env = perception_wrapper.wrap(train_env)

        # 3. Build eval env (no domain rand, fewer envs)
        eval_cfg = override_for_eval(cfg)  # sets domain_rand.enabled=False, n_envs=n_eval_episodes
        eval_env = env_factory.make_eval_env(eval_cfg.domain_rand, eval_cfg.reward, n_envs=cfg.n_eval_episodes)
        eval_env = perception_wrapper.wrap(eval_env)

        # 4. Build PPO
        ppo = build_ppo(cfg)

        # 5. Resume from checkpoint (if cfg.resume_checkpoint is set)
        if cfg.get("resume_checkpoint"):
            ppo.load(cfg.resume_checkpoint, env=train_env)

        # 6. Setup callbacks (checkpoint, eval, gate metrics, trajectory recorder)
        callbacks = setup_callbacks(cfg, eval_env=eval_env, uploader=uploader)

        # 7. Train
        ppo.train(train_env, total_timesteps=cfg.total_timesteps, callbacks=callbacks)

        # 8. Save final model + close envs
        ppo.save(Path(cfg.output_dir) / "final_model")
        train_env.close()
        eval_env.close()
```

`build_ppo()` and `setup_callbacks()` are module-level functions in `rl.py` (moved from `control/__main__.py`). The `_load_dotenv()` helper moves to `training/__main__.py`.

**`loops/supervised.py`** is a stub — filled in when perception training is implemented.

**Invocation:**
```bash
python -m training +experiment=monorace_baseline
```

Note: `config_path="../configs"` continues to work from `training/__main__.py` because `training/` is at the same directory level as the former `control/`.

### 2. `control/` cleanup

After extraction, `control/` is a pure policy/algorithm library:

```
control/
├── __init__.py
├── algorithms/
│   ├── __init__.py
│   ├── base.py          # Algorithm base class (unchanged)
│   └── ppo.py           # PPO wrapper (unchanged)
└── policies/
    ├── __init__.py
    └── gcnet.py          # GCNetExtractor + GCNet (unchanged)
```

Deleted: `control/__main__.py`, `control/callbacks.py`, `control/trajectory_recorder.py`.

### 3. Sim backend abstraction

A factory protocol allows swappable physics backends via `_target_`:

```python
# sim/envs/base.py
class EnvFactory(Protocol):
    def make_vec_env(self, domain_rand_cfg: DictConfig, reward_cfg: DictConfig) -> VecEnv: ...
```

The factory stores sim-specific params (n_envs, dt, vehicle params, etc.) at `__init__` time — Hydra passes all non-`_target_` keys from the config as constructor kwargs. `make_vec_env()` takes only the additional config sections the factory needs to compose the full environment.

**NumpyQuad factory** (`sim/envs/numpy_quad_factory.py`) wraps existing `build_env()` logic: VehicleParams → NumpyQuadDynamics → DomainRandomizer → GateRaceEnv → VecEnvAdapter. Its `__init__` receives `n_envs`, `dt`, `params`, etc. from Hydra instantiation.

**Config** — the existing `_target_` in `configs/sim/numpy_quad.yaml` is **replaced** (it currently points to `GateRaceEnv`):
```yaml
# configs/sim/numpy_quad.yaml
_target_: sim.envs.numpy_quad_factory.NumpyQuadEnvFactory
n_envs: 100
dt: 0.01
params: ...
```

Future backends (e.g., DCL) implement the same protocol with their own config file.

**In `training/loops/rl.py`:**
```python
env_factory = hydra.utils.instantiate(cfg.sim)
train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)
```

New files in `sim/envs/`: `base.py`, `numpy_quad_factory.py`. Existing files unchanged.

### 4. Perception abstraction

Perception has two roles:
1. **Observation wrapper** (in RL loop) — transforms env observations
2. **Standalone model** (in supervised loop) — trained independently

**Important**: The existing `PerceptionNoiseWrapper` is a `gym.Wrapper` (single-env). Since our training pipeline uses `VecEnvAdapter` (SB3's `VecEnv` interface), the new wrappers must operate at the `VecEnv` level using SB3's `VecEnvWrapper` base class. This is a **rewrite**, not a simple move. Key changes:
- HMM state (`_hmm_state`) becomes a per-env array `(n_envs,)` instead of a scalar
- Noise application becomes vectorized across all envs
- Latency buffer becomes per-env

**Wrapper protocol with `_target_`:**

```python
# perception/wrappers/base.py
from stable_baselines3.common.vec_env import VecEnv

class ObservationWrapper(Protocol):
    def wrap(self, env: VecEnv) -> VecEnv: ...
```

**Config** — the `enabled` key is removed from both perception configs (the `_target_` dispatch replaces it):
```yaml
# configs/perception/none.yaml
_target_: perception.wrappers.identity.IdentityWrapper
# (enabled key removed — selecting this config IS the "disabled" path)

# configs/perception/noise_injection.yaml
_target_: perception.wrappers.noise.NoiseInjectionWrapper
dropout_hmm:
  p_detect_to_miss: 0.05
  p_miss_to_detect: 0.3
noise_std: 0.1
latency_frames: 2
# (enabled key removed — Hydra passes remaining keys as constructor kwargs)
```

**In `training/loops/rl.py`:**
```python
perception_wrapper = hydra.utils.instantiate(cfg.perception)
train_env = perception_wrapper.wrap(train_env)
```

**File changes in `perception/`:**
```
perception/
├── wrappers/                  # NEW directory
│   ├── __init__.py
│   ├── base.py                # ObservationWrapper protocol
│   ├── identity.py            # No-op (returns env unchanged)
│   └── noise.py               # Rewritten as VecEnvWrapper (vectorized HMM, noise, latency)
├── noise_injection.py         # DELETE (replaced by wrappers/noise.py)
├── adaptive_crop.py           # Unchanged
├── quadgate.py                # Unchanged
└── detectors/                 # Unchanged
```

### 5. Config changes

**New `configs/loop/` group:**
```yaml
# configs/loop/rl.yaml
_target_: training.loops.rl.RLTrainingLoop

# configs/loop/supervised.yaml
_target_: training.loops.supervised.SupervisedTrainingLoop
```

Note: the existing `configs/training/curriculum/` directory is unrelated (Hydra config paths and Python packages are separate namespaces). No conflict.

**Updated `configs/train.yaml` defaults:**
```yaml
defaults:
  - loop: rl
  - sim: numpy_quad
  - control: ppo
  - perception: none
  - reward: monorace
  - domain_rand: uniform_30pct
  - logging: wandb
  - _self_
```

**`_target_` changes:**
- `configs/sim/numpy_quad.yaml` — **replace** existing `_target_` (currently `GateRaceEnv`) with `NumpyQuadEnvFactory`
- `configs/perception/none.yaml` — add `_target_: perception.wrappers.identity.IdentityWrapper`
- `configs/perception/noise_injection.yaml` — add `_target_: perception.wrappers.noise.NoiseInjectionWrapper`

Experiment presets, control configs, reward configs, domain_rand configs, and logging configs unchanged.

### 6. Docker, Makefile & pyproject.toml

Update `python -m control` → `python -m training` in:
- `docker/control.Dockerfile` CMD (the only Dockerfile with this command; `control-cpu.Dockerfile` runs pytest, unchanged)
- `Makefile` targets

`docker-compose.yml` has no `command:` override — it relies on the Dockerfile CMD, so no change needed there.

Keep Dockerfile name as `control.Dockerfile` to avoid churn in CI/compose service names.

**`pyproject.toml`** changes:
- Add `"training"` to the `packages` list: `packages = ["control", "perception", "sim", "training", "utils"]`
- The existing `[control]` optional-dependency group (SB3, torch) remains — `training` imports `control` which needs those deps, so the dependency chain is preserved

### 7. Trajectory recorder coupling

`TrajectoryRecorderCallback` directly accesses `GateRaceEnv` internals (`env._states`, `env._gates_passed`, `env._gate_indices`, `env._step_reward_components`) via `self.training_env.env`. This is **NumpyQuad-specific** and will not work with future sim backends.

For now, the trajectory recorder remains NumpyQuad-specific — it accesses internals through the `VecEnvAdapter.env` attribute. When a second backend is added, we will need either:
- A trajectory protocol on the env (preferred)
- Backend-specific recorder subclasses

This is deferred — no action in this restructuring beyond moving the file to `training/`.

## File Change Summary

| Action | Path |
|--------|------|
| **New** | `training/__init__.py` |
| **New** | `training/__main__.py` (Hydra entrypoint + W&B init + `_load_dotenv()`) |
| **New** | `training/loops/__init__.py` |
| **New** | `training/loops/base.py` (TrainingLoop protocol) |
| **New** | `training/loops/rl.py` (absorbs `build_env`, `build_ppo`, `setup_callbacks`, eval env logic, resume logic) |
| **New** | `training/loops/supervised.py` (stub) |
| **Move** | `control/callbacks.py` → `training/callbacks.py` |
| **Move** | `control/trajectory_recorder.py` → `training/trajectory_recorder.py` |
| **New** | `sim/envs/base.py` (EnvFactory protocol) |
| **New** | `sim/envs/numpy_quad_factory.py` (extracts build_env logic) |
| **New** | `perception/wrappers/__init__.py` |
| **New** | `perception/wrappers/base.py` (ObservationWrapper protocol) |
| **New** | `perception/wrappers/identity.py` (no-op, returns env unchanged) |
| **New** | `perception/wrappers/noise.py` (rewritten as VecEnvWrapper with vectorized HMM) |
| **Delete** | `perception/noise_injection.py` (replaced by `perception/wrappers/noise.py`) |
| **Edit** | `configs/train.yaml` — add `loop: rl` default |
| **New** | `configs/loop/rl.yaml` |
| **New** | `configs/loop/supervised.yaml` |
| **Edit** | `configs/sim/numpy_quad.yaml` — **replace** `_target_` to point to factory |
| **Edit** | `configs/perception/none.yaml` — replace with `_target_`, remove `enabled` |
| **Edit** | `configs/perception/noise_injection.yaml` — add `_target_`, remove `enabled` |
| **Edit** | `docker/control.Dockerfile` — update CMD |
| **Edit** | `Makefile` — update targets |
| **Edit** | `pyproject.toml` — add `"training"` to packages list |
| **Delete** | `control/__main__.py` |
| **Move** | `tests/test_control/test_train_entrypoint.py` → `tests/test_training/test_train_entrypoint.py` (update imports) |
| **Update** | Tests importing `control.callbacks` / `control.trajectory_recorder` → `training.*` |
