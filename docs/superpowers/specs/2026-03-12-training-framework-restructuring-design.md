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

**`__main__.py`** is the single Hydra entrypoint:
1. Loads config via `@hydra.main(config_path="../configs", config_name="train")`
2. Initializes W&B + ArtifactUploader (same logic as today)
3. Instantiates the training loop via `_target_` from the `loop` config group
4. Calls `loop.run(cfg)`
5. Cleans up (uploader.close, wandb.finish)

**`loops/base.py`** defines the protocol:
```python
class TrainingLoop(Protocol):
    def run(self, cfg: DictConfig) -> None: ...
```

**`loops/rl.py`** absorbs the current `build_env()`, `build_ppo()`, `_setup_callbacks()` logic from `control/__main__.py`.

**`loops/supervised.py`** is a stub — filled in when perception training is implemented.

**Invocation:**
```bash
python -m training +experiment=monorace_baseline
```

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
    def make_vec_env(self, cfg: DictConfig) -> VecEnv: ...
```

**NumpyQuad factory** (`sim/envs/numpy_quad_factory.py`) wraps existing `build_env()` logic: VehicleParams → NumpyQuadDynamics → DomainRandomizer → GateRaceEnv → VecEnvAdapter.

**Config:**
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
train_env = env_factory.make_vec_env(cfg)
```

New files in `sim/envs/`: `base.py`, `numpy_quad_factory.py`. Existing files unchanged.

### 4. Perception abstraction

Perception has two roles:
1. **Observation wrapper** (in RL loop) — transforms env observations
2. **Standalone model** (in supervised loop) — trained independently

For role 1, a wrapper protocol with `_target_`:

```python
# perception/wrappers/base.py
class ObservationWrapper(Protocol):
    def wrap(self, env: VecEnv) -> VecEnv: ...
```

**Config:**
```yaml
# configs/perception/none.yaml
_target_: perception.wrappers.identity.IdentityWrapper

# configs/perception/noise_injection.yaml
_target_: perception.wrappers.noise.NoiseInjectionWrapper
noise_std: 0.1
```

**In `training/loops/rl.py`:**
```python
perception_wrapper = hydra.utils.instantiate(cfg.perception)
train_env = perception_wrapper.wrap(train_env)
```

New directory `perception/wrappers/` with `base.py`, `identity.py`, `noise.py`. Existing `noise_injection.py` logic moves into `perception/wrappers/noise.py`.

### 5. Config changes

**New `configs/loop/` group:**
```yaml
# configs/loop/rl.yaml
_target_: training.loops.rl.RLTrainingLoop

# configs/loop/supervised.yaml
_target_: training.loops.supervised.SupervisedTrainingLoop
```

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

**`_target_` added to:**
- `configs/sim/numpy_quad.yaml`
- `configs/perception/none.yaml`
- `configs/perception/noise_injection.yaml`

Experiment presets, control configs, reward configs, domain_rand configs, and logging configs unchanged.

### 6. Docker & Makefile

Update `python -m control` → `python -m training` in:
- `docker/control.Dockerfile` CMD
- `docker-compose.yml` command
- `Makefile` targets
- `pyproject.toml` entry points (if any)

Keep Dockerfile name as `control.Dockerfile` to avoid churn in CI/compose service names.

## File Change Summary

| Action | Path |
|--------|------|
| **New** | `training/__init__.py` |
| **New** | `training/__main__.py` |
| **New** | `training/loops/__init__.py` |
| **New** | `training/loops/base.py` |
| **New** | `training/loops/rl.py` |
| **New** | `training/loops/supervised.py` |
| **Move** | `control/callbacks.py` → `training/callbacks.py` |
| **Move** | `control/trajectory_recorder.py` → `training/trajectory_recorder.py` |
| **New** | `sim/envs/base.py` |
| **New** | `sim/envs/numpy_quad_factory.py` |
| **New** | `perception/wrappers/__init__.py` |
| **New** | `perception/wrappers/base.py` |
| **New** | `perception/wrappers/identity.py` |
| **Move** | `perception/noise_injection.py` → `perception/wrappers/noise.py` |
| **Edit** | `configs/train.yaml` — add `loop: rl` default |
| **New** | `configs/loop/rl.yaml` |
| **New** | `configs/loop/supervised.yaml` |
| **Edit** | `configs/sim/numpy_quad.yaml` — add `_target_` |
| **Edit** | `configs/perception/none.yaml` — add `_target_` |
| **Edit** | `configs/perception/noise_injection.yaml` — add `_target_` |
| **Edit** | `docker/control.Dockerfile` — update CMD |
| **Edit** | `docker-compose.yml` — update command |
| **Edit** | `Makefile` — update targets |
| **Delete** | `control/__main__.py` |
| **Update** | Tests importing `control.callbacks` / `control.trajectory_recorder` |
