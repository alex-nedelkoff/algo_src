# Unified Metrics Contract & Experiment Onboarder

**Date**: 2026-03-13
**Status**: Draft
**Author**: Claude + Janahan

## Problem

Different environments (GateRaceEnv, RateCtrlEnv) produce metrics in incompatible schemas. Termination codes differ (9 codes vs 3), reward component names differ (6 vs 8), some metrics are missing entirely (avg_speed, first_gate_step). The callback hardcodes GateRaceEnv assumptions, causing broken success_rate (always 0%) and mislabeled termination charts for playground runs. The trajectory recorder reaches into env internals with fragile dual-path `if/elif` logic.

Cross-experiment comparison in W&B is unreliable. Onboarding a new environment requires knowing which internal attributes the callback and recorder expect — there's no contract or guide.

## Solution

Two deliverables:

1. **Metrics contract** — a typed episode info schema that every environment must satisfy, with runtime validation and contract-aware consumers (callback, trajectory recorder).
2. **Experiment onboarder skill** — a Claude Code skill that guides sessions through onboarding new components (env, perception model, control algo, renderer) into the framework, ensuring metrics contract compliance and visualization pipeline integration.

## Deliverable 1: Metrics Contract

### 1.1 EpisodeMetrics TypedDict

A mandatory episode info schema. Every environment must populate these fields in `info["episode"]`.

```python
# metrics/contract.py

from __future__ import annotations

from typing import Any, TypedDict
import numpy as np

class EpisodeMetrics(TypedDict):
    """Mandatory episode metrics contract.

    Every environment must populate all of these fields in info["episode"]
    at the end of each episode. The GateMetricsCallback and VecEnvAdapter
    read only these keys — no env-specific assumptions.

    Non-racing environments (e.g., HoverEnv) should set racing-specific
    fields to their zero values: gates_passed=0, laps_completed=0,
    first_gate_step=-1.
    """
    r: float                        # cumulative episode reward
    l: int                          # episode length (RL steps)
    effective_dt: float             # seconds per RL step (physics_dt * action_repeat)
    gates_passed: int               # gate passages this episode (0 for non-racing)
    laps_completed: int             # full laps completed (0 for non-racing)
    termination: str                # canonical name: "timeout", "crash", "ground", etc.
    success: bool                   # env-defined (see success_criterion)
    success_criterion: str          # human-readable: "survived_full_episode"
    avg_speed: float                # mean velocity magnitude (m/s)
    first_gate_step: int            # step of first gate passage (-1 if none)
    reward_components: dict[str, float]  # env-native component breakdown
```

**Design decisions**:

- **`termination` is a string, not an int code.** Eliminates the TERM_NAMES mapping bug. Each env maps its internal codes to canonical strings at the source. The old `termination_reason` (int) key is removed — clean break, no backwards compatibility shim.
- **`success` is env-defined + documented.** Each env determines what success means for its task and provides a `success_criterion` string. The callback logs the criterion as a W&B run config value so dashboards are unambiguous when comparing across experiments.
- **`effective_dt` enables accurate lap time computation.** The callback currently hardcodes `dt=0.01` for lap time calculation. With `effective_dt` in the contract, `lap_time = (ep_len * effective_dt) / laps` works correctly for any env regardless of its physics timestep or action repeat.
- **`reward_components` uses env-native names.** No forced unification — different reward functions have legitimately different components. Both get logged under `racing/reward_{name}`. Components with the same name (e.g., `progress`) are naturally comparable.
- **`reward_components` is emitted as `dict[str, float]` by the env.** The VecEnvAdapter no longer does array→dict conversion. Each env constructs the dict using its own component names and values. This removes the adapter's dependency on `reward_component_names` and keeps the mapping co-located with the reward function.
- **`first_gate_step` uses -1 as sentinel, not None.** Keeps the TypedDict simple (no Optional). The callback only records entries where `first_gate_step >= 0`.
- **Non-racing envs** (e.g., HoverEnv) set racing-specific fields to zero values: `gates_passed=0`, `laps_completed=0`, `first_gate_step=-1`. They still must define `termination`, `success`, `success_criterion`, `avg_speed`, and `reward_components`.
- **No `extras` in the contract itself.** Env-specific data that doesn't fit the core goes into `info["extras"]` (optional, not validated). Callbacks can read it but are not required to.

### 1.2 TrajectoryProvider Protocol

Environments must implement this protocol for the trajectory recorder. Replaces the current dual-path attribute access.

```python
# metrics/contract.py

from typing import Protocol, runtime_checkable

@runtime_checkable
class TrajectoryProvider(Protocol):
    """Interface for trajectory recording.

    Envs implement these methods so the TrajectoryRecorderCallback
    can extract per-step state without reaching into env internals.

    Gate geometry is assumed to be shared across all envs in a vectorized
    environment (i.e., all envs race on the same track). If per-env track
    randomization is added in the future, this protocol must be updated.
    """

    def get_state(self, env_idx: int) -> dict[str, np.ndarray]:
        """Return current state for one environment.

        Returns dict with keys:
            position: (3,) float64 — x, y, z
            quaternion: (4,) float64 — w, x, y, z
            velocity: (3,) float64 — vx, vy, vz
            body_rates: (3,) float64 — p, q, r
            motor_rpms: (4,) float64 — per-motor RPM
        """
        ...

    def get_gate_geometry(self) -> dict[str, np.ndarray]:
        """Return gate geometry for the track (shared across all envs).

        Returns dict with keys:
            positions: (n_gates, 3) float64
            orientations: (n_gates, 4) float64 — quaternions (w, x, y, z)
            half_extents: (n_gates, 2) float64 — (half_width, half_height)
        """
        ...

    def get_step_reward_components(self, env_idx: int) -> tuple[list[str], np.ndarray]:
        """Return per-step reward component breakdown.

        Returns:
            names: list of component name strings
            values: (n_components,) float64 array
        """
        ...
```

**Note on `@runtime_checkable`**: Python's `isinstance()` check with `@runtime_checkable` only verifies that the required methods exist, not that their return values have the correct structure. The trajectory recorder validates the return dict keys on first call (see section 1.3) and raises a clear error if the dict schema is wrong.

### 1.3 Runtime Validation

Two validation functions, both one-shot at initialization:

```python
# metrics/contract.py

REQUIRED_EPISODE_KEYS = {
    "r": (int, float, np.floating),
    "l": (int, np.integer),
    "effective_dt": (int, float, np.floating),
    "gates_passed": (int, np.integer),
    "laps_completed": (int, np.integer),
    "termination": (str,),
    "success": (bool, np.bool_),
    "success_criterion": (str,),
    "avg_speed": (int, float, np.floating),
    "first_gate_step": (int, np.integer),
    "reward_components": (dict,),
}

class ContractViolation(Exception):
    """Raised when an environment violates the metrics contract."""
    pass

def validate_episode_metrics(episode: dict, env_name: str) -> None:
    """Validate episode info dict satisfies the metrics contract.

    Called once on first episode completion in VecEnvAdapter, then
    never again. Uses explicit raises (not assert) so validation
    cannot be disabled with python -O.
    """
    for key, types in REQUIRED_EPISODE_KEYS.items():
        if key not in episode:
            raise ContractViolation(
                f"{env_name} missing required episode key '{key}'. "
                f"See metrics/contract.py for the full contract."
            )
        if not isinstance(episode[key], types):
            raise ContractViolation(
                f"{env_name}.{key} has type {type(episode[key])}, "
                f"expected one of {types}."
            )

_REQUIRED_STATE_KEYS = {"position", "quaternion", "velocity", "body_rates", "motor_rpms"}

def validate_trajectory_state(state: dict, env_name: str) -> None:
    """Validate get_state() return dict has the required keys.

    Called once on first trajectory extraction, then never again.
    """
    missing = _REQUIRED_STATE_KEYS - set(state.keys())
    if missing:
        raise ContractViolation(
            f"{env_name}.get_state() missing required keys: {missing}. "
            f"See TrajectoryProvider protocol in metrics/contract.py."
        )
```

**Validation trigger points**:
- `validate_episode_metrics()`: Called in VecEnvAdapter on the first completed episode from any env index. The adapter sets `_contract_validated = True` after the first successful validation.
- `validate_trajectory_state()`: Called in TrajectoryRecorderCallback on the first call to `get_state()`. The recorder sets `_state_validated = True` after.

### 1.4 GateMetricsCallback Changes

The callback becomes contract-aware:

- **Remove `TERM_NAMES` dict** — no longer needed.
- **`_term_reasons` buffer stores strings**, not ints. Termination breakdown uses `Counter` on the string buffer to compute fractions dynamically.
- **`_successes` buffer** stores bools. Success rate = `sum(successes) / len(successes)`.
- **`success_criterion`** logged once as `wandb.config["success_criterion"]` on first episode.
- **Lap time uses `effective_dt`** from the contract instead of hardcoded 0.01: `lap_time = (ep_len * effective_dt) / laps`.
- **All other metrics** (gates, laps, reward components, avg_speed, first_gate_step, all-time bests) remain unchanged — they already read from the contract-compatible keys.

### 1.5 TrajectoryRecorderCallback Changes

- Replace all `hasattr(env, "_states")` / `hasattr(env, "_state")` dual-path logic with calls to `TrajectoryProvider` methods.
- Validate `isinstance(env, TrajectoryProvider)` at init. If the env doesn't satisfy the protocol, raise `ContractViolation` with a clear message.
- Validate `get_state()` return dict keys on first call via `validate_trajectory_state()`.
- `_extract_state()` → `env.get_state(idx)`
- `_extract_gate_geometry()` → `env.get_gate_geometry()`
- `_get_reward_components()` → `env.get_step_reward_components(idx)`

### 1.6 Environment Changes

**GateRaceEnv**:
- Episode info: replace `termination_reason` (int) with `termination` (string from `GATE_RACE_TERM_NAMES[code]`). Add `success` (bool: `term_code == TERM_TIMEOUT`), `success_criterion` ("survived_full_episode"), `effective_dt` (0.01).
- Episode info: rename `episode_length` to `l`.
- Episode info: emit `reward_components` as `dict[str, float]` directly (using `REWARD_COMPONENT_NAMES` to zip with the array).
- Implement `TrajectoryProvider` methods wrapping existing internal state.

**RateCtrlEnv**:
- Episode info: replace `termination_reason` (int) with `termination` (string from `RATE_CTRL_TERM_NAMES[code]`). Add `success` (bool: `term_code == TERM_TIMEOUT`), `success_criterion` ("survived_full_episode"), `effective_dt` (`self.dt * self.action_repeat`).
- Episode info: rename `episode_length` to `l`.
- Episode info: emit `reward_components` as `dict[str, float]` directly.
- `avg_speed` and `first_gate_step`: already partially implemented, finalize.
- Add `_step_reward_components` tracking for per-step trajectory recording (currently only episode-level).
- Implement `TrajectoryProvider` methods with Euler→quaternion conversion in `get_state()`.

### 1.7 VecEnvAdapter Changes

- Call `validate_episode_metrics()` on first episode completion from any env index. Set `_contract_validated = True`.
- Remove `reward_component_names` → dict conversion logic — envs now emit `reward_components` as a dict directly.
- Remove `termination_reason` → no longer passed through (envs emit `termination` string).
- `first_gate_step`: pass through as int. No conversion to `None` — sentinel value is `-1`.

### 1.8 Concrete Test: Monorace vs Playground

After implementation, both envs produce identical W&B metric keys:

| W&B Metric | Source |
|------------|--------|
| `racing/gates_per_ep` | `ep["gates_passed"]` mean |
| `racing/laps_per_ep` | `ep["laps_completed"]` mean |
| `racing/success_rate` | `ep["success"]` mean |
| `racing/avg_speed` | `ep["avg_speed"]` mean |
| `racing/steps_to_first_gate` | `ep["first_gate_step"]` mean (only where >= 0) |
| `racing/lap_time_mean` | `(ep["l"] * ep["effective_dt"]) / ep["laps_completed"]` |
| `racing/reward_{name}` | `ep["reward_components"][name]` mean (names differ per env) |
| `termination/{name}` | `Counter(ep["termination"])` fractions (names differ per env) |
| `racing/best_*_ever` | All-time bests from rolling buffers |

`success_criterion` is logged once as `wandb.config["success_criterion"]` so any dashboard comparison shows what "success" means for each run.

## Deliverable 2: Experiment Onboarder Skill

A Claude Code skill created via the `skill-creator` skill. This section defines its behavior; the actual skill file is produced during implementation.

### 2.1 Purpose

Guides sessions through onboarding new components into the framework. Ensures metrics contract compliance and visualization pipeline integration before any training runs.

### 2.2 Trigger Phrases

- "add a new env", "onboard a new environment"
- "port this simulator", "create a new experiment"
- "add a new perception model", "new control algo"
- "onboard", "experiment onboarder"

### 2.3 Component-Type-Aware Checklists

The skill detects which component type is being onboarded and presents the relevant subset.

**For a new environment:**
1. EpisodeMetrics contract — does `step()` return all required keys? Identify termination conditions and map to string names. Define success criterion. Set `effective_dt`.
2. TrajectoryProvider protocol — implement `get_state()`, `get_gate_geometry()`, `get_step_reward_components()`. Ensure `get_state()` returns quaternion-based attitude (convert from Euler if needed).
3. Non-racing fields — if env has no gates/laps, set `gates_passed=0`, `laps_completed=0`, `first_gate_step=-1`.
4. Env factory — create factory class following `NumpyQuadEnvFactory` / `PlaygroundEnvFactory` pattern.
5. Hydra config — add config group entry under `configs/sim/`.
6. VecEnvAdapter compatibility — verify batched info dict structure works with one-shot validation.
7. Smoke test — run 1000 steps, verify W&B metrics appear with correct names, verify rerun trajectory generates.

**For a new perception model:**
1. Hydra config — add config group entry under `configs/perception/`.
2. Observation pipeline — verify obs dim matches env's observation space.
3. W&B logging — ensure model-specific metrics (detection AP, inference time) logged under `perception/` prefix.
4. Smoke test — run with existing env, verify metrics appear.

**For a new control algorithm:**
1. Hydra config — add config group entry under `configs/control/`.
2. Action space compatibility — verify action dim matches env.
3. Training loop — verify callback integration (GateMetricsCallback, TrajectoryRecorder).
4. Smoke test — run 1000 steps, verify W&B metrics.

**For a new renderer / visualizer:**
1. Trajectory schema compatibility — verify .npz fields are consumed correctly.
2. Rerun integration — verify .rrd generation from .npz.
3. W&B panel — verify viewer URLs appear in Rerun HTML panel.

### 2.4 Implementation

Created using the `skill-creator` skill to ensure proper structure, description optimization, and trigger accuracy. Lives at `.claude/skills/experiment-onboarder.md`. The skill references `metrics/contract.py` as the source of truth for required fields and uses `validate_episode_metrics()` and `validate_trajectory_state()` as its programmatic checks.

## Files Changed

| File | Change |
|------|--------|
| `metrics/__init__.py` | New package |
| `metrics/contract.py` | New — EpisodeMetrics, TrajectoryProvider, ContractViolation, validate_episode_metrics, validate_trajectory_state |
| `sim/envs/gate_race_env.py` | Add contract fields to episode info, emit reward_components as dict, implement TrajectoryProvider |
| `sim/envs/rate_ctrl_env.py` | Add contract fields, finalize avg_speed/first_gate_step, add per-step reward components, implement TrajectoryProvider |
| `sim/envs/vec_env_adapter.py` | Add one-shot contract validation, remove reward_component array→dict conversion, remove termination_reason passthrough |
| `training/callbacks.py` | Remove TERM_NAMES, read string termination + bool success, use effective_dt for lap time, dynamic termination breakdown |
| `training/trajectory_recorder.py` | Use TrajectoryProvider protocol instead of dual-path attribute access, add state dict validation |
| `.claude/skills/experiment-onboarder.md` | New skill (via skill-creator) |

## What Doesn't Change

- Reward component names stay env-native (no forced unification)
- .npz trajectory schema stays as-is
- Rerun generator stays env-agnostic
- Hydra config structure stays as-is
- W&B logging via SB3 logger stays as-is
- ArtifactUploader stays as-is
