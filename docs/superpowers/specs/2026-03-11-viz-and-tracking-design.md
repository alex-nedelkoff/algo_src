# Visualization & Experiment Tracking Design

**Date**: 2026-03-11
**Status**: Draft
**Related**: COR-50 (baseline training validation)

## Context

Baseline PPO training runs end-to-end. We have metrics flowing through GateMetricsCallback to TensorBoard/W&B, and checkpoints saving at regular intervals. What's missing:

1. No way to *see* what the policy is doing — no trajectory recording or replay
2. No synthetic camera view showing what a perception model would see
3. Experiment tracking metrics lack reward component breakdown and racing diagnostics

## Goal

Add trajectory recording, rerun-based 3D visualization with pinhole camera rendering, and enhanced experiment tracking metrics to the training pipeline.

## Design

### 1. Trajectory Recorder

**New file**: `control/trajectory_recorder.py`

A new SB3 callback (`TrajectoryRecorderCallback`) that runs at a separate `viz_freq` interval (default 1M total timesteps), independent of eval and checkpoint frequencies.

**Behavior**:
1. Every `viz_freq` total timesteps, use `self.model` (the current policy, already in memory — no disk load)
2. Create a temporary single-env `GateRaceEnv` with domain randomization disabled for deterministic rollouts
3. Roll out `n_viz_episodes` (default 5) eval episodes using `self.model.predict(obs, deterministic=True)`
4. Record per-timestep arrays to `.npz` files

**Frequency note**: `viz_freq` is specified in **total timesteps** (not env steps). The callback internally divides by `n_envs` to convert to the SB3 callback step count, matching the convention used by `CheckpointCallback` and `EvalCallback` in `__main__.py`.

**Recorded data per episode**:
| Array | Shape | Description |
|-------|-------|-------------|
| `schema_version` | scalar | Schema version (1) for future compatibility. Rerun generator checks this and raises a clear error for unsupported versions. |
| `positions` | (T, 3) | Drone xyz world position |
| `quaternions` | (T, 4) | Drone orientation quaternion (w, x, y, z) |
| `velocities` | (T, 3) | World-frame velocity |
| `body_rates` | (T, 3) | Angular velocity in body frame (gyro) |
| `motor_rpms` | (T, 4) | Motor speeds |
| `actions` | (T, 4) | Raw policy output |
| `rewards` | (T,) | Per-step scalar reward |
| `reward_components` | (T, 6) | Breakdown: [progress, body_rate, action_smooth, gate_passage, gate_offset, crash_penalty]. Non-event columns are 0.0 on steps where they don't fire. |
| `reward_component_names` | (6,) | String labels for the 6 columns |
| `gate_events` | (E, 2) | (timestep, gate_idx) for passage events |
| `gate_positions` | (G, 3) | Track gate center positions |
| `gate_orientations` | (G, 4) | Track gate orientation quaternions (w, x, y, z) |
| `gate_half_extents` | (G, 2) | Gate half-width/half-height from `gate_passage_radius` config |
| `dt` | scalar | Timestep for time reconstruction |

**Quaternion convention**: All quaternions stored as (w, x, y, z) matching the codebase standard in `QuadState` and `GateState`. The rerun generator must swizzle to (x, y, z, w) for `rr.Quaternion`.

**Gate dimensions**: The physical gate size is derived from the existing `gate_passage_radius` config value (default 1.0m). This is stored as `gate_half_extents` — a (G, 2) array where each row is `[half_width, half_height]`, both set to `gate_passage_radius`. No changes to `GateState` needed.

**Storage**: Aligned with SB3's flat checkpoint naming convention. SB3 produces `checkpoints/ppo_{N}_steps.zip`. Trajectory recorder creates a parallel directory:
```
checkpoints/
├── ppo_1000000_steps.zip          # SB3 checkpoint (existing)
├── ppo_2000000_steps.zip
├── trajectories/
│   ├── step_1000000/
│   │   ├── eval_ep_0.npz
│   │   ├── eval_ep_1.npz
│   │   └── ...
│   └── step_2000000/
│       └── ...
└── rerun/                          # Generated offline
    ├── step_1000000/
    │   ├── eval_ep_0.rrd
    │   └── ...
    └── step_2000000/
        └── ...
```

**Config additions** to `train.yaml`:
```yaml
viz_freq: 1_000_000        # total timesteps between trajectory recording
n_viz_episodes: 5           # episodes to record per viz checkpoint
```

### 2. Reward Component Decomposition

**Modified file**: `sim/rewards.py`

The `monorace_reward` function currently returns a single scalar. To support both trajectory recording (`reward_components` array) and enhanced metrics, it will be updated to return a `RewardResult` named tuple:

```python
class RewardResult(NamedTuple):
    total: float
    components: dict[str, float]
```

**This is a clean break** — `monorace_reward` will only return `RewardResult`, not `float`. Both `rewards.py` and `gate_race_env.py` are updated atomically. No backward-compatibility shim.

The `components` dict contains:
| Key | Description |
|-----|-------------|
| `progress` | Weighted gate progress reward |
| `body_rate` | Weighted body rate penalty |
| `action_smooth` | Weighted action smoothness penalty |

Gate passage, gate offset, and crash penalty are applied as discrete events in `GateRaceEnv.step()`, not inside `monorace_reward`. These are tracked separately:
| Key | Description |
|-----|-------------|
| `gate_passage` | Discrete reward on gate crossing (weight × 1.0) |
| `gate_offset` | Discrete offset penalty on gate crossing |
| `crash_penalty` | Penalty applied on crash termination |

**GateRaceEnv changes**: The env maintains a per-env `_step_reward_components` array of shape `(n_envs, 6)` initialized to zeros each step. Columns: `[progress, body_rate, action_smooth, gate_passage, gate_offset, crash_penalty]`. At each `step()`:
- Continuous components from `monorace_reward().components` fill columns 0-2
- Discrete gate events fill columns 3-4 (0.0 on non-passage steps)
- Crash penalty fills column 5 (0.0 on non-crash steps)
- Episode totals are accumulated in a per-env `_episode_reward_components` array and surfaced in the `info` dict on episode termination as `reward_components: dict[str, float]`

The env also accumulates `_episode_speed_sum` and `_episode_speed_count` per env (norm of world-frame velocity each step) to surface `avg_speed` in the episode info dict.

### 3. Rerun Generator

**New files**:
- `sim/viz/rerun_generator.py` — main `.npz` → `.rrd` conversion
- `sim/viz/pinhole.py` — pinhole camera projection math
- `sim/viz/__init__.py`
- `sim/viz/__main__.py` — entry point for `python -m sim.viz`

**Invocation**: `python -m sim.viz <trajectory_dir_or_npz_file> [--output-dir <path>]`

Can process a single `.npz` or batch-process all trajectories in a step directory. Outputs `.rrd` files into `checkpoints/rerun/step_{N}/`.

#### 3a. 3D Scene (`/world`)

- **Gates**: Wireframe rectangles via `rr.LineStrips3D`, colored per gate index. Gate corners computed from `gate_positions`, `gate_orientations`, and `gate_half_extents`.
- **Drone**: 3D box (vehicle dimensions from config) with 4 motor points (`rr.Points3D`) to show pitch/roll attitude
- **Flight trail**: `rr.LineStrips3D` accumulated path, colored by speed (norm of world-frame velocity)
- **Coordinate frame**: World-frame axes at origin

Drone pose updated per timestep using `rr.Transform3D` from recorded position + quaternion (swizzled to x, y, z, w for rerun).

#### 3b. Pinhole Camera View (`/drone/camera`)

Forward-facing virtual camera rigidly attached to drone body frame.

**Projection model**:
- Standard pinhole: `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
- Default intrinsics: 90° horizontal FOV, 320×240 resolution
- Camera extrinsics: identity rotation relative to drone body (forward-looking), slight downward tilt configurable

**Rendered elements**:
- Gate rectangles projected as wireframe lines (4 corners → 4 edges)
- Horizon line
- Only gates within FOV and in front of camera are rendered

**Frame decimation**: Render every 10th timestep (configurable `--camera-decimation`) to manage `.rrd` file size. At dt=0.01 and max 1200 steps, this produces ~120 frames per episode. Estimated `.rrd` size: ~2-5MB per episode (vs ~100KB for `.npz`).

**Output**: Per-frame rasterized image logged via `rr.Image` to the `/drone/camera` entity. Pure numpy line rasterization — no OpenGL dependency.

#### 3c. Timeseries Panels (`/telemetry/*`)

Logged as `rr.Scalar` per timestep (every step, no decimation):
- `/telemetry/motors/rpm_{0-3}` — 4 motor RPM traces
- `/telemetry/body_rates/{roll,pitch,yaw}` — angular velocity
- `/telemetry/acceleration/{x,y,z}` — linear acceleration (derived from velocity finite difference)
- `/telemetry/reward` — per-step reward
- `/telemetry/reward_cumulative` — running total

#### 3d. Events (`/events`)

Gate passage events logged as `rr.TextLog` annotations at the corresponding timestep, marking which gate was passed.

### 4. Enhanced Metrics

**Modified file**: `control/callbacks.py` (extend `GateMetricsCallback`)

New metrics logged to TensorBoard/W&B at existing callback frequency:

| Metric | Description |
|--------|-------------|
| `racing/reward_gate_passage` | Mean total gate passage reward per episode |
| `racing/reward_progress` | Mean total progress reward per episode |
| `racing/reward_body_rate` | Mean body rate penalty per episode |
| `racing/reward_crash_penalty` | Mean crash penalty per episode |
| `racing/avg_speed` | Mean drone speed (norm of world-frame velocity) across episodes |
| `racing/time_to_first_gate` | Average steps to first gate passage |

**`time_to_first_gate` implementation**: Add a `_first_gate_step` per-env counter in `GateRaceEnv` that records the step index when `gates_passed` first becomes > 0. Surfaced in the `info` dict on episode termination as `first_gate_step`. If no gates were passed during the episode, the value is `None` — `GateMetricsCallback` filters these out when computing the average. Reset on auto-reset.

**Reward components in info dict**: On episode termination, `info` includes `reward_components: dict[str, float]` with per-component episode totals, plus `avg_speed: float` (mean world-frame speed). `GateMetricsCallback` reads these to compute rolling averages.

### 5. Dependencies

**Training image** (no changes): trajectory recording uses only numpy (`.npz`).

**Visualization** (new, optional):
- `rerun-sdk` — for `.rrd` generation
- No other new dependencies; pinhole projection is pure numpy

Add `rerun-sdk` to an optional `[viz]` dependency group in `pyproject.toml` so it doesn't bloat the training image.

### 6. New/Modified Files Summary

| File | Change |
|------|--------|
| `control/trajectory_recorder.py` | **New** — TrajectoryRecorderCallback |
| `control/__main__.py` | **Modified** — wire TrajectoryRecorderCallback into callback list |
| `control/callbacks.py` | **Modified** — add reward component + speed + time_to_first_gate metrics |
| `sim/rewards.py` | **Modified** — `monorace_reward` returns `RewardResult` with component breakdown |
| `sim/envs/gate_race_env.py` | **Modified** — accumulate reward components, add `_first_gate_step`, `avg_speed`, surface in info |
| `sim/envs/vec_env_adapter.py` | **Modified** — forward `reward_components`, `avg_speed`, `first_gate_step` from episode info |
| `sim/viz/__init__.py` | **New** — package init |
| `sim/viz/__main__.py` | **New** — CLI entry point |
| `sim/viz/rerun_generator.py` | **New** — `.npz` → `.rrd` conversion |
| `sim/viz/pinhole.py` | **New** — pinhole camera projection math |
| `configs/train.yaml` | **Modified** — add `viz_freq`, `n_viz_episodes` |
| `pyproject.toml` | **Modified** — add `[viz]` optional dependency group with `rerun-sdk` |

### 7. Output Directory Structure

```
outputs/{timestamp}/
├── .hydra/
├── tb_logs/
├── checkpoints/
│   ├── ppo_1000000_steps.zip       # SB3 checkpoint (flat file, existing convention)
│   ├── ppo_2000000_steps.zip
│   ├── trajectories/
│   │   ├── step_1000000/
│   │   │   ├── eval_ep_0.npz
│   │   │   └── ...
│   │   └── step_2000000/
│   │       └── ...
│   └── rerun/                       # Generated offline by `python -m sim.viz`
│       ├── step_1000000/
│       │   ├── eval_ep_0.rrd
│       │   └── ...
│       └── step_2000000/
│           └── ...
├── best_model/
│   └── best_model.zip
└── wandb/
```

### 8. Non-Goals

- Photorealistic rendering — wireframe geometric style only
- Live rerun streaming during training — artifacts generated post-rollout via offline script
- Curriculum scheduler wiring — separate concern
- Perception-in-the-loop — future work
- Reward-based trail coloring — default to speed; reward coloring deferred (negative values complicate color mapping)
