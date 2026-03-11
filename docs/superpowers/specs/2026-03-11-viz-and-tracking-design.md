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

**New file**: `sim/viz/trajectory_recorder.py`

A new SB3 callback (`TrajectoryRecorderCallback`) that runs at a separate `viz_freq` interval (default 1M steps), independent of eval and checkpoint frequencies.

**Behavior**:
1. Every `viz_freq` steps, load the current policy
2. Roll out `n_viz_episodes` (default 5) eval episodes with domain randomization disabled
3. Record per-timestep arrays to `.npz` files

**Recorded data per episode**:
| Array | Shape | Description |
|-------|-------|-------------|
| `positions` | (T, 3) | Drone xyz world position |
| `quaternions` | (T, 4) | Drone orientation quaternion |
| `velocities` | (T, 3) | Body-frame velocity |
| `body_rates` | (T, 3) | Angular velocity (gyro) |
| `motor_rpms` | (T, 4) | Motor speeds |
| `actions` | (T, 4) | Raw policy output |
| `rewards` | (T,) | Per-step scalar reward |
| `reward_components` | (T, K) | Breakdown by reward component |
| `gate_events` | (E, 2) | (timestep, gate_idx) for passage events |
| `gate_positions` | (G, 3) | Track gate positions |
| `gate_orientations` | (G, 4) | Track gate orientation quaternions |
| `gate_dimensions` | (G, 2) | Gate width/height |
| `dt` | scalar | Timestep for time reconstruction |

**Storage**: `checkpoints/step_{N}/trajectories/eval_ep_{i}.npz` (~100KB per episode).

**Config additions** to `train.yaml`:
```yaml
viz_freq: 1_000_000        # steps between trajectory recording
n_viz_episodes: 5           # episodes to record per viz checkpoint
```

### 2. Rerun Generator

**New files**:
- `sim/viz/rerun_generator.py` — main `.npz` → `.rrd` conversion
- `sim/viz/pinhole.py` — pinhole camera projection math
- `sim/viz/__init__.py`

**Invocation**: `python -m sim.viz.generate_rerun <trajectory_dir_or_npz_file>`

Can process a single `.npz` or batch-process all trajectories in a checkpoint directory. Outputs `.rrd` files alongside the input `.npz` files in a `rerun/` subdirectory.

**Output path**: `checkpoints/step_{N}/rerun/eval_ep_{i}.rrd`

#### 2a. 3D Scene (`/world`)

- **Gates**: Wireframe rectangles via `rr.LineStrips3D`, colored per gate index
- **Drone**: 3D box (vehicle dimensions from config) with 4 motor points (`rr.Points3D`) to show pitch/roll attitude
- **Flight trail**: `rr.LineStrips3D` accumulated path, optionally colored by speed or reward
- **Coordinate frame**: World-frame axes at origin

Drone pose updated per timestep using `rr.Transform3D` from recorded position + quaternion.

#### 2b. Pinhole Camera View (`/drone/camera`)

Forward-facing virtual camera rigidly attached to drone body frame.

**Projection model**:
- Standard pinhole: `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
- Default intrinsics matching a typical racing drone camera (e.g., 90° horizontal FOV, 320×240 resolution)
- Camera extrinsics: identity rotation relative to drone body (forward-looking), slight downward tilt configurable

**Rendered elements**:
- Gate rectangles projected as wireframe lines (4 corners → 4 edges)
- Horizon line
- Simple ground plane grid (optional)
- Only gates within FOV and in front of camera are rendered

**Output**: Per-timestep rasterized image logged via `rr.Image` to the `/drone/camera` entity. The image is a simple numpy array rendered with basic line drawing (no OpenGL dependency).

#### 2c. Timeseries Panels (`/telemetry/*`)

Logged as `rr.Scalar` per timestep:
- `/telemetry/motors/rpm_{0-3}` — 4 motor RPM traces
- `/telemetry/body_rates/{roll,pitch,yaw}` — angular velocity
- `/telemetry/acceleration/{x,y,z}` — linear acceleration (derived from velocity diff)
- `/telemetry/reward` — per-step reward
- `/telemetry/reward_cumulative` — running total

#### 2d. Events (`/events`)

Gate passage events logged as `rr.TextLog` annotations at the corresponding timestep, marking which gate was passed.

### 3. Enhanced Metrics

**Modified file**: `control/callbacks.py` (extend `GateMetricsCallback`)

New metrics logged to TensorBoard/W&B at existing callback frequency:

| Metric | Description |
|--------|-------------|
| `racing/reward_gate_passage` | Mean reward from gate passage component |
| `racing/reward_progress` | Mean reward from gate progress component |
| `racing/reward_alive` | Mean alive bonus per episode |
| `racing/reward_crash_penalty` | Mean crash penalty per episode |
| `racing/avg_speed` | Mean drone speed across episodes |
| `racing/time_to_first_gate` | Average steps to first gate passage |

These require the environment to expose reward component breakdown in the info dict. The `GateRaceEnv` step function will be updated to include a `reward_components` dict in `info`.

### 4. Dependencies

**Training image** (no changes): trajectory recording uses only numpy (`.npz`).

**Visualization** (new, optional):
- `rerun-sdk` — for `.rrd` generation
- No other new dependencies; pinhole projection is pure numpy

Add `rerun-sdk` to an optional `[viz]` dependency group in `pyproject.toml` so it doesn't bloat the training image.

### 5. Output Directory Structure

```
outputs/{timestamp}/
├── .hydra/
├── tb_logs/
├── checkpoints/
│   ├── step_1000000/
│   │   ├── model.zip
│   │   ├── trajectories/
│   │   │   ├── eval_ep_0.npz
│   │   │   ├── eval_ep_1.npz
│   │   │   ├── eval_ep_2.npz
│   │   │   ├── eval_ep_3.npz
│   │   │   └── eval_ep_4.npz
│   │   └── rerun/
│   │       ├── eval_ep_0.rrd
│   │       ├── eval_ep_1.rrd
│   │       └── ...
│   ├── step_2000000/
│   │   └── ...
│   └── best_model/
│       ├── model.zip
│       ├── trajectories/
│       └── rerun/
└── wandb/
```

### 6. Non-Goals

- Photorealistic rendering — wireframe geometric style only
- Live rerun streaming during training — artifacts generated post-rollout
- Curriculum scheduler wiring — separate concern
- Perception-in-the-loop — future work
