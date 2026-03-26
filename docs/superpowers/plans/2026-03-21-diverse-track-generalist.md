# Diverse Track Generalist Policy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a generalist policy that scores <50% crash rate on the golden set benchmark by training on a much wider distribution of track geometries.

**Architecture:** No new generator code needed — the existing `ProceduralTrackGenerator` already accepts all the parameters we need. We just create a `diverse.yaml` config with wider ranges (±170° turns, 1.5m elevation delta, 1.0-6.0m spacing) and combine it with the `Figure8TrackGenerator` via `MixedTrackGenerator`. Train from scratch with `gate_passage_radius: 0.75` (matching golden set), `arena_bounds: 5` (matching golden set), and all reward components.

**Tech Stack:** Existing code, config-only changes + training run

**Key insight from golden set analysis:**

| Parameter | Our training | Golden set needs | New config |
|-----------|:---:|:---:|:---:|
| Turn angles | ±120° | ~160° (hairpin) | **±170°** |
| Elevation delta | ±0.6m | 2.5m total (elevation_climb) | **±1.5m** |
| Gate spacing | 1.5-4.0m | Varies 1.0-6.0m | **1.0-6.0m** |
| Gate passage radius | 1.5m training | 0.75m (golden set) | **0.75m** |
| Arena bounds | 10m | 5m (golden set) | **5m** |
| Max steps | 3000 | 1200 (golden set) | **1200** |
| Figure-8 mix | 40% | Required | **40%** |
| Elevation range | 1.0-3.5m | 1.0-3.5m | **0.5-4.0m** |

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `configs/track_gen/diverse.yaml` | Wide parameter ranges for diverse tracks |
| Create | `configs/experiment/generalist.yaml` | Full training config (40M, from scratch) |
| — | No code changes | Existing generators handle all parameter ranges |

### Task 1: Diverse Track Config

**Files:**
- Create: `configs/track_gen/diverse.yaml`

- [ ] **Step 1: Create diverse track gen config**

```yaml
# configs/track_gen/diverse.yaml
# Wide parameter ranges covering golden set track diversity.
# Hairpins (±170°), elevation climbs (±1.5m), tight spacing (1.0m+).
n_gates_min: 4
n_gates_max: 10
gate_spacing_min: 1.0
gate_spacing_max: 6.0
turn_angle_min: -170
turn_angle_max: 170
elevation_min: 0.5
elevation_max: 4.0
elevation_delta_max: 1.5
closure_max_angle: 120
closure_max_retries: 30
# Figure-8 mixing
figure8_ratio: 0.3
figure8:
  loop_radius_min: 1.5
  loop_radius_max: 3.0
  gates_per_loop_min: 3
  gates_per_loop_max: 4
  crossing_offset_min: 0.3
  crossing_offset_max: 0.6
  elevation_min: 1.0
  elevation_max: 3.0
  elevation_delta_max: 0.8
```

- [ ] **Step 2: Verify the config generates tracks**

```bash
python -c "
from omegaconf import OmegaConf
from sim.procedural_tracks import ProceduralTrackGenerator, MixedTrackGenerator
from sim.figure8_tracks import Figure8TrackGenerator
import numpy as np

cfg = OmegaConf.load('configs/track_gen/diverse.yaml')
tg_dict = OmegaConf.to_container(cfg, resolve=True)
fig8_cfg = tg_dict.pop('figure8', None)
fig8_ratio = tg_dict.pop('figure8_ratio', 0.0)

proc = ProceduralTrackGenerator(arena_half_width=5.0, **tg_dict)
fig8 = Figure8TrackGenerator(**fig8_cfg)
gen = MixedTrackGenerator(proc, fig8, fig8_ratio)

rng = np.random.default_rng(42)
for i in range(20):
    track = gen.generate(rng)
    print(f'Track {i}: {track.num_gates} gates')
print('All 20 tracks generated successfully')
"
```
Expected: 20 tracks generated, varying gate counts 4-10.

Note: The wider turn angles (±170°) and larger elevation deltas (±1.5m) may cause more closure failures. The `closure_max_retries: 30` handles this — failed attempts fall back to figure-8 tracks.

- [ ] **Step 3: Commit**

```bash
git add configs/track_gen/diverse.yaml
git commit -m "config: add diverse track gen config with wider parameter ranges"
```

### Task 2: Generalist Experiment Config

**Files:**
- Create: `configs/experiment/generalist.yaml`

- [ ] **Step 1: Create experiment config**

```yaml
# configs/experiment/generalist.yaml
# @package _global_
# Generalist policy: train on diverse tracks matching golden set difficulty.
# From scratch with gate_passage_radius=0.75 (golden set standard).

defaults:
  - override /sim: numpy_quad
  - override /control: ppo
  - override /perception: corner_noise
  - override /domain_rand: uniform_30pct
  - override /logging: wandb
  - _self_

sim:
  action_mode: trpy
  n_lookahead_gates: 2
  max_steps: 1200            # Match golden set (12s episodes)
  arena_bounds: 5            # Match golden set (was 10)
  gate_passage_radius: 0.75  # Match golden set (was 1.5!)
  gate_collision: false      # Learn from near-misses

total_timesteps: 40_000_000
# No resume — from scratch with new gate radius

track_gen:
  n_gates_min: 4
  n_gates_max: 10
  gate_spacing_min: 1.0
  gate_spacing_max: 6.0
  turn_angle_min: -170
  turn_angle_max: 170
  elevation_min: 0.5
  elevation_max: 4.0
  elevation_delta_max: 1.5
  closure_max_angle: 120
  closure_max_retries: 30
  figure8_ratio: 0.3
  figure8:
    loop_radius_min: 1.5
    loop_radius_max: 3.0
    gates_per_loop_min: 3
    gates_per_loop_max: 4
    crossing_offset_min: 0.3
    crossing_offset_max: 0.6
    elevation_min: 1.0
    elevation_max: 3.0
    elevation_delta_max: 0.8

perception:
  noise_scale: 1.0
  dropout_onset: null
  dropout_continuation: 0.7

reward:
  weights:
    gate_passage: 1.5
    gate_progress: 1.0
    gate_offset: 1.5
    body_rate: 0.001
    action_smoothness: 0.0
    crash_penalty: 10.0
    spline_proximity: 0.5
    heading_alignment: 0.05
    speed_bonus: 0.0
    boundary_penalty: 5.0
    gate_approach: 0.2
    gate_centering: 3.0
  v_max: 10.0
  action_smoothness_threshold: 0.5

# aRPO bootstrap with slow attenuation
arpo:
  enabled: true
  k_end_fraction: 0.5       # Base policy active for first 20M steps

curriculum:
  enabled: true
  stages:
    - timestep: 0
      reward_weights:
        gate_passage: 1.5
        gate_progress: 1.0
        gate_offset: 1.5
        crash_penalty: 10.0
        spline_proximity: 0.5
        heading_alignment: 0.05
        boundary_penalty: 5.0
        gate_approach: 0.2
        gate_centering: 3.0
      v_max: 8.0
      ent_coef: 0.005
      trigger: null
    - timestep: 15_000_000
      reward_weights:
        gate_passage: 10.0
        gate_progress: 0.5
        gate_offset: 2.0
        crash_penalty: 5.0
        spline_proximity: 0.2
        heading_alignment: 0.03
        speed_bonus: 0.3
        boundary_penalty: 3.0
        gate_approach: 0.3
        gate_centering: 2.0
      v_max: 15.0
      ent_coef: 0.003
      trigger:
        metric: racing/gate_passage_rate
        threshold: 0.70
        window: 200
    - timestep: 30_000_000
      reward_weights:
        gate_passage: 30.0
        gate_progress: 0.0
        gate_offset: 2.0
        crash_penalty: 3.0
        spline_proximity: 0.0
        heading_alignment: 0.0
        speed_bonus: 0.5
        boundary_penalty: 2.0
        gate_approach: 0.3
        gate_centering: 1.0
      v_max: 20.0
      ent_coef: 0.001
      trigger:
        metric: racing/lap_completion_rate
        threshold: 0.3
        window: 200

multi_scene:
  enabled: true
  n_scenes: 10

lr_schedule:
  initial_lr: 0.0003
  final_lr: 0.00005

logging:
  tags: ["trpy", "generalist", "diverse-tracks", "golden-match", "arpo", "40M"]
  group: generalist
  notes: "Generalist policy: diverse tracks (±170° turns, ±1.5m elevation, 0.75m gate radius), 5m arena, 1200 steps, 30% figure-8, aRPO k_end=0.5"
```

- [ ] **Step 2: Commit**

```bash
git add configs/experiment/generalist.yaml
git commit -m "config: add generalist experiment matching golden set track difficulty"
```

### Task 3: Verify + Launch

- [ ] **Step 1: Verify diverse config generates valid tracks**

Run the verification script from Task 1 Step 2.

- [ ] **Step 2: Launch training**

```bash
python -m training +experiment=generalist device=cuda
```

- [ ] **Step 3: After training, run golden set benchmark**

```bash
WANDB_API_KEY=<key> python -m benchmark run \
  --checkpoint <latest_checkpoint> \
  --golden-set-mode holdout \
  --golden-set-dir configs/golden_set \
  --no-trajectories \
  --wandb-project corvidx-drone-racing \
  --sim-config /tmp/benchmark_sim.yaml
```

Update `/tmp/benchmark_sim.yaml` to use the generalist's params (gate_passage_radius=0.75, arena_bounds=5, etc.).

---

## Summary

| Task | What | Effort |
|------|------|--------|
| 1 | `diverse.yaml` track gen config | 5 min |
| 2 | `generalist.yaml` experiment config | 5 min |
| 3 | Verify + launch + benchmark | 10 min + training time |

**Key differences from previous training:**

| Parameter | Previous (figure8_gentle) | Generalist |
|-----------|:---:|:---:|
| Turn angles | ±120° | **±170°** |
| Elevation delta | ±0.6m | **±1.5m** |
| Gate spacing | 1.5-4.0m | **1.0-6.0m** |
| Gate radius | 1.5m | **0.75m** (2× tighter!) |
| Arena | 10m | **5m** (matching golden set) |
| Episodes | 3000 steps (30s) | **1200 steps (12s)** |
| Figure-8 mix | 40% | **30%** |
| Training | Resume | **From scratch** |

**Expected training time:** ~5 hours at ~2500 FPS (GPU) for 40M steps.

**Success criteria:** Golden set aggregate crash rate <50% (currently 80%), no single track at 100% crash.
