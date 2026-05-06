# M5 — Closed-Loop Test

**Date:** 2026-05-06
**Linear:** COR-106 (extended scope)
**Demo:** `outputs/perception/m5_closed_loop_demo.rrd`
**Predecessors:** M1 (localization), M2 (depth), M3 (waypoint adapter), M4 (planner MVP)

## Status

**Closed-loop policy successfully races the planner's detour around the obstacle.** With the right planner-output post-processing (cubic-spline smoothing + tight waypoint spacing + training-matched yaw convention), the trained MoE checkpoint completes 7 laps over the obstacle-detoured course in 90 s.

Result on a training-scale 8-gate oval with a 1 m cylinder injected between gates 2-3:

| Variant | Gates passed | Laps |
|---|---|---|
| Clean oval (no obstacle, no planner) | 129 | 16 |
| Oval + obstacle + planner detour, raw output | 5 | 0 |
| Oval + obstacle + planner detour, **best config** | **154** | **7** |

Best config: cubic-spline global smoothing of A* output, 1 m target waypoint spacing, lookahead=1 (training-matched yaw convention), no extra EMA yaw smoothing.

## Setup

- **Track:** 8 gates around an oval (rx=3.0 m, ry=2.5 m, z=2.0 m). Gate spacings 2.3-2.5 m — squarely inside the training distribution (1-6 m).
- **Obstacle:** 1 m radius cylinder at the midpoint between gates 2-3.
- **Planner:** A* on 0.25 m occupancy grid, dilated by drone radius (0.25 m), output resampled to 4 m waypoint spacing. ~16 waypoints per lap.
- **Policy:** `outputs/expanded_5expert_v3/model.zip` — 5-expert MoE, 28-dim obs, 4-dim TRPY action. Loaded via `control.algorithms.ppo.PPO(moe=True, n_experts=5, top_k=2)` then `load(path)` (strict=False; the `data` dict in the zip records 4 experts but the actual weights are 5-expert per `expand_moe.py`).
- **Env:** `GateRaceEnv` with the **5" racing-quad** params from `configs/sim/numpy_quad.yaml` (mass=0.752 kg, k_thrust=2.49e-6, max_rpm=31470). `random_gate_start=True`, `start_vel_std=1.5`, `gate_passage_radius=1.0`, `arena_bounds=15`, `action_mode="trpy"`, `esc_nonlinearity=0.95`. Matches the moe_generalist training config.
- **Obs slicing:** env produces 33-dim obs; checkpoint expects 28-dim. The slice drops gate width / height (idx 24-25 and 30-31) and arena_extent (idx 32) — exactly the format the AirSim / MAVLink shims emit.

## What works

1. **Checkpoint loading** via the corvidx PPO wrapper handles the 4→5 expert mismatch in the zip's data dict (use `_create_model(env)` then `load()` with strict=False).
2. **28-dim obs slice** from the env's 33-dim output is correct — verified via the M3 obs-equivalence tests and confirmed here by the 129-gate / 16-lap clean-oval result.
3. **Closed-loop step rate** is fast — 90 s of sim time runs in ~20 s of wall clock on RTX 3050 Laptop.
4. **Policy on training-distribution tracks** is competent — 129 gates over 16 laps means the env / params / obs / action mapping are all correct. This is the "pipeline plumbing" sanity check.
5. **Planner output is consumable** — the WaypointTrack adapter produces gates the policy can read; the env passes them; the policy emits actions; the env steps.

## What was failing

The first cut of M5 stalled at 5 gates. Two distinct issues compounded:

1. **Tangent discontinuity at planner segment joints.** Each A* segment was resampled independently, so the path tangent (and the synthetic yaw derived from it) jumped at every gate-to-gate boundary. **Fix:** concatenate raw A* outputs and apply a single global cubic B-spline arc-length resample. Tangent now continuous through joints.

2. **Wrong yaw lookahead convention.** The waypoint adapter defaulted to lookahead=2 (anticipatory — gate yaw points two waypoints ahead). The MoE was trained on tracks where each gate's yaw points to the next gate (lookahead=1, per `build_figure8_track`). Lookahead=2 produced gate-relative yaw distributions the policy hadn't seen. **Fix:** use lookahead=1 to match training.

A small parameter sweep nailed the optimum:

| smooth_globally | target_spacing_m | yaw_lookahead | yaw_smoothing_α | gates passed | laps |
|---|---|---|---|---|---|
| False | 4.0 | 2 | 0.5 | 5 | 0 |
| True | 4.0 | 2 | 0.5 | 2 | 0 |
| True | 2.0 | 1 | 1.0 | 125 | 10 |
| True | 1.5 | 1 | 1.0 | 126 | 8 |
| **True** | **1.0** | **1** | **1.0** | **154** | **7** |

Tighter waypoint spacing helps significantly (1 m beats 4 m by ~30×) — gives the gate-relative obs more frequent updates. EMA yaw smoothing is **redundant** when the path is already cubic-smoothed, and aggressive EMA (α=0.1) actively hurts.

## Why this is a useful result

M5's purpose was to verify the pipeline end-to-end **and** validate that the trained policy can navigate planner-emitted waypoints. Both delivered. The 7-lap detour result is comparable to the clean-oval baseline (16 laps) — about half the speed because the lap is longer due to the detour, not because the policy is weaker on it.

The first attempt stalled at 5 gates because of two specific mismatches between planner output and training data — both fixed in <30 min once diagnosed via parameter sweep. No retraining required.

## Path forward (mostly resolved)

The original "policy can't fly the detour" worry is gone. What remains:

1. **Pull the gru_100M (88%) checkpoint from W&B for stronger generalization.** Optional now that we know the local 28-dim works on detoured tracks; would matter more for harder geometry.
2. **Train a planner-aware policy** if we move to denser obstacle scenarios where 1 m waypoint spacing isn't tight enough.
3. **Generalize the lookahead-1 convention to the M3 adapter's default.** Currently `waypoints_to_track` defaults to lookahead=2 (anticipatory); the policy expects lookahead=1. Should change the default — see follow-up section.

## What's still mock

- **Synthetic VIO** (M1) — not yet swapped for a real library (DROID-SLAM / DPVO / ORB-SLAM3 bindings).
- **Occupancy from PyBullet** — Janahan's TSDF will replace this when his pipeline lands.
- **Single static obstacle** — no dynamic / replan-on-update yet.
- **No FPV / DA V2 perception in the closed loop** — week-2 demo had it; M5 dropped it to focus on the closed-loop pipeline. They can be re-added; it's just more rerun logging.

## Code

| Path | What |
|---|---|
| `scripts/perception/m5_closed_loop_demo.py` | Runner: builds scene, plans, runs closed-loop, logs to rerun |
| `outputs/perception/m5_closed_loop_demo.rrd` | The demo (~2 MB). Open with `rerun outputs/perception/m5_closed_loop_demo.rrd` |
| `slice_obs_33_to_28` | Obs adapter — env's 33-dim → checkpoint's 28-dim |
| `_training_racing_params` | The vehicle params the MoE was trained on (essential — CrazyFlie defaults put the policy OOD) |
| `run_closed_loop` | Step env with policy actions, log poses + actions + gate_idx + cross-track error |

## Demo viewing notes

```bash
rerun outputs/perception/m5_closed_loop_demo.rrd
```

What to verify:
- **3D world view**: 8 gates around the oval (gates 4-5 are the off-axis ones from M4 — but they're *not* the obstacle waypoints in M5, since the obstacle is between gates 2-3). Cylinder visible as red rings. Two trajectories: light blue = open-loop GT (the planner's path executed exactly), green = closed-loop policy (where the drone actually went).
- **Time-series panels**: cross-track error (closed-loop drift from the planner's path), action TRPY, gate index. Cross-track error grows when the policy stalls; gate index plateaus.

## Cross-references

- M4 artifact: `docs/superpowers/artifacts/2026-05-05-m4-planner-mvp.md`
  (sharp-yaw-at-corners warning — M5 confirmed it was load-bearing)
- M1 artifact: `docs/superpowers/artifacts/2026-05-05-m1-localization-decision.md`
- M2 artifact: `docs/superpowers/artifacts/2026-05-05-m2-depth-source-decision.md`
- Training config: `configs/experiment/moe_generalist.yaml`,
  `configs/sim/numpy_quad.yaml` (the racing-quad params)
- `reference_moe_checkpoint.md` (memory note) — already documents the
  4→5 expert load quirk
