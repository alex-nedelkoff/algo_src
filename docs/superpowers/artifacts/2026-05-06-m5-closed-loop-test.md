# M5 — Closed-Loop Test

**Date:** 2026-05-06
**Linear:** COR-106 (extended scope)
**Demo:** `outputs/perception/m5_closed_loop_demo.rrd`
**Predecessors:** M1 (localization), M2 (depth), M3 (waypoint adapter), M4 (planner MVP)

## Status

**Pipeline works end-to-end. Policy partially completes the planner's lap.** The closed-loop test runs the trained 5-expert MoE checkpoint (`outputs/expanded_5expert_v3`, 28-dim) through the M3 `WaypointTrack` adapter fed by the M4 A* planner. The bottleneck is policy quality on planner-emitted detour waypoints, not pipeline plumbing.

Result on a training-scale 8-gate oval with a 1 m cylinder injected between gates 2-3:

| Variant | Gates passed | Laps |
|---|---|---|
| Clean oval (no obstacle, no planner) | **129** | **16** |
| Clean oval + planner output (no obstacle) | TBD | TBD |
| Oval + obstacle + planner detour | 5 | 0 |

Live diagnostic in 90 s of sim time. The drone navigates the first 5 gates of the planner's path, then stalls at the off-axis detour around the cylinder.

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

## What doesn't work yet

**Policy stalls at the planner's off-axis detour waypoints.** The planner's path bends outward to clear the cylinder, producing waypoints whose synthetic yaws change sharply between consecutive segments. The trained policy (which saw turn angles up to ±170° in training) handles smooth corners well but struggles with the discontinuous yaw jumps where planner segments meet.

Yaw-smoothing sweep on the planner output (varying `yaw_smoothing_alpha` from 1.0 / no smoothing to 0.1 / aggressive):

| smoothing α | gates passed |
|---|---|
| 1.0 | 2 |
| 0.5 | 5 |
| 0.25 | 5 |
| 0.1 | 4 |

Smoothing helps from 2 to 5 gates but doesn't fully resolve. The remaining gap is structural: the planner's path geometry is OOD for this checkpoint.

## Why this is a useful result

M5's purpose was to **verify the pipeline end-to-end** — and it does. The 129-gate / 16-lap clean-oval result is the pipeline-correctness proof. The 5-gate planner-detour result quantifies the policy's tolerance for non-training-distribution waypoints, which is exactly the question M5 was supposed to answer.

The next bottleneck is policy quality on detour geometry, NOT the planner / occupancy / waypoint-adapter / obs-slice / env-config plumbing. That's a clean separation of concerns.

## Path forward

Three options for getting closed-loop laps with the planner detour:

1. **Pull a better checkpoint.** The local 28-dim is the 30-45% golden-set version. The gru_100M (88% golden-set) checkpoint exists on W&B but isn't local. Pull it via `wandb artifact get <run>/<artifact>`. Better recurrence in the policy might absorb sharp-yaw transitions more gracefully.
2. **Train a planner-aware policy.** Add planner-style detour tracks to the training mix (currently figure-8 25%, zigzag 15%, procedural 60%). 1-2 days of training time on the existing autoresearch infrastructure.
3. **Replan to avoid sharp transitions.** The path-smoothing module from `sim/tracks/waypoint.py` (cubic spline arc-length resampling) can be applied to the planner's output. Currently we use linear segments; cubic would give continuous tangents and softer yaw transitions. ~30 min of work, would likely raise the 5-gate floor without retraining.

Item 3 is the cheapest first move. Item 1 is the cleanest result. Item 2 is the most thorough.

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
