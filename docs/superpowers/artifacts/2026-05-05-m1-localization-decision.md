# M1 — Localization Architecture Decision

**Date:** 2026-05-05
**Linear:** COR-106 (week 1, M1)
**Spec:** `docs/superpowers/specs/2026-05-05-vision-augmented-racing-design.md`

## Decision

**Vision is augmentation, conditional on a working map-matcher.**

Synthetic-VIO benchmarks calibrated to literature norms show that a
mono-inertial VIO **with periodic map matching against the TSDF**
keeps final-step position drift below ~5 cm over a 30-second race —
well under the 10 cm threshold the spec set for "augmentation works."

Without map matching, the same VIO drifts to ~13 cm by 30 s —
borderline, would force vision into primary navigation. Without VIO
entirely (IMU dead reckoning), drift hits ~4 m — catastrophic.

**Therefore:** the map-matcher is the load-bearing component, not the
VIO itself. Standard mono-inertial VIO accuracy is enough; what
matters is anchoring it against the static map at 1 Hz with ~60 %
correction strength. **This frames the M4 work as "build the map
matcher" first, "build the planner" second.**

## Context

The spec set a binary architecture call:

- **Drift < 10 cm over 30 s → augmentation:** policy keeps using the
  gate-relative obs, vision contributes obstacle detection / minor
  pose corrections through the planner. The trained G&CNet stays in
  the loop unmodified.
- **Drift > 10 cm → primary:** the gate-relative obs frame would
  drift with VIO, breaking the policy. Vision must take over
  navigation, requiring substantial re-architecting (per-frame
  re-localization through landmarks, etc.).

The decision drives M4's planner architecture — a "detour planner"
operating in a trustworthy world frame is a much smaller research
problem than a "primary navigator" that has to handle both planning
and localization.

## Setup

**Why synthetic VIO instead of a real implementation:** every Windows-
friendly VIO library (DPVO, DROID-SLAM, ORB-SLAM3 bindings,
OpenVINS) requires a multi-day Windows build with CUDA/CMake
gymnastics. The architecture decision needs realistic order-of-
magnitude drift numbers, not a measurement of any specific
implementation. A noise model calibrated to literature norms gives
us that without burning a week of integration work.

**Code:** `perception/localization/synthetic_vio.py` (calibrated
noise model) + `scripts/perception/benchmark_localization_drift.py`
(runner, plots). Outputs to `outputs/perception/localization_drift/`.

**Trajectory:** parametric oval, 30 s, 5 m/s target, ±0.5 m
altitude oscillation. 3000 steps at dt=0.01 s. Yaw aligned with
velocity tangent.

**Profiles:**
| Profile | What it represents |
|---|---|
| `orb_slam3_mono_inertial` | Standard VIO, no map matching. Drifts as σ·√t. |
| `orb_slam3_with_map_matching` | Same VIO, ICP correction every 1 s pulling 60 % of position error / 40 % of yaw error. |
| `imu_dead_reckoning` | IMU integration only, no VO at all. Pessimistic baseline. |

**Calibration sources:**
- Campos et al. 2021, ORB-SLAM3 — EuRoC ATE 0.04–0.10 m over ~30 s
  in mono-inertial mode.
- Geneva et al. 2020, OpenVINS — similar 0.05–0.15 m drift bounds.
- Forster et al. 2017, On-Manifold Preintegration — ~1 % per-second
  position drift before any sensor fusion.

**Statistics:** 30 random seeds per profile. Mean / p99 / max
across seeds, mean ± 1σ band per timestep.

## Results (n_seeds = 30, duration = 30 s, dt = 0.01 s)

| Profile | mean pos drift | final-avg | final-p99 | mean yaw drift | final yaw avg |
|---|---|---|---|---|---|
| ORB-SLAM3 mono-inertial (no map) | 0.083 m | **0.129 m** | 0.234 m | 0.32° | 0.50° |
| **ORB-SLAM3 + map matching** | **0.021 m** | **0.029 m** | **0.050 m** | 0.09° | 0.12° |
| IMU dead reckoning | 2.763 m | 4.309 m | 7.902 m | 8.0° | 12.4° |

Plots: `outputs/perception/localization_drift/drift_curves.png`,
`histograms.png`. Per-seed raw data in `raw/<profile>/seed_<n>.npz`
for reproducibility.

## Interpretation

**Map matching is the deciding factor.** The same underlying VIO
with periodic 1 Hz ICP correction shrinks final drift by 4–5×
(from 13 cm to 3 cm). The cost: an ICP solve every second, ~50 ms
on a depth scan against the TSDF. Cheap.

**The 10 cm threshold sits squarely between "with map matching"
and "without."** The spec's threshold is well-calibrated — small
enough that without map matching we'd be in trouble, comfortably
above what map matching delivers.

**Yaw drift is essentially negligible** with map matching (0.12°
final, 0.54° max) and barely concerning even without (0.50° avg,
1.60° max). Yaw is observable from gravity (accelerometer roll/pitch)
and from feature persistence — the random walk on yaw is small in
realistic regimes.

**Dead reckoning would require a fundamentally different policy.**
4 m drift over 30 s is more than typical inter-gate spacing — the
gate-relative obs would point at completely the wrong gate after a
single lap. Worth keeping in mind as a "vision goes offline"
contingency: the policy would need a graceful-degradation path
(e.g., aggressive deceleration + re-localize via gate detection).

## What this means for M4

The M4 (vision-aware planner) plan should be split:

1. **Build the map matcher first.** ICP between current depth scan
   and the static TSDF, running at 1 Hz, fusing VIO pose with the
   match result. This is the load-bearing component for the
   architecture call to hold.
2. **Then build the planner.** Cost-map A* / MPC over the local
   TSDF + depth-derived occupancy, outputting waypoints into the
   M3 `WaypointTrack` adapter.

Step 1 is the only thing that needs to be cheap and robust at race
time; the planner can run at 5–10 Hz and is comparatively forgiving.

## Limitations / what the synthetic numbers don't capture

These are honest gaps the synthetic model can't represent:

- **Feature loss / kidnap recovery.** Real VIO sometimes loses tracking
  entirely (fast motion, motion blur, repetitive textures). Recovery
  takes 100–500 ms during which drift is unbounded. Synthetic VIO has
  no such failure mode. Mitigation: rely on the map-matcher to recover
  pose after a tracking gap.
- **ICP failure modes.** Real ICP can latch onto a wrong local minimum
  (geometric ambiguity in repetitive environments — long corridors,
  symmetric warehouses). The synthetic 60 % strength assumes ICP works.
  Mitigation: gate ICP application by an inlier count threshold; reject
  matches with too few correspondences.
- **Coupling between drift and planner output.** Synthetic VIO doesn't
  affect the trajectory — the drone flies a perfect oval regardless. In
  reality, drift would be fed into the planner's obs and affect waypoint
  selection, potentially compounding error. M5 closed-loop test will
  surface this.

## Architecture call going forward

```
Confirmed week 1 architecture (per spec):
  STATIC: gate poses + TSDF (from exploration phase)
  LIVE:   VIO pose + RGB → depth (DA V2 Small @ 384×512, per M2)
  MATCH:  ICP at 1 Hz against TSDF (NEW load-bearing component for M4)
  PLANNER: cost-map waypoint emission, 5–10 Hz
  POLICY: G&CNet, 100 Hz, gate-relative obs unchanged
```

## Open follow-ups

- **Real VIO swap-in.** Current results are from a noise model. Before
  closed-loop deployment in M5, we should integrate a real VIO library
  and re-run the benchmark. Candidates ranked by expected pain:
    1. **DROID-SLAM** — Python-native, GPU-accelerated, modern. Likely
       cleanest Windows install; needs CUDA build.
    2. **DPVO** — same lineage, lighter, faster. No IMU though.
    3. **ORB-SLAM3 with python bindings** — gold standard for
       accuracy; known to be Windows-friendly with `ORB_SLAM3-python`
       bindings if available.
- **Map matcher implementation.** ICP wrapper around `open3d`'s
  registration API, against the TSDF/occupancy from Janahan's
  exploration. Should land before M4 planner work begins.
- **Sim instrumentation for live drift logging.** Once a real VIO is
  integrated, replace the synthetic noise model with real measurements
  from AirSim trajectories. Code path is the same — same drift script,
  just different `est_poses` source.
- **Failure-mode modeling.** Add "tracking loss" events to the
  synthetic profile (random 100–500 ms windows where VIO produces no
  update) to stress-test the planner's robustness before real VIO is
  integrated.

## Cross-references

- `perception/localization/synthetic_vio.py` — calibrated noise model
- `scripts/perception/benchmark_localization_drift.py` — runner
- `outputs/perception/localization_drift/{summary.json,drift_curves.png,histograms.png}`
- COR-106 — week 1 issue this M1 belongs to
- COR-91 — Janahan's exploration / TSDF (provides the static map this
  decision relies on)
- M2 decision: `docs/superpowers/artifacts/2026-05-05-m2-depth-source-decision.md`
