# M4 — Vision-Aware Planner MVP

**Date:** 2026-05-05
**Linear:** COR-106 (extended scope)
**Plan:** `docs/superpowers/plans/2026-05-05-m4-planner-mvp-plan.md`
**Demo:** `outputs/perception/week2_planner_demo.rrd`

## Status

**Working end-to-end against the warehouse map.** A* planner detours
around a synthetic 1 m cylinder obstacle injected between gates 4 and
5; drone follows the planned path; the existing M1/M2/M3 overlays
(VIO drift shadows, DA V2 depth, synthetic gates) come along for free.

## What's in the demo

- **2.5D occupancy grid** built by raycasting the warehouse mesh at
  drone altitude (per-cell vertical rays through a 2 m slab). 0.25 m
  cell size; ~5400 cells over the warehouse footprint.
- **Synthetic obstacle** — 1 m radius cylinder dropped at the midpoint
  between off-axis gates 4 and 5.
- **Drone-radius dilation** — Minkowski-sum the obstacle set with a
  0.25 m disk so the planner can treat the drone as a point.
- **A\*** with octile heuristic, 8-connected grid, line-of-sight
  smoothing. Goal-snap if the goal cell falls inside the dilated
  occupancy.
- **Per-segment planning** — A\* from each gate to the next. Per-segment
  resampling to ~4 m spacing via the M3 utility.
- **Constant-speed parameterization** — concatenated path walked at
  3 m/s; yaw aligns with velocity tangent. Produces 995-step pose
  trajectory (32.8 s of flight at 30 Hz).

## Code layout

| Path | What |
|---|---|
| `perception/planning/occupancy_grid.py` | `OccupancyGrid2D5` — raycast build, dilation, cylinder injection, world↔grid coords, image export |
| `perception/planning/cost_map_planner.py` | `astar_grid`, `line_of_sight_free`, `smooth_path_los`, `plan` |
| `scripts/perception/week2_planner_demo.py` | Demo runner extending `week1_demo.py` |
| `tests/test_perception/test_occupancy_grid.py` | 12 tests on coords, dilation, cylinder injection |
| `tests/test_perception/test_cost_map_planner.py` | 12 tests on A\*, LoS smoothing, end-to-end planning |

24/24 tests passing.

## Demo viewing notes

```bash
rerun outputs/perception/week2_planner_demo.rrd
```

What to verify:

- **3D world view**:
  - Grey patches = warehouse occupancy at altitude.
  - Red ring between gates 4-5 = the cylinder obstacle (1 m radius).
  - Green line = the A\*-planned path. **Verify it bends around the red ring** (compare to the straight-line route a naive policy would take).
  - Drone follows the green path; VIO shadows still drift behind as before.
- **FPV + depth panels**: same as week 1 — DA V2 Small @ 384×512 every 3rd frame.
- **Drift time series**: same regimes; `with_map` stays under 5 cm, `no_map` ~20 cm by lap end, `dead_reckoning` ~6 m.

Final-step VIO drift on this slightly-longer-than-30 s trajectory:

| Profile | Final drift |
|---|---|
| ORB-SLAM3 + map matching | 0.016 m |
| ORB-SLAM3 mono-inertial | 0.194 m |
| IMU dead reckoning | 6.451 m |

## What's still mock vs real

- **Occupancy** uses PyBullet raycast on the warehouse mesh.
  Janahan's exploration phase produces a TSDF; this is a stand-in
  until we plug into his pipeline.
- **VIO** is still the synthetic noise model from M1; real VIO swap-in
  (DROID-SLAM / DPVO / ORB-SLAM3 bindings) is a follow-up.
- **Drone-following** is open-loop: GT pose follows the planner's
  output exactly. Closed-loop policy testing — actually running the
  G&CNet on the planner's waypoints — is M5.
- **Re-planning** is one-shot at the start; no replan-on-event or
  fixed-rate replan loop. Sufficient for a static scene; not for
  dynamic obstacles or VIO drift updating.
- **Single altitude** — the planner is purely 2.5D. Vertical
  obstacles only. Doesn't handle altitude-dependent constraints
  (e.g., low ceiling).

## Limitations the user should be aware of

- The line-of-sight smoother can push waypoints *very* close to obstacle
  edges. The 0.25 m drone-radius dilation gives some headroom but it's
  worth tuning per scene. Bumping dilation to 0.5 m for safety in the
  v1 deployment is a one-line config change.
- A\* on a 0.25 m grid produces ~12-cm-resolution paths. Fine for the
  current 1 m obstacle; would need finer grid for tight gaps (corridors
  < 1 m wide).
- Yaw at corners is sharp (atan2 of velocity, no smoothing). The G&CNet
  was trained on tracks where each gate's yaw was set in advance, so
  abrupt yaw changes between segments may surprise it. M5 closed-loop
  test will surface this; mitigation is to apply the M3 EMA yaw
  smoother across the concatenated path before parameterizing.

## What this unblocks

With M4 working, M5 (closed-loop test) becomes:

1. Replace the open-loop "drone follows GT" with "drone follows
   policy actions, policy reads gate-relative obs from the planner's
   waypoint stream".
2. Verify the policy actually navigates the detour without stalling
   or oscillating.
3. Quantify: lap time vs the obstacle-free baseline.

That's a focused 2-3 day effort assuming the policy doesn't choke on
the sharp-yaw-at-corners issue noted above.

## Cross-references

- M1 decision: `docs/superpowers/artifacts/2026-05-05-m1-localization-decision.md`
  (the architecture call: vision-as-augmentation, conditional on map matching)
- M2 decision: `docs/superpowers/artifacts/2026-05-05-m2-depth-source-decision.md`
- M3 module: `sim/tracks/waypoint.py` (planner output goes here)
- Week 1 demo: `outputs/perception/week1_demo.rrd`
- Week 2 demo (this): `outputs/perception/week2_planner_demo.rrd`
