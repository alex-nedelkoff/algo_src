# M4 — Vision-aware planner MVP plan

**Date:** 2026-05-05
**Spec:** `docs/superpowers/specs/2026-05-05-vision-augmented-racing-design.md`
**Linear:** COR-106 (extended scope; original was week-1-only M1+M2+M3)
**Predecessors:** M1 (`docs/superpowers/artifacts/2026-05-05-m1-localization-decision.md`), M3 (waypoint adapter)

## Goal

Demonstrate a vision-aware planner that:
1. Builds an occupancy grid from the warehouse map + injected synthetic obstacle.
2. Plans a path from current drone pose to next gate, avoiding obstacles.
3. Emits waypoints into the M3 `WaypointTrack` adapter.
4. Is visible in the existing rerun demo as a detour around the obstacle.

This is M4 v0 — minimum viable. Closed-loop policy testing (drone actually following the planner's waypoints under G&CNet control) is M5 and deferred from this plan.

## Architecture

```
INPUTS:
    drone pose (from VIO + map matching)
    next gate position (from exploration, privileged)
    warehouse occupancy (from Janahan's TSDF; here: PyBullet raycast on warehouse mesh)
    obstacles (synthetic cylinder for the demo)

PLANNER (5 Hz):
    1. Sample 2.5D occupancy at drone altitude
    2. Dilate by drone radius (safety margin)
    3. A* from drone pose → next gate, on the dilated grid
    4. Smooth + downsample path → ~4 m waypoint spacing (matches M3 resampler)
    5. Emit next 1-3 waypoints → WaypointTrack

POLICY (100 Hz, unchanged from week 1):
    G&CNet consumes WaypointTrack as if they were gates.
```

## Tasks

### M4.1 — 2.5D occupancy grid (~3 hours)

`perception/planning/occupancy_grid.py`:

- `OccupancyGrid2D5(grid_size_m, cell_size_m, altitude_m)` — fixed-altitude binary grid in xy.
- `from_pybullet_scene(client_id, body_ids, altitude, cell_size)` — sample by raycasting downward at each cell to detect obstacles at altitude.
- `dilate(radius_m)` — Minkowski-sum the obstacles with a disk of `drone_radius`, returns a new grid.
- `add_cylinder(center_xy, radius_m)` — inject a cylindrical obstacle without re-raycasting.
- `world_to_grid(xy)` / `grid_to_world(ij)` coordinate conversions.

Tests: known-shape inputs (single cylinder, two cylinders) → known cell counts.

### M4.2 — A* planner (~3 hours)

`perception/planning/cost_map_planner.py`:

- `plan(start_xy, goal_xy, occupancy)` — A* with 8-connected grid, Euclidean heuristic.
- Returns waypoint sequence in world coordinates (not grid coordinates).
- Smoothing: greedy line-of-sight reduction (skip a waypoint if start→next is collision-free).
- Resampling: re-use M3's `resample_waypoints` to land on ~4 m spacing.

Tests: open grid → straight line, blocked grid → goes around, no-path → returns None.

### M4.3 — Demo extension (~3 hours)

Extend `scripts/perception/week1_demo.py` (or new `week2_planner_demo.py`):

- Inject a 1 m radius cylinder at the midpoint between gates 4 and 5 (the off-axis ones).
- Replace the parametric oval with a piecewise trajectory that follows planner output:
  - At each gate, run A* to next gate, take the path
  - Concatenate paths into a full lap
- Visualize:
  - Top-down occupancy heatmap (3D plane in rerun, colour by occupancy)
  - Obstacle as a transparent cylinder
  - Planner path as a green line
  - Drone follows the planner path
- Keep the existing M2 (DA V2 depth) + M1 (VIO shadows) + M3 (synthetic gates) overlays.

### M4.4 — Validation + report (~1 hour)

- Run the demo end-to-end, verify drone takes the detour.
- Brief write-up in `docs/superpowers/artifacts/`: what works, what's still mock (occupancy from raycast not TSDF, no real VIO yet, no closed-loop policy yet).

## Out of scope (defer to M5 or follow-ups)

- Closed-loop G&CNet on planner waypoints (M5)
- Replace synthetic VIO with real VIO library (DROID-SLAM / DPVO)
- Replace raycast occupancy with Janahan's actual TSDF
- Multi-altitude / 3D planning (current is 2.5D — planar at drone altitude)
- Dynamic obstacles (e.g., moving objects between gates)
- Re-planning under perception updates (replan-on-event vs planner running at fixed rate)
- Cost beyond occupancy (e.g., minimize energy, smoothness penalties)

## Estimated total: ~10 hours

| Task | Effort |
|---|---|
| M4.1 occupancy grid | 3 h |
| M4.2 A* planner | 3 h |
| M4.3 demo integration | 3 h |
| M4.4 validation | 1 h |

## Cross-references

- Spec: `docs/superpowers/specs/2026-05-05-vision-augmented-racing-design.md`
- M1 decision: `docs/superpowers/artifacts/2026-05-05-m1-localization-decision.md`
  (the architecture call: vision-as-augmentation, conditional on map matching)
- M3 module: `sim/tracks/waypoint.py` (planner output goes here)
- Week 1 demo: `scripts/perception/week1_demo.py` (M4 demo extends this)
