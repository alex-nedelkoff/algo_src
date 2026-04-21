# Warehouse TSDF → PyBullet Phase 1 — 2026-04-21

Session log for the warehouse-pybullet implementation work. Captures context for picking this up tomorrow or after the TSDF artifact lands.

## What this is

Phase 1 of integrating the COR-92 Cosys-AirSim warehouse into a PyBullet training environment. Converts a TSDF voxel grid + the 5 gate poses into URDF assets that PyBullet can load as static collision bodies, plus two validators (silhouette IoU vs AirSim renders, and a fly-through pytest).

Phases 2–4 (occupancy grid + A* planner, gym-pybullet-drones training loop, track config loader) are deferred to later specs.

## Where it lives

- **Branch:** `warehouse-tsdf-pybullet-mvp`
- **Worktree:** `.claude/worktrees/warehouse-tsdf-pybullet-mvp/` (parallel to main repo, isolates the dirty `autoresearch-mvp` working tree)
- **Spec:** `docs/superpowers/specs/2026-04-21-warehouse-tsdf-to-pybullet-design.md`
- **Plan:** `docs/superpowers/plans/2026-04-21-warehouse-tsdf-to-pybullet.md` (18 tasks, T1–T17 done, T18 gated)

## Status

**33/33 tests passing.** 18 commits on the branch. T1–T17 implemented end-to-end on synthetic fixtures.

## Done (T1–T17)

| Component | Path | Notes |
|-----------|------|-------|
| NED↔ENU helpers | `sim/pybullet/coords.py` | `(x,y,z)→(y,x,-z)` for positions; `(w,x,y,z)→(w,y,x,-z)` for quaternions |
| TSDF loader | `scripts/warehouse_to_urdf/tsdf.py` | Adaptive: `.npz` full, `.ply` partial (returns mesh-only artifact), `.bin` stub |
| Mesh ops | `scripts/warehouse_to_urdf/mesh.py` | Gate clipping, marching cubes (skimage), quadric simplification (Open3D), procedural torus, NED→ENU vertex flip |
| URDF writers | `scripts/warehouse_to_urdf/urdf.py` | warehouse.urdf (single static link), gate.urdf (orange ring), gates_enu.json |
| Manifest | `scripts/warehouse_to_urdf/manifest.py` | sha256 hashes + build params + git SHA |
| Build CLI | `scripts/warehouse_to_urdf/__main__.py` | `python -m scripts.warehouse_to_urdf --tsdf X --gates Y --out Z` |
| Runtime loader | `sim/pybullet/warehouse_loader.py` | `WarehouseScene.load_into(client_id) → WarehouseHandles` |
| Fly-through tests | `tests/test_pybullet/test_warehouse_collision.py` | 3 scenarios: clean pass, wall hit, gate-rim clip |
| Trajectories | `tests/fixtures/warehouse_trajectories.yaml` | Hand-authored waypoints |
| AirSim capture | `scripts/capture_airsim_fixtures.py` | Linux-only, ship-and-document |
| Validation harness | `scripts/compare_pybullet_vs_fixtures.py` | Silhouette IoU + HTML report; `python -m scripts.compare_pybullet_vs_fixtures` |

## Gated (T18)

Awaiting:

1. **TSDF artifact** from Janahan — expected evening of 2026-04-21. Format unknown until it lands; loader is adaptive but may need an extra adapter.
2. **AirSim fixtures** — captured one-time on a Linux box with the COR-92 binary. Falls back to `B-fixture` for v1; later may switch to live capture if Janahan provides UE source or a Windows build of the warehouse binary.

When both land, T18 is:
- `python -m scripts.warehouse_to_urdf --tsdf <path> --gates configs/warehouse/warehouse_5gates_v1_gates_ned.json --out sim/assets/warehouse_v1/`
- Author the `warehouse_v1` block in `tests/fixtures/warehouse_trajectories.yaml` with real waypoints
- Run `pytest tests/` — all green
- `python -m scripts.compare_pybullet_vs_fixtures` — calibrate IoU thresholds, fix any quaternion-convention drift in `convert_airsim_camera_orientation_to_pybullet`
- Attach HTML report to Linear sub-issue under COR-92

## Bugs caught & fixed during execution

1. **open3d ↔ Python 3.13 incompatibility** — system Python 3.13 has no open3d wheel. Fix: all Python invocations use `conda run -n monorace` (Python 3.11). Worth pinning `requires-python` if/when others contribute.
2. **`pybullet` declared but not installed** — pyproject.toml had it, conda env didn't. Fix: `conda install -c conda-forge pybullet=3.25` (PyPI source build needs MSVC 14.0).
3. **Three geometry bugs in synthetic test fixtures** (commit `d5b0b39`) — none in pipeline code, all in test fixture authoring:
   - SDF origin `(0,0,0)` put walls underground after NED→ENU flip; fixed to `(0,0,-2)`
   - Synthetic gate orientation was identity (ring lying flat); fixed to `(0.7071, 0.7071, 0, 0)` (NED-wxyz) — confirmed real Cosys-AirSim convention by inspecting the actual `warehouse_5gates_v1_gates_ned.json`, gates use 90° Y-rotations to stand up
   - PyBullet `getContactPoints` requires `mass > 0` (returns empty for kinematic bodies); fixed drone mass `0.0 → 1.0` in `_spawn_drone` helper

## Key conventions baked in

- **World frame:** ENU (Z-up). Single NED→ENU flip applied at build time to mesh + gate poses; runtime is ENU throughout.
- **Gate ring default orientation:** axis along Z (horizontal ring lying flat). Gate JSON's `orientation_wxyz` rotates each gate to its actual pose. Matches Cosys-AirSim convention.
- **Quaternion order:** `(w, x, y, z)` in JSON files; `(x, y, z, w)` for PyBullet API. Loader converts.
- **Asset versioning:** `sim/assets/warehouse_v1/manifest.yaml` records source TSDF/gates hashes + git SHA + build params. Future warehouse versions get `_v2`, etc.

## Next session — pick up here

1. Read this doc + check the latest commit on `warehouse-tsdf-pybullet-mvp`
2. If TSDF artifact has arrived: execute T18 from the plan
3. If not: nothing to do until artifact lands

---

*Generated 2026-04-21 during a Subagent-Driven Development session — 18 implementer dispatches, ~16 review dispatches, ~5 hours of wall-clock execution.*
