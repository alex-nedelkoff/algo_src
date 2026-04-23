# Warehouse → PyBullet MVP: Mesh-Direct (Phase 1, adapted)

**Date:** 2026-04-23
**Linear:** COR-95
**Predecessor plan:** `docs/superpowers/plans/2026-04-21-warehouse-tsdf-to-pybullet.md`
**Spec:** `docs/superpowers/specs/2026-04-21-warehouse-tsdf-to-pybullet-design.md` (unchanged — same outputs, same validation)

---

## ⚠️ Scope correction (discovered 2026-04-23 during execution)

The predecessor plan was **substantially already implemented** on this branch by earlier work (2026-04-21). Baseline state at session start (commit `38fd378`):

- `scripts/warehouse_to_urdf/` — `tsdf.py`, `mesh.py`, `urdf.py`, `manifest.py`, `__main__.py` all present and working
- `sim/pybullet/warehouse_loader.py` — present and working
- `tests/test_warehouse_to_urdf/` + `tests/test_pybullet/` — **33/33 tests green**

**Net effect:** only 5 tasks actually remain in this session:

| Session task | Status |
|---|---|
| Task 1: Verify deps | ✅ no-op, PyYAML already transitive |
| Task 2: UE↔NED coord helpers | ✅ done in commit `fa2ec55` |
| Tasks 3, 6, 7, 8, 9, 11, 12 | ✅ pre-existing, covered by 33/33 baseline tests |
| **Task 4: FBX/OBJ adapter in `tsdf.py`** | ⬜ TODO (small extension to existing `load_tsdf` dispatch) |
| **Task 14 (new): UE→NED mesh transform + spatial bbox clip in `mesh.py`** | ⬜ TODO |
| **Task 5: UE-YAML → NED-JSON gate converter** | ⬜ TODO (new file, small) |
| **Task 10: Extend `__main__.py` for mesh-direct flags** | ⬜ TODO (existing CLI, add `--mesh` / `--gates-ue-yaml` branch) |
| **Task 13: End-to-end run on real FAB inputs + COR-95 update** | ⬜ TODO |

The detailed task sections below still describe the full work as if starting from scratch, but in practice Tasks 3, 6, 7, 8, 9, 11, 12 are already built — TaskList has them marked completed with notes.

---

**Goal:** End-to-end pipeline that turns our exported FAB warehouse FBX + UE-frame gate YAML into a PyBullet scene with floor/walls as collision geometry and 5 gates placed from data, validated by a hand-authored fly-through pytest.

**Scope change vs predecessor:** input is a triangle mesh (`.fbx` or `.obj` exported from UE) + a YAML of UE-frame gate transforms, not a TSDF voxel grid + NED gate JSON. The TSDF code path in `tsdf.py` stays intact for future use when Janahan's COR-91 output lands or when we reconstruct from real competition captures.

## Delta from predecessor

| # | Original task | This plan |
|---|---|---|
| 1 | `open3d` dep + download COR-92 gate JSON | **Modified.** `open3d` already in pyproject.toml (`sim` extras). Verify install + add PyYAML. Skip COR-92 JSON download — we're authoring our own gate JSON from the UE YAML. |
| 2 | NED↔ENU coord helpers | **Unchanged.** Reuse verbatim. |
| 3 | TSDFArtifact + `.npz` adapter | **Unchanged.** Reuse verbatim. TSDF support stays intact. |
| — | — | **NEW.** Mesh adapters (`.fbx`, `.obj`) plug into the existing dispatch; both return a `TSDFArtifact` with `pre_extracted_mesh` populated (same pattern as the `.ply` adapter). |
| — | — | **NEW.** UE YAML → NED gate JSON converter. Handles UE→NED basis change (UE left-handed cm Z-up → NED right-handed m Z-down, origin = PlayerStart). |
| 4 | Gate clipping in SDF | **Dropped.** No SDF to clip. Replaced with: |
| — | — | **NEW.** Spatial bbox clipping of the triangle mesh around `(gate_centroid, playerstart) + 5 m margin` to drop the 75 m extent → ~15 m playable region. |
| 5 | Marching cubes + simplification | **Modified.** Skip marching cubes (we have a mesh). Keep simplification: `simplify_quadric_decimation` to target 30 K triangles. |
| 6 | Procedural torus + NED→ENU vertex flip | **Unchanged.** |
| 7 | URDF + gates_enu.json writers | **Unchanged.** |
| 8 | Manifest writer | **Modified.** Hash FBX + UE YAML instead of TSDF + gate JSON. Record clip bbox and simplification target. |
| 9 | Build-pipeline CLI | **Modified.** New flags: `--mesh` (alternative to `--tsdf`), `--gates-ue-yaml` (alternative to `--gates`), `--clip-margin-m` (default 5.0). At most one of `--tsdf`/`--mesh` and one of `--gates`/`--gates-ue-yaml` required. |
| 10 | Runtime warehouse loader | **Unchanged.** |
| 11 | Fly-through trajectory fixtures | **Modified.** Authored against our 5 FAB gate positions (Gate_01 at ~2 m from PlayerStart, etc.), not COR-92 positions. |
| 12 | `clean_pass_through` test | **Unchanged** in structure. |
| 13 | `wall_collision` + `gate_rim_collision` tests | **Unchanged** in structure. |
| 14 | AirSim fixture capture (Linux-only) | **Dropped.** Defer until AirSim packaging is revisited. |
| 15 | Silhouette IoU function | **Dropped.** Without AirSim fixtures the automated IoU gate doesn't apply. |
| 16 | PyBullet render at fixture pose | **Modified.** Becomes visual-inspection: render PyBullet RGB + depth from 3-5 hand-picked poses, dump PNGs to `outputs/validation/warehouse_fab_v1/`. Manual inspection vs UE Editor screenshots for fidelity check. |
| 17 | HTML report | **Dropped.** Optional future nice-to-have. |
| 18 | End-to-end real run + Linear update | **Kept.** Updated to reference COR-95. |

**Net task count: 13** (vs 18 in the predecessor).

---

## Inputs (exist on disk — no downloads required)

- `data/warehouse_fab/warehouse.fbx` — 37 MB, 1.22 M triangles, UE world frame (cm). Loaded confirmed via Open3D.
- `configs/warehouse/warehouse_fab_v1_gates_ue.yaml` — 5 gate transforms + PlayerStart, UE world frame (cm, degrees RPY).

## Target outputs (same as predecessor spec)

- `sim/assets/warehouse_fab_v1/warehouse.obj` — simplified, ENU, meters
- `sim/assets/warehouse_fab_v1/warehouse.urdf` — single fixed-base link
- `sim/assets/warehouse_fab_v1/gate.urdf` — shared across all 5 gates
- `sim/assets/warehouse_fab_v1/gate_ring.obj` — procedural torus (major 0.80 m, minor 0.05 m)
- `sim/assets/warehouse_fab_v1/gates_enu.json` — 5 gate poses in ENU meters
- `sim/assets/warehouse_fab_v1/manifest.yaml` — hashes, build params, git SHA

---

## Task 1: Verify deps + add PyYAML

**Files:**
- Modify: `pyproject.toml`

- [ ] Verify `open3d` and `scikit-image` are installed: `python -c "import open3d, skimage; print(open3d.__version__, skimage.__version__)"`. They're in `sim` extras already.
- [ ] If `pyyaml` isn't already a transitive dep, add it to base `dependencies`. Check: `python -c "import yaml; print(yaml.__version__)"`. If ImportError, add `pyyaml` to the base `dependencies` list in `pyproject.toml` and `pip install -e .`.
- [ ] Commit only if pyproject.toml changed: `git commit -m "feat(warehouse): ensure pyyaml is available for UE gate YAML parsing"`.

## Task 2: NED↔ENU + UE↔NED coordinate helpers

**Files:**
- Create: `sim/pybullet/__init__.py` (empty)
- Create: `sim/pybullet/coords.py`
- Create: `tests/test_pybullet/__init__.py` (empty)
- Create: `tests/test_pybullet/test_coords.py`

Pattern: TDD — test first, then impl. **Reuse verbatim from predecessor Task 2** (lines 114-235 of the original plan) for `ned_to_enu_position` and `ned_to_enu_quaternion`.

**Additionally in `coords.py`:** add UE → NED helper pair. UE is left-handed cm Z-up; NED is right-handed m Z-down. Origin of NED = PlayerStart UE-world position. Define:

```python
def ue_to_ned_position(pos_ue_cm, playerstart_ue_cm):
    """UE world cm → NED meters, relative to PlayerStart."""
    rel = np.asarray(pos_ue_cm) - np.asarray(playerstart_ue_cm)
    return np.array([rel[0], -rel[1], -rel[2]], dtype=np.float64) / 100.0

def ue_to_ned_quaternion(rpy_deg, playerstart_rpy_deg):
    """UE RPY degrees → NED quaternion (w,x,y,z), accounting for L→R handedness."""
    # Implementation: build UE rotation R_ue from rpy, subtract playerstart rpy,
    # apply basis flip S = diag(1, -1, -1): R_ned = S @ R_ue @ S.T,
    # convert to quaternion wxyz.
```

Tests cover: identity transforms, pure yaw rotation, a gate pose matching a hand-computed NED value from the YAML.

- [ ] Tests fail
- [ ] Implement
- [ ] Tests pass
- [ ] Commit: `feat(pybullet): add NED↔ENU + UE↔NED coord helpers`

## Task 3: TSDFArtifact + `.npz` adapter

**Verbatim from predecessor Task 3** (lines 239-411).

- [ ] Tests fail → implement → tests pass → commit.

## Task 4: Mesh adapters (`.fbx`, `.obj`) in the existing loader

**Files:**
- Modify: `scripts/warehouse_to_urdf/tsdf.py`
- Modify: `tests/test_warehouse_to_urdf/test_tsdf.py`

Extend the dispatch in `load_tsdf`:

```python
if suffix in {".fbx", ".obj"}:
    return _load_mesh(path)
```

Where `_load_mesh` reads the file via `open3d.io.read_triangle_mesh`, returns a `TSDFArtifact` with `pre_extracted_mesh` populated and empty sdf (same pattern as `_load_ply`). If the mesh has zero triangles, raise `ValueError` with a clear message.

Tests: synthetic OBJ fixture (2 triangles, hand-written text) checks the loader returns a non-empty mesh artifact.

- [ ] Tests fail → implement → tests pass → commit: `feat(warehouse): add FBX/OBJ mesh adapters to TSDF loader`

## Task 5: UE YAML → NED gate JSON converter

**Files:**
- Create: `scripts/warehouse_to_urdf/gate_yaml.py`
- Create: `tests/test_warehouse_to_urdf/test_gate_yaml.py`

`convert_ue_yaml_to_ned_json(yaml_path, out_json_path)`:

1. Read YAML, extract `playerstart` and `gates` keys
2. For each gate, compute NED position + quaternion via the UE→NED helpers from Task 2
3. Write JSON matching the COR-92 schema:
   ```json
   {
     "Gate_01": {
       "position_ned": [x, y, z],
       "orientation_wxyz": [w, x, y, z],
       "inner_radius_m": 0.75,
       "outer_radius_m": 0.85
     },
     ...
   }
   ```
4. Radii are hardcoded from our SM_GateRing (0.75 / 0.85) — matches COR-92, and we authored the gates to those dimensions

Tests: round-trip our actual YAML → NED JSON, verify 5 gate keys present, verify PlayerStart becomes origin (gate positions relative to PlayerStart), verify orientations are unit quaternions.

- [ ] Tests fail → implement → tests pass → commit: `feat(warehouse): UE-YAML → NED gate JSON converter`

## Task 6: Mesh spatial clipping

**Files:**
- Create: `scripts/warehouse_to_urdf/mesh.py`
- Create: `tests/test_warehouse_to_urdf/test_mesh.py` (may already exist from Task 3)

`clip_mesh_to_bbox(mesh, bbox_min, bbox_max) -> o3d.geometry.TriangleMesh`:

Keep triangles whose all three vertices are inside the bbox. This is the conservative choice — triangles straddling the boundary get dropped. Given our gate bbox + 5 m margin, we won't be near any important geometry boundaries, so drop-on-straddle is fine.

`compute_default_clip_bbox(gate_positions_ned, margin_m=5.0) -> (bbox_min, bbox_max)`:

Input: N×3 array of gate positions in NED (meters). Returns axis-aligned bbox = gates bbox expanded by margin on each side.

Tests: synthetic 10-triangle mesh, bbox that should keep 3 / 10 triangles, assert correct triangle count.

- [ ] Tests fail → implement → tests pass → commit: `feat(warehouse): mesh spatial-clipping helpers`

## Task 7: Simplify + NED→ENU vertex flip

**Files:**
- Modify: `scripts/warehouse_to_urdf/mesh.py`
- Modify: `tests/test_warehouse_to_urdf/test_mesh.py`

Three functions:

- `simplify_mesh(mesh, target_triangles) -> mesh` — wraps `mesh.simplify_quadric_decimation`. If input has fewer than 2x target triangles, skip (no-op).
- `ned_to_enu_mesh(mesh) -> mesh` — applies `(x, y, z) → (y, x, -z)` to vertices, reverses triangle winding, recomputes normals. Our input mesh is in **UE world frame** (not NED directly), so this function will be called after `ue_to_ned_mesh`.
- `ue_to_ned_mesh(mesh, playerstart_ue_cm) -> mesh` — applies UE→NED basis change + origin shift + cm→m conversion.

Tests: unit cube in each space, verify axis ordering and winding consistency; verify simplify reduces face count monotonically.

Reuse the predecessor plan's procedural torus function (from its Task 6, lines 722-855) verbatim — move it into this `mesh.py` file.

- [ ] Tests fail → implement → tests pass → commit: `feat(warehouse): mesh simplify + coord-frame transforms + procedural torus`

## Task 8: URDF + gates_enu.json writers

**Files:**
- Create: `scripts/warehouse_to_urdf/urdf.py`
- Create: `tests/test_warehouse_to_urdf/test_urdf.py`

**Reuse verbatim from predecessor Task 7** (lines 857-1023). Single shared `gate.urdf`, single warehouse URDF linking `warehouse.obj` as both visual + collision, `gates_enu.json` with NED→ENU converted poses.

- [ ] Tests fail → implement → tests pass → commit.

## Task 9: Manifest writer

**Files:**
- Create: `scripts/warehouse_to_urdf/manifest.py`
- Create: `tests/test_warehouse_to_urdf/test_manifest.py`

**Reuse predecessor Task 8** (lines 1025-1176) with:
- `sources.mesh.path` + `sources.mesh.sha256` (FBX file hash) replaces the TSDF source block
- `sources.gates_ue_yaml.path` + `.sha256` replaces gates JSON hash
- `build_params.clip_bbox_ned` (list of 6 floats) added
- `build_params.target_triangles` unchanged

- [ ] Tests fail → implement → tests pass → commit.

## Task 10: Build-pipeline CLI (`__main__.py`)

**Files:**
- Create: `scripts/warehouse_to_urdf/__main__.py`

Start from predecessor Task 9 (lines 1178-1409) skeleton. Changes:

- Arg group `--tsdf | --mesh` (mutually exclusive, at least one required)
- Arg group `--gates | --gates-ue-yaml` (mutually exclusive, at least one required)
- New arg `--clip-margin-m` (default `5.0`) — margin around gate bbox for the spatial clip
- New arg `--playerstart-x-cm`, `--playerstart-y-cm`, `--playerstart-z-cm` OR read from the UE YAML's `playerstart:` block (preferred — read from YAML)
- Pipeline stages:
  1. Load artifact via `load_tsdf` (dispatches to mesh or TSDF path)
  2. If artifact has `pre_extracted_mesh`, skip marching cubes (same logic as current `.ply` path)
  3. Read gate YAML / JSON; if YAML path, run Task 5 converter → in-memory NED dict
  4. Compute spatial clip bbox from gate NED positions
  5. `ue_to_ned_mesh` (if input was FBX/OBJ in UE frame)
  6. `clip_mesh_to_bbox`
  7. `simplify_mesh`
  8. `ned_to_enu_mesh`
  9. Write outputs: warehouse.obj/urdf, gate.urdf, gate_ring.obj, gates_enu.json, manifest.yaml

- [ ] Invoke via: `python -m scripts.warehouse_to_urdf --mesh data/warehouse_fab/warehouse.fbx --gates-ue-yaml configs/warehouse/warehouse_fab_v1_gates_ue.yaml --out sim/assets/warehouse_fab_v1/ --target-triangles 30000 --clip-margin-m 5.0`
- [ ] Verify all 5 output files exist and have sane sizes (warehouse.obj < 5 MB after simplification)
- [ ] Commit: `feat(warehouse): mesh-direct build pipeline CLI`

## Task 11: Runtime loader (`WarehouseScene`)

**Files:**
- Create: `sim/pybullet/warehouse_loader.py`
- Create: `tests/test_pybullet/test_warehouse_loader.py`

**Reuse verbatim from predecessor Task 10** (lines 1411-1571). Loads warehouse + gates into a PyBullet DIRECT-mode client, returns body IDs.

- [ ] Tests fail → implement → tests pass → commit.

## Task 12: Fly-through pytests

**Files:**
- Create: `tests/fixtures/warehouse_fab_trajectories.yaml`
- Create: `tests/test_pybullet/test_warehouse_collision.py`

**Reuse structure from predecessor Tasks 11-13** (lines 1573-1845). Three tests:
- `clean_pass_through` — straight segments connecting Gate_01→02→03→04→05 at each gate's center
- `wall_collision` — fly into a known wall location (pick from our warehouse extents)
- `gate_rim_collision` — fly into Gate_03's outer ring

Trajectories authored from gate positions in our YAML + hand-picked wall coordinates from the cropped mesh bounds.

- [ ] Tests fail → implement → tests pass → commit: `feat(warehouse): PyBullet fly-through validation tests`

## Task 13: End-to-end run + COR-95 update

- [ ] Run the full CLI on the real inputs and confirm outputs
- [ ] Run pytest over the whole warehouse test tree: `pytest tests/test_warehouse_to_urdf/ tests/test_pybullet/ -v`
- [ ] Spot-check: launch PyBullet GUI mode with `sim/pybullet/warehouse_loader.py` demo, screenshot the scene, compare visually against a UE Editor top-down screenshot of the same gates
- [ ] If PyBullet scene looks like "empty room" rather than warehouse → the Packed Level Actors didn't export. Re-export from UE with PLAs unpacked (right-click each PLA in Outliner → "Unpack Level Instance" → re-run File → Export Selected). Commit the new FBX.
- [ ] Post session-complete comment on COR-95 with: outputs, pytest results, screenshots, outstanding issues
- [ ] Transition COR-95 to "In Review" if cleanly done, or leave "In Progress" with a note if racks need re-exporting

---

## Out of scope for this plan

- AirSim packaging (deferred, see `docs/progress/2026-04-22-ue-airsim-setup.md`)
- AirSim fixture capture + silhouette IoU validation (dropped; no running AirSim)
- Occupancy grid + A* planner (phase 2)
- gym-pybullet-drones training integration (phase 3)
- Movable gate support (`gate_movable.urdf`) — stays fixed-base for phase 1

## Exit criteria

Phase 1 complete when all of these are true:

1. `python -m scripts.warehouse_to_urdf ... ` produces all 5 expected output files + manifest
2. `pytest tests/test_warehouse_to_urdf/ tests/test_pybullet/ -v` passes, ≥ 20 tests
3. `clean_pass_through` fly-through passes: 0 warehouse contacts, drone passes within 0.75 m of each gate center in order
4. `wall_collision` and `gate_rim_collision` tests pass with expected contact patterns
5. Visual inspection: PyBullet scene rendered from ~top-down matches the UE Editor top-down screenshot at recognizable level (same number of walls, rough aisle positions, 5 gates in approximately-right places)
6. COR-95 has a session-close comment with the artifacts + pytest output linked
