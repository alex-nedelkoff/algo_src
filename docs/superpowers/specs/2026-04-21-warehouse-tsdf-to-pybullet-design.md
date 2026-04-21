# Warehouse TSDF → PyBullet Conversion (Phase 1)

**Date:** 2026-04-21
**Status:** Draft
**Linear:** COR-92 (depends on artifact)
**Author:** Alex + Claude

## Scope

Phase 1 of a four-phase initiative to bring the COR-92 Cosys-AirSim warehouse into a PyBullet-based training environment.

**This spec covers phase 1 only:** convert the warehouse TSDF voxel grid + gate pose JSON into URDF assets that load correctly in PyBullet, with collision behavior validated against (a) AirSim render fixtures and (b) a deterministic fly-through test. Phases 2–4 (occupancy grid + A* planner, gym-pybullet-drones training loop integration, track config loader) are explicitly out of scope and will be specified separately.

**Phase 1 deliverable:** PyBullet can load the warehouse + 5 gates as static collision bodies, and a hand-authored fly-through trajectory passes through all 5 gates without false-positive wall collisions and triggers expected collisions on intentionally-bad trajectories.

## Inputs

1. **TSDF artifact** — voxel grid produced by Janahan's reconstruction pipeline. Format unknown until 2026-04-21 evening; loader is adaptive (see Build Pipeline §2). Expected to contain at minimum: signed-distance values, voxel size, world-space origin (NED).
2. **Gate JSON** — `warehouse_5gates_v1_gates_ned.json` from COR-92 attachments. Five gates (`Gate_01..Gate_05`), each with `position_ned`, `orientation_wxyz`, `inner_radius_m: 0.75`, `outer_radius_m: 0.85`.

## Architecture

Two pipelines and two validators:

```
  TSDF artifact          gate JSON (NED)
       │                       │
       ▼                       ▼
  ┌────────────────────────────────────────────────┐
  │  BUILD-TIME  tools/warehouse_to_urdf.py        │
  │   1. Load TSDF (adaptive loader)               │
  │   2. Clip volume around each gate (~1.0 m)     │
  │   3. Marching cubes → triangle mesh            │
  │   4. Simplify (target ~30k faces)              │
  │   5. NED → ENU coordinate flip                 │
  │   6. Write warehouse.{obj,urdf}                │
  │   7. Write gate.urdf + gate_ring.obj           │
  │   8. Write gates_enu.json + manifest.yaml      │
  └────────────────────────────────────────────────┘
                       │
                       ▼
        sim/assets/warehouse_v1/
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  RUNTIME LOADER                VALIDATION
  sim/pybullet/                  B: fixture compare
    warehouse_loader.py          D: fly-through pytest
```

**Key choices:**
- Walls baked into one mesh (`warehouse.obj`), single static URDF body.
- Gates as 5 separately-loaded instances of one shared `gate.urdf` (origin = identity in URDF; pose set at `loadURDF()` call time from `gates_enu.json`).
- Coordinate convention: assets stored in **ENU (Z-up)**. NED → ENU flip applied once at build time to both mesh vertices and gate poses.
- `manifest.yaml` records source TSDF hash + gate JSON hash + build params for provenance.

## Build Pipeline (`tools/warehouse_to_urdf.py`)

**Library:** Open3D — handles TSDF I/O, marching cubes, mesh simplification, OBJ writing in a single dependency.

### Input contract

```python
@dataclass
class TSDFArtifact:
    sdf: np.ndarray          # shape (X, Y, Z), float32, signed distance values
    voxel_size: float        # meters
    origin: np.ndarray       # shape (3,), world-space NED corner of voxel grid
```

Loader auto-dispatches on extension:
- `.npz` — numpy bundle with `sdf`, `voxel_size`, `origin` keys
- `.ply` — pre-extracted Open3D mesh (skip stages 2–3, jump to simplify)
- `.bin` — Open3D `ScalableTSDFVolume` serialized format

Unknown extensions error with a clear message listing supported formats.

### Stages

1. **Load TSDF** → `TSDFArtifact` via the adapter above.
2. **Clip gates** — for each gate position in the JSON, set `sdf[mask] = +large_value` where `mask` is a sphere of radius `--gate-clip-radius` (default 1.0 m, CLI-tunable) around the NED gate position. Carves gate geometry out of the warehouse mesh; we add gates back as separate bodies.
3. **Marching cubes** — extract iso-surface at value 0 via Open3D.
4. **Simplify** — `mesh.simplify_quadric_decimation(target_number_of_triangles=N)` where N defaults to 30000 (CLI-tunable via `--target-triangles`).
5. **NED → ENU flip** — apply `(x, y, z) → (y, x, -z)` to all mesh vertices, recompute normals.
6. **Write outputs:**
   - `warehouse.obj` — final mesh, ENU coordinates
   - `warehouse.urdf` — single fixed-base link, `warehouse.obj` as both `<collision>` and `<visual>`
   - `gate.urdf` — single shared URDF (see Gate URDF section)
   - `gate_ring.obj` — procedurally generated 32-segment torus, ~10 cm thick (outer radius 0.85, inner radius 0.75 baked into the mesh, no scale at load time)
   - `gates_enu.json` — same schema as input gate JSON but with ENU poses
   - `manifest.yaml` — see Provenance section

### Failure modes

- TSDF format not recognized → clear error listing supported extensions
- Mesh simplification reduces to < 100 triangles → abort (TSDF probably empty or all-positive)
- Gate position outside TSDF bounds → warn + skip clip for that gate; continue
- Output asset total > 10 MB → warn that git-lfs may be needed; do not fail

### CLI

```bash
python -m tools.warehouse_to_urdf \
    --tsdf path/to/warehouse_tsdf.npz \
    --gates configs/warehouse/warehouse_5gates_v1_gates_ned.json \
    --out sim/assets/warehouse_v1/ \
    --target-triangles 30000 \
    --gate-clip-radius 1.0
```

## Gate URDF

Single shared `gate.urdf`. All 5 COR-92 gates share `inner_radius_m: 0.75` / `outer_radius_m: 0.85`, so per-gate scale is unnecessary in phase 1. Origin = identity; the runtime loader places N instances at per-gate poses from `gates_enu.json`.

```xml
<robot name="gate">
  <link name="ring">
    <visual>
      <geometry><mesh filename="gate_ring.obj"/></geometry>
      <material name="orange"><color rgba="1.0 0.5 0.1 1.0"/></material>
    </visual>
    <collision>
      <geometry><mesh filename="gate_ring.obj"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
```

`gate_ring.obj` is generated procedurally during the build run (no external mesh dependency): a 32-segment torus with major radius 0.80 m and minor radius 0.05 m, giving 0.75 m inner / 0.85 m outer in the ring plane.

Gates loaded with `useFixedBase=True` for phase 1. The COR-92 gotcha about runtime gate movement (`simSetObjectPose`) is a future concern, addressed by a `gate_movable.urdf` variant in a later phase.

## Runtime Loader (`sim/pybullet/warehouse_loader.py`)

Roughly 80 lines. No env logic, no physics — just turns assets into PyBullet bodies.

```python
@dataclass
class WarehouseHandles:
    warehouse_body_id: int
    gate_body_ids: list[int]   # ordered Gate_01 → Gate_05
    gate_names: list[str]

@dataclass
class WarehouseScene:
    asset_dir: Path             # sim/assets/warehouse_v1/

    def load_into(self, client_id: int) -> WarehouseHandles:
        ...
```

Behavior:
1. `p.setAdditionalSearchPath(str(self.asset_dir))` so URDF mesh paths resolve.
2. Load `warehouse.urdf` with `useFixedBase=True`.
3. Read `gates_enu.json`. For each gate, call `p.loadURDF("gate.urdf", basePosition=g["position_enu"], baseOrientation=g["orientation_enu"], useFixedBase=True)`.
4. Return `WarehouseHandles` with body IDs and ordered gate names.

The env that wraps this loader (phase 3) decides physics step rate, control input, observation construction. Phase 1 only requires the loader to expose body IDs for the validation tests.

## Validation

Combined exit criterion for phase 1: both validations below must pass.

### B-fixture — render comparison vs AirSim

**Note: this is the v1 mechanism.** If a Windows build of the warehouse binary or the UE source becomes available, this migrates to live B-live capture. The fixture-based path is designed so the comparison harness stays unchanged — only the source of `pose_NN_*` files changes.

#### Capture (one-time, runs on Linux box with COR-92 binary)

`tools/capture_airsim_fixtures.py` script. Run once on Janahan's machine or the training rig. For each of ~15 hand-picked ENU camera poses providing coverage of the warehouse from multiple angles and heights:

1. Convert pose ENU → NED.
2. Set drone + camera pose via `simSetVehiclePose` (re: COR-92 gotcha — never call `client.reset()`).
3. `simGetImages` request with `Scene` + `DepthPlanar` (planar Z-buffer, NOT `DepthPerspective` — re: COR-92 gotcha #3).
4. Save: `pose_NN_scene.png`, `pose_NN_depth.npy`, `pose_NN.json` (pose ENU + NED, intrinsics fov/width/height, capture timestamp, AirSim version, warehouse binary version).

Outputs land in `tests/fixtures/airsim_warehouse_v1/` and are checked into the repo (~15 fixtures × ~300 KB each ≈ 5 MB total expected; LFS only if exceeded).

#### Comparison harness

`tools/compare_pybullet_vs_fixtures.py`:

For each fixture in `tests/fixtures/airsim_warehouse_v1/`:
1. Load warehouse scene in PyBullet (DIRECT mode, EGL renderer for headless).
2. `p.computeViewMatrix` from fixture pose, `p.computeProjectionMatrixFOV` matched to fixture intrinsics.
3. `p.getCameraImage(...)` → RGB + depth + segmentation.
4. Build silhouette masks: `(depth > 0) & (depth < far_plane)` for AirSim depth; `(segmask != -1)` for PyBullet.
5. Compute silhouette IoU between the two masks.

**Output:** HTML report `outputs/validation/warehouse_v1/<timestamp>/index.html` with one row per pose: AirSim render | PyBullet render | silhouette overlay | IoU score. Aggregate IoU at top of report.

**Pass bar:** mean silhouette IoU ≥ 0.85, no individual pose < 0.70. Calibrated on the first run; numbers may move once we see real data.

Silhouette IoU isolates geometry/pose correctness from rendering quality (lighting, materials, anti-aliasing differ between Unreal and PyBullet — pure pixel-similarity metrics like SSIM are not reliable across renderers).

### D — fly-through pytest

`tests/test_pybullet/test_warehouse_collision.py` + `tests/fixtures/warehouse_trajectories.yaml`.

Three hand-authored waypoint sequences:
- `clean_pass_through` — straight line passing through all 5 gate centers in order
- `wall_collision` — flies into a known wall location
- `gate_rim_collision` — clips a gate's outer edge

Each test:
1. Boots PyBullet in DIRECT mode, loads warehouse + gates via `WarehouseScene`.
2. Spawns a 5 cm sphere as a kinematic drone (`p.createMultiBody` with collision shape only).
3. Steps the drone along waypoints (linear interp, ~100 substeps), calling `p.performCollisionDetection` then `p.getContactPoints` each step.
4. Records every contact event with body ID and contact location.

**Assertions:**
- `clean_pass_through`: 0 warehouse contacts; drone position passes within `inner_radius_m` (0.75 m) of each gate plane in JSON order; 0 gate-rim contacts.
- `wall_collision`: ≥ 1 warehouse contact; first contact within 10 cm of expected wall point (specified in trajectory fixture).
- `gate_rim_collision`: ≥ 1 contact with the expected gate body ID (specified in trajectory fixture); 0 warehouse contacts.

Tests run in CI (CPU-only, no GPU needed for PyBullet DIRECT mode).

## Provenance (`manifest.yaml`)

```yaml
schema_version: 1
warehouse_version: v1
build:
  timestamp: 2026-04-21T22:30:00Z
  builder: tools/warehouse_to_urdf.py
  git_sha: <commit at build time>
sources:
  tsdf:
    path: <input path used for this build>
    sha256: <hash of TSDF file>
  gates:
    path: <input gates JSON>
    sha256: <hash of gates JSON>
build_params:
  target_triangles: 30000
  gate_clip_radius_m: 1.0
  ned_to_enu_applied: true
outputs:
  warehouse_obj_triangles: <actual count after simplify>
  warehouse_obj_size_bytes: <bytes>
  num_gates: 5
```

The B-fixture comparison harness reads `manifest.yaml` and warns if the fixture set predates the current build (`tests/fixtures/airsim_warehouse_v1/manifest.yaml` carries a matching schema with the build hash it was captured against).

## Dependencies

**Add to `pyproject.toml`:**
- `open3d` — TSDF I/O, marching cubes, mesh simplification, OBJ writing.

**Already present:**
- `pybullet`, `numpy`, `pyyaml`, `pytest`.

**Not added to main deps (Linux capture only):**
- `cosysairsim` — install instructions in `tools/capture_airsim_fixtures.py` header. Run on the Linux box with the COR-92 binary, not on dev boxes.

## File Layout (end state)

```
algo_src/
├── tools/
│   ├── warehouse_to_urdf.py
│   ├── capture_airsim_fixtures.py
│   └── compare_pybullet_vs_fixtures.py
├── sim/
│   ├── pybullet/
│   │   ├── __init__.py
│   │   └── warehouse_loader.py
│   └── assets/
│       └── warehouse_v1/
│           ├── warehouse.urdf
│           ├── warehouse.obj
│           ├── gate.urdf
│           ├── gate_ring.obj
│           ├── gates_enu.json
│           └── manifest.yaml
├── tests/
│   ├── test_pybullet/
│   │   ├── __init__.py
│   │   └── test_warehouse_collision.py
│   └── fixtures/
│       ├── warehouse_trajectories.yaml
│       └── airsim_warehouse_v1/
│           ├── pose_00_scene.png
│           ├── pose_00_depth.npy
│           ├── pose_00.json
│           ├── ... (×15)
│           └── manifest.yaml
└── configs/
    └── warehouse/
        └── warehouse_5gates_v1_gates_ned.json   (downloaded from COR-92)
```

## Success Criteria

Phase 1 is complete when all of the following hold:

1. `python -m tools.warehouse_to_urdf --tsdf X --gates Y --out Z` runs end-to-end and writes all expected files (warehouse.urdf/.obj, gate.urdf, gate_ring.obj, gates_enu.json, manifest.yaml).
2. `manifest.yaml` records reproducible provenance (source hashes, build params, git SHA, timestamp).
3. `pytest tests/test_pybullet/` — all D fly-through scenarios pass in CI.
4. `python -m tools.compare_pybullet_vs_fixtures.py` — mean silhouette IoU ≥ 0.85, no individual pose < 0.70.
5. HTML validation report (`outputs/validation/warehouse_v1/<timestamp>/index.html`) saved and attached to a Linear issue (sub-issue of COR-92, or new COR-93).

## Open Risks

| Risk | Mitigation |
|------|------------|
| TSDF artifact format unknown until 2026-04-21 evening | Adaptive loader handles `.npz` / `.ply` / `.bin`; CLI errors clearly on unknown format. May need one extra adapter when artifact lands. |
| Gate clip radius 1.0 m too aggressive (cuts walls behind gates) or too tight (leaves gate fragments) | CLI flag `--gate-clip-radius`; first build calibrates; visible in B-fixture comparison. |
| 30k triangle target wrong for collision performance | CLI flag `--target-triangles`; D fly-through latency is the practical signal. |
| Asset files exceed 10 MB total | Build-time pre-flight check; warn + suggest git-lfs. |
| PyBullet GL render too ugly for visual inspection | Silhouette IoU is the real metric; RGB renders in HTML report are reference only. |
| Fixture-based B validation rots if warehouse changes | Manifest hash comparison between fixture capture and current URDF build; harness warns when mismatched. Fixture re-capture documented in capture script header. |
| Coordinate-frame errors (NED ↔ ENU) | Single conversion point at build time; `manifest.yaml` records `ned_to_enu_applied: true`; B-fixture comparison would surface mismatches as massive IoU drops. |

## Out of Scope

The following are explicitly deferred to later specs:

- **Phase 2:** 3D occupancy grid extraction + A* planner running during training.
- **Phase 3:** gym-pybullet-drones training loop integration (env, observations, rewards, dynamics).
- **Phase 4:** track config loader for the new env (mirror `sim/tracks/loader.py` schema).
- **B-live migration** — when Windows binary or UE source becomes available, swap fixture capture for live AirSim capture; comparison harness stays the same.
- **Per-gate radius variation** — when procedural tracks introduce gates with different inner/outer radii, replace single shared `gate.urdf` with per-gate variants or runtime mesh scaling.
- **Convex decomposition (V-HACD)** — if simplified mesh is too slow for collision, decompose into convex parts.
- **Movable gates** — `gate_movable.urdf` variant with a free joint, when runtime gate movement is needed.
