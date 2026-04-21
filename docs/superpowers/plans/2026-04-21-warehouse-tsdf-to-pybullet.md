# Warehouse TSDF → PyBullet Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the COR-92 Cosys-AirSim warehouse TSDF + gate JSON into URDF assets that load correctly in PyBullet, validated by silhouette-IoU vs AirSim fixtures and a deterministic fly-through pytest.

**Architecture:** Build-time pipeline (Python + Open3D) extracts/simplifies/converts mesh and writes URDFs + manifest. Runtime loader (~80 lines) turns assets into PyBullet bodies. Validation = comparison harness vs pre-captured AirSim fixtures + pytest fly-through with collision assertions.

**Tech Stack:** Python 3.11, Open3D (mesh ops), PyBullet (physics + render), NumPy, PyYAML, pytest. One new dependency: `open3d` added to `sim` extras.

**Spec:** `docs/superpowers/specs/2026-04-21-warehouse-tsdf-to-pybullet-design.md`

**External inputs (not produced by this plan):**
1. **TSDF artifact** — arriving evening of 2026-04-21 from Janahan; format adaptive (`.npz` / `.ply` / `.bin`).
2. **Gate JSON** — already at https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/exploration-binaries/warehouse_5gates_v1_gates_ned.json (Task 1 downloads it).
3. **AirSim fixtures** — captured one-time on a Linux machine with the COR-92 binary using `scripts/capture_airsim_fixtures.py` (Task 20). All other tasks complete with synthetic test data; final B-validation gates on these fixtures.

---

## File Structure

**New files (created in this plan):**

| Path | Responsibility |
|------|----------------|
| `scripts/warehouse_to_urdf/__init__.py` | Package marker (empty) |
| `scripts/warehouse_to_urdf/__main__.py` | CLI orchestration (~100 LOC) |
| `scripts/warehouse_to_urdf/tsdf.py` | `TSDFArtifact` dataclass + adaptive loader |
| `scripts/warehouse_to_urdf/mesh.py` | Gate clipping, marching cubes, simplification, torus |
| `scripts/warehouse_to_urdf/urdf.py` | warehouse.urdf, gate.urdf, gates_enu.json writers |
| `scripts/warehouse_to_urdf/manifest.py` | manifest.yaml writer + hash helpers |
| `scripts/capture_airsim_fixtures.py` | One-time Linux-only AirSim fixture capture |
| `scripts/compare_pybullet_vs_fixtures.py` | B-fixture validation harness + HTML report |
| `sim/pybullet/__init__.py` | Package marker (empty) |
| `sim/pybullet/coords.py` | NED↔ENU helpers (shared by build + runtime) |
| `sim/pybullet/warehouse_loader.py` | Runtime loader, `WarehouseScene` (~80 LOC) |
| `tests/test_warehouse_to_urdf/__init__.py` | Test package marker |
| `tests/test_warehouse_to_urdf/test_tsdf.py` | TSDF loader unit tests |
| `tests/test_warehouse_to_urdf/test_mesh.py` | Mesh ops unit tests |
| `tests/test_warehouse_to_urdf/test_urdf.py` | URDF writer unit tests |
| `tests/test_warehouse_to_urdf/test_manifest.py` | Manifest writer unit tests |
| `tests/test_warehouse_to_urdf/test_pipeline.py` | End-to-end test with synthetic TSDF |
| `tests/test_pybullet/__init__.py` | Test package marker |
| `tests/test_pybullet/test_coords.py` | Coordinate conversion unit tests |
| `tests/test_pybullet/test_warehouse_loader.py` | Loader smoke test |
| `tests/test_pybullet/test_warehouse_collision.py` | D fly-through tests (3 scenarios) |
| `tests/test_validation/__init__.py` | Test package marker |
| `tests/test_validation/test_silhouette_iou.py` | IoU computation unit test |
| `tests/test_validation/test_html_report.py` | HTML report generation test |
| `tests/fixtures/warehouse_trajectories.yaml` | D fly-through waypoint sequences |
| `configs/warehouse/warehouse_5gates_v1_gates_ned.json` | Downloaded gate poses |

**Modified files:**
- `pyproject.toml` — add `open3d` to `sim` optional-dependencies group.

**Generated artifacts (committed; >10 MB triggers LFS — handled in Task 25):**
- `sim/assets/warehouse_v1/warehouse.{obj,urdf}`, `gate.urdf`, `gate_ring.obj`, `gates_enu.json`, `manifest.yaml`
- `tests/fixtures/airsim_warehouse_v1/pose_NN_*` (×~15) — captured externally in Task 20

---

## Task 1: Add `open3d` dependency, download gate JSON

**Files:**
- Modify: `pyproject.toml`
- Create: `configs/warehouse/warehouse_5gates_v1_gates_ned.json`

- [ ] **Step 1: Add `open3d` to the `sim` optional-dependencies group**

Edit `pyproject.toml` to change the `sim` block:

```toml
sim = [
    "gymnasium",
    "pybullet",
    "open3d",
]
```

- [ ] **Step 2: Install Open3D**

```bash
pip install -e ".[sim]"
python -c "import open3d as o3d; print(o3d.__version__)"
```

Expected: prints a version like `0.18.0` (any 0.17+ is fine).

- [ ] **Step 3: Download the gate JSON**

```bash
mkdir -p configs/warehouse
curl -L -o configs/warehouse/warehouse_5gates_v1_gates_ned.json \
  https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev/exploration-binaries/warehouse_5gates_v1_gates_ned.json
```

- [ ] **Step 4: Verify the JSON parses and contains 5 gates**

```bash
python -c "import json; d=json.load(open('configs/warehouse/warehouse_5gates_v1_gates_ned.json')); print(list(d.keys()))"
```

Expected: `['Gate_01', 'Gate_02', 'Gate_03', 'Gate_04', 'Gate_05']` (or similar ordering).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml configs/warehouse/warehouse_5gates_v1_gates_ned.json
git commit -m "feat(warehouse): add open3d dep and download COR-92 gate JSON"
```

---

## Task 2: NED↔ENU coordinate conversion helpers

**Files:**
- Create: `sim/pybullet/__init__.py`
- Create: `sim/pybullet/coords.py`
- Create: `tests/test_pybullet/__init__.py`
- Create: `tests/test_pybullet/test_coords.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pybullet/__init__.py` (empty).

Create `tests/test_pybullet/test_coords.py`:

```python
"""Tests for NED↔ENU coordinate conversions."""
import numpy as np

from sim.pybullet.coords import ned_to_enu_position, ned_to_enu_quaternion


def test_ned_to_enu_position_swaps_xy_and_negates_z():
    pos_ned = np.array([1.0, 2.0, 3.0])
    pos_enu = ned_to_enu_position(pos_ned)
    np.testing.assert_array_almost_equal(pos_enu, [2.0, 1.0, -3.0])


def test_ned_to_enu_position_zero_is_zero():
    np.testing.assert_array_almost_equal(
        ned_to_enu_position(np.zeros(3)), np.zeros(3)
    )


def test_ned_to_enu_position_array_works_on_batch():
    pts_ned = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    pts_enu = ned_to_enu_position(pts_ned)
    np.testing.assert_array_almost_equal(pts_enu, [[2, 1, -3], [5, 4, -6]])


def test_ned_to_enu_quaternion_identity_stays_identity():
    q_ned = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
    q_enu = ned_to_enu_quaternion(q_ned)
    np.testing.assert_array_almost_equal(q_enu, [1.0, 0.0, 0.0, 0.0])


def test_ned_to_enu_quaternion_180deg_yaw_round_trip():
    # 180° yaw in NED == 180° yaw in ENU (Z axis just flipped sign)
    q_ned = np.array([0.0, 0.0, 0.0, 1.0])  # w=0, z=1 → 180° about Z
    q_enu = ned_to_enu_quaternion(q_ned)
    # After NED→ENU, the rotation should still be valid (unit norm)
    assert abs(np.linalg.norm(q_enu) - 1.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pybullet/test_coords.py -v
```

Expected: ImportError or ModuleNotFoundError on `sim.pybullet.coords`.

- [ ] **Step 3: Create the package + implement the helpers**

Create `sim/pybullet/__init__.py` (empty).

Create `sim/pybullet/coords.py`:

```python
"""NED↔ENU coordinate conversions.

NED: x=North, y=East, z=Down (AirSim, Cosys-AirSim).
ENU: x=East,  y=North, z=Up   (PyBullet world convention used here).

The transform is: (x, y, z)_enu = (y, x, -z)_ned. Equivalent to swapping
the X/Y axes and negating Z. Quaternions are converted by re-expressing
the rotation in the new basis: (w, x, y, z)_enu = (w, y, x, -z)_ned.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ned_to_enu_position(pos_ned: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert position(s) from NED to ENU.

    Works on a single (3,) vector or a batch (N, 3).
    """
    pos = np.asarray(pos_ned, dtype=np.float64)
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 1]
    out[..., 1] = pos[..., 0]
    out[..., 2] = -pos[..., 2]
    return out


def ned_to_enu_quaternion(q_ned_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a quaternion (w, x, y, z) from NED to ENU.

    The rotation is the same physical rotation, expressed in the new basis.
    For the (x, y, z) → (y, x, -z) basis change, the imaginary parts swap
    X/Y and negate Z; the scalar part is unchanged.
    """
    q = np.asarray(q_ned_wxyz, dtype=np.float64)
    return np.array([q[0], q[2], q[1], -q[3]])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pybullet/test_coords.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/__init__.py sim/pybullet/coords.py \
        tests/test_pybullet/__init__.py tests/test_pybullet/test_coords.py
git commit -m "feat(pybullet): add NED↔ENU coordinate conversion helpers"
```

---

## Task 3: TSDFArtifact dataclass + `.npz` adapter

**Files:**
- Create: `scripts/warehouse_to_urdf/__init__.py`
- Create: `scripts/warehouse_to_urdf/tsdf.py`
- Create: `tests/test_warehouse_to_urdf/__init__.py`
- Create: `tests/test_warehouse_to_urdf/test_tsdf.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_warehouse_to_urdf/__init__.py` (empty).

Create `tests/test_warehouse_to_urdf/test_tsdf.py`:

```python
"""Tests for TSDF artifact loading."""
from pathlib import Path

import numpy as np
import pytest

from scripts.warehouse_to_urdf.tsdf import TSDFArtifact, load_tsdf


def _make_synthetic_npz(tmp_path: Path) -> Path:
    """Write a tiny synthetic TSDF .npz: 8x8x8 cube SDF, 0.1 m voxel."""
    sdf = np.ones((8, 8, 8), dtype=np.float32)
    sdf[2:6, 2:6, 2:6] = -0.1  # an interior region < 0
    path = tmp_path / "synthetic.npz"
    np.savez(
        path,
        sdf=sdf,
        voxel_size=np.float32(0.1),
        origin=np.zeros(3, dtype=np.float32),
    )
    return path


def test_load_npz_returns_tsdf_artifact(tmp_path: Path):
    path = _make_synthetic_npz(tmp_path)
    art = load_tsdf(path)
    assert isinstance(art, TSDFArtifact)
    assert art.sdf.shape == (8, 8, 8)
    assert art.sdf.dtype == np.float32
    assert abs(art.voxel_size - 0.1) < 1e-6
    np.testing.assert_array_almost_equal(art.origin, [0, 0, 0])


def test_load_unknown_extension_raises_clear_error(tmp_path: Path):
    bad = tmp_path / "thing.weird"
    bad.write_text("not a tsdf")
    with pytest.raises(ValueError, match="Unsupported TSDF format"):
        load_tsdf(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_warehouse_to_urdf/test_tsdf.py -v
```

Expected: ImportError on `scripts.warehouse_to_urdf.tsdf`.

- [ ] **Step 3: Implement the loader**

Create `scripts/warehouse_to_urdf/__init__.py` (empty).

Create `scripts/warehouse_to_urdf/tsdf.py`:

```python
"""TSDF artifact dataclass and adaptive loader.

The artifact format from Janahan's reconstruction pipeline is unknown
until 2026-04-21 evening. The loader dispatches on file extension and
covers three plausible formats:

    .npz  — numpy bundle with 'sdf', 'voxel_size', 'origin' keys
    .ply  — pre-extracted Open3D triangle mesh (skip MC, return mesh-only)
    .bin  — Open3D ScalableTSDFVolume serialized format

When the artifact lands, add the matching adapter here if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class TSDFArtifact:
    """Loaded TSDF voxel grid + metadata.

    sdf: (X, Y, Z) signed distance values in meters; sign convention
         negative=inside, positive=outside (Open3D and most pipelines).
    voxel_size: edge length of one voxel in meters.
    origin: world-space NED corner of voxel index (0, 0, 0).
    pre_extracted_mesh: if the source artifact was already a mesh
         (e.g. .ply), this holds it and the build pipeline skips
         marching cubes. None for true voxel TSDFs.
    """
    sdf: NDArray[np.float32]
    voxel_size: float
    origin: NDArray[np.float64]
    pre_extracted_mesh: object | None = None  # open3d.geometry.TriangleMesh


def load_tsdf(path: Path | str) -> TSDFArtifact:
    """Load a TSDF artifact, dispatching by extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".ply":
        return _load_ply(path)
    if suffix == ".bin":
        return _load_bin(path)

    raise ValueError(
        f"Unsupported TSDF format: {suffix!r}. "
        f"Supported: .npz, .ply, .bin. File: {path}"
    )


def _load_npz(path: Path) -> TSDFArtifact:
    with np.load(path) as data:
        sdf = np.asarray(data["sdf"], dtype=np.float32)
        voxel_size = float(np.asarray(data["voxel_size"]).item())
        origin = np.asarray(data["origin"], dtype=np.float64)
    return TSDFArtifact(sdf=sdf, voxel_size=voxel_size, origin=origin)


def _load_ply(path: Path) -> TSDFArtifact:
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise ValueError(f"PLY at {path} has no vertices")
    # No SDF — provide an empty grid; pipeline will skip MC.
    return TSDFArtifact(
        sdf=np.zeros((0, 0, 0), dtype=np.float32),
        voxel_size=0.0,
        origin=np.zeros(3),
        pre_extracted_mesh=mesh,
    )


def _load_bin(path: Path) -> TSDFArtifact:
    raise NotImplementedError(
        "Open3D ScalableTSDFVolume .bin loading not yet implemented. "
        "Add adapter here once Janahan confirms format."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_warehouse_to_urdf/test_tsdf.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/warehouse_to_urdf/__init__.py scripts/warehouse_to_urdf/tsdf.py \
        tests/test_warehouse_to_urdf/__init__.py tests/test_warehouse_to_urdf/test_tsdf.py
git commit -m "feat(warehouse): TSDFArtifact dataclass + .npz adaptive loader"
```

---

## Task 4: Gate clipping in the SDF volume

**Files:**
- Create: `scripts/warehouse_to_urdf/mesh.py`
- Modify: `tests/test_warehouse_to_urdf/test_mesh.py` (create new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_warehouse_to_urdf/test_mesh.py`:

```python
"""Tests for mesh operations: clipping, marching cubes, torus."""
import numpy as np
import pytest

from scripts.warehouse_to_urdf.mesh import clip_gates_in_sdf
from scripts.warehouse_to_urdf.tsdf import TSDFArtifact


def _flat_artifact() -> TSDFArtifact:
    """A 20x20x20 SDF entirely below zero (everything inside surface)."""
    sdf = -np.ones((20, 20, 20), dtype=np.float32)
    return TSDFArtifact(
        sdf=sdf, voxel_size=0.1, origin=np.array([0.0, 0.0, 0.0])
    )


def test_clip_gates_sets_voxels_within_radius_to_large_positive():
    art = _flat_artifact()
    # Gate at world (1.0, 1.0, 1.0) NED → voxel (10, 10, 10) at 0.1 m.
    gates_ned = [{"position_ned": [1.0, 1.0, 1.0]}]
    out = clip_gates_in_sdf(art, gates_ned, clip_radius_m=0.3)
    # The voxel at (10, 10, 10) should be +large; far corner unchanged.
    assert out.sdf[10, 10, 10] > 100.0
    assert out.sdf[0, 0, 0] == -1.0


def test_clip_gates_outside_grid_warns_and_skips():
    art = _flat_artifact()
    gates_ned = [{"position_ned": [100.0, 100.0, 100.0]}]
    with pytest.warns(UserWarning, match="outside TSDF bounds"):
        out = clip_gates_in_sdf(art, gates_ned, clip_radius_m=0.3)
    # Original SDF unchanged.
    np.testing.assert_array_equal(out.sdf, art.sdf)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_warehouse_to_urdf/test_mesh.py::test_clip_gates_sets_voxels_within_radius_to_large_positive -v
```

Expected: ImportError on `scripts.warehouse_to_urdf.mesh`.

- [ ] **Step 3: Implement clipping**

Create `scripts/warehouse_to_urdf/mesh.py`:

```python
"""Mesh operations: gate clipping, marching cubes, simplification, torus."""
from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Sequence

import numpy as np

from scripts.warehouse_to_urdf.tsdf import TSDFArtifact

_LARGE_POSITIVE_SDF = 1e6


def clip_gates_in_sdf(
    artifact: TSDFArtifact,
    gates_ned: Sequence[dict],
    clip_radius_m: float,
) -> TSDFArtifact:
    """Carve out gate volumes from the SDF by setting nearby voxels to +∞.

    For each gate, voxels within `clip_radius_m` of the gate's NED
    position have their SDF value set to a large positive value, so
    marching cubes will not extract any geometry there. Gates that fall
    outside the voxel grid are warned about and skipped.
    """
    if artifact.pre_extracted_mesh is not None:
        # No SDF to clip; mesh-cleaning would require a different op.
        return artifact

    sdf = artifact.sdf.copy()
    nx, ny, nz = sdf.shape
    voxel = artifact.voxel_size
    origin = artifact.origin

    for g in gates_ned:
        center_world = np.asarray(g["position_ned"], dtype=np.float64)
        center_idx = (center_world - origin) / voxel
        if np.any(center_idx < -clip_radius_m / voxel) or np.any(
            center_idx > np.array([nx, ny, nz]) + clip_radius_m / voxel
        ):
            warnings.warn(
                f"Gate at {center_world} is outside TSDF bounds; skipping clip."
            )
            continue

        radius_vox = clip_radius_m / voxel
        i_lo = max(0, int(np.floor(center_idx[0] - radius_vox)))
        i_hi = min(nx, int(np.ceil(center_idx[0] + radius_vox)) + 1)
        j_lo = max(0, int(np.floor(center_idx[1] - radius_vox)))
        j_hi = min(ny, int(np.ceil(center_idx[1] + radius_vox)) + 1)
        k_lo = max(0, int(np.floor(center_idx[2] - radius_vox)))
        k_hi = min(nz, int(np.ceil(center_idx[2] + radius_vox)) + 1)

        ii, jj, kk = np.meshgrid(
            np.arange(i_lo, i_hi),
            np.arange(j_lo, j_hi),
            np.arange(k_lo, k_hi),
            indexing="ij",
        )
        d2 = (
            (ii - center_idx[0]) ** 2
            + (jj - center_idx[1]) ** 2
            + (kk - center_idx[2]) ** 2
        )
        mask = d2 <= radius_vox ** 2
        sdf[i_lo:i_hi, j_lo:j_hi, k_lo:k_hi][mask] = _LARGE_POSITIVE_SDF

    return replace(artifact, sdf=sdf)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_warehouse_to_urdf/test_mesh.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/warehouse_to_urdf/mesh.py tests/test_warehouse_to_urdf/test_mesh.py
git commit -m "feat(warehouse): clip gate volumes from SDF before mesh extraction"
```

---

## Task 5: Marching cubes + simplification on the SDF

**Files:**
- Modify: `scripts/warehouse_to_urdf/mesh.py`
- Modify: `tests/test_warehouse_to_urdf/test_mesh.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_warehouse_to_urdf/test_mesh.py`:

```python
import open3d as o3d

from scripts.warehouse_to_urdf.mesh import (
    extract_mesh_from_sdf,
    simplify_mesh,
)


def _sphere_artifact() -> TSDFArtifact:
    """Build a 32^3 SDF representing a sphere of radius 1.0 m centered in a 3.2 m cube."""
    n = 32
    voxel = 0.1
    sdf = np.empty((n, n, n), dtype=np.float32)
    radius = 1.0
    center = np.array([n / 2, n / 2, n / 2]) * voxel
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = np.array([i, j, k]) * voxel
                sdf[i, j, k] = np.linalg.norm(p - center) - radius
    return TSDFArtifact(sdf=sdf, voxel_size=voxel, origin=np.zeros(3))


def test_extract_mesh_from_sphere_sdf_produces_nonempty_mesh():
    art = _sphere_artifact()
    mesh = extract_mesh_from_sdf(art)
    assert isinstance(mesh, o3d.geometry.TriangleMesh)
    assert len(mesh.vertices) > 100
    assert len(mesh.triangles) > 100


def test_extract_mesh_returns_pre_extracted_when_present():
    placeholder = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    art = TSDFArtifact(
        sdf=np.zeros((0, 0, 0), dtype=np.float32),
        voxel_size=0.0,
        origin=np.zeros(3),
        pre_extracted_mesh=placeholder,
    )
    mesh = extract_mesh_from_sdf(art)
    assert mesh is placeholder


def test_simplify_mesh_reduces_triangle_count():
    art = _sphere_artifact()
    mesh = extract_mesh_from_sdf(art)
    n_before = len(mesh.triangles)
    simplified = simplify_mesh(mesh, target_triangles=200)
    assert len(simplified.triangles) < n_before
    assert len(simplified.triangles) <= 250  # decimation is approximate


def test_simplify_aborts_if_result_under_100_triangles():
    art = _sphere_artifact()
    mesh = extract_mesh_from_sdf(art)
    with pytest.raises(ValueError, match="too few triangles"):
        simplify_mesh(mesh, target_triangles=10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_warehouse_to_urdf/test_mesh.py -v
```

Expected: ImportError on `extract_mesh_from_sdf` and `simplify_mesh`.

- [ ] **Step 3: Implement**

Append to `scripts/warehouse_to_urdf/mesh.py`:

```python
def extract_mesh_from_sdf(artifact: TSDFArtifact):
    """Extract triangle mesh from SDF via marching cubes (Open3D).

    If the artifact already carries a pre-extracted mesh (e.g. .ply
    input), return it directly without re-meshing.
    """
    import open3d as o3d
    from skimage import measure

    if artifact.pre_extracted_mesh is not None:
        return artifact.pre_extracted_mesh

    # skimage's marching_cubes is more reliable than Open3D for raw SDF
    # arrays; Open3D's MC API expects a VoxelGrid or VolumeIntegration.
    verts, faces, normals, _ = measure.marching_cubes(
        artifact.sdf,
        level=0.0,
        spacing=(artifact.voxel_size, artifact.voxel_size, artifact.voxel_size),
    )
    verts = verts + artifact.origin  # shift into world frame

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_vertex_normals()
    return mesh


def simplify_mesh(mesh, target_triangles: int):
    """Quadric-decimation simplify; abort if result is too small."""
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )
    if len(simplified.triangles) < 100:
        raise ValueError(
            f"Simplified mesh has too few triangles "
            f"({len(simplified.triangles)} < 100). TSDF likely empty "
            f"or all-positive."
        )
    return simplified
```

We also need `scikit-image` for `measure.marching_cubes`. Add it:

Edit `pyproject.toml` `sim` extras:

```toml
sim = [
    "gymnasium",
    "pybullet",
    "open3d",
    "scikit-image",
]
```

Then:

```bash
pip install -e ".[sim]"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_warehouse_to_urdf/test_mesh.py -v
```

Expected: 6 passed (2 from Task 4 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/warehouse_to_urdf/mesh.py tests/test_warehouse_to_urdf/test_mesh.py pyproject.toml
git commit -m "feat(warehouse): marching cubes + quadric decimation"
```

---

## Task 6: Procedural torus mesh generator + NED→ENU vertex flip

**Files:**
- Modify: `scripts/warehouse_to_urdf/mesh.py`
- Modify: `tests/test_warehouse_to_urdf/test_mesh.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_warehouse_to_urdf/test_mesh.py`:

```python
from scripts.warehouse_to_urdf.mesh import make_torus_mesh, flip_mesh_ned_to_enu


def test_make_torus_mesh_has_expected_topology():
    mesh = make_torus_mesh(major_radius=0.80, minor_radius=0.05, n_segments=32)
    # 32 segments × 32 cross-section = 1024 vertices, 2048 triangles
    assert len(mesh.vertices) == 32 * 32
    assert len(mesh.triangles) == 32 * 32 * 2


def test_make_torus_mesh_bounds_match_inner_outer_radii():
    mesh = make_torus_mesh(major_radius=0.80, minor_radius=0.05, n_segments=32)
    verts = np.asarray(mesh.vertices)
    radial = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
    # Inner edge ≈ 0.75, outer edge ≈ 0.85, ring axis along Z (so |z| ≤ 0.05).
    assert abs(radial.min() - 0.75) < 0.01
    assert abs(radial.max() - 0.85) < 0.01
    assert abs(verts[:, 2]).max() < 0.06


def test_flip_mesh_ned_to_enu_swaps_xy_negates_z():
    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 2.0, 3.0)
    # Box created at origin; verts are {0,1}×{0,2}×{0,3}.
    flipped = flip_mesh_ned_to_enu(mesh)
    verts_in = np.asarray(mesh.vertices)
    verts_out = np.asarray(flipped.vertices)
    # NED→ENU: (x, y, z) → (y, x, -z)
    np.testing.assert_array_almost_equal(verts_out[:, 0], verts_in[:, 1])
    np.testing.assert_array_almost_equal(verts_out[:, 1], verts_in[:, 0])
    np.testing.assert_array_almost_equal(verts_out[:, 2], -verts_in[:, 2])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_warehouse_to_urdf/test_mesh.py -v
```

Expected: ImportError on `make_torus_mesh` and `flip_mesh_ned_to_enu`.

- [ ] **Step 3: Implement**

Append to `scripts/warehouse_to_urdf/mesh.py`:

```python
def make_torus_mesh(
    major_radius: float = 0.80,
    minor_radius: float = 0.05,
    n_segments: int = 32,
):
    """Procedural torus around the Z axis. Used as the gate ring mesh.

    Outer radius (in ring plane) = major + minor; inner = major − minor.
    With defaults: outer 0.85, inner 0.75 — matches COR-92 gates.
    """
    import open3d as o3d

    # Parametric grid: u around the ring axis, v around the tube cross-section.
    u = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = (major_radius + minor_radius * np.cos(vv)) * np.cos(uu)
    y = (major_radius + minor_radius * np.cos(vv)) * np.sin(uu)
    z = minor_radius * np.sin(vv)
    verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Triangulate the (n_segments × n_segments) grid as 2 triangles per quad,
    # wrapping in both directions.
    n = n_segments
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * n + j
            b = ((i + 1) % n) * n + j
            c = ((i + 1) % n) * n + (j + 1) % n
            d = i * n + (j + 1) % n
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.asarray(faces, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def flip_mesh_ned_to_enu(mesh):
    """Return a copy of the mesh with vertices converted NED→ENU.

    (x, y, z) → (y, x, -z). Recomputes normals.
    """
    import open3d as o3d

    verts = np.asarray(mesh.vertices)
    flipped = np.empty_like(verts)
    flipped[:, 0] = verts[:, 1]
    flipped[:, 1] = verts[:, 0]
    flipped[:, 2] = -verts[:, 2]

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(flipped)
    out.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    out.compute_vertex_normals()
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_warehouse_to_urdf/test_mesh.py -v
```

Expected: 9 passed (6 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/warehouse_to_urdf/mesh.py tests/test_warehouse_to_urdf/test_mesh.py
git commit -m "feat(warehouse): procedural torus generator + NED→ENU mesh flip"
```

---

## Task 7: URDF + gates_enu.json writers

**Files:**
- Create: `scripts/warehouse_to_urdf/urdf.py`
- Create: `tests/test_warehouse_to_urdf/test_urdf.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_warehouse_to_urdf/test_urdf.py`:

```python
"""Tests for URDF and gates_enu.json writers."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.warehouse_to_urdf.urdf import (
    write_warehouse_urdf,
    write_gate_urdf,
    write_gates_enu_json,
)


def test_write_warehouse_urdf_produces_valid_xml(tmp_path: Path):
    out = tmp_path / "warehouse.urdf"
    write_warehouse_urdf(out, mesh_filename="warehouse.obj")
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "robot"
    link = root.find("link")
    assert link is not None
    assert link.find("collision/geometry/mesh").attrib["filename"] == "warehouse.obj"
    assert link.find("visual/geometry/mesh").attrib["filename"] == "warehouse.obj"


def test_write_gate_urdf_produces_valid_xml(tmp_path: Path):
    out = tmp_path / "gate.urdf"
    write_gate_urdf(out, mesh_filename="gate_ring.obj")
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib["name"] == "gate"
    link = root.find("link")
    assert link.find("collision/geometry/mesh").attrib["filename"] == "gate_ring.obj"


def test_write_gates_enu_json_converts_ned_to_enu(tmp_path: Path):
    gates_ned = {
        "Gate_01": {
            "position_ned": [1.0, 2.0, -3.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }
    out = tmp_path / "gates_enu.json"
    write_gates_enu_json(out, gates_ned)
    data = json.loads(out.read_text())
    g = data["Gate_01"]
    # NED (1, 2, -3) → ENU (2, 1, 3)
    assert g["position_enu"] == [2.0, 1.0, 3.0]
    # Identity quaternion stays identity in ENU.
    assert g["orientation_enu_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert g["inner_radius_m"] == 0.75
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_warehouse_to_urdf/test_urdf.py -v
```

Expected: ImportError on `scripts.warehouse_to_urdf.urdf`.

- [ ] **Step 3: Implement the writers**

Create `scripts/warehouse_to_urdf/urdf.py`:

```python
"""URDF and gates_enu.json writers."""
from __future__ import annotations

import json
from pathlib import Path

from sim.pybullet.coords import ned_to_enu_position, ned_to_enu_quaternion

_WAREHOUSE_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="warehouse">
  <link name="walls">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
      <material name="grey"><color rgba="0.6 0.6 0.6 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
    </collision>
    <inertial>
      <mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
</robot>
"""

_GATE_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="gate">
  <link name="ring">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
      <material name="orange"><color rgba="1.0 0.5 0.1 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
"""


def write_warehouse_urdf(path: Path, mesh_filename: str) -> None:
    Path(path).write_text(_WAREHOUSE_URDF_TEMPLATE.format(mesh_filename=mesh_filename))


def write_gate_urdf(path: Path, mesh_filename: str) -> None:
    Path(path).write_text(_GATE_URDF_TEMPLATE.format(mesh_filename=mesh_filename))


def write_gates_enu_json(path: Path, gates_ned: dict) -> None:
    """Convert each gate's pose from NED to ENU, write to JSON."""
    out: dict = {}
    for name, g in gates_ned.items():
        pos_enu = ned_to_enu_position(g["position_ned"]).tolist()
        quat_enu = ned_to_enu_quaternion(g["orientation_wxyz"]).tolist()
        out[name] = {
            "position_enu": pos_enu,
            "orientation_enu_wxyz": quat_enu,
            "inner_radius_m": g["inner_radius_m"],
            "outer_radius_m": g["outer_radius_m"],
        }
    Path(path).write_text(json.dumps(out, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_warehouse_to_urdf/test_urdf.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/warehouse_to_urdf/urdf.py tests/test_warehouse_to_urdf/test_urdf.py
git commit -m "feat(warehouse): URDF and gates_enu.json writers"
```

---

## Task 8: Manifest writer with hashes

**Files:**
- Create: `scripts/warehouse_to_urdf/manifest.py`
- Create: `tests/test_warehouse_to_urdf/test_manifest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_warehouse_to_urdf/test_manifest.py`:

```python
"""Tests for manifest writer."""
import hashlib
from pathlib import Path

import yaml

from scripts.warehouse_to_urdf.manifest import sha256_of_file, write_manifest


def test_sha256_of_file_matches_hashlib(tmp_path: Path):
    data = b"hello world\n"
    p = tmp_path / "a.txt"
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_of_file(p) == expected


def test_write_manifest_records_all_provenance(tmp_path: Path):
    tsdf = tmp_path / "tsdf.npz"
    gates = tmp_path / "gates.json"
    obj = tmp_path / "warehouse.obj"
    tsdf.write_bytes(b"fake tsdf")
    gates.write_bytes(b"fake gates")
    obj.write_bytes(b"x" * 4242)

    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(
        manifest_path,
        tsdf_path=tsdf,
        gates_path=gates,
        warehouse_obj_path=obj,
        target_triangles=30000,
        gate_clip_radius_m=1.0,
        warehouse_obj_triangles=27654,
        num_gates=5,
        git_sha="abc1234",
    )
    data = yaml.safe_load(manifest_path.read_text())
    assert data["schema_version"] == 1
    assert data["sources"]["tsdf"]["sha256"] == sha256_of_file(tsdf)
    assert data["sources"]["gates"]["sha256"] == sha256_of_file(gates)
    assert data["build_params"]["target_triangles"] == 30000
    assert data["build_params"]["ned_to_enu_applied"] is True
    assert data["outputs"]["warehouse_obj_size_bytes"] == 4242
    assert data["outputs"]["warehouse_obj_triangles"] == 27654
    assert data["outputs"]["num_gates"] == 5
    assert data["build"]["git_sha"] == "abc1234"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_warehouse_to_urdf/test_manifest.py -v
```

Expected: ImportError on `scripts.warehouse_to_urdf.manifest`.

- [ ] **Step 3: Implement**

Create `scripts/warehouse_to_urdf/manifest.py`:

```python
"""Provenance manifest writer."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    path: Path,
    *,
    tsdf_path: Path,
    gates_path: Path,
    warehouse_obj_path: Path,
    target_triangles: int,
    gate_clip_radius_m: float,
    warehouse_obj_triangles: int,
    num_gates: int,
    git_sha: str,
) -> None:
    obj_size = Path(warehouse_obj_path).stat().st_size
    payload = {
        "schema_version": 1,
        "warehouse_version": "v1",
        "build": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "builder": "scripts/warehouse_to_urdf",
            "git_sha": git_sha,
        },
        "sources": {
            "tsdf": {
                "path": str(tsdf_path),
                "sha256": sha256_of_file(tsdf_path),
            },
            "gates": {
                "path": str(gates_path),
                "sha256": sha256_of_file(gates_path),
            },
        },
        "build_params": {
            "target_triangles": target_triangles,
            "gate_clip_radius_m": gate_clip_radius_m,
            "ned_to_enu_applied": True,
        },
        "outputs": {
            "warehouse_obj_triangles": warehouse_obj_triangles,
            "warehouse_obj_size_bytes": obj_size,
            "num_gates": num_gates,
        },
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_warehouse_to_urdf/test_manifest.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/warehouse_to_urdf/manifest.py tests/test_warehouse_to_urdf/test_manifest.py
git commit -m "feat(warehouse): provenance manifest writer with sha256"
```

---

## Task 9: Build-pipeline CLI orchestration (`__main__.py`)

**Files:**
- Create: `scripts/warehouse_to_urdf/__main__.py`
- Create: `tests/test_warehouse_to_urdf/test_pipeline.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_warehouse_to_urdf/test_pipeline.py`:

```python
"""End-to-end pipeline test on a synthetic TSDF + 1-gate JSON."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _make_synthetic_tsdf(path: Path) -> None:
    """Build a 32^3 SDF representing two boxes (a 'warehouse')."""
    n = 32
    voxel = 0.1
    sdf = np.ones((n, n, n), dtype=np.float32) * 0.5  # outside everywhere
    # A box "wall" along Y axis.
    sdf[5:7, 5:25, 5:25] = -0.1
    # Another box.
    sdf[25:27, 5:25, 5:25] = -0.1
    np.savez(
        path,
        sdf=sdf,
        voxel_size=np.float32(voxel),
        origin=np.zeros(3, dtype=np.float32),
    )


def _make_one_gate_json(path: Path) -> None:
    data = {
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }
    path.write_text(json.dumps(data))


def test_pipeline_end_to_end_produces_all_outputs(tmp_path: Path):
    tsdf = tmp_path / "tsdf.npz"
    gates = tmp_path / "gates.json"
    out = tmp_path / "assets"
    _make_synthetic_tsdf(tsdf)
    _make_one_gate_json(gates)

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.warehouse_to_urdf",
            "--tsdf", str(tsdf),
            "--gates", str(gates),
            "--out", str(out),
            "--target-triangles", "200",
            "--gate-clip-radius", "0.3",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    expected = {
        "warehouse.obj", "warehouse.urdf",
        "gate.urdf", "gate_ring.obj",
        "gates_enu.json", "manifest.yaml",
    }
    actual = {p.name for p in out.iterdir()}
    assert expected.issubset(actual), f"missing: {expected - actual}"

    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["outputs"]["num_gates"] == 1
    assert manifest["build_params"]["ned_to_enu_applied"] is True

    enu = json.loads((out / "gates_enu.json").read_text())
    assert "Gate_01" in enu
    # NED (1.5, 1.5, -1.5) → ENU (1.5, 1.5, 1.5)
    np.testing.assert_array_almost_equal(
        enu["Gate_01"]["position_enu"], [1.5, 1.5, 1.5]
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_warehouse_to_urdf/test_pipeline.py -v
```

Expected: FAIL — `python -m scripts.warehouse_to_urdf` has no `__main__`.

- [ ] **Step 3: Implement the CLI**

Create `scripts/warehouse_to_urdf/__main__.py`:

```python
"""CLI entry point: build warehouse + gate URDFs from TSDF + gate JSON."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

from scripts.warehouse_to_urdf.tsdf import load_tsdf
from scripts.warehouse_to_urdf.mesh import (
    clip_gates_in_sdf,
    extract_mesh_from_sdf,
    simplify_mesh,
    flip_mesh_ned_to_enu,
    make_torus_mesh,
)
from scripts.warehouse_to_urdf.urdf import (
    write_warehouse_urdf,
    write_gate_urdf,
    write_gates_enu_json,
)
from scripts.warehouse_to_urdf.manifest import write_manifest


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsdf", type=Path, required=True)
    ap.add_argument("--gates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-triangles", type=int, default=30000)
    ap.add_argument("--gate-clip-radius", type=float, default=1.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading TSDF: {args.tsdf}")
    artifact = load_tsdf(args.tsdf)
    print(
        f"      shape={artifact.sdf.shape} voxel={artifact.voxel_size} "
        f"origin={artifact.origin}"
    )

    print(f"[2/7] Loading gates: {args.gates}")
    gates_ned = json.loads(args.gates.read_text())
    print(f"      {len(gates_ned)} gates")

    print(f"[3/7] Clipping gate volumes (radius={args.gate_clip_radius} m)")
    artifact = clip_gates_in_sdf(
        artifact,
        list(gates_ned.values()),
        clip_radius_m=args.gate_clip_radius,
    )

    print("[4/7] Marching cubes")
    mesh = extract_mesh_from_sdf(artifact)
    print(f"      raw triangles: {len(mesh.triangles)}")

    print(f"[5/7] Simplify (target {args.target_triangles})")
    mesh = simplify_mesh(mesh, target_triangles=args.target_triangles)
    print(f"      simplified: {len(mesh.triangles)} triangles")

    print("[6/7] NED → ENU coordinate flip")
    mesh_enu = flip_mesh_ned_to_enu(mesh)

    print(f"[7/7] Writing assets to {args.out}")
    obj_path = args.out / "warehouse.obj"
    o3d.io.write_triangle_mesh(str(obj_path), mesh_enu)
    write_warehouse_urdf(args.out / "warehouse.urdf", "warehouse.obj")

    ring = make_torus_mesh()
    o3d.io.write_triangle_mesh(str(args.out / "gate_ring.obj"), ring)
    write_gate_urdf(args.out / "gate.urdf", "gate_ring.obj")

    write_gates_enu_json(args.out / "gates_enu.json", gates_ned)

    write_manifest(
        args.out / "manifest.yaml",
        tsdf_path=args.tsdf,
        gates_path=args.gates,
        warehouse_obj_path=obj_path,
        target_triangles=args.target_triangles,
        gate_clip_radius_m=args.gate_clip_radius,
        warehouse_obj_triangles=len(mesh_enu.triangles),
        num_gates=len(gates_ned),
        git_sha=_git_sha(),
    )

    obj_mb = obj_path.stat().st_size / (1024 * 1024)
    if obj_mb > 10:
        print(f"WARN: warehouse.obj is {obj_mb:.1f} MB — git-lfs recommended")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_warehouse_to_urdf/ -v
```

Expected: all tests pass (15 tests across the test_warehouse_to_urdf/ directory).

- [ ] **Step 5: Commit**

```bash
git add scripts/warehouse_to_urdf/__main__.py tests/test_warehouse_to_urdf/test_pipeline.py
git commit -m "feat(warehouse): CLI orchestration for build pipeline"
```

---

## Task 10: Runtime warehouse loader (`WarehouseScene`)

**Files:**
- Create: `sim/pybullet/warehouse_loader.py`
- Create: `tests/test_pybullet/test_warehouse_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pybullet/test_warehouse_loader.py`:

```python
"""Tests for the runtime PyBullet warehouse loader."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pytest


@pytest.fixture(scope="module")
def synthetic_assets(tmp_path_factory):
    """Build a synthetic warehouse asset bundle once for the test module."""
    out = tmp_path_factory.mktemp("assets")
    tsdf = out.parent / "tsdf.npz"
    gates = out.parent / "gates.json"

    # Synthetic TSDF (one wall).
    n = 24
    sdf = np.ones((n, n, n), dtype=np.float32) * 0.5
    sdf[5:7, 5:20, 5:20] = -0.1
    np.savez(tsdf, sdf=sdf, voxel_size=np.float32(0.1), origin=np.zeros(3))

    gates.write_text(json.dumps({
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }))

    subprocess.run(
        [sys.executable, "-m", "scripts.warehouse_to_urdf",
         "--tsdf", str(tsdf), "--gates", str(gates), "--out", str(out),
         "--target-triangles", "200", "--gate-clip-radius", "0.3"],
        check=True,
    )
    return out


def test_warehouse_scene_loads_warehouse_and_gates(synthetic_assets):
    from sim.pybullet.warehouse_loader import WarehouseScene

    cid = p.connect(p.DIRECT)
    try:
        scene = WarehouseScene(asset_dir=synthetic_assets)
        handles = scene.load_into(cid)
        assert handles.warehouse_body_id >= 0
        assert len(handles.gate_body_ids) == 1
        assert handles.gate_names == ["Gate_01"]

        # Confirm the warehouse body is at the origin and fixed.
        pos, _ = p.getBasePositionAndOrientation(handles.warehouse_body_id, cid)
        np.testing.assert_array_almost_equal(pos, [0, 0, 0])
    finally:
        p.disconnect(cid)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_pybullet/test_warehouse_loader.py -v
```

Expected: ImportError on `sim.pybullet.warehouse_loader`.

- [ ] **Step 3: Implement the loader**

Create `sim/pybullet/warehouse_loader.py`:

```python
"""Runtime PyBullet loader for the warehouse + gate URDFs.

This module turns the build artifacts under `sim/assets/warehouse_v1/`
into PyBullet bodies. It owns no env logic, no physics step rate, no
control. Whatever wraps this loader (a future env class) decides those.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pybullet as p


@dataclass
class WarehouseHandles:
    warehouse_body_id: int
    gate_body_ids: list[int] = field(default_factory=list)
    gate_names: list[str] = field(default_factory=list)


@dataclass
class WarehouseScene:
    asset_dir: Path

    def load_into(self, client_id: int) -> WarehouseHandles:
        asset_dir = Path(self.asset_dir)
        p.setAdditionalSearchPath(str(asset_dir), physicsClientId=client_id)

        warehouse_id = p.loadURDF(
            str(asset_dir / "warehouse.urdf"),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=client_id,
        )

        gates = json.loads((asset_dir / "gates_enu.json").read_text())
        gate_ids: list[int] = []
        gate_names: list[str] = []
        for name in sorted(gates.keys()):
            g = gates[name]
            # gates_enu.json stores wxyz; PyBullet expects xyzw.
            qw, qx, qy, qz = g["orientation_enu_wxyz"]
            gate_id = p.loadURDF(
                str(asset_dir / "gate.urdf"),
                basePosition=g["position_enu"],
                baseOrientation=[qx, qy, qz, qw],
                useFixedBase=True,
                physicsClientId=client_id,
            )
            gate_ids.append(gate_id)
            gate_names.append(name)

        return WarehouseHandles(
            warehouse_body_id=warehouse_id,
            gate_body_ids=gate_ids,
            gate_names=gate_names,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_pybullet/test_warehouse_loader.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/warehouse_loader.py tests/test_pybullet/test_warehouse_loader.py
git commit -m "feat(pybullet): WarehouseScene runtime loader"
```

---

## Task 11: Hand-author fly-through trajectory fixtures

**Files:**
- Create: `tests/fixtures/warehouse_trajectories.yaml`

- [ ] **Step 1: Author the trajectory file**

Create `tests/fixtures/warehouse_trajectories.yaml`:

```yaml
# Hand-authored kinematic drone trajectories for the D fly-through tests.
# Each trajectory is a sequence of (x, y, z) ENU waypoints (meters);
# the test linearly interpolates between consecutive points and steps
# the drone in PyBullet, asserting expected collision behavior.
#
# These coordinates assume the synthetic test asset bundle (one wall,
# one gate at ENU position (1.5, 1.5, 1.5)). When the real warehouse
# v1 assets land, this file gets a parallel section keyed by
# `warehouse_v1` with real waypoints; the test fixture in
# test_warehouse_collision.py picks the right section.

synthetic:
  clean_pass_through:
    waypoints:
      - [0.0, 1.5, 1.5]
      - [1.5, 1.5, 1.5]    # passes through Gate_01 center
      - [3.0, 1.5, 1.5]
    expected_gate_passages: [Gate_01]
    expected_warehouse_contacts: 0

  wall_collision:
    waypoints:
      - [0.5, 1.5, 1.5]
      - [0.55, 1.5, 1.5]    # walks straight into the wall at x≈0.6
      - [0.65, 1.5, 1.5]
    expected_warehouse_contacts_min: 1
    expected_first_contact_within_m_of: [0.6, 1.5, 1.5]

  gate_rim_collision:
    waypoints:
      - [0.0, 1.5, 2.4]
      - [1.5, 1.5, 2.35]   # clips top of gate ring (z≈2.35 = 1.5 + 0.85)
      - [3.0, 1.5, 2.35]
    expected_gate_rim_contacts: [Gate_01]
    expected_warehouse_contacts: 0

# When real assets land, append:
# warehouse_v1:
#   clean_pass_through:
#     waypoints: [...]   # passes through Gate_01..Gate_05 centers
#     ...
```

- [ ] **Step 2: Verify the YAML parses**

```bash
python -c "import yaml; print(list(yaml.safe_load(open('tests/fixtures/warehouse_trajectories.yaml')).keys()))"
```

Expected: `['synthetic']`.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/warehouse_trajectories.yaml
git commit -m "test(warehouse): hand-authored fly-through trajectory fixtures"
```

---

## Task 12: D fly-through test — `clean_pass_through`

**Files:**
- Create: `tests/test_pybullet/test_warehouse_collision.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pybullet/test_warehouse_collision.py`:

```python
"""Fly-through collision tests for the warehouse loader.

Each test uses a kinematic 5 cm sphere ("drone") stepped along
hand-authored waypoints and asserts expected contacts with the
warehouse body and gate bodies.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pytest
import yaml

from sim.pybullet.warehouse_loader import WarehouseScene


@pytest.fixture(scope="module")
def synthetic_assets(tmp_path_factory):
    out = tmp_path_factory.mktemp("assets")
    tsdf = out.parent / "tsdf.npz"
    gates = out.parent / "gates.json"

    n = 24
    sdf = np.ones((n, n, n), dtype=np.float32) * 0.5
    sdf[5:7, 5:20, 5:20] = -0.1   # wall at x ∈ [0.5, 0.7]
    np.savez(tsdf, sdf=sdf, voxel_size=np.float32(0.1), origin=np.zeros(3))

    gates.write_text(json.dumps({
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }))

    subprocess.run(
        [sys.executable, "-m", "scripts.warehouse_to_urdf",
         "--tsdf", str(tsdf), "--gates", str(gates), "--out", str(out),
         "--target-triangles", "200", "--gate-clip-radius", "0.3"],
        check=True,
    )
    return out


@pytest.fixture
def trajectories():
    path = Path("tests/fixtures/warehouse_trajectories.yaml")
    return yaml.safe_load(path.read_text())["synthetic"]


def _spawn_drone(client_id: int, position):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=client_id)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=list(position),
        physicsClientId=client_id,
    )


def _interp_waypoints(waypoints, n_steps_per_segment=20):
    pts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for t in np.linspace(0.0, 1.0, n_steps_per_segment, endpoint=False):
            pts.append((1 - t) * np.array(a) + t * np.array(b))
    pts.append(np.array(waypoints[-1]))
    return pts


def test_clean_pass_through_no_warehouse_contact(synthetic_assets, trajectories):
    cid = p.connect(p.DIRECT)
    try:
        scene = WarehouseScene(asset_dir=synthetic_assets)
        handles = scene.load_into(cid)
        traj = trajectories["clean_pass_through"]
        drone = _spawn_drone(cid, traj["waypoints"][0])

        warehouse_contact_count = 0
        for pt in _interp_waypoints(traj["waypoints"]):
            p.resetBasePositionAndOrientation(
                drone, list(pt), [0, 0, 0, 1], physicsClientId=cid,
            )
            p.performCollisionDetection(physicsClientId=cid)
            for ct in p.getContactPoints(drone, handles.warehouse_body_id, physicsClientId=cid):
                warehouse_contact_count += 1

        assert warehouse_contact_count == traj["expected_warehouse_contacts"]
    finally:
        p.disconnect(cid)
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pytest tests/test_pybullet/test_warehouse_collision.py::test_clean_pass_through_no_warehouse_contact -v
```

Expected: 1 passed.

(This test asserts existing behavior should already work — there's no implementation step. If it fails, the synthetic asset clipping or loader is wrong.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_pybullet/test_warehouse_collision.py
git commit -m "test(warehouse): clean_pass_through fly-through assertion"
```

---

## Task 13: D fly-through test — `wall_collision` and `gate_rim_collision`

**Files:**
- Modify: `tests/test_pybullet/test_warehouse_collision.py`

- [ ] **Step 1: Append the two new tests**

Append to `tests/test_pybullet/test_warehouse_collision.py`:

```python
def test_wall_collision_detected(synthetic_assets, trajectories):
    cid = p.connect(p.DIRECT)
    try:
        scene = WarehouseScene(asset_dir=synthetic_assets)
        handles = scene.load_into(cid)
        traj = trajectories["wall_collision"]
        drone = _spawn_drone(cid, traj["waypoints"][0])

        contacts = []
        for pt in _interp_waypoints(traj["waypoints"]):
            p.resetBasePositionAndOrientation(
                drone, list(pt), [0, 0, 0, 1], physicsClientId=cid,
            )
            p.performCollisionDetection(physicsClientId=cid)
            for ct in p.getContactPoints(drone, handles.warehouse_body_id, physicsClientId=cid):
                contacts.append(ct)

        assert len(contacts) >= traj["expected_warehouse_contacts_min"]
        first_contact_pos = np.array(contacts[0][5])
        expected = np.array(traj["expected_first_contact_within_m_of"])
        assert np.linalg.norm(first_contact_pos - expected) < 0.10
    finally:
        p.disconnect(cid)


def test_gate_rim_collision_detected(synthetic_assets, trajectories):
    cid = p.connect(p.DIRECT)
    try:
        scene = WarehouseScene(asset_dir=synthetic_assets)
        handles = scene.load_into(cid)
        traj = trajectories["gate_rim_collision"]
        drone = _spawn_drone(cid, traj["waypoints"][0])

        warehouse_contacts = 0
        gate_contacts: dict[str, int] = {n: 0 for n in handles.gate_names}
        for pt in _interp_waypoints(traj["waypoints"]):
            p.resetBasePositionAndOrientation(
                drone, list(pt), [0, 0, 0, 1], physicsClientId=cid,
            )
            p.performCollisionDetection(physicsClientId=cid)
            warehouse_contacts += len(
                p.getContactPoints(drone, handles.warehouse_body_id, physicsClientId=cid)
            )
            for name, gid in zip(handles.gate_names, handles.gate_body_ids):
                gate_contacts[name] += len(
                    p.getContactPoints(drone, gid, physicsClientId=cid)
                )

        assert warehouse_contacts == traj["expected_warehouse_contacts"]
        for expected_name in traj["expected_gate_rim_contacts"]:
            assert gate_contacts[expected_name] >= 1, gate_contacts
```

- [ ] **Step 2: Run all 3 tests**

```bash
pytest tests/test_pybullet/test_warehouse_collision.py -v
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_pybullet/test_warehouse_collision.py
git commit -m "test(warehouse): wall + gate_rim collision assertions"
```

---

## Task 14: AirSim fixture capture script (Linux-only, ship-and-document)

**Files:**
- Create: `scripts/capture_airsim_fixtures.py`

- [ ] **Step 1: Author the script**

This script runs **only on a Linux machine with the COR-92 binary running**. It cannot be tested on Windows. Ship it with clear documentation in the header.

Create `scripts/capture_airsim_fixtures.py`:

```python
"""Capture AirSim fixture renders + depth + intrinsics from the warehouse.

REQUIREMENTS (per COR-92):
  - Ubuntu 20.04 / 22.04, NVIDIA driver 545+, Vulkan 1.3
  - Cosys-AirSim Python client installed:
      git clone https://github.com/Cosys-Lab/Cosys-AirSim.git
      cd Cosys-AirSim/PythonClient && pip install -e .
      (imports as `cosysairsim`, NOT the Microsoft `airsim` package)
  - The warehouse binary running:
      DISPLAY=:0 ~/warehouse-packaged/Linux/Warehouse.sh -windowed \\
        -ResX=1280 -ResY=720 &
      sleep 25

  Output (this script's responsibility):
      tests/fixtures/airsim_warehouse_v1/
          pose_NN_scene.png
          pose_NN_depth.npy
          pose_NN.json
          manifest.yaml

  Each capture pose is hand-picked in ENU; converted to NED before
  driving the drone. Camera intrinsics are recorded in the JSON sidecar
  so the comparison harness (compare_pybullet_vs_fixtures.py) can match
  the PyBullet projection exactly.

  GOTCHAS (per COR-92):
   - Never call client.reset() — corrupts spawn permanently.
   - Never .join() on async flight calls — deadlocks on collision.
   - Use ImageType.DepthPlanar (planar Z), NOT DepthPerspective.
   - Use ImageResponse.camera_position / camera_orientation, NOT
     simGetVehiclePose() — the latter drifts up to 0.43 m.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

import cosysairsim as airsim
from sim.pybullet.coords import ned_to_enu_position  # for the inverse


# Hand-picked ENU camera poses giving warehouse coverage.
# Refine after first capture pass.
CAPTURE_POSES_ENU = [
    {"name": "pose_00", "position_enu": [0.0, 0.0, 2.0],  "yaw_deg": 0},
    {"name": "pose_01", "position_enu": [3.0, 0.0, 2.0],  "yaw_deg": 90},
    {"name": "pose_02", "position_enu": [3.0, 3.0, 2.0],  "yaw_deg": 180},
    {"name": "pose_03", "position_enu": [0.0, 3.0, 2.0],  "yaw_deg": 270},
    {"name": "pose_04", "position_enu": [1.5, 1.5, 4.0],  "yaw_deg": 0},
    {"name": "pose_05", "position_enu": [1.5, 1.5, 0.5],  "yaw_deg": 0},
    # Add up to ~15 in total — refine after seeing first pass.
]

WIDTH, HEIGHT = 640, 480
FOV_DEG = 90.0


def enu_to_ned(p_enu):
    return [p_enu[1], p_enu[0], -p_enu[2]]


def main(out_dir: Path = Path("tests/fixtures/airsim_warehouse_v1")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    for pose in CAPTURE_POSES_ENU:
        pos_ned = enu_to_ned(pose["position_enu"])
        # Build orientation quaternion from yaw (NED).
        yaw_rad = np.deg2rad(pose["yaw_deg"])
        cy, sy = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
        q_ned = airsim.Quaternionr(0, 0, sy, cy)  # x, y, z, w

        client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(*pos_ned), q_ned),
            ignore_collision=True,
        )

        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
        ])
        scene_resp, depth_resp = responses
        scene_rgb = np.frombuffer(scene_resp.image_data_uint8, dtype=np.uint8).reshape(
            scene_resp.height, scene_resp.width, 3
        )
        depth_planar = np.array(depth_resp.image_data_float, dtype=np.float32).reshape(
            depth_resp.height, depth_resp.width
        )

        # Use camera_position / camera_orientation from the response.
        cam_pos_ned = [
            scene_resp.camera_position.x_val,
            scene_resp.camera_position.y_val,
            scene_resp.camera_position.z_val,
        ]
        cam_quat_ned_xyzw = [
            scene_resp.camera_orientation.x_val,
            scene_resp.camera_orientation.y_val,
            scene_resp.camera_orientation.z_val,
            scene_resp.camera_orientation.w_val,
        ]

        # Save outputs.
        from PIL import Image
        Image.fromarray(scene_rgb).save(out_dir / f"{pose['name']}_scene.png")
        np.save(out_dir / f"{pose['name']}_depth.npy", depth_planar)

        sidecar = {
            "name": pose["name"],
            "position_ned": cam_pos_ned,
            "position_enu": ned_to_enu_position(np.array(cam_pos_ned)).tolist(),
            "orientation_ned_xyzw": cam_quat_ned_xyzw,
            "intrinsics": {
                "width": scene_resp.width,
                "height": scene_resp.height,
                "fov_deg": FOV_DEG,
            },
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / f"{pose['name']}.json").write_text(json.dumps(sidecar, indent=2))

    manifest = {
        "schema_version": 1,
        "warehouse_binary": "warehouse_5gates_v1.tar.gz",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "num_poses": len(CAPTURE_POSES_ENU),
        "intrinsics": {"width": WIDTH, "height": HEIGHT, "fov_deg": FOV_DEG},
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"Wrote {len(CAPTURE_POSES_ENU)} fixtures to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint-check on dev box (we can't run it, but it should at least parse)**

```bash
python -c "import ast; ast.parse(open('scripts/capture_airsim_fixtures.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/capture_airsim_fixtures.py
git commit -m "feat(warehouse): AirSim fixture capture script (Linux-only)"
```

---

## Task 15: Silhouette IoU function

**Files:**
- Create: `scripts/compare_pybullet_vs_fixtures.py`
- Create: `tests/test_validation/__init__.py`
- Create: `tests/test_validation/test_silhouette_iou.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validation/__init__.py` (empty).

Create `tests/test_validation/test_silhouette_iou.py`:

```python
"""Tests for silhouette IoU computation."""
import numpy as np

from scripts.compare_pybullet_vs_fixtures import silhouette_iou


def test_iou_perfect_match_is_one():
    a = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    assert silhouette_iou(a, a.copy()) == 1.0


def test_iou_disjoint_is_zero():
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[0, 1], [0, 0]], dtype=bool)
    assert silhouette_iou(a, b) == 0.0


def test_iou_half_overlap():
    a = np.array([[1, 1, 0, 0]], dtype=bool)
    b = np.array([[0, 1, 1, 0]], dtype=bool)
    # Intersection = 1 px, Union = 3 px → IoU = 1/3
    assert abs(silhouette_iou(a, b) - 1 / 3) < 1e-9


def test_iou_both_empty_is_one():
    a = np.zeros((2, 2), dtype=bool)
    assert silhouette_iou(a, a.copy()) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_validation/test_silhouette_iou.py -v
```

Expected: ImportError on `scripts.compare_pybullet_vs_fixtures`.

- [ ] **Step 3: Implement the IoU function**

Create `scripts/compare_pybullet_vs_fixtures.py`:

```python
"""B-fixture validation harness: compare PyBullet renders vs AirSim fixtures.

Renders the warehouse + gates in PyBullet at each fixture's camera pose
and intrinsics, builds silhouette masks for both renders, computes
per-pose silhouette IoU, and writes an HTML report.

Pass criteria (per spec): mean IoU ≥ 0.85, no individual pose < 0.70.

Run after fixtures are captured (Task 14 on the Linux box) and after
the build pipeline has produced sim/assets/warehouse_v1/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union for two boolean silhouette masks.

    If both masks are empty, returns 1.0 (degenerate but interpretable).
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validation/test_silhouette_iou.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_pybullet_vs_fixtures.py tests/test_validation/__init__.py tests/test_validation/test_silhouette_iou.py
git commit -m "feat(validation): silhouette IoU function for B-fixture comparison"
```

---

## Task 16: PyBullet render at fixture pose + comparison flow

**Files:**
- Modify: `scripts/compare_pybullet_vs_fixtures.py`
- Modify: `tests/test_validation/test_silhouette_iou.py` (add render test)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_validation/test_silhouette_iou.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pybullet as p
import pytest

from scripts.compare_pybullet_vs_fixtures import (
    render_pybullet_at_pose,
    pybullet_silhouette,
    airsim_depth_silhouette,
)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    out = tmp_path_factory.mktemp("assets")
    tsdf = out.parent / "tsdf.npz"
    gates = out.parent / "gates.json"

    n = 24
    sdf = np.ones((n, n, n), dtype=np.float32) * 0.5
    sdf[5:7, 5:20, 5:20] = -0.1
    np.savez(tsdf, sdf=sdf, voxel_size=np.float32(0.1), origin=np.zeros(3))

    gates.write_text(json.dumps({
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }))

    subprocess.run(
        [sys.executable, "-m", "scripts.warehouse_to_urdf",
         "--tsdf", str(tsdf), "--gates", str(gates), "--out", str(out),
         "--target-triangles", "200", "--gate-clip-radius", "0.3"],
        check=True,
    )
    return out


def test_pybullet_render_returns_segmask(assets):
    cid = p.connect(p.DIRECT)
    try:
        from sim.pybullet.warehouse_loader import WarehouseScene
        WarehouseScene(asset_dir=assets).load_into(cid)
        rgb, depth, segmask = render_pybullet_at_pose(
            cid,
            position_enu=[0.0, 1.5, 1.5],
            orientation_enu_xyzw=[0, 0, 0, 1],
            width=64, height=64, fov_deg=90.0,
        )
        assert rgb.shape == (64, 64, 4)  # PyBullet returns RGBA
        assert depth.shape == (64, 64)
        assert segmask.shape == (64, 64)
        # We're looking at a wall — should see something.
        sil = pybullet_silhouette(segmask)
        assert sil.any()
    finally:
        p.disconnect(cid)


def test_airsim_depth_silhouette_thresholds_correctly():
    depth = np.array([[0.5, 50.0, 0.0], [10.0, 1000.0, 5.0]])
    sil = airsim_depth_silhouette(depth, far_plane=100.0)
    # 0 (no return) and >far_plane both → False; everything else → True.
    expected = np.array([[True, True, False], [True, False, True]])
    np.testing.assert_array_equal(sil, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_validation/test_silhouette_iou.py -v
```

Expected: ImportError on the new symbols.

- [ ] **Step 3: Implement render + silhouette helpers**

Append to `scripts/compare_pybullet_vs_fixtures.py`:

```python
import pybullet as p


def render_pybullet_at_pose(
    client_id: int,
    *,
    position_enu,
    orientation_enu_xyzw,
    width: int,
    height: int,
    fov_deg: float,
    near: float = 0.05,
    far: float = 100.0,
):
    """Render an image from PyBullet at the given camera pose.

    Returns (rgb, depth, segmask). The orientation is the camera's
    world-frame rotation; the camera looks down its local +X axis by
    convention here (matches AirSim camera convention).
    """
    qx, qy, qz, qw = orientation_enu_xyzw
    rot = np.array(p.getMatrixFromQuaternion([qx, qy, qz, qw])).reshape(3, 3)
    forward = rot @ np.array([1.0, 0.0, 0.0])
    up = rot @ np.array([0.0, 0.0, 1.0])
    eye = np.array(position_enu, dtype=np.float64)
    target = eye + forward

    view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(
        fov=fov_deg, aspect=width / height, nearVal=near, farVal=far,
    )
    _w, _h, rgb, depth, segmask = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client_id,
    )
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(height, width, 4)
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
    segmask = np.asarray(segmask, dtype=np.int32).reshape(height, width)
    return rgb, depth, segmask


def pybullet_silhouette(segmask: np.ndarray) -> np.ndarray:
    """Anything not -1 (background) is geometry."""
    return segmask != -1


def airsim_depth_silhouette(depth: np.ndarray, far_plane: float) -> np.ndarray:
    """AirSim DepthPlanar returns 0 for no-return and very large for sky.

    Treat any depth in (0, far_plane) as geometry.
    """
    return (depth > 0.0) & (depth < far_plane)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validation/test_silhouette_iou.py -v
```

Expected: 6 passed (4 from Task 15 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_pybullet_vs_fixtures.py tests/test_validation/test_silhouette_iou.py
git commit -m "feat(validation): PyBullet render at fixture pose + silhouette extraction"
```

---

## Task 17: HTML report + comparison CLI

**Files:**
- Modify: `scripts/compare_pybullet_vs_fixtures.py`
- Create: `tests/test_validation/test_html_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_validation/test_html_report.py`:

```python
"""Tests for HTML validation report generation."""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.compare_pybullet_vs_fixtures import write_html_report


def test_html_report_contains_per_pose_rows_and_aggregate(tmp_path: Path):
    rows = []
    for i in range(3):
        rgb_air = (np.ones((10, 10, 3)) * 50 * i).astype(np.uint8)
        rgb_pyb = (np.ones((10, 10, 3)) * 80 * i).astype(np.uint8)
        air_path = tmp_path / f"a_{i}.png"
        pyb_path = tmp_path / f"b_{i}.png"
        Image.fromarray(rgb_air).save(air_path)
        Image.fromarray(rgb_pyb).save(pyb_path)
        rows.append({
            "name": f"pose_{i:02d}",
            "airsim_png": air_path.name,
            "pybullet_png": pyb_path.name,
            "iou": 0.9 - i * 0.1,
        })
    report_path = tmp_path / "index.html"
    write_html_report(report_path, rows, mean_iou=0.8)
    html = report_path.read_text()
    assert "pose_00" in html
    assert "pose_02" in html
    assert "0.9" in html
    assert "Mean IoU" in html
    assert "0.8" in html
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_validation/test_html_report.py -v
```

Expected: ImportError on `write_html_report`.

- [ ] **Step 3: Implement HTML report + CLI orchestration**

Append to `scripts/compare_pybullet_vs_fixtures.py`:

```python
import argparse
import json as _json
from datetime import datetime, timezone

from sim.pybullet.coords import ned_to_enu_quaternion


def convert_airsim_camera_orientation_to_pybullet(q_ned_xyzw):
    """Convert an AirSim camera orientation quaternion (NED, xyzw) to a
    PyBullet world-frame orientation quaternion (ENU, xyzw).

    AirSim camera convention: body frame is X-forward, Y-right, Z-down,
    expressed in world-NED. PyBullet's camera is built from a forward
    vector derived from this rotation, so the rotation itself just
    needs the NED→ENU basis change applied.
    """
    qx, qy, qz, qw = q_ned_xyzw
    # Reorder xyzw → wxyz, apply NED→ENU flip, reorder back to xyzw.
    q_enu_wxyz = ned_to_enu_quaternion(np.array([qw, qx, qy, qz]))
    return [q_enu_wxyz[1], q_enu_wxyz[2], q_enu_wxyz[3], q_enu_wxyz[0]]


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Warehouse validation</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; }}
  td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
  img {{ max-width: 320px; display: block; }}
  .iou-pass {{ background: #d4edda; }}
  .iou-fail {{ background: #f8d7da; }}
</style></head><body>
<h1>Warehouse v1 — PyBullet vs AirSim fixture comparison</h1>
<p><b>Mean IoU:</b> {mean_iou:.3f} &nbsp; <b>Generated:</b> {ts}</p>
<table>
<tr><th>Pose</th><th>AirSim</th><th>PyBullet</th><th>Silhouette IoU</th></tr>
{rows}
</table>
</body></html>
"""

_ROW_TEMPLATE = """<tr class="{cls}">
<td>{name}</td>
<td><img src="{airsim_png}"></td>
<td><img src="{pybullet_png}"></td>
<td>{iou:.3f}</td>
</tr>"""


def write_html_report(path: Path, rows: list[dict], mean_iou: float) -> None:
    body_rows = "\n".join(
        _ROW_TEMPLATE.format(
            cls="iou-pass" if r["iou"] >= 0.70 else "iou-fail",
            name=r["name"],
            airsim_png=r["airsim_png"],
            pybullet_png=r["pybullet_png"],
            iou=r["iou"],
        )
        for r in rows
    )
    html = _HTML_TEMPLATE.format(
        mean_iou=mean_iou,
        ts=datetime.now(timezone.utc).isoformat(),
        rows=body_rows,
    )
    Path(path).write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=Path("sim/assets/warehouse_v1"))
    ap.add_argument("--fixtures", type=Path,
                    default=Path("tests/fixtures/airsim_warehouse_v1"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/validation/warehouse_v1") /
                            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--far-plane", type=float, default=100.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from sim.pybullet.warehouse_loader import WarehouseScene

    cid = p.connect(p.DIRECT)
    try:
        WarehouseScene(asset_dir=args.assets).load_into(cid)

        rows = []
        ious = []
        for sidecar_path in sorted(args.fixtures.glob("pose_*.json")):
            sidecar = _json.loads(sidecar_path.read_text())
            name = sidecar["name"]
            depth = np.load(args.fixtures / f"{name}_depth.npy")

            rgb, _pdepth, segmask = render_pybullet_at_pose(
                cid,
                position_enu=sidecar["position_enu"],
                orientation_enu_xyzw=convert_airsim_camera_orientation_to_pybullet(
                    sidecar["orientation_ned_xyzw"]
                ),
                width=sidecar["intrinsics"]["width"],
                height=sidecar["intrinsics"]["height"],
                fov_deg=sidecar["intrinsics"]["fov_deg"],
            )
            air_sil = airsim_depth_silhouette(depth, far_plane=args.far_plane)
            pyb_sil = pybullet_silhouette(segmask)
            iou = silhouette_iou(air_sil, pyb_sil)
            ious.append(iou)

            airsim_dst = args.out / f"{name}_airsim.png"
            pybullet_dst = args.out / f"{name}_pybullet.png"
            Image.open(args.fixtures / f"{name}_scene.png").save(airsim_dst)
            Image.fromarray(rgb[..., :3]).save(pybullet_dst)
            rows.append({
                "name": name,
                "airsim_png": airsim_dst.name,
                "pybullet_png": pybullet_dst.name,
                "iou": iou,
            })

        mean_iou = float(np.mean(ious)) if ious else 0.0
        write_html_report(args.out / "index.html", rows, mean_iou=mean_iou)
        print(f"Wrote {len(rows)} comparisons. Mean IoU: {mean_iou:.3f}")
        print(f"Report: {args.out / 'index.html'}")
        if mean_iou < 0.85 or any(r["iou"] < 0.70 for r in rows):
            print("FAIL: validation thresholds not met.")
            return 1
        return 0
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    raise SystemExit(main())
```

> Note on `convert_airsim_camera_orientation_to_pybullet`: the implementation above does a basis-change flip but assumes AirSim's body-frame camera convention (X-forward, Y-right, Z-down) matches what `render_pybullet_at_pose` expects from a forward+up reconstruction. If the first real validation run produces uniformly low IoU values that look like rotated-but-correctly-positioned silhouettes, the body-frame convention needs an additional `R_body_correction` term — calibrate empirically in T18 against a known-pose fixture.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validation/ -v
```

Expected: 7 passed (4 + 2 + 1).

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_pybullet_vs_fixtures.py tests/test_validation/test_html_report.py
git commit -m "feat(validation): HTML report generator + comparison CLI"
```

---

## Task 18: End-to-end real run + Linear update (gated on artifact arrival)

This task only runs once **(a)** the TSDF artifact lands from Janahan and **(b)** AirSim fixtures have been captured on the Linux box (Task 14 is a one-time external execution).

**Files:**
- Generate: `sim/assets/warehouse_v1/*` (committed, possibly via LFS)
- Generate: `outputs/validation/warehouse_v1/<ts>/index.html` (NOT committed)

- [ ] **Step 1: Place the TSDF artifact and (optionally) re-confirm gate JSON freshness**

```bash
# Save the TSDF artifact wherever Janahan delivers it; example:
mkdir -p artifacts/warehouse_v1
cp ~/Downloads/warehouse_tsdf.npz artifacts/warehouse_v1/
ls -la artifacts/warehouse_v1/
```

- [ ] **Step 2: If the artifact is a format the loader doesn't yet handle, add an adapter**

Run a probe:

```bash
python -c "from scripts.warehouse_to_urdf.tsdf import load_tsdf; load_tsdf('artifacts/warehouse_v1/warehouse_tsdf.npz')"
```

If it errors with "Unsupported TSDF format" or `_load_bin` `NotImplementedError`, **stop and add the adapter** to `scripts/warehouse_to_urdf/tsdf.py`. Add a focused unit test in `tests/test_warehouse_to_urdf/test_tsdf.py` that exercises the new adapter on a small synthetic example. Commit.

- [ ] **Step 3: Build assets**

```bash
mkdir -p sim/assets/warehouse_v1
python -m scripts.warehouse_to_urdf \
    --tsdf artifacts/warehouse_v1/warehouse_tsdf.npz \
    --gates configs/warehouse/warehouse_5gates_v1_gates_ned.json \
    --out sim/assets/warehouse_v1/
ls -la sim/assets/warehouse_v1/
cat sim/assets/warehouse_v1/manifest.yaml
```

Expected: 6 files (warehouse.urdf/.obj, gate.urdf, gate_ring.obj, gates_enu.json, manifest.yaml). If `warehouse.obj` exceeds 10 MB, configure git-lfs:

```bash
git lfs install
git lfs track "sim/assets/warehouse_v1/*.obj"
git add .gitattributes
```

- [ ] **Step 4: Author real-warehouse trajectory waypoints**

Append a `warehouse_v1` block to `tests/fixtures/warehouse_trajectories.yaml` mirroring the `synthetic` block structure but with real ENU waypoints derived from `sim/assets/warehouse_v1/gates_enu.json`. Update `tests/test_pybullet/test_warehouse_collision.py` to parametrize on (`synthetic`, `warehouse_v1`) so both run.

- [ ] **Step 5: Run all tests**

```bash
pytest tests/test_pybullet/ tests/test_warehouse_to_urdf/ tests/test_validation/ -v
```

Expected: all green. The new `warehouse_v1` parametrization should pass against the real assets.

- [ ] **Step 6: Capture AirSim fixtures (one-time, on Linux box)**

Hand-off step — Alex runs this on the Linux machine. If WSL2 doesn't work for the warehouse binary or no Linux box is currently available, this step blocks and the spec's "B-fixture portion subject to change" note is invoked: revisit using a Windows build of the binary (Task 14's prerequisite changes).

```bash
# On the Linux machine, with the warehouse binary running:
python -m scripts.capture_airsim_fixtures
ls -la tests/fixtures/airsim_warehouse_v1/
```

Then `scp` the fixture directory back to the dev machine (or commit from the Linux box if convenient).

- [ ] **Step 7: Run B-fixture comparison**

```bash
python -m scripts.compare_pybullet_vs_fixtures \
    --assets sim/assets/warehouse_v1 \
    --fixtures tests/fixtures/airsim_warehouse_v1
```

Expected: prints `Mean IoU: <value>`. If `≥ 0.85` and no individual pose `< 0.70`, exit code 0. If FAIL, the report HTML at `outputs/validation/warehouse_v1/<ts>/index.html` shows which poses failed; iterate by tuning `--target-triangles`, `--gate-clip-radius`, or fixing the `# TODO: reframe` orientation conversion in `compare_pybullet_vs_fixtures.py`.

- [ ] **Step 8: Commit assets + fixtures**

```bash
git add sim/assets/warehouse_v1/ tests/fixtures/airsim_warehouse_v1/ tests/fixtures/warehouse_trajectories.yaml tests/test_pybullet/test_warehouse_collision.py
git commit -m "feat(warehouse): real warehouse_v1 URDF assets + AirSim fixtures"
```

- [ ] **Step 9: Run `/ship-it` to generate completion report and sync to Linear**

```bash
# In Claude Code:
/ship-it
```

Or attach the validation HTML manually to the Linear issue under COR-92.

---

## Self-Review Notes

- **Spec coverage:** all spec sections have a corresponding task — adaptive TSDF loader (T3), gate clipping (T4), MC + simplification (T5), torus + NED→ENU (T6), URDF + gates_enu writers (T7), manifest with hashes (T8), CLI orchestration (T9), runtime loader (T10), trajectory fixtures (T11), three D fly-through tests (T12-T13), AirSim capture (T14), silhouette IoU (T15), render harness (T16), HTML report + CLI (T17), end-to-end real run (T18).
- **Synthetic-data testability:** every task except T14 and T18 runs end-to-end on the dev machine with no external artifacts.
- **Open risks documented in T17:** the body-frame correction inside `convert_airsim_camera_orientation_to_pybullet` may need an additional rotation term, calibrated against real fixtures in T18.
- **Frequent commits:** every task ends with a focused commit; ~18 commits for ~18 well-defined units of work.
- **No placeholders:** every step shows the actual code or command.
