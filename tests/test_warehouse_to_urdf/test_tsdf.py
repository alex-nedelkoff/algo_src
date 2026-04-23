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


def _make_synthetic_obj(tmp_path: Path, filename: str = "synthetic.obj") -> Path:
    """Write a minimal OBJ with 2 triangles (a single quad split into two)."""
    obj_text = (
        "# minimal test mesh — 4 vertices, 2 triangles\n"
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 1.0 1.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "f 1 2 3\n"
        "f 1 3 4\n"
    )
    path = tmp_path / filename
    path.write_text(obj_text)
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


def test_load_obj_returns_mesh_artifact(tmp_path: Path):
    """OBJ input yields a TSDFArtifact with pre_extracted_mesh populated."""
    path = _make_synthetic_obj(tmp_path)
    art = load_tsdf(path)

    assert isinstance(art, TSDFArtifact)
    # No SDF data — pipeline skips marching cubes.
    assert art.sdf.shape == (0, 0, 0)
    assert art.voxel_size == 0.0
    # Mesh must be present with geometry.
    assert art.pre_extracted_mesh is not None
    assert art.pre_extracted_mesh.has_vertices()
    assert art.pre_extracted_mesh.has_triangles()


def test_load_obj_empty_mesh_raises_clear_error(tmp_path: Path):
    """An OBJ with no geometry raises ValueError with a clear message."""
    # An OBJ that declares no vertices/faces — Open3D reads it as empty mesh.
    empty_obj = tmp_path / "empty.obj"
    empty_obj.write_text("# empty obj\n")
    with pytest.raises(ValueError, match="no vertices"):
        load_tsdf(empty_obj)


# NOTE: No separate .fbx fixture test — creating a valid binary FBX by hand
# is impractical. Both .fbx and .obj go through the same _load_mesh() helper
# which calls o3d.io.read_triangle_mesh(), so testing .obj covers the shared
# dispatch logic. The smoke test against the real warehouse.fbx covers the
# actual FBX read path end-to-end (see task instructions).
