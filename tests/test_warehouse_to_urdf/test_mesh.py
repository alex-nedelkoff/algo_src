"""Tests for mesh operations: clipping, marching cubes, torus."""
import numpy as np
import pytest
import open3d as o3d

from scripts.warehouse_to_urdf.mesh import (
    clip_gates_in_sdf,
    extract_mesh_from_sdf,
    simplify_mesh,
    make_torus_mesh,
    flip_mesh_ned_to_enu,
)
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
