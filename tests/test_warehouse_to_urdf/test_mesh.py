"""Tests for mesh operations: clipping, marching cubes, torus."""
import numpy as np
import pytest
import open3d as o3d

from scripts.warehouse_to_urdf.mesh import (
    clip_gates_in_sdf,
    clip_mesh_to_bbox,
    compute_default_clip_bbox,
    extract_mesh_from_sdf,
    flip_mesh_ned_to_enu,
    make_torus_mesh,
    simplify_mesh,
    ue_to_ned_mesh,
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


# ---------------------------------------------------------------------------
# ue_to_ned_mesh
# ---------------------------------------------------------------------------


def _single_triangle_ue(v0_cm, v1_cm, v2_cm):
    """Build a minimal Open3D mesh with one triangle from three UE-frame vertices."""
    verts = np.array([v0_cm, v1_cm, v2_cm], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def test_ue_to_ned_mesh_single_vertex_origin():
    """PlayerStart at origin, vertex (100, 200, 300) cm → NED (1, -2, -3) m."""
    mesh = _single_triangle_ue([100.0, 200.0, 300.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])
    playerstart = np.array([0.0, 0.0, 0.0])
    out = ue_to_ned_mesh(mesh, playerstart)
    verts_out = np.asarray(out.vertices)
    # Vertex 0 is the one we care about.
    np.testing.assert_array_almost_equal(
        verts_out[0], [1.0, -2.0, -3.0], decimal=6
    )


def test_ue_to_ned_mesh_playerstart_shift():
    """PlayerStart (7580, 470, 142) cm, vertex (7570, 270, 150) cm → NED (-0.10, 2.00, -0.08) m."""
    playerstart = np.array([7580.0, 470.0, 142.0])
    # Build a triangle where vertex 0 is the gate-01 hand-computed value.
    mesh = _single_triangle_ue(
        [7570.0, 270.0, 150.0],
        [7580.0, 270.0, 150.0],
        [7570.0, 280.0, 150.0],
    )
    out = ue_to_ned_mesh(mesh, playerstart)
    verts_out = np.asarray(out.vertices)
    np.testing.assert_array_almost_equal(
        verts_out[0], [-0.10, 2.00, -0.08], decimal=6
    )


def test_ue_to_ned_mesh_reverses_winding():
    """Face [a, b, c] in UE should become [a, c, b] in NED output."""
    mesh = _single_triangle_ue(
        [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]
    )
    playerstart = np.array([0.0, 0.0, 0.0])
    out = ue_to_ned_mesh(mesh, playerstart)
    faces_out = np.asarray(out.triangles)
    assert faces_out.shape == (1, 3)
    # Original face was [0, 1, 2]; winding-reversed should be [0, 2, 1].
    np.testing.assert_array_equal(faces_out[0], [0, 2, 1])


def test_ue_to_ned_mesh_recomputes_normals():
    """Returned mesh must have finite vertex normals (compute_vertex_normals called)."""
    mesh = _single_triangle_ue(
        [0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]
    )
    playerstart = np.array([0.0, 0.0, 0.0])
    out = ue_to_ned_mesh(mesh, playerstart)
    normals = np.asarray(out.vertex_normals)
    assert len(normals) == len(np.asarray(out.vertices))
    assert np.all(np.isfinite(normals[0]))


# ---------------------------------------------------------------------------
# clip_mesh_to_bbox
# ---------------------------------------------------------------------------


def _mesh_with_triangles(verts, faces):
    """Build an Open3D TriangleMesh from numpy arrays."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    return mesh


def test_clip_keeps_interior_triangles():
    """3 fully-inside, 3 straddling, 4 outside → result has exactly 3 triangles."""
    # Place vertices at known positions:
    #   v0=(0.5,0.5,0.5)   — inside bbox [0,1]^3
    #   v1=(0.6,0.5,0.5)   — inside
    #   v2=(0.5,0.6,0.5)   — inside
    #   v3=(1.5,0.5,0.5)   — outside (x > 1)
    #   v4=(0.5,1.5,0.5)   — outside (y > 1)
    #   v5=(0.5,0.5,1.5)   — outside (z > 1)
    #   v6=(-0.5,0.5,0.5)  — outside (x < 0)
    verts = [
        [0.5, 0.5, 0.5],  # 0 inside
        [0.6, 0.5, 0.5],  # 1 inside
        [0.5, 0.6, 0.5],  # 2 inside
        [1.5, 0.5, 0.5],  # 3 outside
        [0.5, 1.5, 0.5],  # 4 outside
        [0.5, 0.5, 1.5],  # 5 outside
        [-0.5, 0.5, 0.5], # 6 outside
    ]
    # 3 fully inside: all verts from {0,1,2}
    # 3 straddling: each mixes one inside + one outside vertex
    # 4 outside: remaining combinations among outside verts
    faces = [
        [0, 1, 2],  # fully inside
        [0, 1, 2],  # fully inside (duplicate for count)
        [0, 1, 2],  # fully inside (duplicate for count)
        [0, 1, 3],  # straddling (v3 outside)
        [0, 2, 4],  # straddling (v4 outside)
        [1, 2, 5],  # straddling (v5 outside)
        [3, 4, 5],  # fully outside
        [3, 4, 6],  # fully outside
        [3, 5, 6],  # fully outside
        [4, 5, 6],  # fully outside
    ]
    mesh = _mesh_with_triangles(verts, faces)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([1.0, 1.0, 1.0])
    clipped = clip_mesh_to_bbox(mesh, bbox_min, bbox_max)
    assert len(clipped.triangles) == 3


def test_clip_boundary_inclusive():
    """Triangle with a vertex exactly on the bbox boundary is kept."""
    verts = [
        [0.0, 0.0, 0.0],  # exactly on boundary
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],  # exactly on boundary
    ]
    faces = [[0, 1, 2]]
    mesh = _mesh_with_triangles(verts, faces)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([1.0, 1.0, 1.0])
    clipped = clip_mesh_to_bbox(mesh, bbox_min, bbox_max)
    assert len(clipped.triangles) == 1


def test_clip_empty_output_is_empty_mesh():
    """No triangles survive → return an empty mesh (no error)."""
    verts = [
        [5.0, 5.0, 5.0],
        [6.0, 5.0, 5.0],
        [5.0, 6.0, 5.0],
    ]
    faces = [[0, 1, 2]]
    mesh = _mesh_with_triangles(verts, faces)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([1.0, 1.0, 1.0])
    clipped = clip_mesh_to_bbox(mesh, bbox_min, bbox_max)
    assert len(clipped.triangles) == 0


def test_clip_prunes_unreferenced_vertices():
    """Vertices not referenced by any surviving triangle must be dropped.

    Prevents output .obj bloat and keeps PyBullet's broad-phase AABB
    bounded to the clipped region, not the pre-clip extent.
    """
    verts = [
        [0.5, 0.5, 0.5],   # 0: inside, used by surviving face
        [0.6, 0.5, 0.5],   # 1: inside, used by surviving face
        [0.5, 0.6, 0.5],   # 2: inside, used by surviving face
        [50.0, 50.0, 50.0],  # 3: far outside, orphan after clip
    ]
    faces = [
        [0, 1, 2],         # survives
        [0, 1, 3],         # dropped (v3 outside)
    ]
    mesh = _mesh_with_triangles(verts, faces)
    bbox_min = np.array([0.0, 0.0, 0.0])
    bbox_max = np.array([1.0, 1.0, 1.0])
    clipped = clip_mesh_to_bbox(mesh, bbox_min, bbox_max)
    assert len(clipped.triangles) == 1
    # v3 should no longer be in the output vertex buffer.
    assert len(clipped.vertices) == 3


# ---------------------------------------------------------------------------
# compute_default_clip_bbox
# ---------------------------------------------------------------------------


def test_default_bbox_encloses_all_gates():
    """All gate positions must be strictly inside the returned bbox."""
    gates_ned = np.array([
        [-0.10,  2.00, -0.08],
        [-2.50,  8.40,  0.22],
        [-9.00,  8.40, -0.28],
        [-8.70,  5.00, -0.28],
        [-7.50,  0.30,  0.22],
    ])
    bbox_min, bbox_max = compute_default_clip_bbox(gates_ned, margin_m=5.0)
    assert np.all(gates_ned >= bbox_min)
    assert np.all(gates_ned <= bbox_max)


def test_default_bbox_margin_applied():
    """bbox_min = gates.min(axis=0) - margin, bbox_max = gates.max(axis=0) + margin."""
    gates_ned = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    margin = 2.5
    bbox_min, bbox_max = compute_default_clip_bbox(gates_ned, margin_m=margin)
    expected_min = np.array([1.0, 2.0, 3.0]) - margin
    expected_max = np.array([4.0, 5.0, 6.0]) + margin
    np.testing.assert_array_almost_equal(bbox_min, expected_min)
    np.testing.assert_array_almost_equal(bbox_max, expected_max)


def test_default_bbox_empty_raises_clear_error():
    """Empty gate array raises ValueError with a clear message (not a cryptic numpy error)."""
    empty = np.zeros((0, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        compute_default_clip_bbox(empty)
