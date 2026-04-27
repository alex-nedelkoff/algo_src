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
    """Uniform-grid simplify via vertex clustering; abort if result is too small.

    We use vertex clustering (not quadric decimation) because the latter is
    greedy about preserving curvature: on Megascans-style scenes it spends
    nearly the entire triangle budget on a single high-poly prop (rack, box
    stack) and crushes flat walls / floors to a handful of triangles. Vertex
    clustering by a voxel grid spreads the budget uniformly across the mesh.

    `target_triangles` is treated as a soft target — we pick a voxel size
    that approximately hits it via a bbox-volume heuristic, then iterate
    once if we overshoot or undershoot by a large factor.
    """
    import open3d as o3d
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.max_bound - aabb.min_bound
    bbox_vol = float(extent[0] * extent[1] * extent[2])
    # Heuristic: voxel_size = (bbox_vol / target_triangles)^(1/3) * tuning_factor.
    # Each "occupied" voxel produces ~2 triangles in clustering output.
    voxel_size = (bbox_vol / max(target_triangles, 1)) ** (1.0 / 3.0) * 1.0
    simplified = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )
    if len(simplified.triangles) < 100:
        raise ValueError(
            f"Simplified mesh has too few triangles "
            f"({len(simplified.triangles)} < 100). Mesh likely empty or "
            f"voxel size {voxel_size:.3f} too coarse for bbox {extent}."
        )
    simplified.compute_vertex_normals()
    return simplified


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
    """Return a copy of the mesh with vertices converted NED → PyBullet-world.

    Combined transform: standard NED→ENU `(a, b, c) → (b, a, -c)` followed
    by Ry(-90) `(x, y, z) → (-z, y, x)`, which brings the mesh's "up" axis
    onto PyBullet world +Z. Net: `(a, b, c) → (c, a, b)`.

    Must match `sim/pybullet/coords.py::ned_to_enu_position` so warehouse
    mesh vertices and gate positions stay in the same world frame.
    """
    import open3d as o3d

    verts = np.asarray(mesh.vertices)
    flipped = np.empty_like(verts)
    flipped[:, 0] = verts[:, 2]    # world X = NED.z
    flipped[:, 1] = verts[:, 0]    # world Y = NED.x
    flipped[:, 2] = verts[:, 1]    # world Z = NED.y

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(flipped)
    out.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    out.compute_vertex_normals()
    return out


def ue_to_ned_mesh(mesh, playerstart_ue_cm: np.ndarray):
    """Convert a triangle mesh from UE world frame to NED frame.

    Applies the same basis change as ``sim/pybullet/coords.py::ue_to_ned_position``
    but vertex-wise on the mesh:

        rel_ned_m = (v_ue_cm[0] - ps[0],
                     v_ue_cm[1] - ps[1],
                     -(v_ue_cm[2] - ps[2])) / 100.0

    Only the Z axis is negated — that's the true handedness flip from
    UE left-handed Z-up to NED right-handed Z-down. Triangle winding is
    reversed (``[a, b, c]`` → ``[a, c, b]``) to preserve outward-facing
    normals across the handedness flip. Vertex normals are recomputed.

    Parameters
    ----------
    mesh:
        Open3D ``TriangleMesh`` in UE world coordinates (cm, left-handed,
        Z-up).
    playerstart_ue_cm:
        1-D array of shape (3,) — the UE world position of the PlayerStart
        actor in cm. This becomes the NED origin.

    Returns
    -------
    Open3D ``TriangleMesh`` in NED frame (m, right-handed, Z-down).
    """
    import open3d as o3d

    ps = np.asarray(playerstart_ue_cm, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    rel = verts - ps  # (N, 3)
    ned = np.empty_like(rel)
    ned[:, 0] = rel[:, 0]
    ned[:, 1] = rel[:, 1]      # true handedness flip: only Z is negated
    ned[:, 2] = -rel[:, 2]
    ned /= 100.0

    # Reverse winding to restore correct outward normals after handedness flip.
    faces = np.asarray(mesh.triangles, dtype=np.int32).copy()
    faces[:, [1, 2]] = faces[:, [2, 1]]

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(ned)
    out.triangles = o3d.utility.Vector3iVector(faces)
    out.compute_vertex_normals()
    return out


def clip_mesh_to_bbox(mesh, bbox_min: np.ndarray, bbox_max: np.ndarray):
    """Return a new mesh containing only triangles fully inside an AABB.

    Conservative: a triangle is kept only when **all three** of its vertices
    satisfy ``bbox_min <= v <= bbox_max`` on every axis. Triangles that
    straddle the boundary are dropped. Boundaries are inclusive (a vertex
    exactly on the boundary counts as inside).

    The input mesh is not mutated.

    Parameters
    ----------
    mesh:
        Open3D ``TriangleMesh``.
    bbox_min, bbox_max:
        1-D arrays of shape (3,) defining the axis-aligned bounding box in
        the same coordinate frame as the mesh vertices.

    Returns
    -------
    A new Open3D ``TriangleMesh``. May be empty (0 triangles) if no
    triangle survives the clip.
    """
    import open3d as o3d

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    bmin = np.asarray(bbox_min, dtype=np.float64)
    bmax = np.asarray(bbox_max, dtype=np.float64)

    # inside[i] is True when vertex i is inside the bbox on all axes.
    inside = np.all((verts >= bmin) & (verts <= bmax), axis=1)

    # Keep only faces where every vertex index maps to an inside vertex.
    if len(faces) > 0:
        keep = inside[faces[:, 0]] & inside[faces[:, 1]] & inside[faces[:, 2]]
        kept_faces = faces[keep]
    else:
        kept_faces = faces

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts)
    out.triangles = o3d.utility.Vector3iVector(kept_faces)
    # Drop vertices that no surviving triangle references — otherwise the
    # output retains the full pre-clip vertex buffer, which bloats the .obj
    # and inflates PyBullet's broad-phase AABB back to the pre-clip extent.
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


def compute_default_clip_bbox(
    gate_positions_ned: np.ndarray,
    margin_m: float = 5.0,
):
    """Compute an axis-aligned bbox that encloses all gate positions with margin.

    Parameters
    ----------
    gate_positions_ned:
        Array of shape (N, 3) with N >= 1 — gate centre positions, in NED
        metres. These must already be in NED (i.e. after ``ue_to_ned_mesh``
        has normalised the warehouse to PlayerStart-origin NED).
    margin_m:
        Extra padding added on every side of the tight gate bbox.

    Returns
    -------
    bbox_min, bbox_max : np.ndarray of shape (3,)
        Lower and upper corners of the padded bbox in NED metres.
    """
    pts = np.asarray(gate_positions_ned, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(
            f"gate_positions_ned must have shape (N, 3) with N >= 1, "
            f"got {pts.shape}."
        )
    bbox_min = pts.min(axis=0) - margin_m
    bbox_max = pts.max(axis=0) + margin_m
    return bbox_min, bbox_max
