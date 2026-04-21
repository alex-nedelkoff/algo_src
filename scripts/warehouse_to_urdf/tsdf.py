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
