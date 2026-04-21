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
