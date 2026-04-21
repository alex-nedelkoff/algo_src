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
