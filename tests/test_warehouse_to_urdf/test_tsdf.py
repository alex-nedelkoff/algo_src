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
