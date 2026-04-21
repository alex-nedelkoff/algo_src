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
    sdf[5:25, 5:7, 5:25] = -0.1
    np.savez(tsdf, sdf=sdf, voxel_size=np.float32(0.1), origin=np.array([0.0, 0.0, -2.0], dtype=np.float32))

    gates.write_text(json.dumps({
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
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
