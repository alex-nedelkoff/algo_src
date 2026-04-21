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
    # A box "wall" along Y axis (NED frame: x=0.5-2.5, y=0.5-0.7, z=-1.5 to 0.5).
    sdf[5:25, 5:7, 5:25] = -0.1
    # Another box.
    sdf[25:27, 5:25, 5:25] = -0.1
    np.savez(
        path,
        sdf=sdf,
        voxel_size=np.float32(voxel),
        origin=np.array([0.0, 0.0, -2.0], dtype=np.float32),
    )


def _make_one_gate_json(path: Path) -> None:
    data = {
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
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
