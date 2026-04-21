"""Tests for manifest writer."""
import hashlib
from pathlib import Path

import yaml

from scripts.warehouse_to_urdf.manifest import sha256_of_file, write_manifest


def test_sha256_of_file_matches_hashlib(tmp_path: Path):
    data = b"hello world\n"
    p = tmp_path / "a.txt"
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_of_file(p) == expected


def test_write_manifest_records_all_provenance(tmp_path: Path):
    tsdf = tmp_path / "tsdf.npz"
    gates = tmp_path / "gates.json"
    obj = tmp_path / "warehouse.obj"
    tsdf.write_bytes(b"fake tsdf")
    gates.write_bytes(b"fake gates")
    obj.write_bytes(b"x" * 4242)

    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(
        manifest_path,
        tsdf_path=tsdf,
        gates_path=gates,
        warehouse_obj_path=obj,
        target_triangles=30000,
        gate_clip_radius_m=1.0,
        warehouse_obj_triangles=27654,
        num_gates=5,
        git_sha="abc1234",
    )
    data = yaml.safe_load(manifest_path.read_text())
    assert data["schema_version"] == 1
    assert data["sources"]["tsdf"]["sha256"] == sha256_of_file(tsdf)
    assert data["sources"]["gates"]["sha256"] == sha256_of_file(gates)
    assert data["build_params"]["target_triangles"] == 30000
    assert data["build_params"]["ned_to_enu_applied"] is True
    assert data["outputs"]["warehouse_obj_size_bytes"] == 4242
    assert data["outputs"]["warehouse_obj_triangles"] == 27654
    assert data["outputs"]["num_gates"] == 5
    assert data["build"]["git_sha"] == "abc1234"
