"""Provenance manifest writer."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    path: Path,
    *,
    tsdf_path: Path,
    gates_path: Path,
    warehouse_obj_path: Path,
    target_triangles: int,
    gate_clip_radius_m: float,
    warehouse_obj_triangles: int,
    num_gates: int,
    git_sha: str,
) -> None:
    obj_size = Path(warehouse_obj_path).stat().st_size
    payload = {
        "schema_version": 1,
        "warehouse_version": "v1",
        "build": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "builder": "scripts/warehouse_to_urdf",
            "git_sha": git_sha,
        },
        "sources": {
            "tsdf": {
                "path": str(tsdf_path),
                "sha256": sha256_of_file(tsdf_path),
            },
            "gates": {
                "path": str(gates_path),
                "sha256": sha256_of_file(gates_path),
            },
        },
        "build_params": {
            "target_triangles": target_triangles,
            "gate_clip_radius_m": gate_clip_radius_m,
            "ned_to_enu_applied": True,
        },
        "outputs": {
            "warehouse_obj_triangles": warehouse_obj_triangles,
            "warehouse_obj_size_bytes": obj_size,
            "num_gates": num_gates,
        },
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
