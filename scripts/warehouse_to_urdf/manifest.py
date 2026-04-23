"""Provenance manifest writer."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
    # TSDF path (original path, mutually exclusive with mesh_path)
    tsdf_path: Optional[Path] = None,
    # Mesh-direct path (mutually exclusive with tsdf_path)
    mesh_path: Optional[Path] = None,
    # Gates: NED JSON (original) or UE YAML (mesh-direct path)
    gates_path: Optional[Path] = None,
    gates_ue_yaml_path: Optional[Path] = None,
    warehouse_obj_path: Path,
    target_triangles: int,
    gate_clip_radius_m: float = 0.0,
    clip_margin_m: float = 0.0,
    warehouse_obj_triangles: int,
    num_gates: int,
    git_sha: str,
) -> None:
    obj_size = Path(warehouse_obj_path).stat().st_size

    # Build the sources block based on which inputs were provided
    sources: dict = {}
    if tsdf_path is not None:
        sources["tsdf"] = {
            "path": str(tsdf_path),
            "sha256": sha256_of_file(tsdf_path),
        }
    elif mesh_path is not None:
        sources["mesh"] = {
            "path": str(mesh_path),
            "sha256": sha256_of_file(mesh_path),
        }

    if gates_path is not None:
        sources["gates"] = {
            "path": str(gates_path),
            "sha256": sha256_of_file(gates_path),
        }
    elif gates_ue_yaml_path is not None:
        sources["gates_ue_yaml"] = {
            "path": str(gates_ue_yaml_path),
            "sha256": sha256_of_file(gates_ue_yaml_path),
        }

    # Build params — include only the relevant ones for the pipeline used
    build_params: dict = {
        "target_triangles": target_triangles,
        "ned_to_enu_applied": True,
    }
    if tsdf_path is not None:
        build_params["gate_clip_radius_m"] = gate_clip_radius_m
    if mesh_path is not None:
        build_params["clip_margin_m"] = clip_margin_m

    payload = {
        "schema_version": 1,
        "warehouse_version": "v1",
        "build": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "builder": "scripts/warehouse_to_urdf",
            "git_sha": git_sha,
        },
        "sources": sources,
        "build_params": build_params,
        "outputs": {
            "warehouse_obj_triangles": warehouse_obj_triangles,
            "warehouse_obj_size_bytes": obj_size,
            "num_gates": num_gates,
        },
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
