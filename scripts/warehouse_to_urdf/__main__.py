"""CLI entry point: build warehouse + gate URDFs from TSDF + gate JSON."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d

from scripts.warehouse_to_urdf.tsdf import load_tsdf
from scripts.warehouse_to_urdf.mesh import (
    clip_gates_in_sdf,
    extract_mesh_from_sdf,
    simplify_mesh,
    flip_mesh_ned_to_enu,
    make_torus_mesh,
)
from scripts.warehouse_to_urdf.urdf import (
    write_warehouse_urdf,
    write_gate_urdf,
    write_gates_enu_json,
)
from scripts.warehouse_to_urdf.manifest import write_manifest


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsdf", type=Path, required=True)
    ap.add_argument("--gates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-triangles", type=int, default=30000)
    ap.add_argument("--gate-clip-radius", type=float, default=1.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading TSDF: {args.tsdf}")
    artifact = load_tsdf(args.tsdf)
    print(
        f"      shape={artifact.sdf.shape} voxel={artifact.voxel_size} "
        f"origin={artifact.origin}"
    )

    print(f"[2/7] Loading gates: {args.gates}")
    gates_ned = json.loads(args.gates.read_text())
    print(f"      {len(gates_ned)} gates")

    print(f"[3/7] Clipping gate volumes (radius={args.gate_clip_radius} m)")
    artifact = clip_gates_in_sdf(
        artifact,
        list(gates_ned.values()),
        clip_radius_m=args.gate_clip_radius,
    )

    print("[4/7] Marching cubes")
    mesh = extract_mesh_from_sdf(artifact)
    print(f"      raw triangles: {len(mesh.triangles)}")

    print(f"[5/7] Simplify (target {args.target_triangles})")
    mesh = simplify_mesh(mesh, target_triangles=args.target_triangles)
    print(f"      simplified: {len(mesh.triangles)} triangles")

    print("[6/7] NED → ENU coordinate flip")
    mesh_enu = flip_mesh_ned_to_enu(mesh)

    print(f"[7/7] Writing assets to {args.out}")
    obj_path = args.out / "warehouse.obj"
    o3d.io.write_triangle_mesh(str(obj_path), mesh_enu)
    write_warehouse_urdf(args.out / "warehouse.urdf", "warehouse.obj")

    ring = make_torus_mesh()
    o3d.io.write_triangle_mesh(str(args.out / "gate_ring.obj"), ring)
    write_gate_urdf(args.out / "gate.urdf", "gate_ring.obj")

    write_gates_enu_json(args.out / "gates_enu.json", gates_ned)

    write_manifest(
        args.out / "manifest.yaml",
        tsdf_path=args.tsdf,
        gates_path=args.gates,
        warehouse_obj_path=obj_path,
        target_triangles=args.target_triangles,
        gate_clip_radius_m=args.gate_clip_radius,
        warehouse_obj_triangles=len(mesh_enu.triangles),
        num_gates=len(gates_ned),
        git_sha=_git_sha(),
    )

    obj_mb = obj_path.stat().st_size / (1024 * 1024)
    if obj_mb > 10:
        print(f"WARN: warehouse.obj is {obj_mb:.1f} MB — git-lfs recommended")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
