"""CLI entry point: build warehouse + gate URDFs from TSDF + gate JSON."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from scripts.warehouse_to_urdf.tsdf import load_tsdf
from scripts.warehouse_to_urdf.mesh import (
    clip_gates_in_sdf,
    clip_mesh_to_bbox,
    compute_default_clip_bbox,
    extract_mesh_from_sdf,
    flip_mesh_ned_to_enu,
    make_torus_mesh,
    simplify_mesh,
    ue_to_ned_mesh,
)
from scripts.warehouse_to_urdf.gate_yaml import convert_ue_yaml_to_ned_json
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


def _write_common_assets(out_dir: Path, mesh_enu, gates_ned: dict) -> None:
    """Write the 5 output files shared by both pipelines."""
    obj_path = out_dir / "warehouse.obj"
    o3d.io.write_triangle_mesh(str(obj_path), mesh_enu)
    write_warehouse_urdf(out_dir / "warehouse.urdf", "warehouse.obj")

    ring = make_torus_mesh()
    o3d.io.write_triangle_mesh(str(out_dir / "gate_ring.obj"), ring)
    write_gate_urdf(out_dir / "gate.urdf", "gate_ring.obj")

    write_gates_enu_json(out_dir / "gates_enu.json", gates_ned)
    return obj_path


def _run_tsdf_pipeline(args) -> int:
    """Original 7-stage TSDF pipeline."""
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


def _run_mesh_pipeline(args) -> int:
    """8-stage mesh-direct pipeline (FBX/OBJ + UE YAML)."""
    print(f"[1/8] Loading mesh: {args.mesh}")
    artifact = load_tsdf(args.mesh)
    mesh = artifact.pre_extracted_mesh
    if mesh is None:
        raise RuntimeError(
            f"load_tsdf({args.mesh}) did not return a pre_extracted_mesh. "
            "Expected .fbx or .obj path."
        )
    print(f"      raw triangles: {len(mesh.triangles)}")

    print(f"[2/8] Converting UE gate YAML to NED: {args.gates_ue_yaml}")
    intermediate_json = args.out / "gates_ned_intermediate.json"
    gates_ned = convert_ue_yaml_to_ned_json(args.gates_ue_yaml, intermediate_json)
    print(f"      {len(gates_ned)} gates → {intermediate_json}")

    print("[3/8] Extracting PlayerStart location from YAML")
    raw_yaml = yaml.safe_load(args.gates_ue_yaml.read_text(encoding="utf-8"))
    playerstart_ue_cm = np.array(
        raw_yaml["playerstart"]["location_cm"], dtype=np.float64
    )
    print(f"      playerstart_ue_cm={playerstart_ue_cm.tolist()}")

    print("[4/8] UE world frame → NED metres")
    mesh = ue_to_ned_mesh(mesh, playerstart_ue_cm)
    print(f"      triangles after frame transform: {len(mesh.triangles)}")

    print(f"[5/8] Computing clip bbox (margin={args.clip_margin_m} m)")
    gate_positions_ned = np.array(
        [v["position_ned"] for v in gates_ned.values()], dtype=np.float64
    )
    bmin, bmax = compute_default_clip_bbox(
        gate_positions_ned, margin_m=args.clip_margin_m
    )
    print(f"      bbox min={bmin.tolist()}  max={bmax.tolist()}")

    print("[6/8] Clipping mesh to bbox")
    mesh = clip_mesh_to_bbox(mesh, bmin, bmax)
    print(f"      triangles after clip: {len(mesh.triangles)}")

    print(f"[7/8] Simplify (target {args.target_triangles})")
    mesh = simplify_mesh(mesh, target_triangles=args.target_triangles)
    print(f"      simplified: {len(mesh.triangles)} triangles")

    print("[8/8] NED → ENU flip + writing assets")
    mesh_enu = flip_mesh_ned_to_enu(mesh)

    obj_path = args.out / "warehouse.obj"
    o3d.io.write_triangle_mesh(str(obj_path), mesh_enu)
    write_warehouse_urdf(args.out / "warehouse.urdf", "warehouse.obj")

    ring = make_torus_mesh()
    o3d.io.write_triangle_mesh(str(args.out / "gate_ring.obj"), ring)
    write_gate_urdf(args.out / "gate.urdf", "gate_ring.obj")

    write_gates_enu_json(args.out / "gates_enu.json", gates_ned)

    write_manifest(
        args.out / "manifest.yaml",
        mesh_path=args.mesh,
        gates_ue_yaml_path=args.gates_ue_yaml,
        warehouse_obj_path=obj_path,
        target_triangles=args.target_triangles,
        clip_margin_m=args.clip_margin_m,
        warehouse_obj_triangles=len(mesh_enu.triangles),
        num_gates=len(gates_ned),
        git_sha=_git_sha(),
    )

    obj_mb = obj_path.stat().st_size / (1024 * 1024)
    if obj_mb > 10:
        print(f"WARN: warehouse.obj is {obj_mb:.1f} MB — git-lfs recommended")

    print("Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build warehouse + gate URDFs from TSDF or mesh input."
    )

    # Mutually exclusive input source: TSDF or mesh
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--tsdf", type=Path,
        help="TSDF artifact path (.npz, .ply, .bin). Use for voxel-grid inputs.",
    )
    src_group.add_argument(
        "--mesh", type=Path,
        help="Triangle mesh input (.fbx or .obj) in UE world frame.",
    )

    # Mutually exclusive gate source: NED JSON or UE YAML
    gate_group = ap.add_mutually_exclusive_group(required=True)
    gate_group.add_argument(
        "--gates", type=Path,
        help="Gate positions in NED JSON format (COR-92 schema).",
    )
    gate_group.add_argument(
        "--gates-ue-yaml", type=Path,
        help="UE-frame gate YAML (includes playerstart). Used with --mesh.",
    )

    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-triangles", type=int, default=30000)
    ap.add_argument("--gate-clip-radius", type=float, default=1.0,
                    help="SDF gate clip radius in metres (TSDF path only).")
    ap.add_argument("--clip-margin-m", type=float, default=5.0,
                    help="Spatial clip bbox margin in metres (mesh-direct path only).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Dispatch to the appropriate pipeline
    if args.mesh is not None:
        if args.gates_ue_yaml is None:
            ap.error("--mesh requires --gates-ue-yaml")
        return _run_mesh_pipeline(args)
    else:
        if args.gates is None:
            ap.error("--tsdf requires --gates")
        return _run_tsdf_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
