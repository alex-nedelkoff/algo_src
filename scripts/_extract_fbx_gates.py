"""Extract per-gate geometry from the warehouse FBX for visual overlay validation.

For each gate in gates_enu.json, clips a small bbox around the gate's ENU
position in the FBX-derived mesh and writes a small .obj file. Use as a
debug overlay in scripts/_fly_around.py to visually verify the procedural
torus positions match the actually-placed gate geometry from UE.

Inputs:  data/warehouse_fab/warehouse.fbx, configs/.../warehouse_fab_v1_gates_ue.yaml
Outputs: sim/assets/warehouse_fab_v1/fbx_gate_NN.obj  (one per gate)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from scripts.warehouse_to_urdf.tsdf import load_tsdf
from scripts.warehouse_to_urdf.mesh import (
    ue_to_ned_mesh,
    flip_mesh_ned_to_enu,
    clip_mesh_to_bbox,
)

FBX = Path("data/warehouse_fab/warehouse.fbx")
YAML = Path("configs/warehouse/warehouse_fab_v1_gates_ue.yaml")
GATES_JSON = Path("sim/assets/warehouse_fab_v1/gates_enu.json")
OUT_DIR = Path("sim/assets/warehouse_fab_v1")
HALF_EXTENT = 1.0  # meters around each gate centroid

print(f"Loading FBX...")
artifact = load_tsdf(FBX)
mesh = artifact.pre_extracted_mesh
print(f"  raw triangles: {len(mesh.triangles):,}")

ps = yaml.safe_load(YAML.read_text(encoding="utf-8"))["playerstart"]
ps_ue = np.array(ps["location_cm"], dtype=np.float64)
print(f"  PlayerStart UE cm: {ps_ue.tolist()}")

# Bring full mesh into ENU once (same path the build pipeline uses)
mesh_ned = ue_to_ned_mesh(mesh, ps_ue)
mesh_enu = flip_mesh_ned_to_enu(mesh_ned)
print(f"  ENU triangles: {len(mesh_enu.triangles):,}")

gates = json.loads(GATES_JSON.read_text(encoding="utf-8"))
for name, g in gates.items():
    pos = np.array(g["position_enu"], dtype=np.float64)
    bmin = pos - HALF_EXTENT
    bmax = pos + HALF_EXTENT
    clipped = clip_mesh_to_bbox(mesh_enu, bmin, bmax)
    out_path = OUT_DIR / f"fbx_{name}.obj"
    o3d.io.write_triangle_mesh(str(out_path), clipped)
    print(f"  {name}: {len(clipped.triangles):,} tri at ENU{tuple(round(x,2) for x in pos)} "
          f"-> {out_path.name}")

print("Done.")
