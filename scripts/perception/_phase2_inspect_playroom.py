"""Inspect playroom mesh + training cameras to figure out:

- which axis is "up" (cameras vary least along this axis — typical
  photographer's head height stays roughly constant)
- where's the floor (min along up-axis)
- mesh extent + camera traversable bbox
- a suggested world_offset_<axis> to put the floor at z=0 in PyBullet's
  ENU/Z-up convention

Run on laptop where the assets live::

    cd C:\\Users\\alexj\\Documents\\algo_src\\.claude\\worktrees\\warehouse-tsdf-pybullet-mvp
    python scripts\\perception\\_phase2_inspect_playroom.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# Resolve asset paths relative to repo root
HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
ASSET_DIR = REPO / "sim" / "assets" / "3dgs" / "playroom"
MESH_PATH = ASSET_DIR / "mesh.obj"
CAMS_PATH = ASSET_DIR / "cameras.json"


def main() -> int:
    print(f"[mesh] loading {MESH_PATH}")
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))
    verts = np.asarray(mesh.vertices)
    print(f"       {len(verts)} verts, {len(np.asarray(mesh.triangles))} tris")
    print(f"       per-axis min: {verts.min(0).round(3)}")
    print(f"       per-axis max: {verts.max(0).round(3)}")
    print(f"       per-axis extent: {(verts.max(0) - verts.min(0)).round(3)}")

    print(f"\n[cams] loading {CAMS_PATH}")
    cams = json.loads(CAMS_PATH.read_text())
    cam_pos = np.array([c["position"] for c in cams])
    print(f"       {len(cams)} training cameras")
    print(f"       per-axis std: {cam_pos.std(0).round(3)}")
    print(f"       per-axis min: {cam_pos.min(0).round(3)}")
    print(f"       per-axis max: {cam_pos.max(0).round(3)}")
    print(f"       per-axis extent: {(cam_pos.max(0) - cam_pos.min(0)).round(3)}")

    # Identify "up" axis: photographers' camera position varies *least* along it
    # (they're mostly at head height, moving laterally around the room).
    up_axis = int(np.argmin(cam_pos.std(0)))
    axis_names = "xyz"
    print(f"\n[analysis] up-axis (cam std minimum) = {axis_names[up_axis]}")
    # Use percentile-based floor/ceiling rather than min/max — 3DGS
    # reconstructions always have out-of-room Gaussian outliers.
    cam_mean_up = cam_pos[:, up_axis].mean()
    mesh_p01 = float(np.percentile(verts[:, up_axis], 1))
    mesh_p99 = float(np.percentile(verts[:, up_axis], 99))
    print(f"  camera mean along {axis_names[up_axis]}: {cam_mean_up:.3f}")
    print(f"  mesh p1  along {axis_names[up_axis]}: {mesh_p01:.3f}")
    print(f"  mesh p99 along {axis_names[up_axis]}: {mesh_p99:.3f}")
    print(f"  mesh min/max with outliers: {verts[:, up_axis].min():.3f} / {verts[:, up_axis].max():.3f}")

    # Sign of "up": floor is the end further from cam_mean (cameras at
    # head height, ~1.5 m above the floor; ceiling much closer typically).
    dist_to_p01 = abs(cam_mean_up - mesh_p01)
    dist_to_p99 = abs(cam_mean_up - mesh_p99)
    if dist_to_p01 > dist_to_p99:
        floor = mesh_p01; ceiling = mesh_p99; up_sign = +1
    else:
        floor = mesh_p99; ceiling = mesh_p01; up_sign = -1
    print(f"  inferred floor: {floor:.3f}  ceiling: {ceiling:.3f}")
    print(f"  inferred room height: {abs(ceiling - floor):.2f} m")
    print(f"  +{axis_names[up_axis]} direction is {'UP' if up_sign>0 else 'DOWN'}")

    cam_above_floor = abs(cam_mean_up - floor)
    print(f"  mean camera height above floor: {cam_above_floor:.2f} m")
    print(f"  {'OK' if 0.8 < cam_above_floor < 2.2 else 'WARN'} "
          f"({'consistent with human photographer' if 0.8 < cam_above_floor < 2.2 else 'unusual'})")

    print(f"\n[suggested PyBullet setup]")
    if up_axis == 2 and up_sign == +1:
        print(f"  Already ENU/Z-up. world_offset_z = {-floor:.3f}")
    elif up_axis == 1 and up_sign == +1:
        print(f"  Mesh is +Y up. Pre-transform: rotate -90 deg about X axis "
              f"(new_y = -old_z, new_z = old_y). Post-transform floor at z = "
              f"{floor:.3f}, world_offset_z = {-floor:.3f}")
    elif up_axis == 1 and up_sign == -1:
        print(f"  Mesh is -Y up. Pre-transform: rotate +90 deg about X axis "
              f"(new_y = old_z, new_z = -old_y). Post-transform floor at z = "
              f"{-floor:.3f}, world_offset_z = {floor:.3f}")
    elif up_axis == 2 and up_sign == -1:
        print(f"  Mesh is -Z up (inverted). Mirror Z, then world_offset_z = {floor:.3f}")
    else:
        print(f"  Up-axis is X — unusual, needs custom handling")

    print(f"\n[open-space estimate]")
    # Crude: along the two non-up axes, where do camera positions cluster?
    h_axes = [a for a in (0, 1, 2) if a != up_axis]
    for a in h_axes:
        print(f"  {axis_names[a]}: cam p10={np.percentile(cam_pos[:, a], 10):+.3f}  "
              f"p50={np.percentile(cam_pos[:, a], 50):+.3f}  "
              f"p90={np.percentile(cam_pos[:, a], 90):+.3f}")
    print(f"  -> the camera p10..p90 box on the two horizontal axes is "
          f"a reasonable proxy for 'where a drone can fly'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
