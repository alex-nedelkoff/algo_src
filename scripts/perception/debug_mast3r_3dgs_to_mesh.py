"""3DGS → triangle mesh via TSDF fusion.

Renders the 3DGS scene from N camera poses (training cameras by default,
or sampled poses) and fuses the resulting RGBD frames into an Open3D
ScalableTSDFVolume, then extracts a triangle mesh via marching cubes.
Output: an .obj that can be loaded into PyBullet for collision and into
Blender for gate-insertion editing.

Empirical: on Deep Blending 'playroom' (2.5M gaussians, 225 training cams)
with --voxel 0.05 --max-cams 100 --depth-trunc 12.0, produces a 511 K
vertex / 972 K triangle mesh (~83 MB .obj) over a 15.6 × 11.7 × 13.6 m
extent in ~5 min on RTX 3090. Usable for PyBullet collision out of the box.

Usage::
    python -m scripts.perception.debug_mast3r_3dgs_to_mesh \\
        --ply sim/assets/3dgs/<scene>/point_cloud/iteration_30000/point_cloud.ply \\
        --cameras-json sim/assets/3dgs/<scene>/cameras.json \\
        --out sim/assets/3dgs/<scene>/mesh.obj \\
        --voxel 0.05 --max-cams 100 --depth-trunc 12.0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import open3d as o3d

from scripts.perception.debug_mast3r_3dgs_validate import (
    load_3dgs_ply, to_gsplat_inputs, load_cameras_json, cam_dict_to_viewmat,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--cameras-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="TSDF voxel size in scene units (m). Smaller → denser mesh, more VRAM")
    ap.add_argument("--sdf-trunc", type=float, default=0.20,
                    help="TSDF truncation distance")
    ap.add_argument("--max-cams", type=int, default=100,
                    help="How many training cameras to use (uniformly subsampled)")
    ap.add_argument("--depth-trunc", type=float, default=15.0,
                    help="Max depth to integrate (m). Beyond this, treated as empty.")
    ap.add_argument("--render-w", type=int, default=512)
    ap.add_argument("--render-h", type=int, default=384)
    args = ap.parse_args()

    print(f"[1/4] Loading 3DGS scene: {args.ply}")
    raw = load_3dgs_ply(args.ply)
    means, quats, scales, opacities, colors_sh = to_gsplat_inputs(raw)
    sh_degree = raw["sh_degree"]
    cams = load_cameras_json(args.cameras_json)
    n_cams = len(cams)
    print(f"      {n_cams} cameras available, using up to {args.max_cams}")

    if n_cams > args.max_cams:
        idx = np.linspace(0, n_cams - 1, args.max_cams).astype(int)
        cams = [cams[i] for i in idx]
        print(f"      subsampled to {len(cams)} cameras")

    print(f"[2/4] Setting up TSDF volume (voxel={args.voxel} m, trunc={args.sdf_trunc} m)")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    print(f"[3/4] Rendering + integrating {len(cams)} views...")
    from gsplat.rendering import rasterization
    for ci, cam in enumerate(cams):
        src_w, src_h = int(cam["width"]), int(cam["height"])
        src_fx, src_fy = float(cam["fx"]), float(cam["fy"])
        sx, sy = args.render_w / src_w, args.render_h / src_h
        fx, fy = src_fx * sx, src_fy * sy
        cx, cy = args.render_w / 2.0, args.render_h / 2.0
        K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        viewmat = cam_dict_to_viewmat(cam)

        with torch.no_grad():
            render, _, _ = rasterization(
                means, quats, scales, opacities, colors_sh,
                torch.from_numpy(viewmat).cuda()[None],
                torch.from_numpy(K_np).cuda()[None],
                args.render_w, args.render_h,
                sh_degree=sh_degree, render_mode="RGB+ED",
            )
        rgb = (render[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth = render[0, ..., 3].cpu().numpy().astype(np.float32)
        # Mask out invalid / too-far / zero depths
        depth[depth > args.depth_trunc] = 0
        depth[depth < 0.05] = 0

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            args.render_w, args.render_h, fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, viewmat)
        if (ci + 1) % 10 == 0 or ci == len(cams) - 1:
            print(f"      {ci+1}/{len(cams)} integrated")

    print(f"[4/4] Extracting mesh via marching cubes...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"      vertices={len(mesh.vertices)}  triangles={len(mesh.triangles)}")
    bbox = mesh.get_axis_aligned_bounding_box()
    print(f"      bbox extent (m): {np.asarray(bbox.get_extent()).round(2)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.out), mesh, write_triangle_uvs=True)
    print(f"      saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
