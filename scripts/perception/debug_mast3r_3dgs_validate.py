"""Phase 2 validation — does MASt3R produce in-distribution depth on 3DGS renders?

Loads a pre-trained 3D Gaussian Splatting scene (.ply file in the standard
3DGS format), renders one frame using gsplat at the original training-camera
pose (so we know it's a "valid" view), runs mast3r_inference_mono on the
result, and reports depth-distribution statistics.

Result (COR-106 Phase 2): 3DGS renders are *also* OOD for MASt3R. Both
MASt3R and DA V2 Metric Indoor produced ~2.77 m median on a render with
~6.81 m rasterizer-GT, agreeing to 3 decimals — same biased prediction.
Real-camera input (e.g., ETH3D, see debug_mast3r_eth3d_slam.py) is
required for in-distribution validation.

Usage::

    python -m scripts.perception.debug_mast3r_3dgs_validate \\
        --ply sim/assets/3dgs/<scene>/point_cloud/iteration_30000/point_cloud.ply \\
        --cameras-json sim/assets/3dgs/<scene>/cameras.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from plyfile import PlyData


# ---------------------------------------------------------------------------
# 3DGS .ply loader
# ---------------------------------------------------------------------------

def load_3dgs_ply(path: str | Path) -> dict:
    """Load a 3DGS-format .ply into raw tensors (no activations applied)."""
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    # SH DC term (3 channels)
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
    # SH higher-order terms — ordered f_rest_0 .. f_rest_(K*3-1) where K is
    # the number of higher-order SH coefficients per channel
    rest_names = sorted(
        [n for n in v.data.dtype.names if n.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_names:
        f_rest = np.stack([v[n] for n in rest_names], axis=1).astype(np.float32)
        # Stored interleaved as (N, 3*K) → reshape to (N, K, 3)
        K_higher = f_rest.shape[1] // 3
        f_rest = f_rest.reshape(-1, 3, K_higher).transpose(0, 2, 1)
    else:
        f_rest = np.zeros((means.shape[0], 0, 3), dtype=np.float32)
    opacity = v["opacity"][:, None].astype(np.float32)  # (N, 1)
    scale = np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float32)

    n_total_sh = 1 + f_rest.shape[1]
    sh_degree = int(round(math.sqrt(n_total_sh) - 1))
    print(f"[ply] {means.shape[0]} gaussians, sh_degree={sh_degree} "
          f"(1 DC + {f_rest.shape[1]} higher-order)")
    return dict(means=means, f_dc=f_dc, f_rest=f_rest,
                opacity=opacity, scale=scale, rot=rot, sh_degree=sh_degree)


def to_gsplat_inputs(raw: dict, device: str = "cuda"):
    """Apply 3DGS activations and pack into gsplat-shaped tensors."""
    import torch as t
    means = t.from_numpy(raw["means"]).to(device)
    # Quaternions stored as (rot_0, rot_1, rot_2, rot_3) in 3DGS = wxyz already.
    quats = t.from_numpy(raw["rot"]).to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = t.exp(t.from_numpy(raw["scale"]).to(device))
    opacities = t.sigmoid(t.from_numpy(raw["opacity"][:, 0]).to(device))
    # Colors: (N, K_total, 3) where K_total = (sh_degree + 1)^2
    f_dc = t.from_numpy(raw["f_dc"]).to(device)[:, None, :]   # (N, 1, 3)
    f_rest = t.from_numpy(raw["f_rest"]).to(device)            # (N, K-1, 3)
    colors_sh = t.cat([f_dc, f_rest], dim=1)                   # (N, K, 3)
    return means, quats, scales, opacities, colors_sh


# ---------------------------------------------------------------------------
# Camera handling — load the training cameras from cameras.json
# ---------------------------------------------------------------------------

def load_cameras_json(path: str | Path) -> list[dict]:
    """3DGS' cameras.json schema: list of {id, img_name, width, height, position,
    rotation, fy, fx}.
    """
    with open(path) as f:
        return json.load(f)


def cam_dict_to_viewmat(cam: dict) -> np.ndarray:
    """Build a 4x4 world-to-camera matrix from a cameras.json entry.

    3DGS' cameras.json stores R (3x3) as 'rotation' and T (3,) as 'position'.
    The convention is: the camera-to-world transform is [R | position].
    World-to-cam is the inverse: [R.T | -R.T @ position].
    """
    R = np.array(cam["rotation"], dtype=np.float32)   # 3x3
    t = np.array(cam["position"], dtype=np.float32)   # 3
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -R.T @ t
    return Rt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def stats_line(name: str, depth: np.ndarray) -> None:
    valid = (depth > 0.05) & (depth < 200) & np.isfinite(depth)
    d = depth[valid]
    if len(d) == 0:
        print(f"  {name}: no valid depths"); return
    print(f"  {name:<28} median={np.median(d):7.3f}  mean={d.mean():7.3f}  "
          f"p10={np.percentile(d, 10):7.3f}  p90={np.percentile(d, 90):7.3f}  "
          f"p99={np.percentile(d, 99):7.3f}  range={d.max()-d.min():7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--cameras-json", type=Path, required=True)
    ap.add_argument("--cam-idx", type=int, default=0,
                    help="Which training camera to render from")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--render-native", action="store_true",
                    help="Render at the training camera's native resolution, "
                         "then downsample to (width, height) before MASt3R")
    ap.add_argument("--save-rgb", type=Path, default=None,
                    help="Optional path to save the rendered RGB as PNG")
    args = ap.parse_args()

    print(f"[1/4] Loading 3DGS .ply: {args.ply}")
    raw = load_3dgs_ply(args.ply)
    means, quats, scales, opacities, colors_sh = to_gsplat_inputs(raw)
    sh_degree = raw["sh_degree"]

    print(f"[2/4] Loading cameras.json: {args.cameras_json}")
    cams = load_cameras_json(args.cameras_json)
    print(f"      {len(cams)} cameras, using cam_idx={args.cam_idx}")
    cam = cams[args.cam_idx]
    viewmat = cam_dict_to_viewmat(cam)
    # The training camera's K — but we want to render at our (W, H), so
    # rescale fx, fy and recenter cx, cy.
    src_w, src_h = int(cam["width"]), int(cam["height"])
    src_fx, src_fy = float(cam["fx"]), float(cam["fy"])
    if args.render_native:
        # Render at native, downsample later
        render_w, render_h = src_w, src_h
        rfx, rfy = src_fx, src_fy
    else:
        render_w, render_h = args.width, args.height
        sx, sy = args.width / src_w, args.height / src_h
        rfx, rfy = src_fx * sx, src_fy * sy
    rcx, rcy = render_w / 2.0, render_h / 2.0
    K_render = np.array([[rfx, 0, rcx], [0, rfy, rcy], [0, 0, 1]], dtype=np.float32)
    # K used for MASt3R inference (output resolution)
    K_np = np.array([[src_fx * args.width / src_w, 0, args.width / 2.0],
                     [0, src_fy * args.height / src_h, args.height / 2.0],
                     [0, 0, 1]], dtype=np.float32)
    print(f"      src K: fx={src_fx:.1f} fy={src_fy:.1f} ({src_w}x{src_h})")
    print(f"      render K: fx={rfx:.1f} fy={rfy:.1f} ({render_w}x{render_h})")
    print(f"      mast3r K: fx={K_np[0,0]:.1f} fy={K_np[1,1]:.1f} ({args.width}x{args.height})")

    print(f"[3/4] Rasterizing with gsplat (sh_degree={sh_degree}) "
          f"at {render_w}x{render_h}...")
    from gsplat.rendering import rasterization
    viewmat_t = torch.from_numpy(viewmat).cuda()[None]
    K_render_t = torch.from_numpy(K_render).cuda()[None]
    render, alpha, meta = rasterization(
        means, quats, scales, opacities, colors_sh,
        viewmat_t, K_render_t, render_w, render_h,
        sh_degree=sh_degree, render_mode="RGB+ED",
    )
    rgb = render[0, ..., :3].clamp(0, 1).detach().cpu().numpy()
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    depth_ed = render[0, ..., 3].detach().cpu().numpy()
    # Also get median depth (closer to actual surface) via "D" mode
    render_d, _, _ = rasterization(
        means, quats, scales, opacities, colors_sh,
        viewmat_t, K_render_t, render_w, render_h,
        sh_degree=sh_degree, render_mode="D",
    )
    depth_d = render_d[0, ..., 0].detach().cpu().numpy()
    depth_3dgs = depth_ed
    if args.render_native:
        # Downsample RGB to args.width x args.height for MASt3R
        from PIL import Image
        rgb_pil = Image.fromarray(rgb_uint8).resize(
            (args.width, args.height), Image.BILINEAR)
        rgb_uint8 = np.asarray(rgb_pil, dtype=np.uint8)
        # Downsample depth too (for fair stat comparison)
        d_pil = Image.fromarray(depth_3dgs.astype(np.float32))
        d_pil = d_pil.resize((args.width, args.height), Image.BILINEAR)
        depth_3dgs = np.asarray(d_pil, dtype=np.float32)
    print(f"      RGB shape={rgb_uint8.shape}  "
          f"3DGS depth median={np.median(depth_3dgs[depth_3dgs > 0]):.3f} m")

    if args.save_rgb is not None:
        from PIL import Image
        args.save_rgb.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_uint8).save(args.save_rgb)
        print(f"      saved RGB to {args.save_rgb}")

    print(f"[4/4] Running MASt3R inference + depth stats...")
    from perception.localization.icp_localizer import CameraIntrinsics
    from perception.localization.mast3r_localizer import Mast3rLocalizer
    R_BODY_TO_CAM = np.array([
        [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
    ])
    loc = Mast3rLocalizer(K=K_np.astype(np.float64),
                          R_body_to_cam=R_BODY_TO_CAM,
                          img_size_wh=(args.width, args.height))
    from mast3r_slam.frame import create_frame
    from mast3r_slam.mast3r_utils import mast3r_inference_mono
    import lietorch
    rgb_f = rgb_uint8.astype(np.float32) / 255.0
    T_id = lietorch.Sim3.Identity(1, device="cuda")
    f = create_frame(0, rgb_f, T_id, img_size=512, device="cuda")
    X, _ = mast3r_inference_mono(loc._model, f)
    z_mast3r = X[..., 2].detach().cpu().numpy().reshape(args.height, args.width)

    # DA V2 Metric Indoor for cross-check
    try:
        from transformers import pipeline
        from PIL import Image as _I
        print("\n[bonus] DA V2 Metric Indoor on the same render...")
        pipe = pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                        device="cuda")
        out = pipe(_I.fromarray(rgb_uint8))
        d = out["predicted_depth"]
        z_dav2 = d.detach().cpu().numpy() if hasattr(d, "detach") else np.array(d)
    except Exception as e:
        print(f"\n[bonus] DA V2 inference skipped: {e}")
        z_dav2 = None

    if args.render_native:
        from PIL import Image as _I3
        d_pil = _I3.fromarray(depth_d.astype(np.float32))
        d_pil = d_pil.resize((args.width, args.height), _I3.BILINEAR)
        depth_d = np.asarray(d_pil, dtype=np.float32)

    print(f"\n=== Depth comparison (cam_idx={args.cam_idx}) ===")
    stats_line("MAST  3DGS render", z_mast3r)
    if z_dav2 is not None:
        stats_line("DAV2  3DGS render", z_dav2)
    stats_line("3DGS GT (ED-mode)", depth_ed)
    stats_line("3DGS GT (D-mode)", depth_d)
    print(f"\nReference (earlier real-photo test):")
    print(f"  Chateau1.png            median=  6.405  range=43.7m")
    print(f"  Chateau2.png            median=  8.616  range=17.0m")
    print(f"  PyBullet (OOD)          median=  3.352  range= 1.12m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
