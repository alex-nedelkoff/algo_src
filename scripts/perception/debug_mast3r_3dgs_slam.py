"""Phase 2 SLAM test — does MASt3R-SLAM track through 3DGS-rendered frames?

Renders a smooth slow trajectory by interpolating between two training
cameras, runs MASt3R-SLAM on the RGB sequence, and compares estimated
pose to the interpolated GT.

This is the *actual* validation of Phase 2: even if single-frame depth
predictions disagree with the rasterizer's GT, what matters for SLAM
is whether the per-frame depths are consistent enough across views to
let the optimizer track motion. PyBullet failed this; real photos
pass; 3DGS is the open question.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from plyfile import PlyData

from scripts.perception.debug_mast3r_3dgs_validate import (
    load_3dgs_ply, to_gsplat_inputs, load_cameras_json, cam_dict_to_viewmat,
)
from perception.localization.icp_localizer import CameraIntrinsics
from perception.localization.mast3r_localizer import Mast3rLocalizer

# Body-to-cam rotation (same as PyBullet smoketests). For the 3DGS test we
# treat the SLAM "body" as the cam itself — but the localizer wraps anyway,
# so a known rotation is fine to keep using.
R_BODY_TO_CAM = np.array([
    [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
])


def slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    sin_t0 = np.sin(theta_0)
    s0 = np.sin((1 - t) * theta_0) / sin_t0
    s1 = np.sin(t * theta_0) / sin_t0
    return s0 * q0 + s1 * q1


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 R → wxyz quaternion."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--cameras-json", type=Path, required=True)
    ap.add_argument("--cam-a", type=int, default=0)
    ap.add_argument("--cam-b", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=384)
    args = ap.parse_args()

    print(f"[1/4] Loading 3DGS scene...")
    raw = load_3dgs_ply(args.ply)
    means, quats, scales, opacities, colors_sh = to_gsplat_inputs(raw)
    sh_degree = raw["sh_degree"]
    cams = load_cameras_json(args.cameras_json)

    cam_a, cam_b = cams[args.cam_a], cams[args.cam_b]
    src_w, src_h = int(cam_a["width"]), int(cam_a["height"])
    src_fx, src_fy = float(cam_a["fx"]), float(cam_a["fy"])
    sx, sy = args.width / src_w, args.height / src_h
    fx, fy = src_fx * sx, src_fy * sy
    cx, cy = args.width / 2.0, args.height / 2.0
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    print(f"      cam_a={args.cam_a}, cam_b={args.cam_b}, "
          f"interpolating {args.n_frames} frames")

    # Build interpolated GT poses
    R_a = np.array(cam_a["rotation"], dtype=np.float32)  # cam-to-world rotation
    R_b = np.array(cam_b["rotation"], dtype=np.float32)
    t_a = np.array(cam_a["position"], dtype=np.float32)
    t_b = np.array(cam_b["position"], dtype=np.float32)
    qa = rot_to_quat(R_a)
    qb = rot_to_quat(R_b)
    distance = np.linalg.norm(t_b - t_a)
    print(f"      interp distance = {distance:.3f} m (in 3DGS world units)")

    print(f"[2/4] Rendering {args.n_frames} interpolated frames with gsplat...")
    from gsplat.rendering import rasterization
    K_t = torch.from_numpy(K_np).cuda()[None]
    rgb_buf = []
    gt_T_world_cam = []  # (N, 4, 4) — cam-to-world (matrices we'll compare)
    gt_T_w2c_buf = []     # (N, 4, 4) — world-to-cam (for rasterization)
    for i in range(args.n_frames):
        u = i / max(1, args.n_frames - 1)
        q = slerp_quaternion(qa, qb, u)
        R_i = quat_to_rot(q)
        t_i = (1 - u) * t_a + u * t_b
        # cameras.json convention: R_i is cam-to-world rotation, t_i is camera
        # position in world. World-to-cam: viewmat = [R_i.T | -R_i.T @ t_i]
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R_i.T
        viewmat[:3, 3] = -R_i.T @ t_i
        gt_T_w2c_buf.append(viewmat)
        T_c2w = np.eye(4, dtype=np.float32)
        T_c2w[:3, :3] = R_i
        T_c2w[:3, 3] = t_i
        gt_T_world_cam.append(T_c2w)
        viewmat_t = torch.from_numpy(viewmat).cuda()[None]
        render, _, _ = rasterization(
            means, quats, scales, opacities, colors_sh,
            viewmat_t, K_t, args.width, args.height,
            sh_degree=sh_degree, render_mode="RGB",
        )
        rgb = render[0, ..., :3].clamp(0, 1).detach().cpu().numpy()
        rgb_buf.append((rgb * 255).astype(np.uint8))
    print(f"      rendered {len(rgb_buf)} frames")

    print(f"[3/4] Running MASt3R-SLAM on the sequence...")
    loc = Mast3rLocalizer(K=K_np.astype(np.float64),
                          R_body_to_cam=R_BODY_TO_CAM,
                          img_size_wh=(args.width, args.height))
    # Use the first GT cam pose as the reset anchor (treating "body" as
    # equivalent to "cam" by passing identity R_body_to_cam-like alignment).
    # For pure pose-error eval we just need a consistent reference.
    T_env_body0 = gt_T_world_cam[0] @ np.linalg.inv(_t_body_cam())
    loc.reset(T_env_body0)

    est_T_world_body = []
    for t in range(args.n_frames):
        res = loc.localize(rgb_buf[t], t=t)
        est_T_world_body.append(res.refined_T_world_body)

    est_pos = np.array([T[:3, 3] for T in est_T_world_body])
    gt_body_pos = np.array(
        [(T @ np.linalg.inv(_t_body_cam()))[:3, 3] for T in gt_T_world_cam]
    )

    print(f"\n=== Pose tracking results ===")
    err = np.linalg.norm(est_pos - gt_body_pos, axis=1) * 100
    for t in (0, 1, 5, 10, 15, 20, 25, args.n_frames - 1):
        if t < args.n_frames:
            print(f"  t={t:>2}  err={err[t]:.2f} cm  "
                  f"GT={gt_body_pos[t].round(3)}  est={est_pos[t].round(3)}")
    print(f"\n  pos_err mean={err.mean():.2f} cm  max={err.max():.2f} cm")
    print(f"  GT total motion = {distance * 100:.1f} cm  "
          f"(scale ratio est/GT = "
          f"{np.linalg.norm(est_pos[-1] - est_pos[0]) / distance:.3f})")
    return 0


def _t_body_cam() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_BODY_TO_CAM
    return T


if __name__ == "__main__":
    raise SystemExit(main())
