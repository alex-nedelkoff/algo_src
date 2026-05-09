"""GT-depth-anchored MASt3R-SLAM — does the SLAM stack work given correct depth?

Monkey-patches Frame.update_pointmap so the per-pixel pointmap stored in
each frame is computed from PyBullet's GT depth instead of MASt3R's
prediction. Matcher features and confidences come from MASt3R as
normal, so correspondences are still found by the foundation model;
only the metric geometry is replaced.

If pose error drops to ~ICP-with-good-prior level, the SLAM stack is
sound and the failure is purely depth-prediction OOD. If pose error
stays high, there is a deeper issue (matching quality, optimizer
sensitivity to bad Xkf, etc.).

Supports two depth sources via --depth-source:
  gt    : PyBullet's z-buffer converted to metric (sub-cm pose tracking)
  da_v2 : Depth-Anything-V2-Metric-Indoor-Small (also OOD on PyBullet,
          requires `pip install transformers>=4.45,<4.50`)

Kept regression harness — useful to validate any new depth source
(e.g., 3DGS-rendered RGB → MASt3R native depth, real camera input)
without modifying the production localizer. See COR-106 Phase 1.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import numpy as np
import pybullet as pb
import torch

from perception.localization.icp_localizer import CameraIntrinsics
from perception.localization.mast3r_localizer import Mast3rLocalizer
from scripts.perception.mast3r_slam_smoketest_easy import (
    R_BODY_TO_CAM, WIDTH, HEIGHT, VFOV_DEG,
    line_trajectory, oval_trajectory, pose_to_matrix, setup_scene, render_rgb_depth,
)


def zbuf_to_metric(zbuf: np.ndarray, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    """PyBullet z-buffer → metric depth (linear from camera)."""
    return far * near / (far - (far - near) * zbuf)


def gt_pointmap(depth_metric: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W) metric depth + 3×3 K → (H*W, 3) pointmap in cam coords."""
    H, W = depth_metric.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    z = depth_metric.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["line", "oval"], default="line")
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--depth-source", choices=["gt", "da_v2"], default="gt",
                    help="gt: PyBullet z-buffer. da_v2: DA V2 Metric Indoor Small")
    args = ap.parse_args()

    n_frames, dt = args.n_frames, 0.03
    if args.mode == "line":
        positions, yaws = line_trajectory(n_frames, dt)
    else:
        positions, yaws = oval_trajectory(n_frames, dt)

    print(f"[scene] rendering {n_frames} frames, mode={args.mode}")
    cid = setup_scene(use_egl=True)
    rgb_buf, depth_buf = [], []
    for t in range(n_frames):
        rgb, zbuf = render_rgb_depth(cid, positions[t], yaws[t])
        rgb_buf.append(rgb)
        depth_buf.append(zbuf_to_metric(zbuf))
    pb.disconnect(cid)

    intr = CameraIntrinsics.from_vfov(VFOV_DEG, WIDTH, HEIGHT)
    K_np = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]],
                    dtype=np.float32)

    # Choose depth source.
    if args.depth_source == "gt":
        chosen_depth = depth_buf
        print(f"[depth] source=gt (PyBullet z-buffer → metric)")
    else:  # da_v2
        from transformers import pipeline
        from PIL import Image
        print(f"[depth] source=da_v2  loading Depth-Anything-V2-Metric-Indoor-Small")
        pipe = pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                        device="cuda")
        chosen_depth = []
        for t in range(n_frames):
            pil = Image.fromarray(rgb_buf[t])
            out = pipe(pil)
            d = out["predicted_depth"]
            d_np = d.detach().cpu().numpy() if hasattr(d, "detach") else np.array(d)
            # The pipeline may return depth at native (518×518-ish) — resize
            # back to (HEIGHT, WIDTH) for our pointmap function.
            if d_np.shape != (HEIGHT, WIDTH):
                from PIL import Image as _I
                d_pil = _I.fromarray(d_np.astype(np.float32))
                d_pil = d_pil.resize((WIDTH, HEIGHT), _I.BILINEAR)
                d_np = np.asarray(d_pil, dtype=np.float32)
            chosen_depth.append(d_np)
            if t == 0:
                gt_med = float(np.median(depth_buf[0]))
                da_med = float(np.median(d_np))
                print(f"[depth] t=0 GT_median={gt_med:.3f}m  "
                      f"DAV2_median={da_med:.3f}m  ratio={da_med/gt_med:.3f}")

    gt_pointmaps = [gt_pointmap(d, K_np) for d in chosen_depth]
    print(f"[depth] pointmap shape per frame: {gt_pointmaps[0].shape}")

    # ----- Monkey-patch Frame.update_pointmap to override X with GT ------
    # tracker.track() calls update_pointmap TWICE per step: once on the
    # current frame (line 44) and once on the keyframe (line 99). They
    # refer to different frames, so we index by the Frame object's own
    # frame_id rather than a global counter.
    import mast3r_slam.frame as frame_mod
    original_update = frame_mod.Frame.update_pointmap
    counters = {"applied": 0, "missed": 0}

    def patched_update(self, X, C):
        fid = int(getattr(self, "frame_id", -1))
        if 0 <= fid < len(gt_pointmaps):
            X_gt = torch.from_numpy(gt_pointmaps[fid]).to(
                device=X.device, dtype=X.dtype)
            if X.dim() == 3:
                X_gt = X_gt.unsqueeze(0)
            elif X.dim() == 4:
                _, H, W, _ = X.shape
                X_gt = X_gt.reshape(H, W, 3).unsqueeze(0)
            X = X_gt
            counters["applied"] += 1
        else:
            counters["missed"] += 1
        return original_update(self, X, C)

    frame_mod.Frame.update_pointmap = patched_update

    # ---------------------------------------------------------------------
    loc = Mast3rLocalizer(K=K_np, R_body_to_cam=R_BODY_TO_CAM,
                          img_size_wh=(WIDTH, HEIGHT))
    gt_T0 = pose_to_matrix(positions[0], yaws[0])
    loc.reset(gt_T0)

    print(f"\n{'t':>3} {'mode':<10} {'scale':>7} "
          f"{'x_gt':>7} {'y_gt':>7} {'x_est':>7} {'y_est':>7} {'err_cm':>7}")
    print("-" * 70)
    errs = []
    for t in range(n_frames):
        res = loc.localize(rgb_buf[t], t=t)
        T = res.extra["T_S_C"]
        s = abs(np.linalg.det(T[:3, :3])) ** (1.0 / 3.0)
        gt = positions[t]
        est = res.refined_T_world_body[:3, 3]
        err_cm = float(np.linalg.norm(gt - est) * 100)
        errs.append(err_cm)
        if t < 5 or t % 5 == 0 or t == n_frames - 1:
            print(f"{t:>3} {res.mode:<10} {s:>7.4f} "
                  f"{gt[0]:>7.3f} {gt[1]:>7.3f} {est[0]:>7.3f} {est[1]:>7.3f} "
                  f"{err_cm:>7.2f}")

    errs = np.array(errs)
    print(f"\n=== Summary (mode={args.mode}, GT depth anchored) ===")
    print(f"  pos_err mean={errs.mean():.2f} cm  max={errs.max():.2f} cm")
    print(f"  pointmap overrides applied={counters['applied']}  missed={counters['missed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
