"""One-off — does MASt3R-SLAM keep Sim(3) scale=1 with use_calib=True?

Now also: compare MASt3R's predicted depth to PyBullet's GT depth at a
single frame, to test the hypothesis that MASt3R underpredicts metric
depth on PyBullet renders (out-of-distribution data) and that's the real
cause of translation under-estimation, not Sim(3) scale drift.

Drives 10 frames of slow forward translation and dumps, per frame:

  - log_s         : the lietorch.Sim3 log-scale parameter (data[-1])
  - scale_det     : (det(T_S_C[:3,:3]))^(1/3), cross-check on scale
  - GT vs est x   : ground-truth and estimated x position
  - err_cm        : world-frame body position error

If log_s != 0 and grows, the Sim(3) optimizer isn't pinning scale despite
use_calib=True.

Kept regression harness — run after any changes to mast3r_localizer.py
or the upstream MASt3R-SLAM tracker. See COR-106 Phase 1 (May 2026).
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pybullet as pb

from perception.localization.icp_localizer import CameraIntrinsics
from perception.localization.mast3r_localizer import Mast3rLocalizer
from scripts.perception.mast3r_slam_smoketest_easy import (
    R_BODY_TO_CAM, WIDTH, HEIGHT, VFOV_DEG,
    line_trajectory, pose_to_matrix, setup_scene, render_rgb_depth,
)


def main() -> int:
    n_frames, dt = 50, 0.03
    positions, yaws = line_trajectory(n_frames, dt)

    print(f"[scene] rendering {n_frames} frames")
    cid = setup_scene(use_egl=True)
    rgb_buf = []
    depth_buf = []
    for t in range(n_frames):
        rgb, depth = render_rgb_depth(cid, positions[t], yaws[t])
        rgb_buf.append(rgb)
        depth_buf.append(depth)
    pb.disconnect(cid)

    # PyBullet's depth buffer is normalised z-buffer in [0, 1]. Convert to
    # metric Z using its near/far convention (near=0.1, far=100 from
    # render_rgb_depth's projection matrix).
    near, far = 0.1, 100.0
    def zbuf_to_z(zbuf):
        # PyBullet docs: linear_depth = far * near / (far - (far - near) * zbuf)
        return far * near / (far - (far - near) * zbuf)

    gt_metric_depth = [zbuf_to_z(d) for d in depth_buf]

    intr = CameraIntrinsics.from_vfov(VFOV_DEG, WIDTH, HEIGHT)
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
    print(f"[K] cx,cy={intr.cx:.1f},{intr.cy:.1f}  W/2,H/2={WIDTH/2:.1f},{HEIGHT/2:.1f}  "
          f"fx,fy={intr.fx:.1f},{intr.fy:.1f}")

    loc = Mast3rLocalizer(K=K, R_body_to_cam=R_BODY_TO_CAM, img_size_wh=(WIDTH, HEIGHT))

    # Verify config + K actually flowed into the tracker layer.
    from mast3r_slam.config import config as mcfg
    print(f"[cfg] use_calib={mcfg.get('use_calib')}  "
          f"img_downsample={mcfg.get('dataset', {}).get('img_downsample')}  "
          f"center_principle_point={mcfg.get('dataset', {}).get('center_principle_point')}")
    K_in_kfs = loc._keyframes.K.detach().cpu().numpy()
    print(f"[K-in-kfs]\n{K_in_kfs}")

    gt_T0 = pose_to_matrix(positions[0], yaws[0])
    loc.reset(gt_T0)

    # T_env_S = body0 → cam (we set this in reset). It has scale=1.
    T_env_S = loc._T_env_slam
    # Cam-to-body for re-projection (after applying alternative Sim3 normalisation)
    T_cam_body = loc.T_cam_body

    hdr = (f"{'t':>2} {'mode':<10} {'scale':>7} "
           f"{'tS_C[0]':>8} "
           f"{'x_gt':>7} {'x_raw':>7} {'x_div':>7} {'x_mul':>7} "
           f"{'err_raw':>7} {'err_div':>7} {'err_mul':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for t in range(n_frames):
        res = loc.localize(rgb_buf[t], t=t)
        T = res.extra["T_S_C"]
        s = abs(np.linalg.det(T[:3, :3])) ** (1.0 / 3.0)

        # Three candidate extractions of the env-frame body position:
        #   raw : current code — compose T_env_S @ T_S_C @ T_cam_body as 4×4 matmul
        #   div : same, but with T_S_C translation divided by s before composition
        #   mul : same, but with T_S_C translation multiplied by s before composition
        def compose(t_scale: float) -> np.ndarray:
            T_local = T.copy()
            T_local[:3, 3] = T[:3, 3] * t_scale
            T_local[:3, :3] = T[:3, :3] / s  # always normalise rotation block
            return T_env_S @ T_local @ T_cam_body

        T_raw = T_env_S @ T @ T_cam_body  # current path (no normalisation at all)
        T_div = compose(1.0 / s)
        T_mul = compose(s)

        gt = positions[t]
        x_gt = gt[0]
        x_raw = T_raw[0, 3]; err_raw = np.linalg.norm(T_raw[:3, 3] - gt) * 100
        x_div = T_div[0, 3]; err_div = np.linalg.norm(T_div[:3, 3] - gt) * 100
        x_mul = T_mul[0, 3]; err_mul = np.linalg.norm(T_mul[:3, 3] - gt) * 100

        print(f"{t:>2} {res.mode:<10} {s:>7.4f} {T[0, 3]:>+8.4f} "
              f"{x_gt:>7.3f} {x_raw:>7.3f} {x_div:>7.3f} {x_mul:>7.3f} "
              f"{err_raw:>7.2f} {err_div:>7.2f} {err_mul:>7.2f}")

    # ------------------------------------------------------------------
    # H6 test: does MASt3R underpredict metric depth?
    # Run mast3r inference on a few frames in isolation and compare the
    # predicted point-cloud Z to PyBullet's GT depth.
    # ------------------------------------------------------------------
    print("\n=== Depth-prediction comparison (H6 test) ===")
    from mast3r_slam.frame import create_frame
    from mast3r_slam.mast3r_utils import mast3r_inference_mono
    for ti in (0, 25, 49):
        rgb_f = rgb_buf[ti].astype(np.float32) / 255.0
        gt_d = gt_metric_depth[ti]
        # Mast3R inference in isolation (no tracker / Sim3 — just the model)
        import lietorch
        import torch
        T_id = lietorch.Sim3.Identity(1, device="cuda")
        f = create_frame(0, rgb_f, T_id, img_size=512, device="cuda")
        X_pred, C_pred = mast3r_inference_mono(loc._model, f)
        # X_pred is (1, H*W, 3) — last channel is camera-frame Z
        X_pred_np = X_pred.detach().cpu().numpy().reshape(-1, 3)
        pred_z = X_pred_np[:, 2]
        # Match the resolution of GT (downsample/resize if needed)
        # Frame.create_frame resizes to img_size=512; the resulting H,W
        # are not 384x512 but the long-side-512 resize. Just compare
        # central crop statistics for now.
        valid_pred = (pred_z > 0.1) & (pred_z < 50)
        pred_z_v = pred_z[valid_pred]
        gt_z_v = gt_d.flatten()
        gt_z_v = gt_z_v[(gt_z_v > 0.1) & (gt_z_v < 50)]
        print(f"  t={ti:>2}  GT  depth median={np.median(gt_z_v):6.3f}  mean={gt_z_v.mean():6.3f}  p10={np.percentile(gt_z_v, 10):6.3f}  p90={np.percentile(gt_z_v, 90):6.3f}")
        print(f"        MAST median={np.median(pred_z_v):6.3f}  mean={pred_z_v.mean():6.3f}  p10={np.percentile(pred_z_v, 10):6.3f}  p90={np.percentile(pred_z_v, 90):6.3f}")
        print(f"        ratio MAST/GT (median) = {np.median(pred_z_v)/np.median(gt_z_v):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
