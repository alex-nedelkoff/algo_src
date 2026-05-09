"""Does MASt3R produce in-distribution depth on real photos?

Compare its depth output on:
  - Chateau1.png + Chateau2.png — real photos from MASt3R's own demo assets
    (in-distribution by definition)
  - PyBullet render of our textured room (suspected OOD)

If real-photo predictions show wide dynamic range while PyBullet's stay
compressed in a narrow 1.9-3.6 m band, the OOD theory is directly
confirmed.

Kept reference harness — quick sanity check that the MASt3R model is
healthy on in-distribution input. Useful before debugging anything
SLAM-related: if real-photo depth dynamic range is broken too, the
problem is the model or its loading, not the rendering. See COR-106.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from pathlib import Path
from PIL import Image

import lietorch  # noqa: F401
import pybullet as pb
from perception.localization.icp_localizer import CameraIntrinsics
from perception.localization.mast3r_localizer import Mast3rLocalizer
from scripts.perception.mast3r_slam_smoketest_easy import (
    R_BODY_TO_CAM, WIDTH, HEIGHT, VFOV_DEG,
    line_trajectory, pose_to_matrix, setup_scene, render_rgb_depth,
)

CHATEAU_DIR = Path("/workspace/algo_src/external_packages/MASt3R-SLAM/"
                   "thirdparty/mast3r/dust3r/croco/assets")


def stats(name: str, depth: np.ndarray) -> None:
    valid = (depth > 0.05) & (depth < 200) & np.isfinite(depth)
    d = depth[valid]
    if len(d) == 0:
        print(f"  {name}: no valid depths")
        return
    print(f"  {name:<22} median={np.median(d):7.3f}  mean={d.mean():7.3f}  "
          f"p10={np.percentile(d, 10):7.3f}  p90={np.percentile(d, 90):7.3f}  "
          f"p99={np.percentile(d, 99):7.3f}  range={d.max() - d.min():7.2f}")


def infer_depth(loc: Mast3rLocalizer, rgb_uint8: np.ndarray) -> np.ndarray:
    """Run mast3r_inference_mono and return the per-pixel Z."""
    from mast3r_slam.frame import create_frame
    from mast3r_slam.mast3r_utils import mast3r_inference_mono

    rgb_f = rgb_uint8.astype(np.float32) / 255.0
    T_id = lietorch.Sim3.Identity(1, device="cuda")
    f = create_frame(0, rgb_f, T_id, img_size=512, device="cuda")
    X, _ = mast3r_inference_mono(loc._model, f)
    H, W = rgb_uint8.shape[:2]
    z = X[..., 2].detach().cpu().numpy().reshape(H, W)
    return z


def main() -> int:
    intr = CameraIntrinsics.from_vfov(VFOV_DEG, WIDTH, HEIGHT)
    K_np = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]],
                    dtype=np.float32)
    loc = Mast3rLocalizer(K=K_np, R_body_to_cam=R_BODY_TO_CAM,
                          img_size_wh=(WIDTH, HEIGHT))

    print("\n=== Real photos (MASt3R demo assets — in distribution) ===")
    for name in ("Chateau1.png", "Chateau2.png"):
        path = CHATEAU_DIR / name
        if not path.exists():
            print(f"  skipping {name}: not found at {path}")
            continue
        img = Image.open(path).convert("RGB")
        # Resize to (HEIGHT, WIDTH) like our PyBullet renders for fair comparison
        img = img.resize((WIDTH, HEIGHT), Image.BILINEAR)
        rgb = np.asarray(img, dtype=np.uint8)
        z = infer_depth(loc, rgb)
        stats(f"MAST {name}", z)

    print("\n=== PyBullet renders (suspected OOD) ===")
    positions, yaws = line_trajectory(2, 0.03)
    cid = setup_scene(use_egl=True)
    rgb0, zbuf0 = render_rgb_depth(cid, positions[0], yaws[0])
    pb.disconnect(cid)
    near, far = 0.1, 100.0
    gt0 = far * near / (far - (far - near) * zbuf0)
    z_pred = infer_depth(loc, rgb0)
    stats("MAST PyBullet t=0", z_pred)
    stats("GT  PyBullet t=0", gt0)

    print("\nVerdict:")
    print("  If real-photo medians/p99 span tens of m (broad dynamic range)")
    print("  while PyBullet stays compressed in 1.9-3.6 m, OOD is confirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
