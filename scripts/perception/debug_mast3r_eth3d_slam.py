"""In-distribution validation — run MASt3R-SLAM on ETH3D sofa_1.

ETH3D's mono SLAM benchmark provides real-camera RGB sequences with GT
poses from a laser scanner. By definition this is in-distribution for
MASt3R's foundation model. If MASt3R-SLAM tracks well here, our
production stack is sound; the failures we saw on PyBullet and 3DGS
are purely synthetic-rendering OOD.

Format expected at --root:
  rgb.txt          # lines: <timestamp> <image_path>
  calibration.txt  # whitespace: fx fy cx cy [k1 k2 p1 p2]
  groundtruth.txt  # TUM format: <timestamp> tx ty tz qx qy qz qw
  rgb/             # the actual image directory
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import cv2

from perception.localization.icp_localizer import CameraIntrinsics
from perception.localization.mast3r_localizer import Mast3rLocalizer


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (xyzw) → 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12: return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


import math


def load_eth3d(root: Path):
    rgb_lines = (root / "rgb.txt").read_text().strip().splitlines()
    rgb_lines = [l for l in rgb_lines if l and not l.startswith("#")]
    rgb_entries = []
    for ln in rgb_lines:
        parts = ln.split()
        ts = float(parts[0])
        rel = parts[1]
        rgb_entries.append((ts, root / rel))

    calib = np.loadtxt(root / "calibration.txt")
    if calib.ndim == 0:
        calib = calib.reshape(1)
    fx, fy, cx, cy = float(calib[0]), float(calib[1]), float(calib[2]), float(calib[3])

    gt_lines = (root / "groundtruth.txt").read_text().strip().splitlines()
    gt_lines = [l for l in gt_lines if l and not l.startswith("#")]
    gt = []
    for ln in gt_lines:
        parts = ln.split()
        ts = float(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = (float(parts[4]), float(parts[5]),
                          float(parts[6]), float(parts[7]))
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(qx, qy, qz, qw)
        T[:3, 3] = [tx, ty, tz]
        gt.append((ts, T))
    return rgb_entries, (fx, fy, cx, cy), gt


def closest_gt(ts: float, gt_list) -> np.ndarray:
    best = min(gt_list, key=lambda x: abs(x[0] - ts))
    return best[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--target-w", type=int, default=512,
                    help="MASt3R input width; height auto-derived from native aspect")
    args = ap.parse_args()

    print(f"[1/4] Loading ETH3D scene at {args.root}")
    rgb_entries, (fx, fy, cx, cy), gt = load_eth3d(args.root)
    print(f"      {len(rgb_entries)} frames, calib fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}, {len(gt)} GT poses")

    img0 = cv2.imread(str(rgb_entries[0][1]))
    src_h, src_w = img0.shape[:2]
    print(f"      native resolution: {src_w}x{src_h}")

    # MASt3R needs long-side 512 with patch-aligned dims. We'll resize to
    # (target_w, target_h) where target_h preserves aspect ratio rounded to 16.
    target_w = args.target_w
    target_h = int(round(src_h * target_w / src_w / 16) * 16)
    sx, sy = target_w / src_w, target_h / src_h
    K_new = np.array([[fx * sx, 0, target_w / 2.0],
                      [0, fy * sy, target_h / 2.0],
                      [0, 0, 1]], dtype=np.float32)
    print(f"      resize to {target_w}x{target_h}, K_new fx={K_new[0,0]:.1f} "
          f"fy={K_new[1,1]:.1f}")

    print(f"[2/4] Loading frames ({args.start}..{args.start + args.n_frames})")
    frames = rgb_entries[args.start:args.start + args.n_frames]
    rgb_buf, gt_T_buf, ts_buf = [], [], []
    for ts, p in frames:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb_buf.append(img.astype(np.uint8))
        ts_buf.append(ts)
        gt_T_buf.append(closest_gt(ts, gt))

    print(f"[3/4] Running MASt3R-SLAM live tracking...")
    R_id = np.eye(3)
    loc = Mast3rLocalizer(K=K_new.astype(np.float64),
                          R_body_to_cam=R_id,
                          img_size_wh=(target_w, target_h))
    loc.reset(gt_T_buf[0])  # anchor at frame-0 GT

    est_T = []
    for t in range(len(rgb_buf)):
        res = loc.localize(rgb_buf[t], t=t)
        est_T.append(res.refined_T_world_body)

    print(f"\n[4/4] Pose error vs GT")
    est_pos = np.array([T[:3, 3] for T in est_T])
    gt_pos = np.array([T[:3, 3] for T in gt_T_buf])
    err_cm = np.linalg.norm(est_pos - gt_pos, axis=1) * 100
    motion_gt = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1).sum()
    motion_est = np.linalg.norm(np.diff(est_pos, axis=0), axis=1).sum()

    for t in (0, 1, 5, 10, 20, 30, 50, 75, len(rgb_buf) - 1):
        if t < len(rgb_buf):
            print(f"  t={t:>3}  err={err_cm[t]:6.2f} cm  "
                  f"GT={gt_pos[t].round(3)}  est={est_pos[t].round(3)}")

    print(f"\n=== Summary ===")
    print(f"  pos_err mean={err_cm.mean():.2f} cm  max={err_cm.max():.2f} cm  "
          f"final={err_cm[-1]:.2f} cm")
    print(f"  total path GT={motion_gt*100:.1f} cm  est={motion_est*100:.1f} cm  "
          f"ratio={motion_est/motion_gt:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
