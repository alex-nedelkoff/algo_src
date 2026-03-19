"""Undistort + stereo rectify EuRoC VI-Sensor stereo pairs.

EuRoC uses pinhole cameras with radial-tangential distortion.

Usage:
  python -m perception.training.data.euroc.rectify_stereo \
      --input-dir ~/corvidx/data/euroc/machine_hall/MH_01_easy/mav0 \
      --output-dir ~/corvidx/data/euroc/rectified/MH_01_easy/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def load_euroc_calibration(mav_dir: Path) -> dict:
    """Load EuRoC ASL calibration from sensor.yaml files."""
    cam0_yaml = mav_dir / "cam0" / "sensor.yaml"
    cam1_yaml = mav_dir / "cam1" / "sensor.yaml"

    with open(cam0_yaml) as f:
        cam0 = yaml.safe_load(f)
    with open(cam1_yaml) as f:
        cam1 = yaml.safe_load(f)

    # Intrinsics [fu, fv, cu, cv]
    fu0, fv0, cu0, cv0 = cam0["intrinsics"]
    K0 = np.array([[fu0, 0, cu0], [0, fv0, cv0], [0, 0, 1]], dtype=np.float64)

    fu1, fv1, cu1, cv1 = cam1["intrinsics"]
    K1 = np.array([[fu1, 0, cu1], [0, fv1, cv1], [0, 0, 1]], dtype=np.float64)

    # Distortion (radial-tangential: k1, k2, p1, p2)
    D0 = np.array(cam0["distortion_coefficients"], dtype=np.float64)
    D1 = np.array(cam1["distortion_coefficients"], dtype=np.float64)

    w, h = cam0["resolution"]

    # T_BS: sensor (camera) to body transform
    T_BS0 = np.array(cam0["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    T_BS1 = np.array(cam1["T_BS"]["data"], dtype=np.float64).reshape(4, 4)

    # Stereo extrinsics: T_cam1_cam0 = inv(T_BS1) @ T_BS0
    T_cam1_cam0 = np.linalg.inv(T_BS1) @ T_BS0
    R = T_cam1_cam0[:3, :3]
    T = T_cam1_cam0[:3, 3]

    return {
        "K0": K0, "K1": K1,
        "D0": D0, "D1": D1,
        "R": R, "T": T,
        "T_BS0": T_BS0, "T_BS1": T_BS1,
        "image_size": (w, h),
    }


def compute_rectification(calib: dict, alpha: float = 0.0):
    """Compute stereo rectification maps."""
    K0, K1 = calib["K0"], calib["K1"]
    D0, D1 = calib["D0"], calib["D1"]
    R, T = calib["R"], calib["T"]
    w, h = calib["image_size"]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K0, D0, K1, D1,
        imageSize=(w, h),
        R=R, T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(K0, D0, R1, P1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K1, D1, R2, P2, (w, h), cv2.CV_32FC1)

    K_rect = P1[:3, :3].copy()
    baseline = abs(P2[0, 3] / P2[0, 0]) if abs(P2[0, 0]) > 0 else abs(T[0])

    rect_params = {
        "fx": float(K_rect[0, 0]),
        "fy": float(K_rect[1, 1]),
        "cx": float(K_rect[0, 2]),
        "cy": float(K_rect[1, 2]),
        "baseline": float(baseline),
        "width": w,
        "height": h,
    }

    log.info("Rectified: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, baseline=%.4fm",
             rect_params["fx"], rect_params["fy"],
             rect_params["cx"], rect_params["cy"], rect_params["baseline"])

    return map1_l, map2_l, map1_r, map2_r, rect_params


def rectify_sequence(mav_dir: Path, output_dir: Path, alpha: float = 0.0) -> dict:
    """Rectify all stereo pairs in an EuRoC sequence."""
    cam0_dir = mav_dir / "cam0" / "data"
    cam1_dir = mav_dir / "cam1" / "data"

    left_images = sorted(cam0_dir.glob("*.png"))
    right_images = sorted(cam1_dir.glob("*.png"))

    n_frames = min(len(left_images), len(right_images))
    log.info("Rectifying %d stereo pairs from %s", n_frames, mav_dir)

    calib = load_euroc_calibration(mav_dir)
    map1_l, map2_l, map1_r, map2_r, rect_params = compute_rectification(calib, alpha)

    out_left = output_dir / "left"
    out_right = output_dir / "right"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    # Save timestamps (filenames are timestamps in ns)
    timestamps = []
    for i in range(n_frames):
        img_l = cv2.imread(str(left_images[i]), cv2.IMREAD_UNCHANGED)
        img_r = cv2.imread(str(right_images[i]), cv2.IMREAD_UNCHANGED)

        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        cv2.imwrite(str(out_left / f"{i:06d}.png"), rect_l)
        cv2.imwrite(str(out_right / f"{i:06d}.png"), rect_r)

        ts = int(left_images[i].stem)
        timestamps.append(ts)

        if (i + 1) % 500 == 0 or i == n_frames - 1:
            log.info("  Rectified %d/%d frames", i + 1, n_frames)

    # Save rectification params
    with open(output_dir / "rectification_params.json", "w") as f:
        json.dump(rect_params, f, indent=2)

    # Save timestamps
    np.savetxt(str(output_dir / "timestamps.txt"), np.array(timestamps, dtype=np.int64), fmt="%d")

    # Copy GT poses
    gt_src = mav_dir / "state_groundtruth_estimate0" / "data.csv"
    if gt_src.exists():
        import shutil
        shutil.copy2(gt_src, output_dir / "groundtruth.csv")

    # Save T_BS0 for body→camera conversion
    calib_out = {
        "T_BS0": calib["T_BS0"].tolist(),
        "T_BS1": calib["T_BS1"].tolist(),
    }
    with open(output_dir / "calibration.json", "w") as f:
        json.dump(calib_out, f, indent=2)

    return rect_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Undistort + rectify EuRoC stereo pairs")
    parser.add_argument("--input-dir", required=True, type=Path, help="mav0 directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    rectify_sequence(args.input_dir, args.output_dir, args.alpha)


if __name__ == "__main__":
    main()
