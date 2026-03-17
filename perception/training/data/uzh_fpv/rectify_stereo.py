"""Undistort fisheye + stereo rectify UZH-FPV Snapdragon stereo pairs.

UZH-FPV Snapdragon cameras have 186° FOV fisheye lenses with equidistant
distortion model (Kalibr convention). FoundationStereo requires rectified,
undistorted input.

Pipeline:
  1. Load Kalibr YAML calibration (equidistant distortion, T_cam_imu)
  2. Compute stereo extrinsics: T_left_right = T_left_imu @ inv(T_right_imu)
  3. cv2.fisheye.stereoRectify → R1, R2, P1, P2
  4. cv2.fisheye.initUndistortRectifyMap → remap tables
  5. cv2.remap each frame pair

Usage:
  python -m perception.training.data.uzh_fpv.rectify_stereo \
      --input-dir ~/corvidx/data/uzh-fpv/extracted/indoor_forward_3_snapdragon/ \
      --calib ~/corvidx/data/uzh-fpv/calib/indoor_forward_snapdragon.yaml \
      --output-dir ~/corvidx/data/uzh-fpv/rectified/indoor_forward_3_snapdragon/
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


def load_kalibr_calibration(calib_path: Path) -> dict:
    """Parse Kalibr camchain YAML into camera parameters.

    Returns dict with keys:
        K0, D0: left camera intrinsics and distortion (equidistant)
        K1, D1: right camera intrinsics and distortion
        T_cn_cnm1: (4, 4) transform from cam0 (left) to cam1 (right)
        image_size: (width, height)
    """
    with open(calib_path) as f:
        calib = yaml.safe_load(f)

    # cam0 = left, cam1 = right (Kalibr convention)
    cam0 = calib["cam0"]
    cam1 = calib["cam1"]

    # Intrinsics: [fx, fy, cx, cy]
    fu0, fv0, cu0, cv0 = cam0["intrinsics"]
    K0 = np.array([[fu0, 0, cu0], [0, fv0, cv0], [0, 0, 1]], dtype=np.float64)

    fu1, fv1, cu1, cv1 = cam1["intrinsics"]
    K1 = np.array([[fu1, 0, cu1], [0, fv1, cv1], [0, 0, 1]], dtype=np.float64)

    # Distortion coefficients (equidistant/fisheye: k1, k2, k3, k4)
    D0 = np.array(cam0["distortion_coeffs"], dtype=np.float64).reshape(4, 1)
    D1 = np.array(cam1["distortion_coeffs"], dtype=np.float64).reshape(4, 1)

    # Image resolution
    w, h = cam0["resolution"]
    image_size = (w, h)

    # Stereo extrinsics: T_cn_cnm1 in cam1 gives T_cam1_cam0
    # This is the transform from cam0 (left) frame to cam1 (right) frame
    T_cn_cnm1 = np.array(cam1["T_cn_cnm1"], dtype=np.float64).reshape(4, 4)

    # Verify distortion model
    model0 = cam0.get("distortion_model", "equidistant")
    model1 = cam1.get("distortion_model", "equidistant")
    if model0 != "equidistant" or model1 != "equidistant":
        log.warning(
            "Expected equidistant distortion, got: cam0=%s, cam1=%s",
            model0, model1,
        )

    return {
        "K0": K0,
        "K1": K1,
        "D0": D0,
        "D1": D1,
        "T_cn_cnm1": T_cn_cnm1,
        "image_size": image_size,
    }


def compute_rectification_maps(
    calib: dict,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute stereo rectification undistort+remap tables.

    Args:
        calib: Output of load_kalibr_calibration().
        alpha: Free scaling parameter (0 = no black borders, 1 = keep all pixels).

    Returns:
        map1_left, map2_left: Remap tables for left camera.
        map1_right, map2_right: Remap tables for right camera.
        rect_params: Dict with rectified K, baseline, image size.
    """
    K0, K1 = calib["K0"], calib["K1"]
    D0, D1 = calib["D0"], calib["D1"]
    T_cn_cnm1 = calib["T_cn_cnm1"]
    w, h = calib["image_size"]
    image_size = (w, h)

    # Extract rotation and translation from T_cam1_cam0
    R = T_cn_cnm1[:3, :3]
    T = T_cn_cnm1[:3, 3]

    # Stereo rectify using fisheye model
    # flags=cv2.CALIB_ZERO_DISPARITY centers the principal points
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        K0, D0, K1, D1,
        imageSize=(w, h),
        R=R, tvec=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=image_size,
        fov_scale=1.0,
    )

    # Compute undistort+rectify maps
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K0, D0, R1, P1, image_size, cv2.CV_32FC1,
    )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R2, P2, image_size, cv2.CV_32FC1,
    )

    # Extract rectified intrinsics from P1 (3x4 projection matrix)
    # P1 = [K_rect | 0], P2 = [K_rect | t]
    K_rect = P1[:3, :3].copy()

    # Baseline = -P2[0,3] / P2[0,0] (in meters)
    baseline = abs(float(T[0]))  # Use raw translation magnitude
    if abs(P2[0, 3]) > 0:
        baseline = abs(P2[0, 3] / P2[0, 0])

    rect_params = {
        "fx": float(K_rect[0, 0]),
        "fy": float(K_rect[1, 1]),
        "cx": float(K_rect[0, 2]),
        "cy": float(K_rect[1, 2]),
        "baseline": float(baseline),
        "width": w,
        "height": h,
    }

    log.info("Rectified params: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, baseline=%.4fm",
             rect_params["fx"], rect_params["fy"],
             rect_params["cx"], rect_params["cy"],
             rect_params["baseline"])

    return map1_left, map2_left, map1_right, map2_right, rect_params


def rectify_sequence(
    input_dir: Path,
    calib_path: Path,
    output_dir: Path,
    alpha: float = 0.0,
) -> dict:
    """Rectify all stereo pairs in a sequence.

    Returns:
        rect_params dict with rectified intrinsics and baseline.
    """
    left_dir = input_dir / "left"
    right_dir = input_dir / "right"

    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Expected left/ and right/ dirs in {input_dir}")

    left_images = sorted(left_dir.glob("*.png"))
    right_images = sorted(right_dir.glob("*.png"))

    if len(left_images) != len(right_images):
        log.warning(
            "Frame count mismatch: %d left vs %d right",
            len(left_images), len(right_images),
        )
    n_frames = min(len(left_images), len(right_images))
    log.info("Rectifying %d stereo pairs from %s", n_frames, input_dir)

    # Load calibration and compute maps (once)
    calib = load_kalibr_calibration(calib_path)
    map1_l, map2_l, map1_r, map2_r, rect_params = compute_rectification_maps(
        calib, alpha=alpha
    )

    # Create output directories
    out_left = output_dir / "left"
    out_right = output_dir / "right"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        img_l = cv2.imread(str(left_images[i]), cv2.IMREAD_UNCHANGED)
        img_r = cv2.imread(str(right_images[i]), cv2.IMREAD_UNCHANGED)

        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        cv2.imwrite(str(out_left / f"{i:06d}.png"), rect_l)
        cv2.imwrite(str(out_right / f"{i:06d}.png"), rect_r)

        if (i + 1) % 200 == 0 or i == n_frames - 1:
            log.info("  Rectified %d/%d frames", i + 1, n_frames)

    # Save rectification params
    params_path = output_dir / "rectification_params.json"
    with open(params_path, "w") as f:
        json.dump(rect_params, f, indent=2)
    log.info("Saved rectification params to %s", params_path)

    # Copy timestamps if available
    for ts_file in ("left_timestamps.txt", "right_timestamps.txt", "groundtruth.txt"):
        src = input_dir / ts_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / ts_file)

    return rect_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Undistort + stereo rectify UZH-FPV fisheye stereo pairs"
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Extracted sequence directory (with left/ and right/)")
    parser.add_argument("--calib", required=True, type=Path,
                        help="Kalibr camchain YAML calibration file")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for rectified images")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Free scaling parameter (0=no borders, 1=keep all)")

    args = parser.parse_args()
    rectify_sequence(args.input_dir, args.calib, args.output_dir, args.alpha)


if __name__ == "__main__":
    main()
