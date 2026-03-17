"""Tests for UZH-FPV stereo rectification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


def _make_synthetic_calibration(tmp_path: Path) -> Path:
    """Create a synthetic Kalibr-style calibration YAML for testing."""
    # Simple pinhole-like fisheye with minimal distortion
    calib = {
        "cam0": {
            "camera_model": "pinhole",
            "distortion_model": "equidistant",
            "intrinsics": [250.0, 250.0, 320.0, 240.0],
            "distortion_coeffs": [0.01, 0.001, 0.0001, 0.00001],
            "resolution": [640, 480],
            "T_cam_imu": [
                [1, 0, 0, 0.0],
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1],
            ],
        },
        "cam1": {
            "camera_model": "pinhole",
            "distortion_model": "equidistant",
            "intrinsics": [250.0, 250.0, 320.0, 240.0],
            "distortion_coeffs": [0.01, 0.001, 0.0001, 0.00001],
            "resolution": [640, 480],
            # T_cn_cnm1: cam1 in cam0 frame — 10cm baseline along x
            "T_cn_cnm1": [
                [1, 0, 0, 0.1],
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1],
            ],
            "T_cam_imu": [
                [1, 0, 0, 0.1],
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.0],
                [0, 0, 0, 1],
            ],
        },
    }

    calib_path = tmp_path / "calib.yaml"
    with open(calib_path, "w") as f:
        yaml.dump(calib, f)
    return calib_path


class TestKalibrParsing:
    """Test calibration YAML parsing."""

    def test_load_kalibr_calibration(self, tmp_path: Path):
        from perception.training.data.uzh_fpv.rectify_stereo import (
            load_kalibr_calibration,
        )

        calib_path = _make_synthetic_calibration(tmp_path)
        calib = load_kalibr_calibration(calib_path)

        assert calib["K0"].shape == (3, 3)
        assert calib["K1"].shape == (3, 3)
        assert calib["D0"].shape == (4, 1)
        assert calib["D1"].shape == (4, 1)
        assert calib["T_cn_cnm1"].shape == (4, 4)
        assert calib["image_size"] == (640, 480)

        # Check intrinsics values
        assert calib["K0"][0, 0] == pytest.approx(250.0)
        assert calib["K0"][0, 2] == pytest.approx(320.0)

        # Check baseline (10cm along x)
        assert calib["T_cn_cnm1"][0, 3] == pytest.approx(0.1)


class TestRectification:
    """Test stereo rectification pipeline."""

    def test_rectification_produces_horizontal_epilines(self, tmp_path: Path):
        """Rectified stereo pairs should have horizontal epipolar lines.

        Corresponding points in left and right images should have the same
        y-coordinate after rectification.
        """
        from perception.training.data.uzh_fpv.rectify_stereo import (
            compute_rectification_maps,
            load_kalibr_calibration,
        )

        calib_path = _make_synthetic_calibration(tmp_path)
        calib = load_kalibr_calibration(calib_path)
        map1_l, map2_l, map1_r, map2_r, rect_params = compute_rectification_maps(calib)

        # Create synthetic stereo pair with a known point
        w, h = calib["image_size"]
        img_l = np.zeros((h, w), dtype=np.uint8)
        img_r = np.zeros((h, w), dtype=np.uint8)

        # Draw circles at corresponding points
        # Left image: point at center
        cv2.circle(img_l, (320, 240), 10, 255, -1)
        # Right image: same point shifted by ~disparity
        cv2.circle(img_r, (300, 240), 10, 255, -1)

        # Rectify
        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # Find centroids of bright regions
        def find_centroid(img):
            moments = cv2.moments(img)
            if moments["m00"] == 0:
                return None
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return cx, cy

        c_l = find_centroid(rect_l)
        c_r = find_centroid(rect_r)

        assert c_l is not None, "Left rectified image has no bright region"
        assert c_r is not None, "Right rectified image has no bright region"

        # Y-coordinates should be approximately equal (horizontal epipolar lines)
        assert abs(c_l[1] - c_r[1]) < 3.0, (
            f"Epipolar constraint violated: left y={c_l[1]:.1f}, right y={c_r[1]:.1f}"
        )

    def test_rectification_params_saved(self, tmp_path: Path):
        """rectification_params.json should contain expected fields."""
        from perception.training.data.uzh_fpv.rectify_stereo import (
            compute_rectification_maps,
            load_kalibr_calibration,
        )

        calib_path = _make_synthetic_calibration(tmp_path)
        calib = load_kalibr_calibration(calib_path)
        _, _, _, _, rect_params = compute_rectification_maps(calib)

        assert "fx" in rect_params
        assert "fy" in rect_params
        assert "cx" in rect_params
        assert "cy" in rect_params
        assert "baseline" in rect_params
        assert "width" in rect_params
        assert "height" in rect_params
        assert rect_params["baseline"] > 0
        assert rect_params["fx"] > 0
