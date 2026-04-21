"""Tests for HTML validation report generation."""
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.compare_pybullet_vs_fixtures import write_html_report


def test_html_report_contains_per_pose_rows_and_aggregate(tmp_path: Path):
    rows = []
    for i in range(3):
        rgb_air = (np.ones((10, 10, 3)) * 50 * i).astype(np.uint8)
        rgb_pyb = (np.ones((10, 10, 3)) * 80 * i).astype(np.uint8)
        air_path = tmp_path / f"a_{i}.png"
        pyb_path = tmp_path / f"b_{i}.png"
        Image.fromarray(rgb_air).save(air_path)
        Image.fromarray(rgb_pyb).save(pyb_path)
        rows.append({
            "name": f"pose_{i:02d}",
            "airsim_png": air_path.name,
            "pybullet_png": pyb_path.name,
            "iou": 0.9 - i * 0.1,
        })
    report_path = tmp_path / "index.html"
    write_html_report(report_path, rows, mean_iou=0.8)
    html = report_path.read_text()
    assert "pose_00" in html
    assert "pose_02" in html
    assert "0.9" in html
    assert "Mean IoU" in html
    assert "0.8" in html
