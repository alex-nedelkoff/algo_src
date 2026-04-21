"""Tests for silhouette IoU computation."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from scripts.compare_pybullet_vs_fixtures import (
    silhouette_iou,
    render_pybullet_at_pose,
    pybullet_silhouette,
    airsim_depth_silhouette,
)


def test_iou_perfect_match_is_one():
    a = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    assert silhouette_iou(a, a.copy()) == 1.0


def test_iou_disjoint_is_zero():
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[0, 1], [0, 0]], dtype=bool)
    assert silhouette_iou(a, b) == 0.0


def test_iou_half_overlap():
    a = np.array([[1, 1, 0, 0]], dtype=bool)
    b = np.array([[0, 1, 1, 0]], dtype=bool)
    # Intersection = 1 px, Union = 3 px → IoU = 1/3
    assert abs(silhouette_iou(a, b) - 1 / 3) < 1e-9


def test_iou_both_empty_is_one():
    a = np.zeros((2, 2), dtype=bool)
    assert silhouette_iou(a, a.copy()) == 1.0


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    out = tmp_path_factory.mktemp("assets")
    tsdf = out.parent / "tsdf.npz"
    gates = out.parent / "gates.json"

    n = 24
    sdf = np.ones((n, n, n), dtype=np.float32) * 0.5
    sdf[5:25, 5:7, 5:25] = -0.1
    np.savez(tsdf, sdf=sdf, voxel_size=np.float32(0.1),
             origin=np.array([0.0, 0.0, -2.0], dtype=np.float32))

    gates.write_text(json.dumps({
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }))

    subprocess.run(
        [sys.executable, "-m", "scripts.warehouse_to_urdf",
         "--tsdf", str(tsdf), "--gates", str(gates), "--out", str(out),
         "--target-triangles", "200", "--gate-clip-radius", "0.3"],
        check=True,
    )
    return out


def test_pybullet_render_returns_segmask(assets):
    cid = p.connect(p.DIRECT)
    try:
        from sim.pybullet.warehouse_loader import WarehouseScene
        WarehouseScene(asset_dir=assets).load_into(cid)
        rgb, depth, segmask = render_pybullet_at_pose(
            cid,
            position_enu=[0.0, 1.5, 1.5],
            orientation_enu_xyzw=[0, 0, 0, 1],
            width=64, height=64, fov_deg=90.0,
        )
        assert rgb.shape == (64, 64, 4)  # PyBullet returns RGBA
        assert depth.shape == (64, 64)
        assert segmask.shape == (64, 64)
        # We're looking at a wall — should see something.
        sil = pybullet_silhouette(segmask)
        assert sil.any()
    finally:
        p.disconnect(cid)


def test_airsim_depth_silhouette_thresholds_correctly():
    depth = np.array([[0.5, 50.0, 0.0], [10.0, 1000.0, 5.0]])
    sil = airsim_depth_silhouette(depth, far_plane=100.0)
    # 0 (no return) and >far_plane both → False; everything else → True.
    expected = np.array([[True, True, False], [True, False, True]])
    np.testing.assert_array_equal(sil, expected)
