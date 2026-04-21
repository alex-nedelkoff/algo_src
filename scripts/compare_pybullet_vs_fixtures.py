"""B-fixture validation harness: compare PyBullet renders vs AirSim fixtures.

Renders the warehouse + gates in PyBullet at each fixture's camera pose
and intrinsics, builds silhouette masks for both renders, computes
per-pose silhouette IoU, and writes an HTML report.

Pass criteria (per spec): mean IoU ≥ 0.85, no individual pose < 0.70.

Run after fixtures are captured (Task 14 on the Linux box) and after
the build pipeline has produced sim/assets/warehouse_v1/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union for two boolean silhouette masks.

    If both masks are empty, returns 1.0 (degenerate but interpretable).
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def render_pybullet_at_pose(
    client_id: int,
    *,
    position_enu,
    orientation_enu_xyzw,
    width: int,
    height: int,
    fov_deg: float,
    near: float = 0.05,
    far: float = 100.0,
):
    """Render an image from PyBullet at the given camera pose.

    Returns (rgb, depth, segmask). The orientation is the camera's
    world-frame rotation; the camera looks down its local +X axis by
    convention here (matches AirSim camera convention).
    """
    qx, qy, qz, qw = orientation_enu_xyzw
    rot = np.array(p.getMatrixFromQuaternion([qx, qy, qz, qw])).reshape(3, 3)
    forward = rot @ np.array([1.0, 0.0, 0.0])
    up = rot @ np.array([0.0, 0.0, 1.0])
    eye = np.array(position_enu, dtype=np.float64)
    target = eye + forward

    view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(
        fov=fov_deg, aspect=width / height, nearVal=near, farVal=far,
    )
    _w, _h, rgb, depth, segmask = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client_id,
    )
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(height, width, 4)
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
    segmask = np.asarray(segmask, dtype=np.int32).reshape(height, width)
    return rgb, depth, segmask


def pybullet_silhouette(segmask: np.ndarray) -> np.ndarray:
    """Anything not -1 (background) is geometry."""
    return segmask != -1


def airsim_depth_silhouette(depth: np.ndarray, far_plane: float) -> np.ndarray:
    """AirSim DepthPlanar returns 0 for no-return and very large for sky.

    Treat any depth in (0, far_plane) as geometry.
    """
    return (depth > 0.0) & (depth < far_plane)
