"""3D gate-corner map + pixel projection + position Jacobian.

Foundation for tight per-corner KF updates (playbook 0.3 / ADR-VINS):
  uv_pred = project(map corner, drone position, trusted attitude)
  innovation = detected corner - uv_pred ;  H = d(uv)/d(position)
Attitude is a trusted input (wfix chain), so H has only a position block
and the existing PosVelKF absorbs corner updates without restructuring.

Frames: spawn/world (x downcourse, y right, z down). Camera model from
vq2/camera.py (nose cam, 20 deg up, fx=fy=320).
"""
from __future__ import annotations

import numpy as np

from .camera import CX, CY, FX, FY, M_BODY_CAM, R_world_body

APERTURE_HALF = 0.75   # m; gate_physical_size=1.5 (postprocess.py), VQ1-size


def _square(center, normal_axis="x"):
    """4 aperture corners for a gate whose plane is perpendicular to the
    course x-axis (G1/HIGH) -- corners span y and z."""
    cx, cy, cz = center
    h = APERTURE_HALF
    return np.array([
        [cx, cy - h, cz - h],   # top-left    (z down: -h is UP)
        [cx, cy + h, cz - h],   # top-right
        [cx, cy + h, cz + h],   # bottom-right
        [cx, cy - h, cz + h],   # bottom-left
    ])


def _square_yawed(center, normal_xy):
    """Corners for a gate with aperture plane perpendicular to normal_xy."""
    n = np.asarray(normal_xy, float)
    n = n / np.linalg.norm(n)
    lat = np.array([-n[1], n[0], 0.0])   # in-plane horizontal direction
    up = np.array([0.0, 0.0, -1.0])
    c = np.asarray(center, float)
    h = APERTURE_HALF
    return np.array([
        c - h * lat + h * up,
        c + h * lat + h * up,
        c + h * lat - h * up,
        c - h * lat - h * up,
    ])


GATE_CORNERS = {
    "G1": _square((11.0, 0.0, -1.3)),
    "HIGH": _square((10.4, 0.0, -4.0)),
    "G2": _square_yawed((30.5, 8.5, -1.5), (0.95, 0.2)),  # N2, vq2wp.py:76
}


def project_corners(p, att, gate: str):
    """Project a gate's 4 map corners into pixels for a drone at world
    position p with attitude att=(roll, pitch, yaw).

    Returns (uv (4,2), visible (4,) bool) -- visible = in front of the
    camera and inside the 640x360 frame."""
    corners = GATE_CORNERS[gate]
    R_wb = R_world_body(*att)
    rel_w = corners - np.asarray(p, float)          # world
    rel_b = rel_w @ R_wb                             # rows: R_wb.T @ v
    rel_c = rel_b @ M_BODY_CAM                       # body -> camera
    z = rel_c[:, 2]
    ok = z > 0.2
    zs = np.where(ok, z, 1.0)
    uv = np.stack([rel_c[:, 0] / zs * FX + CX,
                   rel_c[:, 1] / zs * FY + CY], axis=1)
    vis = ok & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & \
        (uv[:, 1] >= 0) & (uv[:, 1] < 360)
    return uv, vis


def corner_jacobian(p, att, gate: str):
    """d(uv)/d(p) for each corner: (4, 2, 3). Analytic.

    rel_c = M^T R_wb^T (c_w - p)  =>  d(rel_c)/dp = -(R_wb M)^T = -A
    u = fx * rc_x / rc_z + cx     =>  du/dp = fx (rc_z * -A_x - rc_x * -A_z)/rc_z^2
    """
    corners = GATE_CORNERS[gate]
    R_wb = R_world_body(*att)
    A = (R_wb @ M_BODY_CAM)          # rel_c = A.T @ (c_w - p)
    rel_c = (corners - np.asarray(p, float)) @ A
    H = np.zeros((len(corners), 2, 3))
    for i, rc in enumerate(rel_c):
        x, y, z = rc
        if z <= 0.2:
            continue
        dx, dy, dz = -A[:, 0], -A[:, 1], -A[:, 2]   # d(rc_x)/dp etc.
        H[i, 0] = FX * (z * dx - x * dz) / (z * z)
        H[i, 1] = FY * (z * dy - y * dz) / (z * z)
    return H
