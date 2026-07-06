"""VQ2 camera model (constants flight-validated in vq2/live/vq2wp.py).

Pixel convention: u right, v down, origin top-left (cv2). Camera frame:
x right, y down, z forward (optical axis). Body frame: x fwd, y right,
z down. Camera pitched CAM_TILT up in body, no yaw offset (VQ2 nose cam,
flight-validated 07-06; VQ1's camyaw +5.7 deg does NOT apply here).
"""
from __future__ import annotations

import math

import numpy as np

W, H = 640, 360
FX = FY = 226.0  # px; GateNet PnP default (vq2/gatenet/postprocess.py)
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
CAM_TILT = math.radians(20.0)

_ct, _st = math.cos(CAM_TILT), math.sin(CAM_TILT)
_CZ = np.array([_ct, 0.0, -_st])
_CY = np.array([_st, 0.0, _ct])
_CX = np.cross(_CY, _CZ)
M_BODY_CAM = np.stack([_CX, _CY, _CZ], axis=1)


def pixel_rays_body(uv) -> np.ndarray:
    """(N,2) pixel coords -> (N,3) unit rays in the body frame."""
    uv = np.atleast_2d(np.asarray(uv, float))
    rc = np.stack(
        [(uv[:, 0] - CX) / FX, (uv[:, 1] - CY) / FY, np.ones(len(uv))], axis=1
    )
    rc /= np.linalg.norm(rc, axis=1, keepdims=True)
    return rc @ M_BODY_CAM.T


def R_level_body(roll: float, pitch: float) -> np.ndarray:
    """Identical composition to eskf.accel_level (Ry @ Rx). Keep in sync."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx


def R_world_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) @ R_level_body — matches vq2wp.py's a_w / g_w rotations."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ R_level_body(roll, pitch)
