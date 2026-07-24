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
# fx=fy=width/2=320 @640x360 (90 deg HFOV, zero distortion). Adjudicated from
# the 1.5 m G0 aperture geometry + certified 10.595 m depth and Janahan's Kalibr
# chain (docs/handoff/g0_focal_adjudication.md); Alex GREEN-LIT 320 over the
# old 226. GateNet PnP default (vq2/gatenet/postprocess.py) tracks this.
FX = FY = 320.0
# Pixel-centre convention (W-1)/2 = 319.5,179.5. The adjudication's 320.0/180.0
# principal point is within corner-localization noise (~0.5 px) -- not churned.
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
# CAM_TILT 20 deg is REAL and correct (settled 2026-07-24,
# docs/handoff/cam_tilt_pair_experiment.md): hover-VP measures cam-in-body
# +18.6..+21.4 deg; the spawn sits ~-18 deg nose-down (honest rest accel),
# which is why a pad view still sees the gate near image centre.
CAM_TILT = math.radians(20.0)

# Single source of the pinhole intrinsic for every mapping/estimator consumer.
# (fx, fy, cx, cy) @ (W, H). DPVO bridge/route, GateNet PnP, fusion, vq2wp and
# the offline tools derive from these — do not re-hardcode the numbers.
FULL_INTRINSICS = (FX, FY, CX, CY)

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
    # einsum, not @: numpy-on-Accelerate emits spurious RuntimeWarnings in
    # the tall-matmul kernel (outputs verified finite); einsum path is clean
    return np.einsum("ij,kj->ik", rc, M_BODY_CAM)


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
