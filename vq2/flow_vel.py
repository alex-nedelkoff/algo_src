"""Ground-plane optical-flow velocity for VQ2.

Static-landmark identity for a floor feature seen in two frames:
    L = x0 + d0*r0_w = x1 + d1*r1_w
      =>  drone displacement  x1 - x0 = d0*r0_w - d1*r1_w
with depth d = h / r_w_z from the flat-floor assumption (z down, floor
below the drone => ray z component positive). Height error scales the
recovered speed proportionally (10% h error -> 10% v error) — h comes
from the KF z which carries vision fixes.

Attitude must be the flight-proven wfix chain (vq2wp.py), passed per
frame; derotation is implicit because each frame's rays are rotated by
its own full attitude.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from .camera import pixel_rays_body, R_world_body
except ImportError:  # flat deploy next to vq2wp.py on the laptop
    from camera import pixel_rays_body, R_world_body

Z_FLOOR = 0.15  # m, floor z in the arming frame (z down); Task 6 calibrates

SIGMA_FLOOR = 0.15      # m/s, never report tighter than this
MAD_K = 3.0             # inlier band = MAD_K * 1.4826 * MAD + 0.05 m/s


def velocity_from_tracks(
    uv0,
    uv1,
    att0,
    att1,
    h0: float,
    h1: float,
    dt: float,
    min_decl_deg: float = 8.0,
    max_depth: float = 25.0,
):
    """Per-track world-frame drone velocity from floor-feature pairs.

    Returns (v_w (M,3) for the used tracks, used (N,) bool mask)."""
    uv0 = np.atleast_2d(np.asarray(uv0, float))
    uv1 = np.atleast_2d(np.asarray(uv1, float))
    r0 = pixel_rays_body(uv0) @ R_world_body(*att0).T
    r1 = pixel_rays_body(uv1) @ R_world_body(*att1).T
    min_z = math.sin(math.radians(min_decl_deg))
    used = (r0[:, 2] > min_z) & (r1[:, 2] > min_z)
    if h0 <= 0.05 or h1 <= 0.05 or dt <= 0:
        return np.zeros((0, 3)), np.zeros(len(uv0), dtype=bool)
    d0 = np.where(used, h0 / np.maximum(r0[:, 2], 1e-9), 0.0)
    d1 = np.where(used, h1 / np.maximum(r1[:, 2], 1e-9), 0.0)
    used &= (d0 < max_depth) & (d1 < max_depth)
    disp = d0[used, None] * r0[used] - d1[used, None] * r1[used]
    return disp / dt, used


def robust_velocity(v_tracks, min_tracks: int = 6):
    """Median + MAD inlier mean. Returns (v (3,), sigma, n_inliers) or None."""
    v = np.atleast_2d(np.asarray(v_tracks, float))
    if len(v) < min_tracks:
        return None
    med = np.median(v, axis=0)
    dev = np.linalg.norm(v - med, axis=1)
    mad = np.median(dev)
    inl = dev <= MAD_K * 1.4826 * mad + 0.05
    if inl.sum() < min_tracks:
        return None
    v_in = v[inl]
    v_est = v_in.mean(axis=0)
    spread = float(np.linalg.norm(v_in.std(axis=0)))
    sigma = max(SIGMA_FLOOR, spread / math.sqrt(len(v_in)))
    return v_est, sigma, int(inl.sum())
