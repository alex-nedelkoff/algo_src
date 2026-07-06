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


import cv2  # noqa: E402  (kept below the pure-geometry half: geometry tests
#             must not require cv2; move to top only if a linter forces it)

_GRID_STEP = 8


class FlowVelocity:
    """Sparse-LK floor-flow velocity. One instance per stream; call
    process() once per decoded frame in timestamp order."""

    def __init__(
        self,
        max_corners: int = 80,
        quality: float = 0.01,
        min_dist: int = 12,
        win: int = 21,
        levels: int = 3,
        fb_max_px: float = 1.0,
        min_decl_deg: float = 8.0,
        max_depth: float = 25.0,
        min_tracks: int = 6,
    ):
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist
        self.lk = dict(
            winSize=(win, win),
            maxLevel=levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.fb_max_px = fb_max_px
        self.min_decl_deg = min_decl_deg
        self.max_depth = max_depth
        self.min_tracks = min_tracks
        self._prev = None  # (gray, t_s, att, h)
        # precompute body rays on a coarse pixel grid for the floor mask
        try:
            from .camera import H, W, pixel_rays_body
        except ImportError:
            from camera import H, W, pixel_rays_body
        gy, gx = np.mgrid[0:H:_GRID_STEP, 0:W:_GRID_STEP]
        self._grid_shape = gy.shape
        self._grid_rays_b = pixel_rays_body(
            np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
        )
        self._img_shape = (H, W)

    def _floor_mask(self, att) -> np.ndarray:
        r_w = self._grid_rays_b @ R_world_body(*att).T
        below = (r_w[:, 2] > math.sin(math.radians(self.min_decl_deg)))
        m = below.reshape(self._grid_shape).astype(np.uint8) * 255
        return cv2.resize(
            m, (self._img_shape[1], self._img_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def reset(self) -> None:
        self._prev = None

    def process(self, gray, t_s: float, att, h: float):
        prev, self._prev = self._prev, (gray, t_s, att, h)
        if prev is None:
            return None
        pgray, pt, patt, ph = prev
        dt = t_s - pt
        if dt < 0.005 or dt > 0.15 or h <= 0.05 or ph <= 0.05:
            return None
        p0 = cv2.goodFeaturesToTrack(
            pgray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_dist,
            mask=self._floor_mask(patt),
        )
        if p0 is None or len(p0) < self.min_tracks:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pgray, gray, p0, None, **self.lk)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gray, pgray, p1, None, **self.lk)
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < self.fb_max_px)
        if good.sum() < self.min_tracks:
            return None
        uv0 = p0.reshape(-1, 2)[good]
        uv1 = p1.reshape(-1, 2)[good]
        v_tracks, used = velocity_from_tracks(
            uv0, uv1, patt, att, ph, h, dt,
            min_decl_deg=self.min_decl_deg, max_depth=self.max_depth,
        )
        est = robust_velocity(v_tracks, self.min_tracks)
        if est is None:
            return None
        v_w, sigma, ninl = est
        return v_w, sigma, ninl, int(good.sum())
