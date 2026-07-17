"""Manhattan-vertical vanishing-point attitude: absolute roll/pitch from
image line structure (T6c architecture harvest, Tier A).

Why: the gyro chain has no absolute tilt reference IN FLIGHT — the accel CF
is quasi-static-only (fusion.AttitudeTracker), and 1-3 deg of accumulated
gap error is a 0.17-0.5 m/s^2 phantom lateral force (the blind-leg killer,
VQ2-DRIFT-01). World-vertical structure (gate posts, wall edges, shelving)
projects to segments whose common vanishing point IS the gravity direction
in camera coordinates — an absolute roll/pitch observation available every
frame, in flight, at any speed.

Method: prior-gated weighted least squares, NOT cold-start RANSAC. The gyro
attitude is always within a few degrees, so we use it to (a) gate segments
to those consistent with a near-vertical VP and (b) resolve the +-v sign
and reject horizontal-VP misidentification. Deterministic (no seed), cheap
(one LSD pass + a 3x3 eigensolve), and honest about what it is: a drift
corrector, not an absolute-from-scratch solver.

  1. LSD segments (cv2.createLineSegmentDetector — present in cv2 5.x;
     ximgproc FastLineDetector/ELSED are NOT in the monorace env).
  2. Segment interpretation-plane normal n = p1 x p2 (unit endpoint rays):
     any VP v on the segment's line satisfies n . v = 0.
  3. Gate: keep segments with |n . g_pred_cam| < sin(gate_deg), g_pred from
     the prior attitude. Excludes floor/horizontal structure.
  4. v = smallest-eigenvalue eigenvector of sum(w n n^T), length-weighted,
     with IRLS Huber reweighting on |n . v| to shed stragglers.
  5. g_cam = +-v (sign toward prior) -> g_body = M_BODY_CAM @ g_cam ->
     (roll, pitch) from g_b = R_level_body(r,p).T @ [0,0,1]
                          = [-sin p, sin r cos p, cos r cos p].

Rejects (returns None): too few gated segments, weak eigengap (no single
dominant VP), or solution further than max_dev_deg from the prior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CX, CY, FX, FY, M_BODY_CAM


@dataclass
class VpConfig:
    min_len_px: float = 14.0     # LSD segments shorter than this are noise
    gate_deg: float = 10.0       # prior cone half-angle for segment gating
    max_dev_deg: float = 8.0     # solution beyond this from prior = wrong VP
    min_segments: int = 6        # below this the eigensolve is not trustworthy
    irls_iters: int = 2
    huber_delta: float = 0.01    # |n.v| residual scale for IRLS weights
    min_eigengap: float = 4.0    # lambda1/lambda0 of the normal matrix


@dataclass
class VpResult:
    roll: float
    pitch: float
    n_seg: int                   # gated segments used in the final solve
    quality: float               # min(1, eigengap/min_eigengap) in [0, 1]
    g_body: np.ndarray           # measured gravity direction, body frame


def gravity_body(roll: float, pitch: float) -> np.ndarray:
    """Gravity (world +z, down) expressed in the body frame at (roll, pitch).
    Identically R_level_body(roll, pitch).T @ [0,0,1] — kept closed-form so
    vp.py stays consistent with camera.R_level_body by construction."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    return np.array([-sp, sr * cp, cr * cp])


def roll_pitch_from_gravity(g_body: np.ndarray) -> tuple:
    """Invert gravity_body: g_b = [-sin p, sin r cos p, cos r cos p]."""
    pitch = math.atan2(-g_body[0], math.hypot(g_body[1], g_body[2]))
    roll = math.atan2(g_body[1], g_body[2])
    return roll, pitch


_LSD = None


def _segments(gray: np.ndarray, min_len_px: float) -> tuple:
    """LSD line segments -> ((N,2,2) endpoints px, (N,) lengths px)."""
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector()
    lines = _LSD.detect(gray)[0]
    if lines is None or len(lines) == 0:
        return np.zeros((0, 2, 2)), np.zeros(0)
    seg = lines.reshape(-1, 4).astype(float)
    d = seg[:, 2:] - seg[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    keep = length >= min_len_px
    return seg[keep].reshape(-1, 2, 2), length[keep]


def vp_roll_pitch(
    gray: np.ndarray,
    roll_prior: float,
    pitch_prior: float,
    cfg: VpConfig = VpConfig(),
) -> VpResult | None:
    ends, w = _segments(gray, cfg.min_len_px)
    if len(ends) < cfg.min_segments:
        return None

    # endpoint pixel coords -> unit rays in the CAMERA frame (not body:
    # the VP eigensolve lives in camera coords, M_BODY_CAM applied at the end)
    rays = np.concatenate(
        [(ends[..., 0:1] - CX) / FX, (ends[..., 1:2] - CY) / FY,
         np.ones_like(ends[..., 0:1])], axis=2)
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    n = np.cross(rays[:, 0], rays[:, 1])
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-9
    n, w = n[ok] / nn[ok, None], w[ok]

    g_pred_body = gravity_body(roll_prior, pitch_prior)
    g_pred_cam = M_BODY_CAM.T @ g_pred_body
    gated = np.abs(n @ g_pred_cam) < math.sin(math.radians(cfg.gate_deg))
    n, w = n[gated], w[gated]
    if len(n) < cfg.min_segments:
        return None

    # weighted eigensolve, IRLS-refined: v minimises sum w (n.v)^2
    wi = w.copy()
    v = g_pred_cam
    for _ in range(cfg.irls_iters + 1):
        M = (n * wi[:, None]).T @ n
        evals, evecs = np.linalg.eigh(M)
        v = evecs[:, 0]
        r = np.abs(n @ v)
        wi = w * np.minimum(1.0, cfg.huber_delta / np.maximum(r, 1e-12))
    eigengap = float(evals[1] / max(evals[0], 1e-12))
    if eigengap < cfg.min_eigengap:
        return None

    if v @ g_pred_cam < 0:
        v = -v
    g_body = M_BODY_CAM @ v
    dev = math.degrees(math.acos(np.clip(g_body @ g_pred_body, -1.0, 1.0)))
    if dev > cfg.max_dev_deg:
        return None

    roll, pitch = roll_pitch_from_gravity(g_body)
    return VpResult(
        roll=roll, pitch=pitch, n_seg=int(len(n)),
        quality=min(1.0, eigengap / cfg.min_eigengap), g_body=g_body)
