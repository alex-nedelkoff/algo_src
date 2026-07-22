"""Build calibration-gated, quarantined pillar text-panel proposals.

This module deliberately has no dependency on the flight pillar map.  A
proposal is an auditable statement about a *text panel* seen while hovering;
it is never evidence that the corresponding physical pillar axis is known.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .corpus import ImuSample


@dataclass(frozen=True)
class ApprovedCameraModel:
    """Camera intrinsics allowed to enter reset-NED mapping.

    The approval flag is intentionally separate from a numerical camera model:
    merely selecting either historic focal-length hypothesis must not unlock a
    map transform.
    """

    id: str
    K: np.ndarray
    approved: bool
    sigma_px: float = 1.0


@dataclass(frozen=True)
class GateAnchor:
    """Known G0 datum and its camera-relative PnP pose at hover start.

    ``R_cam_gate`` is OpenCV's object-to-camera rotation.  ``R_world_gate``
    must use an upright, right-handed G0 convention; its construction is kept
    at the call site so a 90-degree in-plane ambiguity cannot be hidden here.
    """

    p_world_gate: np.ndarray
    R_world_gate: np.ndarray
    R_cam_gate: np.ndarray
    t_cam_gate: np.ndarray
    t_rx_wall: float
    translation_sigma_m: float


@dataclass(frozen=True)
class TextPanelObservation:
    """One already-OCR'd and PnP-solved station-text observation."""

    frame: str
    number: str
    ocr_confidence: float
    t_rx_wall: float
    t_cam_text: np.ndarray
    pnp_rms_px: float
    edge_center_u: float | None = None
    edge_width_px: float | None = None
    edge_quality: float | None = None


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def anchor_camera_pose(anchor: GateAnchor) -> tuple[np.ndarray, np.ndarray]:
    """Return world-from-camera rotation and camera origin at the G0 anchor."""
    R_wc = np.asarray(anchor.R_world_gate, float) @ np.asarray(anchor.R_cam_gate, float).T
    p_wc = np.asarray(anchor.p_world_gate, float) - R_wc @ np.asarray(anchor.t_cam_gate, float)
    return R_wc, p_wc


def integrate_hover_yaw(imu: Iterable[ImuSample], t0_wall: float, t1_wall: float) -> tuple[float, float]:
    """Integrate received IMU z gyro between wall-clock endpoints.

    Returns ``(delta_yaw, largest_gap_s)``.  The caller must reject a result
    whose gap exceeds its association policy; no false precision is claimed.
    """
    samples = sorted((s for s in imu if t0_wall <= s.rx_wall <= t1_wall), key=lambda s: s.rx_wall)
    if len(samples) < 2:
        raise ValueError("need at least two IMU samples spanning the observation")
    yaw = 0.0
    max_gap = 0.0
    for a, b in zip(samples, samples[1:]):
        dt = b.rx_wall - a.rx_wall
        if dt <= 0.0:
            continue
        max_gap = max(max_gap, dt)
        yaw += 0.5 * (a.gyr[2] + b.gyr[2]) * dt
    return yaw, max_gap


def robust_fuse(points: Iterable[np.ndarray], translation_sigma_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Median/MAD centre and diagonal covariance, inflated for hover drift."""
    p = np.asarray(list(points), dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] != 3:
        raise ValueError("need at least two finite 3D observations to fuse")
    if not np.all(np.isfinite(p)):
        raise ValueError("observations must be finite")
    centre = np.median(p, axis=0)
    # 1.4826 * MAD estimates a Gaussian sigma without letting one bad PnP
    # solve dominate the candidate; translation is an explicit independent
    # uncertainty rather than a hidden zero-motion assumption.
    sigma = 1.4826 * np.median(np.abs(p - centre), axis=0)
    covariance = np.diag(sigma * sigma + float(translation_sigma_m) ** 2)
    return centre, covariance


def build_candidate(
    anchor: GateAnchor,
    camera: ApprovedCameraModel,
    imu: Iterable[ImuSample],
    observations: Iterable[TextPanelObservation],
    *,
    min_ocr_confidence: float = 0.90,
    max_pnp_rms_px: float = 3.0,
    max_imu_gap_s: float = 0.10,
    min_edge_quality: float = 0.75,
) -> dict:
    """Build one quarantined text-panel candidate and its complete sidecar.

    Observations are rejected individually.  A physical pillar axis is never
    emitted because the panel-face-to-axis offset has not been calibrated.
    """
    if not camera.approved:
        raise ValueError("camera calibration is not approved; refusing reset-NED map output")
    K = np.asarray(camera.K, float)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("approved camera K must be a finite 3x3 matrix")
    obs = list(observations)
    imu = list(imu)
    if not obs:
        raise ValueError("no text-panel observations")
    numbers = {o.number for o in obs}
    if len(numbers) != 1:
        raise ValueError("one candidate may contain exactly one station number")

    R_wc0, p_wc = anchor_camera_pose(anchor)
    accepted, points = [], []
    for o in obs:
        reason = None
        if o.ocr_confidence < min_ocr_confidence:
            reason = "low_ocr_confidence"
        elif o.pnp_rms_px > max_pnp_rms_px:
            reason = "poor_pnp_reprojection"
        elif o.edge_quality is not None and o.edge_quality < min_edge_quality:
            reason = "unstable_pillar_edges"
        if reason is None:
            try:
                dyaw, gap = integrate_hover_yaw(imu, anchor.t_rx_wall, o.t_rx_wall)
            except ValueError:
                dyaw, gap, reason = 0.0, float("inf"), "missing_imu_association"
            if reason is None and gap > max_imu_gap_s + 1e-9:
                reason = "imu_gap_too_large"
        if reason is None:
            R_wc = _rz(dyaw) @ R_wc0
            p_world = p_wc + R_wc @ np.asarray(o.t_cam_text, float)
            points.append(p_world)
        accepted.append({
            "frame": o.frame, "station_number": o.number,
            "ocr_confidence": o.ocr_confidence, "pnp_rms_px": o.pnp_rms_px,
            "t_rx_wall": o.t_rx_wall, "yaw_delta_rad": None if reason else dyaw,
            "imu_max_gap_s": None if reason else gap,
            "edge_fit": {"center_u": o.edge_center_u, "width_px": o.edge_width_px,
                         "quality": o.edge_quality},
            "accepted": reason is None, "rejection": reason,
        })
    if len(points) < 2:
        raise ValueError("fewer than two observations passed candidate gates")
    centre, covariance = robust_fuse(points, anchor.translation_sigma_m)
    number = next(iter(numbers))
    return {
        "version": "pillar-candidate-1",
        "frame": "reset-NED",
        "units": "m",
        "camera_calibration": {"id": camera.id, "approved": True, "sigma_px": camera.sigma_px},
        "text_panels": [{
            "candidate_id": f"station-{number}-panel-1", "number": number,
            "position_m": centre.tolist(), "covariance_m2": covariance.tolist(),
            "observations": accepted,
        }],
        "quarantined": [{
            "id": f"station-{number}-pillar-axis-1", "number": number,
            "reason": "text panel fused; panel-face-to-pillar-axis offset is uncalibrated",
            "human_approved": False,
        }],
    }
