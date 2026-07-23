"""Fail-closed visual-servo policy for short calibration captures.

This is intentionally *not* the race controller.  It only authorizes a small
forward reference while one aperture track is stable, centred, in-frame, and
well below the close-range crop limit.  Any ambiguous observation commands a
hold; callers remain responsible for landing on a sustained hold.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aigp.gate_detect import DEFAULT_PARAMS, red_mask


@dataclass(frozen=True)
class GateObservation:
    u: float
    v: float
    width_px: float
    height_px: float


def detect_aperture_gate(bgr: np.ndarray, params: dict | None = None) -> GateObservation | None:
    """Return the largest square red frame that contains a substantial dark hole.

    The ordinary HSV detector is deliberately broad and accepts red signs.  A
    calibration capture needs a gate aperture specifically, so solid red
    components and red components without a nested hole are rejected here.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    contours, hierarchy = cv2.findContours(red_mask(bgr, p), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    candidates: list[tuple[float, GateObservation]] = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            continue
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if area < p["min_area_px"] or w <= 0 or h <= 0:
            continue
        if abs(w / float(h) - 1.0) > p["square_tol"] or (y + h / 2.0) > p["max_v_frac"] * bgr.shape[0]:
            continue
        best_hole = 0.0
        hole_box = None
        child = hierarchy[0][i][2]
        while child != -1:
            hole_area = float(cv2.contourArea(contours[child]))
            if hole_area > best_hole:
                best_hole = hole_area
                hole_box = cv2.boundingRect(contours[child])
            child = hierarchy[0][child][0]
        if hole_box is None or best_hole < 0.05 * area:
            continue
        hx, hy, hw, hh = hole_box
        candidates.append((area, GateObservation(hx + hw / 2.0, hy + hh / 2.0, float(w), float(h))))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


@dataclass(frozen=True)
class SafeServoCommand:
    yaw_rate_rad_s: float
    forward_m_s: float
    accepted: bool
    reason: str


@dataclass
class SafeGateServo:
    """Continuity-gated yaw/forward policy, expressed in image coordinates."""

    cx: float = 320.0
    fx: float = 320.0
    min_width_px: float = 50.0
    max_width_px: float = 105.0
    max_center_v_px: float = 245.0
    max_step_u_px: float = 55.0
    max_size_ratio: float = 1.35
    stable_required: int = 6
    yaw_gain: float = -0.35  # empirical legacy sign: gate-right -> negative yaw rate
    yaw_rate_limit: float = 0.08
    center_for_forward: float = 0.075
    forward_m_s: float = 0.08
    _last: GateObservation | None = None
    _stable: int = 0
    _track_lost: bool = False

    def step(self, obs: GateObservation | None) -> SafeServoCommand:
        if self._track_lost:
            return SafeServoCommand(0.0, 0.0, False, "track was lost; reset required")
        if obs is None:
            if self._last is not None:
                self._track_lost = True
            self._stable = 0
            return SafeServoCommand(0.0, 0.0, False, "no gate")
        if not (self.min_width_px <= obs.width_px <= self.max_width_px):
            self._stable = 0
            return SafeServoCommand(0.0, 0.0, False, "gate size outside safe band")
        if obs.v > self.max_center_v_px:
            self._stable = 0
            return SafeServoCommand(0.0, 0.0, False, "gate is low or clipped")
        if self._last is not None:
            size_ratio = max(obs.width_px, self._last.width_px) / min(obs.width_px, self._last.width_px)
            if abs(obs.u - self._last.u) > self.max_step_u_px or size_ratio > self.max_size_ratio:
                self._stable = 0
                self._last = None
                self._track_lost = True
                return SafeServoCommand(0.0, 0.0, False, "identity continuity rejected")
        self._last = obs
        self._stable += 1
        ex = (obs.u - self.cx) / self.fx
        yaw_rate = max(-self.yaw_rate_limit, min(self.yaw_rate_limit, self.yaw_gain * ex))
        forward = self.forward_m_s if self._stable >= self.stable_required and abs(ex) <= self.center_for_forward else 0.0
        reason = "stable centred track" if forward else "centering / acquiring"
        return SafeServoCommand(yaw_rate, forward, True, reason)
