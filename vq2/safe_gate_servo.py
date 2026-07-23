"""Fail-closed visual-servo policy for short calibration captures.

This is intentionally *not* the race controller.  It only authorizes a small
forward reference while one aperture track is stable, centred, in-frame, and
well below the close-range crop limit.  Any ambiguous observation commands a
hold; callers remain responsible for landing on a sustained hold.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateObservation:
    u: float
    v: float
    width_px: float
    height_px: float


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
