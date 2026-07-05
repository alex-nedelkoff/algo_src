"""VQ2 estimators under test.

Frame conventions (flight-verified 2026-07-03, see vault frames CANON):
  * HIGHRES_IMU gyro pitch-rate is MIRRORED vs physical: wfix = [1, -1, 1].
    Integrate pitch with -gyro_y or the estimate mirrors away from truth.
  * Accelerometer axes are physical; at rest the gravity split reads the
    known ~18 deg spawn tilt directly.
  * Accel tilt correction is INVALID under thrust (specific force points
    along the thrust axis with |f| ~ g regardless of tilt): seed attitude
    from accel AT REST, then integrate gyro only while flying.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .corpus import ImuSample

GRAVITY = 9.81
WFIX = (1.0, -1.0, 1.0)


@dataclass
class BaselineState:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0  # open-loop (unobservable without vision)
    vx_b: float = 0.0
    vy_b: float = 0.0
    vz_up: float = 0.0

    @property
    def tilt(self) -> float:
        return math.sqrt(self.roll**2 + self.pitch**2)


@dataclass
class BaselineImuEstimator:
    """The v9 flight math, extracted verbatim: wfix'd gyro attitude +
    gravity-compensated body-velocity integration. No vision, no bleed
    (bleed was a flight-safety hack; replay wants the honest drift)."""

    state: BaselineState = field(default_factory=BaselineState)
    _last_us: int | None = None

    def seed_from_rest(self, sample: ImuSample) -> None:
        ax, ay, az = sample.acc
        self.state.roll = math.atan2(ay, -az)
        self.state.pitch = math.atan2(ax, math.sqrt(ay * ay + az * az))
        self.state.vx_b = self.state.vy_b = self.state.vz_up = 0.0
        self._last_us = sample.t_us

    def predict(self, sample: ImuSample) -> None:
        if self._last_us is None or sample.t_us <= self._last_us:
            self._last_us = sample.t_us
            return
        dt = (sample.t_us - self._last_us) / 1e6
        self._last_us = sample.t_us
        s = self.state
        gx, gy, gz = (g * w for g, w in zip(sample.gyr, WFIX))
        s.roll += gx * dt
        s.pitch += gy * dt
        s.yaw += gz * dt
        sr, cr = math.sin(s.roll), math.cos(s.roll)
        sp, cp = math.sin(s.pitch), math.cos(s.pitch)
        ax, ay, az = sample.acc
        a_up = -(ax * (-sp) + ay * (sr * cp) + az * (cr * cp)) - GRAVITY
        s.vz_up += a_up * dt
        s.vx_b += (ax + GRAVITY * (-sp)) * dt
        s.vy_b += (ay + GRAVITY * (sr * cp)) * dt


def accel_implied_attitude(sample: ImuSample) -> tuple:
    """(roll, pitch) from the gravity direction. Valid ONLY at rest /
    unpowered quasi-static moments — see module docstring."""
    ax, ay, az = sample.acc
    roll = math.atan2(ay, -az)
    pitch = math.atan2(ax, math.sqrt(ay * ay + az * az))
    return roll, pitch


def is_at_rest(sample: ImuSample, gyro_tol: float = 0.005, acc_tol: float = 0.15) -> bool:
    """Rest detector: gyro ~ 0 and |specific force| ~ g."""
    gn = math.sqrt(sum(g * g for g in sample.gyr))
    an = math.sqrt(sum(a * a for a in sample.acc))
    return gn < gyro_tol and abs(an - GRAVITY) < acc_tol
