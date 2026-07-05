"""Position/velocity error-state filter for VQ2 (attitude is a trusted input).

Design (flight-derived, 2026-07-05):
  * The wfix'd gyro integration is ~exact on the sim IMU (measured 0.01-0.05 deg
    in-air drift), so attitude is NOT a filter state -- it is a trusted input,
    like time. The filter carries position + velocity in the level/world frame
    (yaw-aligned at the pad, gravity-vertical, origin at arming point).
  * Prediction at IMU rate: p += v dt ; v += a_lvl dt, where a_lvl is the
    gravity-compensated specific force rotated by the trusted attitude.
    Process noise on v absorbs the attitude-error gravity leak (~g*sin(err)).
  * Update: a gate detection matched to a map gate is a direct position
    measurement of the DRONE: z = map_pos - g_lvl(obs). Sigma grows with range;
    Mahalanobis gate rejects mismatches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81


@dataclass
class PosVelKF:
    # continuous-time noise densities
    q_vel: float = 0.12          # m/s^2 / sqrt(Hz): attitude-leak + model slop
    q_pos: float = 0.0
    sigma_meas_base: float = 0.35    # m at close range
    sigma_meas_per_m: float = 0.06   # + per metre of gate range
    maha_gate: float = 11.34         # chi2 0.99, 3 dof

    x: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [p(3), v(3)]
    P: np.ndarray = field(default_factory=lambda: np.diag([0.01] * 3 + [0.01] * 3))

    @property
    def p(self) -> np.ndarray:
        return self.x[:3]

    @property
    def v(self) -> np.ndarray:
        return self.x[3:]

    def predict(self, a_lvl: np.ndarray, dt: float) -> None:
        if dt <= 0 or dt > 0.2:
            return
        self.x[:3] += self.x[3:] * dt + 0.5 * a_lvl * dt * dt
        self.x[3:] += a_lvl * dt
        # F = [[I, I dt], [0, I]]
        P = self.P
        Ppp = P[:3, :3]; Ppv = P[:3, 3:]; Pvv = P[3:, 3:]
        Ppp_n = Ppp + dt * (Ppv + Ppv.T) + dt * dt * Pvv
        Ppv_n = Ppv + dt * Pvv
        self.P[:3, :3] = Ppp_n
        self.P[:3, 3:] = Ppv_n
        self.P[3:, :3] = Ppv_n.T
        self.P[3:, 3:] = Pvv + np.eye(3) * (self.q_vel ** 2) * dt
        self.P[:3, :3] += np.eye(3) * (self.q_pos ** 2) * dt

    def update_position(self, z: np.ndarray, rng: float) -> bool:
        """Direct drone-position measurement (map gate minus gate-relative obs).

        Returns True if accepted (passed the Mahalanobis gate)."""
        sig = self.sigma_meas_base + self.sigma_meas_per_m * rng
        R = np.eye(3) * (sig ** 2)
        # H = [I 0]
        S = self.P[:3, :3] + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        r = z - self.x[:3]
        d2 = float(r @ Sinv @ r)
        if d2 > self.maha_gate:
            return False
        K = self.P[:, :3] @ Sinv            # (6,3)
        self.x += K @ r
        IKH = np.eye(6)
        IKH[:, :3] -= K
        self.P = IKH @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return True

    def reset_at_rest(self, p0: np.ndarray | None = None) -> None:
        self.x[:3] = 0.0 if p0 is None else p0
        self.x[3:] = 0.0
        self.P = np.diag([0.01] * 3 + [0.01] * 3)


def accel_level(acc, roll: float, pitch: float) -> np.ndarray:
    """Gravity-compensated linear acceleration in the level frame from the
    trusted attitude (aero pitch sign handled: Ry built from -pitch)."""
    # HARD-validated against rest gravity (spawn tilt test): sin(+pitch) here.
    # (The detection chain uses sin(-pitch) with a compensating sign inside its
    # camera matrix -- composed chain is flight-validated; do not "unify" blindly.)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_lb = Ry @ Rx
    f_lvl = R_lb @ np.asarray(acc, float)
    # NED-ish level frame: z down; gravity vector = +g z
    return f_lvl + np.array([0.0, 0.0, GRAVITY])
