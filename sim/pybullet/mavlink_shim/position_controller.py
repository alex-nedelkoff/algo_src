"""SE(3) geometric position controller (Mellinger & Kumar 2011).

Outputs (q_des_enu_wxyz, thrust_norm) — byte-compatible with the SET_ATTITUDE_TARGET
payload format that AttitudeController already consumes. This stage runs ONLY
when the shim is in position-mode (last inbound message was SET_POSITION_TARGET_LOCAL_NED).

Algorithm (all vectors in ENU world frame except where noted):
  e_p = pos - pos_target                       # position error
  e_v = vel - vel_target                       # velocity error
  f_des = -K_p·e_p - K_d·e_v + m·g·ẑ + m·a_target
  thrust_norm = clip(f_des · z_b_current / max_thrust, 0, 1)
  z_b_des = f_des / ‖f_des‖
  x_c = [cos(yaw_target), sin(yaw_target), 0]   # heading reference
  y_b_des = normalize(z_b_des × x_c)
  x_b_des = y_b_des × z_b_des
  R_des = [x_b_des | y_b_des | z_b_des]
  q_des = mat_to_quat_wxyz(R_des)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState


# Conservative Mellinger-ish gains for a 5"-class racing quad. Vertical gains
# higher than horizontal because altitude error is corrected through pure
# thrust magnitude (cheap) while horizontal error needs to tilt the body first.
_K_P_DEFAULT = np.array([4.0, 4.0, 8.0])
_K_D_DEFAULT = np.array([3.0, 3.0, 6.0])
_G = 9.81


def _rot_to_quat_wxyz(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix → unit quaternion (wxyz). Stable Shepperd's method."""
    t = float(np.trace(R))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _quat_rotate_vector(q_wxyz: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate v by quaternion (wxyz) — i.e. R_body_to_world @ v."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])
    return R @ v


@dataclass
class PositionController:
    params: VehicleParams
    k_p: NDArray[np.float64] = field(default_factory=lambda: _K_P_DEFAULT.copy())
    k_d: NDArray[np.float64] = field(default_factory=lambda: _K_D_DEFAULT.copy())

    def __post_init__(self) -> None:
        self._max_thrust = 4.0 * self.params.k_thrust * self.params.max_omega ** 2
        self._k_p = np.asarray(self.k_p, dtype=np.float64)
        self._k_d = np.asarray(self.k_d, dtype=np.float64)

    def compute(self, *, target, state: DroneState) -> tuple[NDArray[np.float64], float]:
        """Return (q_des_enu_wxyz, thrust_normalized in [0,1]).

        target: PositionTarget (ENU). Disabled fields are zero.
        """
        m = float(self.params.mass)
        # Position / velocity error (ENU).
        e_p = state.pos_enu - target.pos_enu
        e_v = state.vel_enu - target.vel_enu

        # Desired thrust vector in world.
        f_des = -self._k_p * e_p - self._k_d * e_v + m * np.array([0.0, 0.0, _G]) + m * target.accel_enu

        # Current body-z in world (rotate (0,0,1) by current attitude).
        z_b_current = _quat_rotate_vector(state.quat_wxyz, np.array([0.0, 0.0, 1.0]))
        thrust_n_world = float(np.dot(f_des, z_b_current))
        thrust_norm = float(np.clip(thrust_n_world / self._max_thrust, 0.0, 1.0))

        # Desired body-z direction.
        f_norm = float(np.linalg.norm(f_des))
        if f_norm < 1e-6:
            # Degenerate: no thrust needed → keep current attitude.
            return state.quat_wxyz.copy(), 0.0
        z_b_des = f_des / f_norm

        # Heading reference in horizontal plane.
        yaw_t = float(target.yaw_enu) if target.use_yaw else 0.0
        x_c = np.array([np.cos(yaw_t), np.sin(yaw_t), 0.0])

        # Build desired rotation. Guard near-singular case (z_b parallel to x_c).
        cross = np.cross(z_b_des, x_c)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-3:
            # Pick any horizontal axis perpendicular to z_b_des as fallback.
            x_c = np.array([0.0, 1.0, 0.0])
            cross = np.cross(z_b_des, x_c)
            cross_norm = float(np.linalg.norm(cross))
        y_b_des = cross / cross_norm
        x_b_des = np.cross(y_b_des, z_b_des)

        R_des = np.column_stack([x_b_des, y_b_des, z_b_des])
        q_des = _rot_to_quat_wxyz(R_des)
        # Force positive w (canonical hemisphere).
        if q_des[0] < 0.0:
            q_des = -q_des
        return q_des, thrust_norm
