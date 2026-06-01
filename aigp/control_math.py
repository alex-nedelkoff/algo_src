"""Pure body-rate controller math (NED world, FRD body). No sim deps."""
from __future__ import annotations

import numpy as np

from .geometry import quat_to_R

G = 9.81
_G_VEC = np.array([0.0, 0.0, G])  # NED gravity acceleration (down positive)


def desired_accel(pos, vel, pos_sp, vel_sp, kp_pos, kd_pos) -> np.ndarray:
    """PD on position + velocity error -> desired translational accel (NED)."""
    pos = np.asarray(pos, float); vel = np.asarray(vel, float)
    return (np.asarray(kp_pos, float) * (np.asarray(pos_sp, float) - pos)
            + np.asarray(kd_pos, float) * (np.asarray(vel_sp, float) - vel))


def collective_accel(a_des, quat, g: float = G) -> float:
    """Required thrust acceleration projected onto current body-up. ~g at hover."""
    t_vec = np.asarray(a_des, float) - np.array([0.0, 0.0, g])  # hover -> [0,0,-g]
    body_up = -quat_to_R(quat)[:, 2]                            # body up axis in world
    return float(t_vec @ body_up)


def accel_to_thrust_norm(c: float, hover_thrust: float, k_a: float, g: float = G) -> float:
    """Map collective accel magnitude -> normalized thrust [0,1] (mass-agnostic)."""
    return float(np.clip(hover_thrust + (c - g) / k_a, 0.0, 1.0))


def desired_attitude(a_des, yaw_sp: float, g: float = G) -> np.ndarray:
    """Desired body->world rotation: thrust along desired up, heading from yaw_sp."""
    t_vec = np.asarray(a_des, float) - np.array([0.0, 0.0, g])
    zb = -t_vec / np.linalg.norm(t_vec)              # body-down axis in world (hover ->[0,0,1])
    x_c = np.array([np.cos(yaw_sp), np.sin(yaw_sp), 0.0])  # desired heading (N,E)
    yb = np.cross(zb, x_c); yb = yb / np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])


def attitude_error(R_cur, R_des) -> np.ndarray:
    """Geometric attitude error (body frame), 0.5*vee(R_des^T R_cur - R_cur^T R_des)."""
    M = R_des.T @ R_cur - R_cur.T @ R_des
    return 0.5 * np.array([M[2, 1], M[0, 2], M[1, 0]])


def body_rate_cmd(R_cur, R_des, kp_att: float, max_rate: float | None = None) -> np.ndarray:
    """Body-rate setpoint that drives R_cur -> R_des."""
    w = -kp_att * attitude_error(R_cur, R_des)
    if max_rate is not None:
        w = np.clip(w, -max_rate, max_rate)
    return w
