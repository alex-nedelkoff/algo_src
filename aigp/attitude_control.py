"""Mass-agnostic body-rate cascade controller for the AI-GP sim."""
from __future__ import annotations

import numpy as np

from .control_math import (
    G, accel_to_thrust_norm, body_rate_cmd, collective_accel,
    desired_accel, desired_attitude,
)
from .geometry import quat_to_R


class BodyRateController:
    def __init__(self, hover_thrust, k_a, kp_pos=(6, 6, 6), kd_pos=(4, 4, 4),
                 kp_att=8.0, max_rate=4.0, g=G):
        self.hover_thrust = hover_thrust
        self.k_a = k_a
        self.kp_pos = np.asarray(kp_pos, float)
        self.kd_pos = np.asarray(kd_pos, float)
        self.kp_att = kp_att
        self.max_rate = max_rate
        self.g = g

    def update(self, pos, vel, quat, omega, pos_sp, vel_sp, yaw_sp):
        """Returns (body_rates (3,), thrust_norm) for send_attitude_target."""
        a_des = desired_accel(pos, vel, pos_sp, vel_sp, self.kp_pos, self.kd_pos)
        c = collective_accel(a_des, quat, self.g)
        thrust = accel_to_thrust_norm(c, self.hover_thrust, self.k_a, self.g)
        R_cur = quat_to_R(quat)
        R_des = desired_attitude(a_des, yaw_sp, self.g)
        w = body_rate_cmd(R_cur, R_des, self.kp_att, self.max_rate)
        return w, thrust
