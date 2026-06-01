"""Mass-agnostic body-rate cascade controller for the AI-GP sim."""
from __future__ import annotations

import numpy as np

from .control_math import (
    G, accel_to_thrust_norm, attitude_error_quat, collective_accel,
    desired_accel, desired_attitude, mat_to_quat,
)


class BodyRateController:
    def __init__(self, hover_thrust, k_a, kp_pos=(2, 2, 2), kd_pos=(3, 3, 3),
                 kp_att=4.0, max_rate=3.0, rate_gain=1.0, g=G):
        self.hover_thrust = hover_thrust
        self.k_a = k_a
        self.kp_pos = np.asarray(kp_pos, float)
        self.kd_pos = np.asarray(kd_pos, float)
        self.kp_att = kp_att
        self.max_rate = max_rate
        # The sim's measured (omega_actual / omega_cmd). For the AI-GP sim this
        # is ~ -1.93 (inverted + amplified), so we divide our desired rate by it
        # to get the setpoint that yields the intended actual rate.
        self.rate_gain = rate_gain
        self.g = g

    def update(self, pos, vel, quat, omega, pos_sp, vel_sp, yaw_sp):
        """Returns (body_rates (3,), thrust_norm) for send_attitude_target."""
        a_des = desired_accel(pos, vel, pos_sp, vel_sp, self.kp_pos, self.kd_pos)
        c = collective_accel(a_des, quat, self.g)
        thrust = accel_to_thrust_norm(c, self.hover_thrust, self.k_a, self.g)
        R_des = desired_attitude(a_des, yaw_sp, self.g)
        q_des = mat_to_quat(R_des)
        e = attitude_error_quat(quat, q_des)        # body-frame rotation cur->des (robust thru 180)
        w = (self.kp_att * e) / self.rate_gain
        w = np.clip(w, -self.max_rate, self.max_rate)
        return w, thrust
