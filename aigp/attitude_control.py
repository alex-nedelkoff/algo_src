"""Mass-agnostic body-rate cascade controller for the AI-GP sim."""
from __future__ import annotations

import numpy as np

from .control_math import (
    G, accel_to_thrust_norm, attitude_error_quat, collective_accel,
    desired_accel, desired_attitude, mat_to_quat, setpoint_send_remap,
)


class AttitudeSetpointController:
    """Position-hold / guidance via the sim's STABLE attitude-setpoint interface.

    pos+vel PD -> desired accel (horizontal-tilt clamped) -> desired attitude ->
    sim send-frame remap. Returns (q_send_wxyz, thrust_norm) for
    Commander.send_attitude_setpoint. Mass-agnostic thrust via probed hover/k_a.
    """

    def __init__(self, hover_thrust, k_a, kp_pos=(1.0, 1.0, 1.8),
                 kd_pos=(3.2, 3.2, 3.0), tilt_max_deg=10.0, g=G):
        self.hover_thrust = hover_thrust
        self.k_a = k_a
        self.kp_pos = np.asarray(kp_pos, float)
        self.kd_pos = np.asarray(kd_pos, float)
        self.tilt_max_acc = float(np.tan(np.radians(tilt_max_deg)) * g)
        self.g = g
        self._q_prev = None

    def update(self, pos, vel, quat, pos_sp, vel_sp, yaw_sp):
        a_des = desired_accel(pos, vel, pos_sp, vel_sp, self.kp_pos, self.kd_pos)
        ah = a_des[:2]
        n = float(np.linalg.norm(ah))
        if n > self.tilt_max_acc:
            a_des = a_des.copy()
            a_des[:2] = ah / n * self.tilt_max_acc
        q_des = mat_to_quat(desired_attitude(a_des, yaw_sp, self.g))
        q_send = setpoint_send_remap(q_des)
        if self._q_prev is not None and np.dot(q_send, self._q_prev) < 0:
            q_send = -q_send                 # sign continuity
        self._q_prev = q_send
        thrust = accel_to_thrust_norm(collective_accel(a_des, quat, self.g),
                                      self.hover_thrust, self.k_a, self.g)
        return q_send, thrust


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
