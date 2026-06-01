"""Send arm / reset / velocity-setpoint commands to the sim."""
from __future__ import annotations

import time

import numpy as np
from pymavlink import mavutil

from .protocol import SIM_RESET_CMD

# SET_POSITION_TARGET_LOCAL_NED type_mask: ignore position, accel, yaw_rate;
# use velocity (bits 3,4,5 = 0) + yaw (bit 10 = 0).
_VEL_YAW_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


class Commander:
    def __init__(self, conn, system_boot_ms: int):
        self.conn = conn
        self.system_boot_ms = system_boot_ms

    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def sim_reset(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            SIM_RESET_CMD, 0, 0, 0, 0, 0, 0, 0, 0,
        )

    def send_velocity_setpoint(self, vx, vy, vz, yaw):
        now_ms = int(time.time() * 1000) - self.system_boot_ms
        self.conn.mav.set_position_target_local_ned_send(
            now_ms,
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            _VEL_YAW_MASK,
            0.0, 0.0, 0.0,        # position (ignored)
            float(vx), float(vy), float(vz),
            0.0, 0.0, 0.0,        # accel (ignored)
            float(yaw), 0.0,      # yaw, yaw_rate (ignored)
        )

    def send_attitude_setpoint(self, q_wxyz, thrust_norm):
        """Desired attitude (quaternion, sim send-frame) + normalized thrust via
        SET_ATTITUDE_TARGET in ATTITUDE mode (body-rate fields ignored). This is the
        STABLE interface for the AI-GP sim (rate mode is explosive)."""
        now_ms = int(time.time() * 1000) - self.system_boot_ms
        mask = (mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
                | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
                | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE)
        self.conn.mav.set_attitude_target_send(
            now_ms, self.conn.target_system, self.conn.target_component, mask,
            [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])],
            0.0, 0.0, 0.0, float(np.clip(thrust_norm, 0.0, 1.0)))

    def send_attitude_target(self, body_rates, thrust_norm):
        """Body-rate setpoint + normalized thrust via SET_ATTITUDE_TARGET (rate mode).
        NOTE: the AI-GP sim's rate loop is explosive/nonlinear — prefer
        send_attitude_setpoint (attitude mode) instead."""
        now_ms = int(time.time() * 1000) - self.system_boot_ms
        self.conn.mav.set_attitude_target_send(
            now_ms,
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
            [1.0, 0.0, 0.0, 0.0],   # quaternion ignored in rate mode
            float(body_rates[0]), float(body_rates[1]), float(body_rates[2]),
            float(np.clip(thrust_norm, 0.0, 1.0)),
        )
