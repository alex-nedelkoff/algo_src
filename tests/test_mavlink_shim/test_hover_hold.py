"""Integration test: hover hold with identity attitude + hover thrust."""
import socket
import time

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.numpy_quad import GRAVITY
from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hover_thrust(params: VehicleParams) -> float:
    """Normalized [0,1] thrust that achieves hover for given params."""
    weight_n = params.mass * 9.81
    max_thrust_per_motor = params.k_thrust * (params.max_omega ** 2)
    return weight_n / (4.0 * max_thrust_per_motor)


def _hover_motor_speed(params: VehicleParams) -> float:
    """Motor speed (rad/s) at hover equilibrium."""
    return float(np.sqrt(params.mass * GRAVITY / (4.0 * params.k_thrust)))


def test_hover_hold_keeps_drone_within_tolerance():
    port = _free_port()
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )

    # Initialize motors at hover equilibrium speed to avoid the first-order motor
    # lag (tau_motor=0.02s) causing an initial sag from zero.
    w_hover = _hover_motor_speed(params)
    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.full(4, w_hover),
        timestamp_us=0,
    )

    try:
        shim.start()
        shim.reset(initial)
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )

        thrust = _hover_thrust(params)
        for _ in range(500):
            client.mav.set_attitude_target_send(
                time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                q=[1.0, 0.0, 0.0, 0.0],
                body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                thrust=float(thrust),
            )
            stepped = shim.step(timeout_s=0.5)
            assert stepped, "shim.step did not advance"

        final = shim._last_state
        pos_err = np.linalg.norm(final.pos_enu - initial.pos_enu)
        att_err_deg = np.degrees(2.0 * np.arccos(abs(np.clip(final.quat_wxyz[0], -1.0, 1.0))))
        assert pos_err < 0.05, f"position drifted {pos_err:.3f} m"
        assert att_err_deg < 1.0, f"attitude drifted {att_err_deg:.2f}°"
    finally:
        shim.stop()
