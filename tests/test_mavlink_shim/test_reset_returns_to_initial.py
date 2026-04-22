"""Integration test: reset returns drone to specified initial state exactly."""
import socket

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hover_motor_speed(params: VehicleParams) -> float:
    return float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))


def test_reset_after_disturbance_returns_state_to_initial():
    port = _free_port()
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )

    w_hover = _hover_motor_speed(params)
    initial = DroneState(
        pos_enu=np.array([1.0, 2.0, 3.0]),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, w_hover),
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
        # Disturb: 100 commands at full thrust + 30° pitch.
        for _ in range(100):
            client.mav.set_attitude_target_send(
                time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                q=[0.966, 0.0, 0.259, 0.0],  # ~30° pitch
                body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                thrust=1.0,
            )
            assert shim.step(timeout_s=0.5)
        # State has now diverged. Reset → should be back to initial.
        shim.reset(initial)
        np.testing.assert_array_equal(shim._last_state.pos_enu, initial.pos_enu)
        np.testing.assert_array_equal(shim._last_state.vel_enu, initial.vel_enu)
        np.testing.assert_array_equal(shim._last_state.quat_wxyz, initial.quat_wxyz)
    finally:
        shim.stop()
