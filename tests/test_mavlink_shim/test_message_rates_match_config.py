"""Integration test: message rates match configured Hz over 5s window."""
import socket
import time

import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
import numpy as np


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hover_motor_speed(params: VehicleParams) -> float:
    return float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))


def test_message_rates_within_5_percent_of_config():
    port = _free_port()
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    rates = {"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200}
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=False,
        rates_hz=rates,
    )

    w_hover = _hover_motor_speed(params)
    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, w_hover),
        timestamp_us=0,
    )

    counts = {"HEARTBEAT": 0, "ATTITUDE": 0, "ODOMETRY": 0, "HIGHRES_IMU": 0}

    try:
        with shim:
            shim.reset(initial)
            client = mavutil.mavlink_connection(
                f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
            )
            # First heartbeat from client so shim learns where to send.
            client.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
            )
            # Give shim time to receive the heartbeat and register the client address.
            time.sleep(0.1)

            t_end = time.monotonic() + 5.0
            while time.monotonic() < t_end:
                msg = client.recv_match(blocking=True, timeout=0.5)
                if msg is None:
                    continue
                t = msg.get_type()
                if t in counts:
                    counts[t] += 1

        for name_count, name_rate in [
            ("HEARTBEAT", "heartbeat"), ("ATTITUDE", "attitude"),
            ("ODOMETRY", "odometry"), ("HIGHRES_IMU", "highres_imu"),
        ]:
            expected = rates[name_rate] * 5  # over 5 seconds
            actual = counts[name_count]
            tol = max(1, expected * 0.15)
            assert expected - tol <= actual <= expected + tol, (
                f"{name_count}: got {actual}, expected {expected} ± {tol}"
            )
    finally:
        # context manager already calls stop, but be defensive.
        if shim.is_running():
            shim.stop()
