"""MISSION_REQUEST_LIST -> MISSION_COUNT(0); LOG_REQUEST_LIST -> LOG_ENTRY(0)."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover_state(params: VehicleParams) -> DroneState:
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def _make_shim(port: int) -> MavlinkShim:
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state(params))
    return shim


def _wait_for(client, msg_type: str, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None:
            return m
        time.sleep(0.005)
    return None


def test_mission_request_list_returns_count_zero():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.mission_request_list_send(target_system=1, target_component=1)
        msg = _wait_for(client, "MISSION_COUNT")
        assert msg is not None
        assert msg.count == 0
    finally:
        shim.stop()


def test_log_request_list_returns_num_logs_zero():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.log_request_list_send(
            target_system=1, target_component=1, start=0, end=0xFFFF,
        )
        msg = _wait_for(client, "LOG_ENTRY")
        assert msg is not None
        assert msg.num_logs == 0
    finally:
        shim.stop()
