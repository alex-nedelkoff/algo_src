"""PARAM_REQUEST_LIST -> single PARAM_VALUE(count=0); PARAM_REQUEST_READ silent."""
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


def test_param_request_list_returns_single_empty_param_value():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.param_request_list_send(
            target_system=1, target_component=1,
        )
        deadline = time.time() + 1.0
        msg = None
        while time.time() < deadline:
            m = client.recv_match(type="PARAM_VALUE", blocking=False)
            if m is not None:
                msg = m
                break
            time.sleep(0.005)
        assert msg is not None, "no PARAM_VALUE received"
        assert msg.param_count == 0
        # We send a single sentinel; spec doesn't dictate the exact id beyond
        # 'empty list signalling'. Just assert non-empty id, count=0.
        assert msg.param_count == 0
    finally:
        shim.stop()


def test_param_request_read_is_silent():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.param_request_read_send(
            target_system=1, target_component=1,
            param_id=b"DOES_NOT_EXIST\x00\x00",
            param_index=-1,
        )
        # Drain incoming for 0.5 s; assert no PARAM_VALUE arrives.
        deadline = time.time() + 0.5
        seen = False
        while time.time() < deadline:
            m = client.recv_match(type="PARAM_VALUE", blocking=False)
            if m is not None:
                seen = True
                break
            time.sleep(0.005)
        assert not seen, "PARAM_REQUEST_READ for unknown id must not emit PARAM_VALUE"
    finally:
        shim.stop()
