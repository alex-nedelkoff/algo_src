"""COMMAND_LONG dispatch: three known commands + UNSUPPORTED fallback."""
from __future__ import annotations

import os
import socket
import time

import pytest

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


# pymavlink 2.4.49 does not bundle PROTOCOL_VERSION (msg id 300) in any
# dialect. The dispatcher still ACKs MAV_CMD_REQUEST_PROTOCOL_VERSION via
# the no-op send_protocol_version path, but no PROTOCOL_VERSION message
# can arrive at the client. Skip the version-receive assertion accordingly.
_HAS_PROTOCOL_VERSION = hasattr(mavutil.mavlink, "MAVLink_protocol_version_message")


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


def _wait_for(client, msg_type: str, timeout_s: float = 1.0, **match):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None and all(getattr(m, k, None) == v for k, v in match.items()):
            return m
        time.sleep(0.005)
    return None


def _send_cmd_long(client, cmd):
    client.mav.command_long_send(
        target_system=1, target_component=1,
        command=cmd, confirmation=0,
        param1=0, param2=0, param3=0, param4=0,
        param5=0, param6=0, param7=0,
    )


def _make_shim(port):
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state(params))
    return shim


def test_request_autopilot_capabilities_sends_version_and_ack():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        _send_cmd_long(client, mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES)
        ver = _wait_for(client, "AUTOPILOT_VERSION")
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
        )
        assert ver is not None, "no AUTOPILOT_VERSION"
        assert ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    finally:
        shim.stop()


@pytest.mark.skipif(
    not _HAS_PROTOCOL_VERSION,
    reason="pymavlink build lacks PROTOCOL_VERSION (msg id 300); send is no-op",
)
def test_request_protocol_version_sends_version_and_ack():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        _send_cmd_long(client, mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION)
        ver = _wait_for(client, "PROTOCOL_VERSION")
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION,
        )
        assert ver is not None
        assert ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    finally:
        shim.stop()


def test_request_protocol_version_acks_even_when_send_is_noop():
    """Even on pymavlink builds without PROTOCOL_VERSION, the dispatcher
    still ACKs MAV_CMD_REQUEST_PROTOCOL_VERSION as ACCEPTED. The send is a
    silent no-op via server.send_protocol_version's getattr guard.
    """
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        _send_cmd_long(client, mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION)
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION,
        )
        assert ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    finally:
        shim.stop()


def test_request_message_known_id_sends_message_and_ack():
    """REQUEST_MESSAGE(param1=msgid) -> we send that message + ACK ACCEPTED."""
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        # MAVLINK_MSG_ID_AUTOPILOT_VERSION = 148
        client.mav.command_long_send(
            target_system=1, target_component=1,
            command=mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, confirmation=0,
            param1=148, param2=0, param3=0, param4=0, param5=0, param6=0, param7=0,
        )
        ver = _wait_for(client, "AUTOPILOT_VERSION")
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        )
        assert ver is not None
        assert ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    finally:
        shim.stop()


def test_unknown_command_acks_unsupported():
    """MAV_CMD_DO_SET_MODE is not in our table -> ACK UNSUPPORTED, no crash."""
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        _send_cmd_long(client, mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        )
        assert ack is not None
        assert ack.result == mavutil.mavlink.MAV_RESULT_UNSUPPORTED
    finally:
        shim.stop()
