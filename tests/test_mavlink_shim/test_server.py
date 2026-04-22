"""Tests for MavlinkServer — UDP loopback I/O via pymavlink."""
import socket
import time

import numpy as np
import pytest
from pymavlink import mavutil

from sim.pybullet.mavlink_shim.server import MavlinkServer


def _free_port() -> int:
    """Find an unused UDP port for test isolation."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_sends_heartbeat_received_by_pymavlink_client():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port, sysid=1, compid=1)
    client = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}",
                                         source_system=255, source_component=0)
    # Client must send something first so server learns its address.
    client.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
    )
    # Give server a chance to receive client heartbeat.
    time.sleep(0.05)
    server.recv_pending()
    server.send_heartbeat()

    # Client should receive the heartbeat back.
    msg = client.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
    assert msg is not None, "client did not receive heartbeat from server"
    assert msg.get_srcSystem() == 1


def test_server_recv_pending_returns_set_attitude_target_payload():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port, sysid=1, compid=1)
    client = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}",
                                         source_system=255, source_component=0)
    # Send a SET_ATTITUDE_TARGET from client.
    client.mav.set_attitude_target_send(
        time_boot_ms=0, target_system=1, target_component=1,
        type_mask=0,
        q=[1.0, 0.0, 0.0, 0.0],
        body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
        thrust=0.5,
    )
    time.sleep(0.05)
    msgs = server.recv_pending()
    assert any(m.get_type() == "SET_ATTITUDE_TARGET" for m in msgs), \
        f"got {[m.get_type() for m in msgs]}"
