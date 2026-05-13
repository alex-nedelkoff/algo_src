"""AUTOPILOT_VERSION and PROTOCOL_VERSION minimal-field senders."""
from __future__ import annotations

import os
import socket
import time

import pytest

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

from sim.pybullet.mavlink_shim.server import MavlinkServer


# pymavlink 2.4.49 does not bundle PROTOCOL_VERSION (msg id 300) in any
# dialect — it has been removed from upstream common.xml. Skip the
# corresponding test on builds that lack the encoder; the server-side
# send_protocol_version method is still wired up as a no-op so Task 5's
# dispatcher can call it uniformly.
_HAS_PROTOCOL_VERSION = hasattr(mavutil.mavlink, "MAVLink_protocol_version_message")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect_client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def _wait_for(client, msg_type: str, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None:
            return m
        time.sleep(0.01)
    return None


def test_send_autopilot_version_emits_zero_capability_packet():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()

        server.send_autopilot_version()

        msg = _wait_for(client, "AUTOPILOT_VERSION")
        assert msg is not None
        # Spec: capabilities=0, all versions=0, vendor=0, product=0.
        assert msg.capabilities == 0
        assert msg.flight_sw_version == 0
        assert msg.middleware_sw_version == 0
        assert msg.vendor_id == 0
        assert msg.product_id == 0
    finally:
        server.close()


@pytest.mark.skipif(
    not _HAS_PROTOCOL_VERSION,
    reason="pymavlink build lacks PROTOCOL_VERSION (msg id 300); send is no-op",
)
def test_send_protocol_version_emits_200_100_200():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()

        server.send_protocol_version()

        msg = _wait_for(client, "PROTOCOL_VERSION")
        assert msg is not None
        assert msg.version == 200
        assert msg.min_version == 100
        assert msg.max_version == 200
    finally:
        server.close()
