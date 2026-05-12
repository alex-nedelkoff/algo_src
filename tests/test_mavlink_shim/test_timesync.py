"""TIMESYNC outbound — server-initiated periodic + client-initiated reply."""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

from sim.pybullet.mavlink_shim.server import MavlinkServer


def _connect_client(port: int):
    """Connect a pymavlink client to a running MavlinkServer."""
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def test_send_timesync_emits_packet():
    server = MavlinkServer(host="127.0.0.1", port=14770)
    client = _connect_client(14770)
    # Send one packet from the client so the server captures its address.
    client.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
    )
    time.sleep(0.05)
    server.recv_pending()  # capture client address

    tc1 = 1234567890
    ts1 = 9876543210
    server.send_timesync(tc1=tc1, ts1=ts1)

    deadline = time.time() + 1.0
    msg = None
    while time.time() < deadline:
        msg = client.recv_match(type="TIMESYNC", blocking=False)
        if msg is not None:
            break
        time.sleep(0.01)
    server.close()

    assert msg is not None, "client never received TIMESYNC"
    assert msg.tc1 == tc1
    assert msg.ts1 == ts1
