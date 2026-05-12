"""TIMESYNC outbound — server-initiated periodic + client-initiated reply."""
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
from sim.pybullet.mavlink_shim.server import MavlinkServer


def _free_port() -> int:
    """Find an unused UDP port for test isolation."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect_client(port: int):
    """Connect a pymavlink client to a running MavlinkServer."""
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def test_send_timesync_emits_packet():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()

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

        assert msg is not None, "client never received TIMESYNC"
        assert msg.tc1 == tc1
        assert msg.ts1 == ts1
    finally:
        server.close()


def _initial_hover_state(params: VehicleParams) -> DroneState:
    hover_omega = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.array([0.0, 0.0, 1.0]),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover_omega),
        timestamp_us=0,
    )


def test_shim_emits_periodic_timesync():
    """Server-initiated TIMESYNC fires at the configured rate."""
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    port = _free_port()
    shim = MavlinkShim(
        backend=backend, params=params, host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0, "timesync": 10.0},
    )
    shim.reset(_initial_hover_state(params))
    shim.start()
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)

        count = 0
        deadline = time.time() + 1.5
        while time.time() < deadline:
            msg = client.recv_match(type="TIMESYNC", blocking=False)
            if msg is not None and msg.ts1 == 0:
                # Server-initiated has ts1=0; tc1=our time.
                assert msg.tc1 > 0
                count += 1
            time.sleep(0.005)

        # Expect ≥10 in 1.5s @ 10 Hz (allow ample slack for scheduler jitter).
        assert count >= 8, f"expected ≥8 server-initiated TIMESYNC in 1.5s, got {count}"
    finally:
        shim.stop()


def test_shim_replies_to_client_initiated_timesync():
    """Inbound TIMESYNC with tc1=0 gets replied to: ts1 echoed, tc1=our time."""
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    port = _free_port()
    shim = MavlinkShim(
        backend=backend, params=params, host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0},  # disable periodic timesync so we only see replies
    )
    shim.reset(_initial_hover_state(params))
    shim.start()
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        ts1_sent = 123456789
        client.mav.timesync_send(tc1=0, ts1=ts1_sent)

        deadline = time.time() + 1.0
        reply = None
        while time.time() < deadline:
            msg = client.recv_match(type="TIMESYNC", blocking=False)
            if msg is not None and msg.ts1 == ts1_sent and msg.tc1 != 0:
                reply = msg
                break
            time.sleep(0.005)
        assert reply is not None, "no reply with echoed ts1 + nonzero tc1"
    finally:
        shim.stop()
