"""HOME_POSITION emitted once after the first GCS HEARTBEAT arrives."""
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
from sim.pybullet.mavlink_shim.geo_origin import ANDURIL_HQ_LAT_LON


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover_state():
    params = VehicleParams()
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def test_home_position_emitted_after_first_gcs_heartbeat():
    port = _free_port()
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port, rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state())
    shim.start()
    try:
        client = _client(port)
        # First HEARTBEAT from the GCS -> shim should send one HOME_POSITION.
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        deadline = time.time() + 2.0
        msg = None
        while time.time() < deadline:
            m = client.recv_match(type="HOME_POSITION", blocking=False)
            if m is not None:
                msg = m
                break
            time.sleep(0.005)
        assert msg is not None, "no HOME_POSITION received"
        assert msg.latitude == int(ANDURIL_HQ_LAT_LON[0] * 1e7)
        assert msg.longitude == int(ANDURIL_HQ_LAT_LON[1] * 1e7)
        # Local NED (x,y,z) origin.
        assert abs(msg.x) < 1e-6
        assert abs(msg.y) < 1e-6
        assert abs(msg.z) < 1e-6
    finally:
        shim.stop()


def test_home_position_not_emitted_repeatedly():
    """A second client HEARTBEAT should NOT trigger another HOME_POSITION."""
    port = _free_port()
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port, rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state())
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        # Drain the first HOME_POSITION.
        time.sleep(0.5)
        seen = 0
        while True:
            m = client.recv_match(type="HOME_POSITION", blocking=False)
            if m is None:
                break
            seen += 1
        assert seen >= 1
        # Second heartbeat after a delay.
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        deadline = time.time() + 0.5
        seen2 = 0
        while time.time() < deadline:
            m = client.recv_match(type="HOME_POSITION", blocking=False)
            if m is not None:
                seen2 += 1
            time.sleep(0.005)
        assert seen2 == 0, f"HOME_POSITION re-emitted ({seen2}) on repeat heartbeat"
    finally:
        shim.stop()
