"""GPS_RAW_INT and GLOBAL_POSITION_INT byte-correctness via the shim."""
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


def _client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _state_at(east, north, up) -> DroneState:
    params = VehicleParams()
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.array([east, north, up]),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover),
        timestamp_us=0,
    )


def _make_shim(port: int, origin=ANDURIL_HQ_LAT_LON) -> MavlinkShim:
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={
            "heartbeat": 2.0,
            "gps_raw_int": 1.0,
            "global_position_int": 5.0,
        },
        geo_origin_lat_lon=origin,
    )
    return shim


def _wait_for(client, msg_type: str, timeout_s: float = 2.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None:
            return m
        time.sleep(0.005)
    return None


def test_gps_raw_int_reports_3d_fix_at_origin_for_zero_offset():
    port = _free_port()
    shim = _make_shim(port)
    shim.reset(_state_at(0.0, 0.0, 0.0))
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        msg = _wait_for(client, "GPS_RAW_INT")
        assert msg is not None
        # fix_type=3 (3D fix), 12 satellites.
        assert msg.fix_type == 3
        assert msg.satellites_visible == 12
        # lat/lon scaled by 1e7. At origin offset (0,0,0) we expect the
        # origin itself.
        assert msg.lat == int(ANDURIL_HQ_LAT_LON[0] * 1e7)
        assert msg.lon == int(ANDURIL_HQ_LAT_LON[1] * 1e7)
    finally:
        shim.stop()


def test_global_position_int_offsets_lat_lon_by_local_position():
    port = _free_port()
    shim = _make_shim(port)
    # 100 m north, 0 east, 5 m up.
    shim.reset(_state_at(east=0.0, north=100.0, up=5.0))
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        msg = _wait_for(client, "GLOBAL_POSITION_INT")
        assert msg is not None
        # +100 m north -> lat increases by ~100/111320 deg, scaled by 1e7.
        expected_dlat_e7 = int((100.0 / 111_320.0) * 1e7)
        actual_dlat_e7 = msg.lat - int(ANDURIL_HQ_LAT_LON[0] * 1e7)
        # tolerance: 1 int7 unit = ~1.1 cm.
        assert abs(actual_dlat_e7 - expected_dlat_e7) <= 2
        # alt and relative_alt in mm. The drone has been free-falling for
        # ~one tick by the time the first GLOBAL_POSITION_INT goes out
        # (no thrust-target set in this test), so allow up to ~250 mm of drop.
        assert 4750 <= msg.relative_alt <= 5000
    finally:
        shim.stop()
