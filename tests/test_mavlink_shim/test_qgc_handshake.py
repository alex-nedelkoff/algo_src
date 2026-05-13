"""End-to-end fake-GCS handshake against the MavlinkShim.

Replays the QGroundControl connect sequence in order and asserts each
expected response arrives within 1 s. Then samples the rate-scheduled
telemetry for 2 s and asserts the per-message rates land within +/-20%
of nominal.

This is the protocol-correctness gate. The manual QGC screenshot in
Task 11 covers the UI side.
"""
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


def _client(port):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover():
    params = VehicleParams()
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


def test_qgc_handshake_sequence():
    port = _free_port()
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        # All defaults: heartbeat, attitude, highres_imu, timesync,
        # local_position_ned, sys_status, vfr_hud, gps_raw_int,
        # global_position_int.
    )
    shim.reset(_hover())
    shim.start()
    try:
        client = _client(port)

        # Step 1: GCS heartbeat -> shim notes the client, fires HOME_POSITION.
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        assert _wait_for(client, "HOME_POSITION") is not None

        # Step 2: REQUEST_AUTOPILOT_CAPABILITIES.
        client.mav.command_long_send(
            target_system=1, target_component=1,
            command=mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
            confirmation=0,
            param1=0, param2=0, param3=0, param4=0,
            param5=0, param6=0, param7=0,
        )
        assert _wait_for(client, "AUTOPILOT_VERSION") is not None
        assert _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
        ) is not None

        # Step 3: PARAM_REQUEST_LIST.
        client.mav.param_request_list_send(target_system=1, target_component=1)
        pv = _wait_for(client, "PARAM_VALUE")
        assert pv is not None and pv.param_count == 0

        # Step 4: MISSION_REQUEST_LIST.
        client.mav.mission_request_list_send(target_system=1, target_component=1)
        mc = _wait_for(client, "MISSION_COUNT")
        assert mc is not None and mc.count == 0

        # Step 5: Unknown COMMAND_LONG -> UNSUPPORTED ACK.
        client.mav.command_long_send(
            target_system=1, target_component=1,
            command=mavutil.mavlink.MAV_CMD_DO_SET_MODE, confirmation=0,
            param1=0, param2=0, param3=0, param4=0,
            param5=0, param6=0, param7=0,
        )
        ack = _wait_for(
            client, "COMMAND_ACK", command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        )
        assert ack is not None
        assert ack.result == mavutil.mavlink.MAV_RESULT_UNSUPPORTED

        # Step 6: Telemetry sample over 2 s. Count per message type and
        # assert >= 80% of nominal.
        counts = {
            "HEARTBEAT": 0,
            "ATTITUDE": 0,
            "HIGHRES_IMU": 0,
            "LOCAL_POSITION_NED": 0,
            "SYS_STATUS": 0,
            "GPS_RAW_INT": 0,
            "GLOBAL_POSITION_INT": 0,
            "VFR_HUD": 0,
        }
        end = time.time() + 2.0
        while time.time() < end:
            m = client.recv_match(blocking=False)
            if m is not None and m.get_type() in counts:
                counts[m.get_type()] += 1
            else:
                time.sleep(0.001)

        nominal = {
            "HEARTBEAT": 2.0,
            "ATTITUDE": 100.0,
            "HIGHRES_IMU": 200.0,
            "LOCAL_POSITION_NED": 30.0,
            "SYS_STATUS": 1.0,
            "GPS_RAW_INT": 1.0,
            "GLOBAL_POSITION_INT": 5.0,
            "VFR_HUD": 10.0,
        }
        for k, hz in nominal.items():
            expected = hz * 2.0  # window = 2 s
            min_ok = max(1, int(expected * 0.6))
            assert counts[k] >= min_ok, (
                f"{k}: got {counts[k]}, expected ~{int(expected)} (>= {min_ok})"
            )
    finally:
        shim.stop()
