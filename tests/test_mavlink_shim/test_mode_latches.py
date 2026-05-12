"""Mode latching: last inbound message wins (position vs attitude)."""
from __future__ import annotations

import os
import socket
import time

import numpy as np

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mode_flips_with_each_message_type():
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    port = _free_port()
    shim = MavlinkShim(
        backend=backend, params=params, host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0},
    )
    shim.reset(DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.zeros(4), timestamp_us=0,
    ))
    shim.start()
    try:
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}",
            source_system=2, source_component=1, dialect="ardupilotmega",
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)

        # Send POSITION target first.
        client.mav.set_position_target_local_ned_send(
            time_boot_ms=0, target_system=1, target_component=1,
            coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask=0,
            x=1.0, y=0.0, z=-1.0, vx=0.0, vy=0.0, vz=0.0,
            afx=0.0, afy=0.0, afz=0.0, yaw=0.0, yaw_rate=0.0,
        )
        time.sleep(0.1)
        assert shim._mode == "position"

        # Then ATTITUDE target.
        client.mav.set_attitude_target_send(
            time_boot_ms=0, target_system=1, target_component=1,
            type_mask=0, q=[1.0, 0.0, 0.0, 0.0],
            body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0, thrust=0.5,
        )
        time.sleep(0.1)
        assert shim._mode == "attitude"

        # Position again — flips back.
        client.mav.set_position_target_local_ned_send(
            time_boot_ms=0, target_system=1, target_component=1,
            coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED, type_mask=0,
            x=0.0, y=0.0, z=-1.0, vx=0.0, vy=0.0, vz=0.0,
            afx=0.0, afy=0.0, afz=0.0, yaw=0.0, yaw_rate=0.0,
        )
        time.sleep(0.1)
        assert shim._mode == "position"
    finally:
        shim.stop()
