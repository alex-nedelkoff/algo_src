"""SET_POSITION_TARGET_LOCAL_NED end-to-end: drone reaches the target."""
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


def _spec_consistent_params() -> VehicleParams:
    """5\" racing quad consistent with VADR-TS-002 §3.6 chassis."""
    return VehicleParams(
        mass=0.65, arm_length=0.115,
        inertia=np.diag([0.0028, 0.0028, 0.0046]),
        k_thrust=2.0e-6, k_torque=7.0e-8, tau_motor=0.04,
        max_rpm=31470.0,
    )


def _client(port):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def _initial_hover(params):
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def test_position_target_converges():
    params = _spec_consistent_params()
    backend = NumpyQuadBackend(params=params)
    port = _free_port()
    shim = MavlinkShim(
        backend=backend, params=params, host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0, "attitude": 100.0, "highres_imu": 200.0},
    )
    shim.reset(_initial_hover(params))
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)

        # Target: NED (5, 0, -2) → ENU (0, 5, 2). All fields enabled (mask=0).
        t_start = time.time()
        last_final_pos = None
        while time.time() - t_start < 10.0:
            client.mav.set_position_target_local_ned_send(
                time_boot_ms=int((time.time() - t_start) * 1000),
                target_system=1, target_component=1,
                coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask=0,
                x=5.0, y=0.0, z=-2.0,
                vx=0.0, vy=0.0, vz=0.0,
                afx=0.0, afy=0.0, afz=0.0,
                yaw=0.0, yaw_rate=0.0,
            )
            time.sleep(0.05)
            while client.recv_match(blocking=False) is not None:
                pass
            last_final_pos = shim._last_state.pos_enu.copy()

        err = np.linalg.norm(last_final_pos - np.array([0.0, 5.0, 2.0]))
        assert err < 0.5, f"position error {err:.2f} m exceeded 0.5 m after 10 s"
    finally:
        shim.stop()
