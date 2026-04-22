"""Integration test: 30° pitch step → drone tracks it within 0.5s."""
import socket

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.coords_mavlink import enu_quat_to_ned_euler


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hover_thrust(params: VehicleParams) -> float:
    weight_n = params.mass * 9.81
    max_thrust_per_motor = params.k_thrust * (params.max_omega ** 2)
    return weight_n / (4.0 * max_thrust_per_motor)


def _hover_motor_speed(params: VehicleParams) -> float:
    """rad/s per motor for hover equilibrium."""
    return float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))


def test_30deg_pitch_step_reaches_target_within_0_5s():
    port = _free_port()
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
        controller_k_att=5.0,
    )

    w_hover = _hover_motor_speed(params)
    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, w_hover),  # start at hover equilibrium
        timestamp_us=0,
    )

    angle = np.deg2rad(30.0)
    # Pitch +30° in NED frame: rotation about NED +Y axis (East).
    q_target_ned = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])

    try:
        shim.start()
        shim.reset(initial)
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )

        thrust = _hover_thrust(params)
        # Run for 0.5 s = 100 steps at 200 Hz.
        for i in range(100):
            client.mav.set_attitude_target_send(
                time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                q=list(q_target_ned),
                body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                thrust=float(thrust),
            )
            assert shim.step(timeout_s=0.5)
            if i % 20 == 0:
                roll, pitch, yaw = enu_quat_to_ned_euler(shim._last_state.quat_wxyz)
                print(f"t={i*0.005:.2f}s pitch={np.degrees(pitch):.2f}°")

        final = shim._last_state
        roll, pitch, yaw = enu_quat_to_ned_euler(final.quat_wxyz)
        pitch_deg = np.degrees(pitch)
        print(f"Final pitch: {pitch_deg:.2f}°")
        assert 22.0 <= pitch_deg <= 32.0, f"pitch={pitch_deg:.2f}°, expected 22-32"
    finally:
        shim.stop()
