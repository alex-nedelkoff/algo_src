"""Tests for the MavlinkShim ↔ DroneBackend interface contract using a MockBackend."""
import socket
from dataclasses import dataclass, field

import numpy as np
import pytest
from pymavlink import mavutil

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import (
    DroneBackend,
    DroneState,
    ImuSample,
    MavlinkShim,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class MockBackend:
    """Records every call and returns canned state."""
    reset_calls: list = field(default_factory=list)
    step_calls: list = field(default_factory=list)
    imu_calls: int = 0
    _state: DroneState = field(default_factory=lambda: DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    ))

    def reset(self, initial_state: DroneState) -> None:
        self.reset_calls.append(initial_state)
        self._state = initial_state

    def step(self, motor_commands, dt: float) -> DroneState:
        self.step_calls.append((motor_commands.copy(), dt))
        return self._state

    def get_imu(self) -> ImuSample:
        self.imu_calls += 1
        return ImuSample(
            accel_body=np.array([0.0, 0.0, 9.81]),
            gyro_body=np.zeros(3),
            timestamp_us=0,
        )

    def close(self) -> None:
        pass


def test_shim_step_calls_backend_step_with_motor_commands_from_controller():
    port = _free_port()
    backend = MockBackend()
    shim = MavlinkShim(
        backend=backend,
        params=VehicleParams(),
        host="127.0.0.1", port=port,
        lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )
    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )

    try:
        shim.start()
        shim.reset(initial)
        # No client command yet — step with timeout returns False.
        stepped = shim.step(timeout_s=0.05)
        assert stepped is False
        assert len(backend.step_calls) == 0

        # Send a SET_ATTITUDE_TARGET → step should advance and call backend.step.
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{port}", source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )
        client.mav.set_attitude_target_send(
            time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
            q=[1.0, 0.0, 0.0, 0.0],
            body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
            thrust=0.5,
        )
        stepped = shim.step(timeout_s=0.5)
        assert stepped is True
        assert len(backend.step_calls) == 1
        cmd, dt = backend.step_calls[0]
        assert cmd.shape == (4,)
        assert dt > 0.0
    finally:
        shim.stop()


def test_shim_reset_propagates_to_backend():
    port = _free_port()
    backend = MockBackend()
    shim = MavlinkShim(
        backend=backend, params=VehicleParams(),
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )
    initial = DroneState(
        pos_enu=np.array([1.0, 2.0, 3.0]),
        vel_enu=np.zeros(3), quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3), motor_speed=np.zeros(4),
        timestamp_us=0,
    )
    try:
        shim.start()
        shim.reset(initial)
        assert len(backend.reset_calls) == 1
        np.testing.assert_array_equal(backend.reset_calls[0].pos_enu, [1.0, 2.0, 3.0])
    finally:
        shim.stop()


def test_shim_context_manager_starts_and_stops():
    port = _free_port()
    backend = MockBackend()
    shim = MavlinkShim(
        backend=backend, params=VehicleParams(),
        host="127.0.0.1", port=port, lockstep=True,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    )
    with shim:
        assert shim.is_running()
    assert not shim.is_running()
