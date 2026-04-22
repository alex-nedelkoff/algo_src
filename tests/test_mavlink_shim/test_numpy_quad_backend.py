"""Tests for NumpyQuadBackend — single-vehicle wrapper of numpy_quad dynamics."""
import numpy as np
import pytest

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _initial_state(timestamp_us: int = 0) -> DroneState:
    return DroneState(
        pos_enu=np.zeros(3),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.zeros(4),
        timestamp_us=timestamp_us,
    )


def test_reset_then_step_zero_motor_drops_drone_due_to_gravity():
    backend = NumpyQuadBackend(params=VehicleParams())
    backend.reset(_initial_state())

    # Zero motor speeds, dt = 0.1s. Drone should fall ~ 0.5 * g * dt² ≈ 0.049 m.
    state = backend.step(motor_commands=np.zeros(4), dt=0.1)
    assert state.pos_enu[2] < -0.04, f"drone did not fall: z={state.pos_enu[2]}"
    assert state.vel_enu[2] < 0.0, f"drone z-velocity should be negative: {state.vel_enu}"


def test_reset_returns_state_to_initial():
    backend = NumpyQuadBackend(params=VehicleParams())
    initial = _initial_state()
    backend.reset(initial)

    # Step a few times to disturb state.
    for _ in range(10):
        backend.step(motor_commands=np.array([100.0, 100.0, 100.0, 100.0]), dt=0.01)

    backend.reset(initial)
    state = backend.step(motor_commands=np.zeros(4), dt=0.0)  # zero-time step to read state
    np.testing.assert_array_almost_equal(state.pos_enu, initial.pos_enu)
    np.testing.assert_array_almost_equal(state.vel_enu, initial.vel_enu)
    np.testing.assert_array_almost_equal(state.quat_wxyz, initial.quat_wxyz)


def test_get_imu_returns_gravity_when_at_rest():
    """Stationary drone should sense ~9.81 m/s² downward in body frame.

    Body frame matches world ENU at identity attitude, so accel_body[2] ≈ +9.81
    (the proper acceleration sensed by an IMU at rest = -gravity vector = up).
    """
    backend = NumpyQuadBackend(params=VehicleParams())
    backend.reset(_initial_state())

    imu = backend.get_imu()
    assert imu.accel_body.shape == (3,)
    assert imu.gyro_body.shape == (3,)
    # IMU at rest measures acceleration opposing gravity → ~+9.81 in body Z.
    assert abs(imu.accel_body[2] - 9.81) < 0.5
    np.testing.assert_array_almost_equal(imu.gyro_body, np.zeros(3))
