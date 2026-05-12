"""PyBulletBackend conforms to the DroneBackend Protocol."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneBackend, DroneState, ImuSample
from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend


def test_satisfies_drone_backend_protocol():
    backend = PyBulletBackend(params=VehicleParams(), gui=False)
    try:
        # isinstance() against a Protocol checks structural conformance.
        assert isinstance(backend, DroneBackend)
    finally:
        backend.close()


def test_step_returns_drone_state_dataclass():
    backend = PyBulletBackend(params=VehicleParams(), gui=False)
    try:
        s0 = DroneState(
            pos_enu=np.zeros(3), vel_enu=np.zeros(3),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_vel_body=np.zeros(3),
            motor_speed=np.zeros(4), timestamp_us=0,
        )
        backend.reset(s0)
        s1 = backend.step(np.zeros(4), dt=0.01)
        assert isinstance(s1, DroneState)
        assert s1.pos_enu.shape == (3,)
        assert s1.quat_wxyz.shape == (4,)
        assert s1.motor_speed.shape == (4,)
    finally:
        backend.close()


def test_get_imu_returns_imu_sample():
    backend = PyBulletBackend(params=VehicleParams(), gui=False)
    try:
        s0 = DroneState(
            pos_enu=np.zeros(3), vel_enu=np.zeros(3),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_vel_body=np.zeros(3),
            motor_speed=np.zeros(4), timestamp_us=0,
        )
        backend.reset(s0)
        backend.step(np.zeros(4), dt=0.01)
        imu = backend.get_imu()
        assert isinstance(imu, ImuSample)
        assert imu.accel_body.shape == (3,)
        assert imu.gyro_body.shape == (3,)
    finally:
        backend.close()
