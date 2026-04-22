"""DroneBackend Protocol and shared state dataclasses.

The Protocol isolates the MAVLink shim from any specific physics engine.
Phase 1 ships NumpyQuadBackend; future backends (PyBullet warehouse
collision, real drone passthrough) plug in by implementing this Protocol.

All state is in ENU world frame; the MAVLink server handles ENU↔NED
conversion at the message boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class DroneState:
    """Full kinematic state of a single drone in ENU world frame."""
    pos_enu: NDArray[np.float64]            # (3,) world position, ENU
    vel_enu: NDArray[np.float64]            # (3,) world velocity, ENU
    quat_wxyz: NDArray[np.float64]          # (4,) attitude in ENU world frame
    angular_vel_body: NDArray[np.float64]   # (3,) body-frame rotation rates rad/s
    motor_speed: NDArray[np.float64]        # (4,) per-motor angular velocity rad/s
    timestamp_us: int                       # monotonic microseconds since shim start


@dataclass
class ImuSample:
    """Single IMU reading in body frame."""
    accel_body: NDArray[np.float64]         # (3,) body-frame accel including gravity, m/s²
    gyro_body: NDArray[np.float64]          # (3,) body-frame angular rate, rad/s
    timestamp_us: int


class DroneBackend(Protocol):
    """Interface for any physics engine driving a single quadrotor.

    motor_commands convention: per-motor angular velocity in rad/s
    (matches the output of sim/dynamics/trpy_mixer.py).
    """

    def reset(self, initial_state: DroneState) -> None:
        """Reset internal state to the given snapshot."""
        ...

    def step(self, motor_commands: NDArray[np.float64], dt: float) -> DroneState:
        """Advance physics by dt seconds and return the new state."""
        ...

    def get_imu(self) -> ImuSample:
        """Return the latest IMU sample derived from current state."""
        ...
