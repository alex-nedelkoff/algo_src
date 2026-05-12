"""Single-vehicle DroneBackend wrapping sim/dynamics/numpy_quad.py.

NumpyQuadDynamics is a class (not a top-level function). It must be
initialised with params, then reset(n_envs=1) called before step().
The step signature is step(states, actions, dt) — actions are per-motor
angular velocities in rad/s.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
from sim.dynamics.numpy_quad import NumpyQuadDynamics, GRAVITY

from sim.pybullet.mavlink_shim.backend import DroneState, ImuSample

# State vector indices (mirrors sim/dynamics/numpy_quad.py)
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)   # wxyz
OMEGA = slice(10, 13)
MOTOR = slice(13, 17)


@dataclass
class NumpyQuadBackend:
    """Single-vehicle wrapper around NumpyQuadDynamics.

    Internally holds a (1, 17) state array and delegates all physics to
    NumpyQuadDynamics. Converts between DroneState (ENU, wxyz) and the
    underlying batch arrays.
    """

    params: VehicleParams

    def __post_init__(self) -> None:
        self._dynamics = NumpyQuadDynamics(params=self.params)
        # Initialise with 1 env so internal param arrays are allocated.
        self._state: NDArray[np.float64] = self._dynamics.reset(n_envs=1)
        self._t0_ns = time.monotonic_ns()
        self._last_accel_body = np.array([0.0, 0.0, GRAVITY])
        self._last_omega_body = np.zeros(3)

    def _now_us(self) -> int:
        return (time.monotonic_ns() - self._t0_ns) // 1000

    def reset(self, initial_state: DroneState) -> None:
        """Reset internal state to the given DroneState snapshot."""
        # Re-allocate param arrays (in case n_envs changes in future, defensive).
        self._state = self._dynamics.reset(n_envs=1)
        # Overwrite with the caller-supplied initial state.
        self._state[0, POS] = initial_state.pos_enu
        self._state[0, VEL] = initial_state.vel_enu
        self._state[0, QUAT] = initial_state.quat_wxyz
        self._state[0, OMEGA] = initial_state.angular_vel_body
        self._state[0, MOTOR] = initial_state.motor_speed
        self._t0_ns = time.monotonic_ns()
        self._last_accel_body = np.array([0.0, 0.0, GRAVITY])
        self._last_omega_body = initial_state.angular_vel_body.copy()

    # Maximum sub-step size for integration accuracy.
    _MAX_SUB_DT: float = 0.01

    def step(self, motor_commands: NDArray[np.float64], dt: float) -> DroneState:
        """Advance physics by dt seconds, return updated DroneState.

        Large dt values are automatically sub-stepped in increments of
        _MAX_SUB_DT to preserve integration accuracy and ensure position
        reflects velocity changes within the same call.
        """
        vel_before = self._state[0, VEL].copy()

        if dt > 0.0:
            actions = motor_commands.reshape(1, 4)
            remaining = dt
            while remaining > 0.0:
                sub_dt = min(remaining, self._MAX_SUB_DT)
                self._state = self._dynamics.step(
                    states=self._state,
                    actions=actions,
                    dt=sub_dt,
                )
                remaining -= sub_dt

        # Compute body-frame acceleration for IMU emulation.
        vel_after = self._state[0, VEL]
        if dt > 0.0:
            world_accel = (vel_after - vel_before) / dt
        else:
            world_accel = np.zeros(3)

        q = self._state[0, QUAT]
        # Kinematic acceleration in body frame.
        accel_body_kinematic = _rotate_world_to_body(world_accel, q)
        # An IMU at rest reads +g upward (proper acceleration = −gravity).
        # In ENU world, gravity is [0, 0, -g]. The reaction in body frame:
        gravity_body = _rotate_world_to_body(np.array([0.0, 0.0, -GRAVITY]), q)
        # Proper acceleration = kinematic - gravity_body (subtract the free-fall component).
        self._last_accel_body = accel_body_kinematic - gravity_body
        self._last_omega_body = self._state[0, OMEGA].copy()

        return DroneState(
            pos_enu=self._state[0, POS].copy(),
            vel_enu=self._state[0, VEL].copy(),
            quat_wxyz=self._state[0, QUAT].copy(),
            angular_vel_body=self._state[0, OMEGA].copy(),
            motor_speed=self._state[0, MOTOR].copy(),
            timestamp_us=self._now_us(),
        )

    def get_imu(self) -> ImuSample:
        """Return the latest IMU sample derived from current state."""
        return ImuSample(
            accel_body=self._last_accel_body.copy(),
            gyro_body=self._last_omega_body.copy(),
            timestamp_us=self._now_us(),
        )

    def close(self) -> None:
        """No-op: NumpyQuadBackend holds no external resources."""
        pass


def _rotate_world_to_body(
    v_world: NDArray[np.float64],
    q_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate a world-frame vector into body frame using quaternion q (wxyz).

    The quaternion encodes body-to-world rotation (as used by numpy_quad).
    World-to-body is the transpose of that rotation matrix.
    """
    w, x, y, z = q_wxyz
    # Body-to-world rotation matrix R (column vectors are body axes in world).
    R_body_to_world = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])
    # World-to-body is the transpose.
    return R_body_to_world.T @ v_world
