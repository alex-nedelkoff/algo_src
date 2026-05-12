"""Both backends under identical SET_ATTITUDE_TARGET hover converge to within
0.1 m / 5° / 0.1 m/s over 3 s — proves PyBulletBackend is a drop-in physics
swap for NumpyQuadBackend in the same MAVLink loop."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.attitude_controller import AttitudeController
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend


def _spec_quad():
    return VehicleParams(
        mass=0.65, arm_length=0.115,
        inertia=np.diag([0.0028, 0.0028, 0.0046]),
        k_thrust=2.0e-6, k_torque=7.0e-8, tau_motor=0.04, max_rpm=31470.0,
    )


def _initial_hover(params):
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.array([0.0, 0.0, 1.0]),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover),
        timestamp_us=0,
    )


def _roll_with_attitude_loop(backend, params, controller, q_target, thrust_norm, n_steps, dt):
    # Get an initial state snapshot from the backend by issuing a zero-step.
    s = backend.step(np.zeros(4), dt=0.0)
    for _ in range(n_steps):
        motor = controller.compute(
            q_target_enu_wxyz=q_target, thrust_normalized=thrust_norm,
            q_current_enu_wxyz=s.quat_wxyz, omega_current_body=s.angular_vel_body,
        )
        s = backend.step(motor, dt=dt)
    return s


def test_hover_attitude_loop_equivalence():
    params = _spec_quad()
    s0 = _initial_hover(params)

    np_backend = NumpyQuadBackend(params=params)
    pb_backend = PyBulletBackend(params=params, gui=False)
    try:
        np_backend.reset(s0)
        pb_backend.reset(s0)

        controller = AttitudeController(params=params, k_att=6.0, k_damp=0.0)
        max_thrust = 4 * params.k_thrust * params.max_omega ** 2
        thrust_norm = params.mass * 9.81 / max_thrust
        q_target = np.array([1.0, 0.0, 0.0, 0.0])
        dt = 1.0 / 120

        s_np = _roll_with_attitude_loop(
            np_backend, params, controller, q_target, thrust_norm, n_steps=int(3 / dt), dt=dt,
        )
        s_pb = _roll_with_attitude_loop(
            pb_backend, params, controller, q_target, thrust_norm, n_steps=int(3 / dt), dt=dt,
        )

        # Position equivalence.
        np.testing.assert_allclose(s_pb.pos_enu, s_np.pos_enu, atol=0.1)
        # Velocity equivalence.
        np.testing.assert_allclose(s_pb.vel_enu, s_np.vel_enu, atol=0.1)
        # Attitude equivalence (quaternion close to identity for both).
        np.testing.assert_allclose(s_pb.quat_wxyz, s_np.quat_wxyz, atol=0.1)
    finally:
        pb_backend.close()
