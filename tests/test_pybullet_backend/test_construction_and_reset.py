"""PyBulletBackend construction, programmatic body, reset semantics."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend


def _spec_quad():
    return VehicleParams(
        mass=0.65, arm_length=0.115,
        inertia=np.diag([0.0028, 0.0028, 0.0046]),
        k_thrust=2.0e-6, k_torque=7.0e-8, tau_motor=0.04, max_rpm=31470.0,
    )


def test_constructs_in_direct_mode():
    backend = PyBulletBackend(params=_spec_quad(), gui=False)
    try:
        assert backend._body_id is not None
    finally:
        backend.close()


@pytest.mark.xfail(reason="step() implemented in Task 8")
def test_reset_places_drone_at_initial_state():
    backend = PyBulletBackend(params=_spec_quad(), gui=False)
    try:
        s0 = DroneState(
            pos_enu=np.array([1.0, 2.0, 3.0]),
            vel_enu=np.array([0.1, -0.2, 0.05]),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_vel_body=np.zeros(3),
            motor_speed=np.zeros(4), timestamp_us=0,
        )
        backend.reset(s0)
        # First step with zero motor commands → drone falls under gravity.
        s1 = backend.step(np.zeros(4), dt=0.01)
        np.testing.assert_allclose(s1.pos_enu[:2], [1.0, 2.0], atol=1e-3)
        # Should have lost some altitude (gravity pulls -Z in ENU).
        assert s1.pos_enu[2] < 3.0
        # Quaternion stays roughly identity.
        np.testing.assert_allclose(s1.quat_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-3)
    finally:
        backend.close()


def test_motor_positions_in_body_frame():
    """Motor positions derived from trpy_mixer allocation matrix.

    | Idx | Position (body) |
    |---|---|
    | 0 (FR) | (+L/√2, -L/√2, 0) |
    | 1 (FL) | (+L/√2, +L/√2, 0) |
    | 2 (RL) | (-L/√2, +L/√2, 0) |
    | 3 (RR) | (-L/√2, -L/√2, 0) |
    """
    backend = PyBulletBackend(params=_spec_quad(), gui=False)
    try:
        L = 0.115
        s = L / np.sqrt(2.0)
        expected = np.array([
            [+s, -s, 0],
            [+s, +s, 0],
            [-s, +s, 0],
            [-s, -s, 0],
        ])
        np.testing.assert_allclose(backend._motor_positions_body, expected, atol=1e-6)
    finally:
        backend.close()
