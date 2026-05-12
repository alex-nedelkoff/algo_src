"""At hover_omega thrust, the drone holds altitude over 2 s (within 0.1 m)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend


def test_constant_hover_thrust_holds_altitude():
    params = VehicleParams(
        mass=0.65, arm_length=0.115,
        inertia=np.diag([0.0028, 0.0028, 0.0046]),
        k_thrust=2.0e-6, k_torque=7.0e-8, tau_motor=0.04, max_rpm=31470.0,
    )
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    backend = PyBulletBackend(params=params, gui=False)
    try:
        s0 = DroneState(
            pos_enu=np.array([0.0, 0.0, 1.0]),
            vel_enu=np.zeros(3),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_vel_body=np.zeros(3),
            motor_speed=np.full(4, hover),
            timestamp_us=0,
        )
        backend.reset(s0)
        for _ in range(int(2.0 / (1.0 / 120))):
            s = backend.step(np.full(4, hover), dt=1.0 / 120)
        assert abs(s.pos_enu[2] - 1.0) < 0.1, f"altitude drift {s.pos_enu[2]-1.0:+.3f} m exceeds ±0.1 m"
    finally:
        backend.close()
