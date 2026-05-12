"""Step in motor command produces a first-order response with τ within 20% of params.tau_motor."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend


def test_motor_lag_step_response():
    tau = 0.04
    params = VehicleParams(
        mass=0.65, arm_length=0.115,
        inertia=np.diag([0.0028, 0.0028, 0.0046]),
        k_thrust=2.0e-6, k_torque=7.0e-8, tau_motor=tau, max_rpm=31470.0,
    )
    backend = PyBulletBackend(params=params, gui=False)
    try:
        hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
        s0 = DroneState(
            pos_enu=np.array([0.0, 0.0, 1.0]),
            vel_enu=np.zeros(3),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_vel_body=np.zeros(3),
            motor_speed=np.full(4, hover), timestamp_us=0,
        )
        backend.reset(s0)
        # Command 1.5× hover; sample omega vs time.
        cmd = 1.5 * hover
        dt = 1.0 / 120
        times = []
        omegas = []
        for k in range(int(0.5 / dt)):
            backend.step(np.full(4, cmd), dt=dt)
            times.append((k + 1) * dt)
            omegas.append(float(backend._omega_actual[0]))
        # Fit a first-order curve: ω(t) = cmd + (hover - cmd)·exp(-t/τ_fit)
        ratios = (np.array(omegas) - cmd) / (hover - cmd)
        valid = (ratios > 0.01) & (ratios < 0.99)
        slope, _ = np.polyfit(np.array(times)[valid], np.log(np.clip(ratios[valid], 1e-6, None)), 1)
        tau_fit = -1.0 / slope
        assert abs(tau_fit - tau) / tau < 0.2, f"tau_fit={tau_fit:.4f} differs from {tau} by >20%"
    finally:
        backend.close()
