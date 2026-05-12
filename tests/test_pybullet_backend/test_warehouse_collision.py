"""Loading the warehouse scene into PyBulletBackend produces working collision."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pybullet")

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.backend import DroneState
from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend
from sim.pybullet.warehouse_loader import WarehouseScene


_ASSET_CANDIDATES = [
    Path("sim/assets/playroom_v1"),
    Path("sim/assets/warehouse_v1"),
]


def _find_asset_dir():
    for path in _ASSET_CANDIDATES:
        if (path / "warehouse.urdf").exists():
            return path
    return None


@pytest.mark.skipif(_find_asset_dir() is None, reason="no warehouse asset available")
def test_drone_stops_at_floor():
    """Drop the drone with no thrust; it falls until colliding with the floor (z=0)
    and stops within one chassis-height of the surface."""
    asset_dir = _find_asset_dir()
    params = VehicleParams(
        mass=0.65, arm_length=0.115,
        inertia=np.diag([0.0028, 0.0028, 0.0046]),
        k_thrust=2.0e-6, k_torque=7.0e-8, tau_motor=0.04, max_rpm=31470.0,
    )
    backend = PyBulletBackend(params=params, gui=False)
    try:
        # Load the warehouse INTO the backend's client.
        scene = WarehouseScene(asset_dir=asset_dir)
        scene.load_into(backend._client)

        s0 = DroneState(
            pos_enu=np.array([0.0, 0.0, 5.0]),
            vel_enu=np.zeros(3),
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_vel_body=np.zeros(3),
            motor_speed=np.zeros(4), timestamp_us=0,
        )
        backend.reset(s0)
        s = backend.step(np.zeros(4), dt=0.01)
        for _ in range(int(3.0 / (1.0 / 120))):
            s = backend.step(np.zeros(4), dt=1.0 / 120)
        # After 3 s of falling, drone should be on or near the floor.
        assert 0.0 <= s.pos_enu[2] < 0.5, f"drone z={s.pos_enu[2]:.3f} after free-fall onto floor"
        # And not moving (settled).
        assert np.linalg.norm(s.vel_enu) < 0.5
    finally:
        backend.close()
