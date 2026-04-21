"""Fly-through collision tests for the warehouse loader.

Each test uses a kinematic 5 cm sphere ("drone") stepped along
hand-authored waypoints and asserts expected contacts with the
warehouse body and gate bodies.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pytest
import yaml

from sim.pybullet.warehouse_loader import WarehouseScene


@pytest.fixture(scope="module")
def synthetic_assets(tmp_path_factory):
    out = tmp_path_factory.mktemp("assets")
    tsdf = out.parent / "tsdf.npz"
    gates = out.parent / "gates.json"

    n = 24
    sdf = np.ones((n, n, n), dtype=np.float32) * 0.5
    sdf[5:7, 5:20, 5:20] = -0.1   # wall at x ∈ [0.5, 0.7]
    np.savez(tsdf, sdf=sdf, voxel_size=np.float32(0.1), origin=np.zeros(3))

    gates.write_text(json.dumps({
        "Gate_01": {
            "position_ned": [1.5, 1.5, -1.5],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }))

    subprocess.run(
        [sys.executable, "-m", "scripts.warehouse_to_urdf",
         "--tsdf", str(tsdf), "--gates", str(gates), "--out", str(out),
         "--target-triangles", "200", "--gate-clip-radius", "0.3"],
        check=True,
    )
    return out


@pytest.fixture
def trajectories():
    path = Path("tests/fixtures/warehouse_trajectories.yaml")
    return yaml.safe_load(path.read_text())["synthetic"]


def _spawn_drone(client_id: int, position):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=client_id)
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=list(position),
        physicsClientId=client_id,
    )


def _interp_waypoints(waypoints, n_steps_per_segment=20):
    pts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        for t in np.linspace(0.0, 1.0, n_steps_per_segment, endpoint=False):
            pts.append((1 - t) * np.array(a) + t * np.array(b))
    pts.append(np.array(waypoints[-1]))
    return pts


def test_clean_pass_through_no_warehouse_contact(synthetic_assets, trajectories):
    cid = p.connect(p.DIRECT)
    try:
        scene = WarehouseScene(asset_dir=synthetic_assets)
        handles = scene.load_into(cid)
        traj = trajectories["clean_pass_through"]
        drone = _spawn_drone(cid, traj["waypoints"][0])

        warehouse_contact_count = 0
        for pt in _interp_waypoints(traj["waypoints"]):
            p.resetBasePositionAndOrientation(
                drone, list(pt), [0, 0, 0, 1], physicsClientId=cid,
            )
            p.performCollisionDetection(physicsClientId=cid)
            for ct in p.getContactPoints(drone, handles.warehouse_body_id, physicsClientId=cid):
                warehouse_contact_count += 1

        assert warehouse_contact_count == traj["expected_warehouse_contacts"]
    finally:
        p.disconnect(cid)
