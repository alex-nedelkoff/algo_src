"""Interactive PyBullet GUI fly-through of the FAB warehouse.

Spawns a small kinematic drone (5 cm sphere) at the PlayerStart origin
(ENU 0,0,1) and lets you steer it with the keyboard. Third-person camera
follows the drone.

Controls
--------
W / S        translate forward / back along drone heading
A / D        strafe left / right
Q / E        descend / ascend
LEFT / RIGHT yaw left / right
SHIFT        hold to move 4x faster
R            reset to spawn pose
K            quit (or just close the window)

Run from worktree root:
    $env:PYTHONPATH = "."
    conda run -n monorace python scripts/_fly_around.py

Note: 5 cm sphere has BlockAll-style contact reporting via PyBullet but
movement is *kinematic* (we set the pose each tick) — collisions don't
push the drone back. You'll fly through walls. This script is for visual
inspection only, not for testing collision behavior.
"""
from __future__ import annotations
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from sim.pybullet.warehouse_loader import WarehouseScene

ASSETS = Path("sim/assets/warehouse_fab_v1")
SPAWN_POS = np.array([0.5, -2.0, 2.0])  # inside the walled warehouse interior, well above the floor
SPAWN_YAW = math.radians(-90)            # face -Y (south, toward the open side of the diorama)
DRONE_RADIUS = 0.10                       # bigger so it's visible against the warehouse
BASE_SPEED_MPS = 1.5
BASE_YAW_DPS = 90.0
DT = 1.0 / 60.0


def _make_drone(cid: int) -> int:
    coll = p.createCollisionShape(p.GEOM_SPHERE, radius=DRONE_RADIUS, physicsClientId=cid)
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=DRONE_RADIUS, rgbaColor=[1.0, 0.2, 0.2, 1.0],
        physicsClientId=cid,
    )
    bid = p.createMultiBody(
        baseMass=0.0,                 # mass=0 → kinematic
        baseCollisionShapeIndex=coll,
        baseVisualShapeIndex=vis,
        basePosition=SPAWN_POS.tolist(),
        physicsClientId=cid,
    )
    return bid


def _draw_orientation_anchors(cid: int, floor_z: float) -> None:
    """Big RGB axis arrows at origin + a green grid disc at floor level + gate labels."""
    # World axes at origin: X=red (east), Y=green (north), Z=blue (up). 2 m long.
    p.addUserDebugLine([0, 0, 0], [2, 0, 0], [1, 0, 0], lineWidth=3, physicsClientId=cid)
    p.addUserDebugLine([0, 0, 0], [0, 2, 0], [0, 1, 0], lineWidth=3, physicsClientId=cid)
    p.addUserDebugLine([0, 0, 0], [0, 0, 2], [0, 0, 1], lineWidth=3, physicsClientId=cid)
    p.addUserDebugText("X (E)", [2.1, 0, 0], [1, 0, 0], textSize=1.5, physicsClientId=cid)
    p.addUserDebugText("Y (N)", [0, 2.1, 0], [0, 1, 0], textSize=1.5, physicsClientId=cid)
    p.addUserDebugText("Z (UP)", [0, 0, 2.1], [0, 0, 1], textSize=1.5, physicsClientId=cid)

    # Ground plane: a faint grid at floor level so the eye has a horizontal anchor.
    grid_extent = 12
    for i in range(-grid_extent, grid_extent + 1, 2):
        p.addUserDebugLine([i, -grid_extent, floor_z], [i, grid_extent, floor_z],
                           [0.3, 0.5, 0.3], lineWidth=1, physicsClientId=cid)
        p.addUserDebugLine([-grid_extent, i, floor_z], [grid_extent, i, floor_z],
                           [0.3, 0.5, 0.3], lineWidth=1, physicsClientId=cid)


def _label_gates(cid: int, handles) -> None:
    """Float each gate's name above it in the viewport."""
    for gid, gname in zip(handles.gate_body_ids, handles.gate_names):
        pos, _ = p.getBasePositionAndOrientation(gid, physicsClientId=cid)
        p.addUserDebugText(
            gname, [pos[0], pos[1], pos[2] + 1.0],
            [1, 0.6, 0.1], textSize=1.5, physicsClientId=cid,
        )


def _overlay_fbx_gates(cid: int, handles) -> int:
    """Load FBX-extracted gate geometry as cyan visual overlays.

    The extracts contain world-space ENU vertices that exactly overlap the
    warehouse mesh, so they'd z-fight if rendered in place. Lift each one
    +4 m on Z so it floats above the warehouse as a clearly-visible preview
    while keeping its X/Y matched to the gate below.
    """
    n_loaded = 0
    for gid_pb, gname in zip(handles.gate_body_ids, handles.gate_names):
        obj_path = (ASSETS / f"fbx_{gname}.obj").resolve()
        if not obj_path.exists() or obj_path.stat().st_size < 1024:
            print(f"  skip {gname}: no FBX extract on disk")
            continue
        try:
            vis = p.createVisualShape(
                p.GEOM_MESH, fileName=str(obj_path),
                rgbaColor=[0.1, 0.9, 0.9, 1.0],
                physicsClientId=cid,
            )
        except p.error as e:
            print(f"  failed to load {obj_path.name}: {e}")
            continue
        if vis < 0:
            print(f"  createVisualShape returned -1 for {obj_path.name}")
            continue
        bid = p.createMultiBody(
            baseMass=0.0, baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, 4.0],     # lift overlay above warehouse
            physicsClientId=cid,
        )
        # Floating label so the eye links cyan blob to gate name
        gate_pos, _ = p.getBasePositionAndOrientation(gid_pb, physicsClientId=cid)
        p.addUserDebugText(
            f"FBX {gname}", [gate_pos[0], gate_pos[1], gate_pos[2] + 4.5],
            [0.1, 0.9, 0.9], textSize=1.2, physicsClientId=cid,
        )
        print(f"  overlay {gname}: bid={bid}  size={obj_path.stat().st_size//1024} KB")
        n_loaded += 1
    return n_loaded


def _set_chase_camera(cid: int, drone_pos: np.ndarray, drone_yaw: float) -> None:
    behind_dist = 2.5
    above = 1.0
    cam_x = drone_pos[0] - behind_dist * math.cos(drone_yaw)
    cam_y = drone_pos[1] - behind_dist * math.sin(drone_yaw)
    cam_z = drone_pos[2] + above
    p.resetDebugVisualizerCamera(
        cameraDistance=behind_dist,
        cameraYaw=math.degrees(drone_yaw) - 90.0,  # PyBullet yaw convention
        cameraPitch=-15.0,
        cameraTargetPosition=drone_pos.tolist(),
        physicsClientId=cid,
    )


def main() -> int:
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    # Keep GUI panel ENABLED so the warehouse-rotation sliders are visible.
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)

    print("Loading warehouse scene...")
    handles = WarehouseScene(asset_dir=ASSETS).load_into(cid)
    wh_min, wh_max = p.getAABB(handles.warehouse_body_id, physicsClientId=cid)
    floor_z = wh_min[2]
    print(f"  warehouse + {len(handles.gate_body_ids)} gates loaded")
    print(f"  warehouse AABB ENU min={tuple(round(x,2) for x in wh_min)}  "
          f"max={tuple(round(x,2) for x in wh_max)}")
    print(f"  floor at z={floor_z:.2f} m  ceiling at z={wh_max[2]:.2f} m")

    _draw_orientation_anchors(cid, floor_z)
    _label_gates(cid, handles)
    n_overlays = _overlay_fbx_gates(cid, handles)
    print(f"  loaded {n_overlays} FBX gate overlays (cyan)")


    drone_id = _make_drone(cid)
    drone_pos = SPAWN_POS.copy()
    drone_yaw = SPAWN_YAW

    print(f"  spawn: ENU{tuple(round(x,2) for x in SPAWN_POS)}  "
          f"yaw={math.degrees(SPAWN_YAW):.0f}°")
    print("Controls: WASD move, QE up/down, LEFT/RIGHT yaw, SHIFT=fast, R=reset, K=quit")

    last_print = time.monotonic()
    try:
        while p.isConnected(physicsClientId=cid):
            keys = p.getKeyboardEvents(physicsClientId=cid)

            shift = (p.B3G_SHIFT in keys
                     and (keys[p.B3G_SHIFT] & p.KEY_IS_DOWN))
            speed = BASE_SPEED_MPS * (4.0 if shift else 1.0)
            yaw_speed = math.radians(BASE_YAW_DPS) * (2.0 if shift else 1.0)

            # Heading vector in world frame (ENU): yaw=0 → +X
            fwd = np.array([math.cos(drone_yaw), math.sin(drone_yaw), 0.0])
            right = np.array([math.sin(drone_yaw), -math.cos(drone_yaw), 0.0])
            up = np.array([0.0, 0.0, 1.0])

            def held(k):
                return k in keys and (keys[k] & p.KEY_IS_DOWN)

            move = np.zeros(3)
            if held(ord('w')): move += fwd
            if held(ord('s')): move -= fwd
            if held(ord('d')): move += right
            if held(ord('a')): move -= right
            if held(ord('e')): move += up
            if held(ord('q')): move -= up
            if np.linalg.norm(move) > 0:
                move = move / np.linalg.norm(move) * speed * DT
                drone_pos = drone_pos + move

            if held(p.B3G_LEFT_ARROW):  drone_yaw += yaw_speed * DT
            if held(p.B3G_RIGHT_ARROW): drone_yaw -= yaw_speed * DT

            if ord('r') in keys and (keys[ord('r')] & p.KEY_WAS_TRIGGERED):
                drone_pos = SPAWN_POS.copy()
                drone_yaw = SPAWN_YAW
                print("reset")

            if ord('k') in keys and (keys[ord('k')] & p.KEY_WAS_TRIGGERED):
                print("quit")
                break

            quat = p.getQuaternionFromEuler([0, 0, drone_yaw])
            p.resetBasePositionAndOrientation(
                drone_id, drone_pos.tolist(), quat, physicsClientId=cid
            )
            _set_chase_camera(cid, drone_pos, drone_yaw)
            p.stepSimulation(physicsClientId=cid)

            now = time.monotonic()
            if now - last_print > 1.0:
                print(f"pos ENU=({drone_pos[0]:+.2f}, {drone_pos[1]:+.2f}, "
                      f"{drone_pos[2]:+.2f})  yaw={math.degrees(drone_yaw):+.1f}°")
                last_print = now

            time.sleep(DT)
    finally:
        p.disconnect(cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
