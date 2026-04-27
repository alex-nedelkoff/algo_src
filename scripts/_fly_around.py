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
from scipy.spatial.transform import Rotation

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
    behind_dist = 4.0
    p.resetDebugVisualizerCamera(
        cameraDistance=behind_dist,
        cameraYaw=math.degrees(drone_yaw) - 90.0,
        cameraPitch=-30.0,
        cameraTargetPosition=drone_pos.tolist(),
        physicsClientId=cid,
    )


def _set_orbit_camera(cid: int, target: np.ndarray, distance: float, yaw_deg: float) -> None:
    """God's-eye orbit camera fixed at target. Use mouse drag to look around."""
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw_deg,
        cameraPitch=-35.0,
        cameraTargetPosition=target.tolist(),
        physicsClientId=cid,
    )


def main() -> int:
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=cid)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=cid)
    # Explicitly turn OFF wireframe-only rendering — PyBullet sometimes
    # remembers this from a previous session and you end up seeing only
    # edges of a 3M-triangle mesh, which looks like noise.
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=cid)

    print("Loading warehouse scene...")
    handles = WarehouseScene(asset_dir=ASSETS).load_into(cid)
    wh_min, wh_max = p.getAABB(handles.warehouse_body_id, physicsClientId=cid)
    floor_z = wh_min[2]
    print(f"  warehouse + {len(handles.gate_body_ids)} gates loaded")
    print(f"  warehouse AABB ENU min={tuple(round(x,2) for x in wh_min)}  "
          f"max={tuple(round(x,2) for x in wh_max)}")
    print(f"  floor at z={floor_z:.2f} m  ceiling at z={wh_max[2]:.2f} m")

    # Recolor the warehouse a high-contrast orange so it pops against the
    # dark background — easier to see roof shape, walls, etc.
    p.changeVisualShape(
        handles.warehouse_body_id, -1,
        rgbaColor=[0.95, 0.55, 0.20, 1.0],
        physicsClientId=cid,
    )

    _draw_orientation_anchors(cid, floor_z)
    _label_gates(cid, handles)
    # Skip cyan overlays for now — they were only useful for the position-check session.
    # n_overlays = _overlay_fbx_gates(cid, handles)

    # Capture each gate's ORIGINAL ENU position so we can re-rotate the whole
    # scene around origin together with the warehouse, preserving relative geometry.
    original_gate_positions = []
    for gid in handles.gate_body_ids:
        pos, orn = p.getBasePositionAndOrientation(gid, physicsClientId=cid)
        original_gate_positions.append((np.array(pos), np.array(orn)))

    # After the ue_to_ned handedness-flip fix (2026-04-27), no URDF rpy or
    # runtime rotation should be needed. If alignment looks right at 0/0/0,
    # the pipeline fix worked. The buttons are still here in case it didn't.
    LOCKED_WAREHOUSE_RPY_DEG = (0, 0, 0)
    wh_quat = p.getQuaternionFromEuler([
        math.radians(LOCKED_WAREHOUSE_RPY_DEG[0]),
        math.radians(LOCKED_WAREHOUSE_RPY_DEG[1]),
        math.radians(LOCKED_WAREHOUSE_RPY_DEG[2]),
    ])
    wh_rot_mat = np.array(p.getMatrixFromQuaternion(wh_quat)).reshape(3, 3)
    p.resetBasePositionAndOrientation(
        handles.warehouse_body_id, [0, 0, 0], wh_quat, physicsClientId=cid,
    )
    print(f"  warehouse LOCKED at rpy={LOCKED_WAREHOUSE_RPY_DEG}°")

    # Gate POSITIONS automatically track the warehouse rotation so they stay
    # in sync (no translation drift). The 1/2/3 buttons control only the
    # torus AXIS rotation (gate orientation), not positions.
    for gid, (orig_pos, orig_orn) in zip(handles.gate_body_ids, original_gate_positions):
        new_pos = wh_rot_mat @ orig_pos
        new_orn = p.multiplyTransforms([0, 0, 0], wh_quat, [0, 0, 0], orig_orn.tolist())[1]
        p.resetBasePositionAndOrientation(gid, new_pos.tolist(), new_orn, physicsClientId=cid)
        # Stash the auto-tracked pose so the buttons can compose against it
    auto_tracked_gate_poses = []
    for gid in handles.gate_body_ids:
        pos, orn = p.getBasePositionAndOrientation(gid, physicsClientId=cid)
        auto_tracked_gate_poses.append((np.array(pos), np.array(orn)))

    # GLOBAL gate adjustments — same rpy + offset + mirror flags for every gate.
    gate_rpy_deg = [0, 0, 0]
    gate_offset = np.zeros(3)
    gate_mirror = [False, False, False]   # X/Y/Z mirror flags (5/6/7 to toggle)
    rotation_step_sign = 1
    hud_id = p.addUserDebugText("gates rpy 0/0/0  off (0,0,0)", [0, 0, 1.5],
                                [1.0, 1.0, 0.2], textSize=3.0, physicsClientId=cid)

    # Compute a spawn point outside the warehouse AABB, facing back at the
    # AABB centre. Otherwise the chase camera ends up inside the mesh and
    # back-face culling makes the warehouse invisible (only debug labels
    # remain visible since text isn't culled).
    aabb_center = np.array([(a + b) / 2 for a, b in zip(wh_min, wh_max)])
    aabb_extent = np.array([b - a for a, b in zip(wh_min, wh_max)])
    standoff = 1.5 * aabb_extent.max() / 2
    # Spawn outside AABB on +X, ELEVATED (so the down-tilted chase camera sees
    # the warehouse from above + slightly to the side — you can tell roof shape).
    drone_spawn = aabb_center + np.array([standoff, 0, aabb_extent.max() * 0.5])
    drone_yaw = math.pi

    drone_id = _make_drone(cid)
    drone_pos = drone_spawn.copy()

    print(f"  spawn: ENU{tuple(round(x,2) for x in drone_spawn)}  "
          f"yaw={math.degrees(drone_yaw):.0f}°  (outside AABB, facing centre)")
    print("Drone      : WASD move, QE up/down, LEFT/RIGHT yaw, SHIFT=fast, R=reset, ESC=quit")
    print(f"Warehouse LOCKED at rpy={LOCKED_WAREHOUSE_RPY_DEG}°.")
    print("All gates  :")
    print("  ORIENT : 1=±90 about X, 2=±90 about Y, 3=±90 about Z, 4=flip ± sign, 0=reset orient")
    print("  MIRROR : 5=toggle X mirror, 6=toggle Y mirror, 7=toggle Z mirror")
    print("  MOVE   : JL=±worldX, IK=±worldY, UO=±worldZ, N=reset offset")
    print("           (SHIFT held = 0.05m fine, default 0.25m)")
    print("Camera     : C=toggle chase/orbit. Orbit = mouse-drag, scroll to zoom.")

    # Start in orbit mode at a god's-eye view of the warehouse so you can see
    # the overall building shape. Press C to switch to chase camera.
    orbit_distance = float(aabb_extent.max() * 1.2)
    chase_mode = False
    _set_orbit_camera(cid, aabb_center, orbit_distance, yaw_deg=45.0)

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

            # Quit on ESC (ASCII 27) — K was double-bound with the I/J/K/L
            # translate cluster. Also exits if the window is closed (loop guard).
            if 27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED):
                print("quit")
                break

            if ord('c') in keys and (keys[ord('c')] & p.KEY_WAS_TRIGGERED):
                chase_mode = not chase_mode
                print(f"camera: {'CHASE' if chase_mode else 'ORBIT'}")
                if not chase_mode:
                    _set_orbit_camera(cid, aabb_center, orbit_distance, yaw_deg=45.0)

            # GLOBAL gate orient + translate. Same delta applied to every gate.
            scene_changed = False
            if ord('4') in keys and (keys[ord('4')] & p.KEY_WAS_TRIGGERED):
                rotation_step_sign = -rotation_step_sign
                print(f"rotation step: {'+90°' if rotation_step_sign > 0 else '-90°'}")
                scene_changed = True
            for ax_idx, ax_key in [(0, '1'), (1, '2'), (2, '3')]:
                if ord(ax_key) in keys and (keys[ord(ax_key)] & p.KEY_WAS_TRIGGERED):
                    gate_rpy_deg[ax_idx] += 90 * rotation_step_sign
                    while gate_rpy_deg[ax_idx] >= 180: gate_rpy_deg[ax_idx] -= 360
                    while gate_rpy_deg[ax_idx] < -180: gate_rpy_deg[ax_idx] += 360
                    scene_changed = True
            if ord('0') in keys and (keys[ord('0')] & p.KEY_WAS_TRIGGERED):
                gate_rpy_deg[:] = [0, 0, 0]
                scene_changed = True

            step = 0.05 if shift else 0.25
            if held(ord('l')): gate_offset[0] += step; scene_changed = True
            if held(ord('j')): gate_offset[0] -= step; scene_changed = True
            if held(ord('i')): gate_offset[1] += step; scene_changed = True
            if held(ord('k')): gate_offset[1] -= step; scene_changed = True
            if held(ord('u')): gate_offset[2] += step; scene_changed = True
            if held(ord('o')): gate_offset[2] -= step; scene_changed = True
            if ord('n') in keys and (keys[ord('n')] & p.KEY_WAS_TRIGGERED):
                gate_offset[:] = 0
                scene_changed = True
                print("reset gate offset")

            for ax_idx, ax_key in [(0, '5'), (1, '6'), (2, '7')]:
                if ord(ax_key) in keys and (keys[ord(ax_key)] & p.KEY_WAS_TRIGGERED):
                    gate_mirror[ax_idx] = not gate_mirror[ax_idx]
                    scene_changed = True
                    print(f"mirror {'XYZ'[ax_idx]} = {gate_mirror[ax_idx]}")

            if scene_changed:
                # Treat gates as a single rigid constellation. 1/2/3 rotate
                # the whole group around the world origin (positions AND
                # orientations); offset translates the rotated group.
                extra_quat = p.getQuaternionFromEuler([
                    math.radians(gate_rpy_deg[0]),
                    math.radians(gate_rpy_deg[1]),
                    math.radians(gate_rpy_deg[2]),
                ])
                rot_mat = np.array(p.getMatrixFromQuaternion(extra_quat)).reshape(3, 3)
                mirror_mat = np.diag([(-1.0 if m else 1.0) for m in gate_mirror])
                mirror_active = any(gate_mirror)
                # Apply rotation + mirror to position AND orientation. For
                # orientation we use the similarity transform M @ R @ M^T so
                # the result stays a valid rotation (det +1) — important for
                # off-axis-yawed gates like Gate_05 whose torus axis depends
                # on the rotation, not just position.
                for gid, (auto_pos, auto_orn) in zip(handles.gate_body_ids, auto_tracked_gate_poses):
                    rotated = rot_mat @ auto_pos
                    new_pos = (mirror_mat @ rotated) + gate_offset
                    composed_xyzw = p.multiplyTransforms(
                        [0, 0, 0], extra_quat, [0, 0, 0], auto_orn.tolist(),
                    )[1]
                    if mirror_active:
                        R_orn = Rotation.from_quat(composed_xyzw).as_matrix()
                        R_new = mirror_mat @ R_orn @ mirror_mat   # M = M^T for diag mirror
                        new_orn = Rotation.from_matrix(R_new).as_quat().tolist()
                    else:
                        new_orn = list(composed_xyzw)
                    p.resetBasePositionAndOrientation(
                        gid, new_pos.tolist(), new_orn, physicsClientId=cid,
                    )
                mirror_str = "".join("XYZ"[i] for i, m in enumerate(gate_mirror) if m) or "-"
                msg = (f"gates  rpy {gate_rpy_deg[0]:+d}/{gate_rpy_deg[1]:+d}/{gate_rpy_deg[2]:+d}  "
                       f"off ({gate_offset[0]:+.2f},{gate_offset[1]:+.2f},{gate_offset[2]:+.2f})  "
                       f"mir={mirror_str}  step{'+' if rotation_step_sign > 0 else '-'}90")
                print(msg)
                hud_id = p.addUserDebugText(
                    msg, [drone_pos[0], drone_pos[1], drone_pos[2] + 1.5],
                    [1.0, 1.0, 0.2], textSize=3.0,
                    replaceItemUniqueId=hud_id, physicsClientId=cid,
                )

            quat = p.getQuaternionFromEuler([0, 0, drone_yaw])
            p.resetBasePositionAndOrientation(
                drone_id, drone_pos.tolist(), quat, physicsClientId=cid
            )
            if chase_mode:
                _set_chase_camera(cid, drone_pos, drone_yaw)
            # else: leave camera alone — user controls via mouse drag / scroll
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
