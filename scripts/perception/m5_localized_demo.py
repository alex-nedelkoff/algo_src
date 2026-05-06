"""M5+ — Closed-loop racing with realistic localization.

Same scenario as ``m5_closed_loop_demo.py`` (8-gate oval, 4 cylinders,
A*-planned detour, MoE policy in the loop) but with one critical
change: the policy's gate-relative observation is built from a
**localized pose** — the output of ICP-against-mesh — not the env's
ground-truth state. This is the realistic deployment path: in a real
race, the drone has to estimate its own pose from sensors; the obs is
inevitably noisier than the true state.

Pipeline per step:
    1. Render depth from the env's true pose (the drone's "camera").
    2. Use the previous step's localized pose as the ICP initial guess
       (with velocity-based extrapolation between ICP frames).
    3. Run point-to-plane ICP against the static map cloud.
    4. Build the gate-relative obs from the ICP-refined pose
       (motor speeds / body rates / prev action come from env, since
       VIO would also estimate those — out of scope for the MVP).
    5. Slice to 28-dim, feed policy, get action.
    6. Step env (env physics integrates the true state — only the
       *obs* is noisy, not the dynamics).

The static map is built from the warehouse mesh + the 4 cylinder
obstacles. Warehouse is shifted so its xy-center aligns with the oval
(center 0, 0) and floor is at z=0; the oval at z=2 sits cleanly inside.

Cost: ~50 ms per step (PyBullet render ~5 ms + ICP ~30 ms + book-keeping)
with ICP running every step. For a 30-s sim that's ~2.5 min wall clock.
ICP stride parameter (--icp-period) can be raised to amortize.

Usage::

    python -m scripts.perception.m5_localized_demo
    rerun outputs/perception/m5_localized_demo.rrd
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import pybullet as pb  # noqa: E402

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402

from control.algorithms.ppo import PPO  # noqa: E402
from perception.localization.icp_localizer import (  # noqa: E402
    CameraIntrinsics, IcpLocalizer, matrix_to_pose,
    sample_mesh_surface, world_pose_to_matrix,
)
from sim.envs.gate_race_env import GateRaceEnv  # noqa: E402
from sim.tracks.waypoint import waypoints_to_track  # noqa: E402
from scripts.perception.m5_closed_loop_demo import (  # noqa: E402
    _Dummy28, _training_racing_params, slice_obs_33_to_28,
    build_planner_path, load_moe_policy,
)
from scripts.perception.week2_planner_demo import OBSTACLE_RADIUS_M  # noqa: E402
from scripts.perception.week1_demo import quat_to_rotmat  # noqa: E402


APP_ID = "vision_aug_racing_m5_localized_demo_v1"
CHECKPOINT = Path("C:/Users/alexj/Documents/algo_src/outputs/expanded_5expert_v3/model.zip")
WAREHOUSE_ASSETS = Path("sim/assets/warehouse_fab_v1")
MESH_PATH = WAREHOUSE_ASSETS / "warehouse.obj"

R_BODY_TO_CAM = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0],
])


def yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)])


def quat_yaw(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def render_depth_from_state(
    cid: int, state17: np.ndarray, width: int, height: int, vfov_deg: float,
    near: float = 0.1, far: float = 100.0,
) -> np.ndarray:
    """Render PyBullet z-buffer depth from a 17-dim drone state."""
    pos = state17[0:3]
    qw, qx, qy, qz = state17[6:10]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    eye = pos
    target = pos + 5.0 * forward
    up = np.array([0.0, 0.0, 1.0])
    view = pb.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    )
    proj = pb.computeProjectionMatrixFOV(
        fov=vfov_deg, aspect=width / height, nearVal=near, farVal=far,
    )
    _, _, _, depth, _ = pb.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_TINY_RENDERER, physicsClientId=cid,
    )
    return np.asarray(depth, dtype=np.float32).reshape(height, width)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/m5_localized_demo.rrd"))
    ap.add_argument("--max-time-s", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--icp-period", type=int, default=1,
                    help="run ICP every N sim steps (1 = every step)")
    ap.add_argument("--cell-size-m", type=float, default=0.25)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Scene ---------------------------------------------------------------

    print("Building scene (clean oval @ z=2.0)...")
    n_gates = 8
    theta = np.linspace(0, 2 * np.pi, n_gates, endpoint=False)
    gates_env = np.stack([
        3.0 * np.cos(theta),
        2.5 * np.sin(theta),
        np.full(n_gates, 2.0),
    ], axis=1)
    obstacles = [
        (0.5 * (gates_env[0, 0:2] + gates_env[1, 0:2]), 0.6),
        (0.5 * (gates_env[2, 0:2] + gates_env[3, 0:2]), 0.9),
        (0.5 * (gates_env[4, 0:2] + gates_env[5, 0:2]), 0.5),
        (0.5 * (gates_env[6, 0:2] + gates_env[7, 0:2]), 0.7),
    ]
    aabb_min_env = np.array([gates_env[:, 0].min() - 1.0,
                             gates_env[:, 1].min() - 1.0, 0.0])
    aabb_max_env = np.array([gates_env[:, 0].max() + 1.0,
                             gates_env[:, 1].max() + 1.0, 5.0])
    altitude_env = 2.0

    print("Planning A* path around obstacles (cubic-spline @ 1m spacing)...")
    full_path_env = build_planner_path(
        obstacles, gates_env, aabb_min_env, aabb_max_env,
        cell_size_m=args.cell_size_m, altitude_m=altitude_env,
        smooth_globally=True, target_spacing_m=1.0,
    )
    print(f"  {full_path_env.shape[0]} waypoints over the lap")

    track = waypoints_to_track(
        full_path_env, yaw_lookahead=1, yaw_smoothing_alpha=1.0, closed_loop=True,
    )

    # ---- Localization map ---------------------------------------------------

    # PyBullet client for live depth rendering. Spawn the warehouse
    # mesh shifted so its xy-center is at world origin and its floor is
    # at z=0 — fits the oval's clean coordinate frame.
    cid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(str(WAREHOUSE_ASSETS), physicsClientId=cid)
    # Original warehouse AABB: (-7.5, -13.31, -5.7) to (9.52, 2.11, 1.49).
    # Shift to center xy on origin, floor at 0.
    wh_center = np.array([1.01, -5.6, -5.7])  # original AABB center xy + min z
    wh_shift = -wh_center                       # to put center at origin / floor at 0
    print(f"\nSpawning warehouse mesh shifted by {wh_shift.round(2).tolist()}")
    warehouse_id = pb.loadURDF(
        str(WAREHOUSE_ASSETS / "warehouse.urdf"),
        basePosition=wh_shift.tolist(),
        useFixedBase=True, physicsClientId=cid,
    )

    # Spawn the cylinder obstacles in the same client (so depth rendering
    # sees them too).
    print("Spawning cylinder obstacles...")
    obstacle_ids = []
    for xy, r in obstacles:
        col = pb.createCollisionShape(pb.GEOM_CYLINDER, radius=r, height=4.0,
                                      physicsClientId=cid)
        vis = pb.createVisualShape(pb.GEOM_CYLINDER, radius=r, length=4.0,
                                   rgbaColor=(0.85, 0.25, 0.25, 1.0),
                                   physicsClientId=cid)
        bid = pb.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[float(xy[0]), float(xy[1]), altitude_env],
            physicsClientId=cid,
        )
        obstacle_ids.append(bid)

    # Build the map point cloud: warehouse surface (shifted) + cylinder
    # surfaces, sampled with normals.
    print("Building static map point cloud...")
    t0 = time.perf_counter()
    map_pcd = sample_mesh_surface(MESH_PATH, n_points=40_000)
    map_pcd.translate(wh_shift)
    # Add cylinder surface samples (uniform around the cylinder, sampled
    # heights — open3d doesn't have GEOM_CYLINDER sampling directly, so
    # do it manually).
    rng = np.random.default_rng(0)
    cyl_pts = []
    cyl_normals = []
    for xy, r in obstacles:
        n_pts = max(200, int(2 * math.pi * r * 4.0 * 100))   # ~100 pts/m²
        thetas = rng.uniform(0, 2 * math.pi, size=n_pts)
        zs = rng.uniform(altitude_env - 2.0, altitude_env + 2.0, size=n_pts)
        for t_, z in zip(thetas, zs):
            x = float(xy[0]) + r * math.cos(t_)
            y = float(xy[1]) + r * math.sin(t_)
            cyl_pts.append([x, y, float(z)])
            cyl_normals.append([math.cos(t_), math.sin(t_), 0.0])
    if cyl_pts:
        cyl_arr = np.asarray(cyl_pts)
        cyl_norm = np.asarray(cyl_normals)
        cyl_pcd = o3d.geometry.PointCloud()
        cyl_pcd.points = o3d.utility.Vector3dVector(cyl_arr)
        cyl_pcd.normals = o3d.utility.Vector3dVector(cyl_norm)
        map_pcd += cyl_pcd
    print(f"  map cloud: {len(map_pcd.points)} points "
          f"({time.perf_counter() - t0:.1f} s)")

    # ---- Localizer ----------------------------------------------------------

    width, height, vfov = 512, 384, 70.0
    intrinsics = CameraIntrinsics.from_vfov(vfov, width, height)
    localizer = IcpLocalizer(
        map_pcd=map_pcd,
        intrinsics=intrinsics,
        R_body_to_cam=R_BODY_TO_CAM,
        max_iterations=30,
        max_correspondence_distance_m=0.5,
        min_inlier_fraction=0.30,
        body_cloud_voxel_size_m=0.05,
        depth_stride=2,
    )

    # ---- Policy + env -------------------------------------------------------

    print(f"\nLoading checkpoint: {CHECKPOINT.name}")
    policy = load_moe_policy(CHECKPOINT)

    n_steps_max = int(round(args.max_time_s / args.dt))
    print(f"Closed-loop run: {args.max_time_s}s = {n_steps_max} steps "
          f"(ICP every {args.icp_period} step)")

    env = GateRaceEnv(
        track=track, params=_training_racing_params(),
        n_envs=1, dt=args.dt, max_steps=n_steps_max,
        n_lookahead_gates=2, gate_passage_radius=1.0, gate_collision=False,
        action_mode="trpy", esc_nonlinearity=0.95, ceiling=10.0,
        random_gate_start=True, start_vel_std=1.5, start_att_std=0.1,
        arena_bounds=15.0,
    )
    env.reset(seed=0)

    # Initial localized pose = env's starting truth.
    estimated_pos = env._states[0, 0:3].copy()
    estimated_quat = env._states[0, 6:10].copy()

    # Per-step state logging.
    gt_trajectory = np.zeros((n_steps_max, 3), dtype=np.float64)
    est_trajectory = np.zeros((n_steps_max, 3), dtype=np.float64)
    pose_err_per_step = np.zeros(n_steps_max, dtype=np.float64)
    icp_fitness_per_step = np.zeros(n_steps_max, dtype=np.float64)
    actions_log = np.zeros((n_steps_max, 4), dtype=np.float64)
    icp_period = max(1, args.icp_period)

    print("\nStepping closed-loop with ICP-localized obs...")
    t0 = time.perf_counter()
    max_gates = 0
    max_laps = 0
    last_icp_ms = 0.0

    for t in range(n_steps_max):
        true_state = env._states[0].copy()
        gt_trajectory[t] = true_state[0:3]

        # Use the env's true orientation as the IMU-attitude proxy
        # (sub-degree accurate in real flight); only POSITION is refined
        # by ICP. The ICP initial guess uses the IMU attitude + the
        # previous step's estimated position for a clean "rotation
        # known, translation noisy" setup.
        imu_quat = true_state[6:10].copy()
        estimated_quat = imu_quat   # for downstream uses

        if t % icp_period == 0:
            depth = render_depth_from_state(cid, true_state, width, height, vfov)
            t_icp = time.perf_counter()
            T_init = world_pose_to_matrix(estimated_pos, imu_quat)
            result = localizer.localize(depth, T_init,
                                        is_zbuffer=True, near=0.1, far=100.0)
            last_icp_ms = (time.perf_counter() - t_icp) * 1000.0
            if result.success:
                # Take the refined position; keep IMU attitude.
                refined_pos, _ = matrix_to_pose(result.refined_T_world_body)
                estimated_pos = refined_pos
                icp_fitness_per_step[t] = result.fitness
            else:
                icp_fitness_per_step[t] = 0.0
        else:
            # Between ICP frames: integrate true velocity (VIO proxy) onto
            # estimated position.
            estimated_pos = estimated_pos + true_state[3:6] * args.dt
            icp_fitness_per_step[t] = icp_fitness_per_step[t - 1] if t > 0 else 0.0

        est_trajectory[t] = estimated_pos
        pose_err_per_step[t] = float(np.linalg.norm(estimated_pos - true_state[0:3]))

        # Build obs from estimated POSITION (from ICP) but env-truth
        # ORIENTATION. In real deployment the orientation comes from
        # IMU attitude estimation (accelerometer gravity vector + gyro
        # integration), which is sub-degree accurate at our timescales.
        # ICP-derived attitude has 1-2 deg error that destabilises the
        # gate-yaw frame transformations in the obs builder — verified
        # empirically: pos_only variant runs without crashing, pos+quat
        # variant crashes in <3s. So we mirror the realistic IMU+ICP
        # split here (env truth standing in for IMU until VIO is wired).
        saved_pos = env._states[0, 0:3].copy()
        env._states[0, 0:3] = estimated_pos
        obs_33 = env._compute_obs_batched()[0]
        env._states[0, 0:3] = saved_pos

        obs_28 = slice_obs_33_to_28(obs_33)
        action, _ = policy._model.predict(obs_28, deterministic=True)
        actions_log[t] = action

        _, _, term, trunc, info = env.step(action[None, :])
        max_gates = max(max_gates, int(env._gates_passed[0]))
        max_laps = max(max_laps, int(env._laps_completed[0]))

        if t % 100 == 0:
            elapsed = time.perf_counter() - t0
            term_name = (info.get("episode", {}).get("termination", ["?"])[0]
                         if "episode" in info else "?")
            print(f"  t={t:>5} elapsed={elapsed:.1f}s gates={max_gates} laps={max_laps} "
                  f"pose_err={pose_err_per_step[t]*100:.1f}cm "
                  f"icp_ms={last_icp_ms:.0f} term={term_name}")

        if term[0] or trunc[0]:
            term_name = info.get("episode", {}).get("termination", ["?"])[0]
            print(f"\n  TERMINATED at t={t}: {term_name}")
            break

    actual_steps = t + 1
    elapsed = time.perf_counter() - t0
    print(f"\nDone — {actual_steps} steps in {elapsed:.1f} s "
          f"({actual_steps / elapsed:.1f} steps/s)")
    print(f"  gates passed: {max_gates}")
    print(f"  laps:         {max_laps}")
    print(f"  pose error:   mean={pose_err_per_step[:actual_steps].mean()*100:.2f}cm  "
          f"max={pose_err_per_step[:actual_steps].max()*100:.2f}cm")

    # ---- Visualisation ------------------------------------------------------

    print(f"\nWriting {args.out}...")
    rr.init(APP_ID, spawn=False)
    rr.save(str(args.out))
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # Map cloud (subsampled for visualization).
    map_arr = np.asarray(map_pcd.points)
    sub_idx = np.random.default_rng(0).choice(len(map_arr), min(8000, len(map_arr)), replace=False)
    rr.log(
        "/world/map_cloud",
        rr.Points3D(positions=map_arr[sub_idx].tolist(),
                    colors=[(80, 90, 110)] * len(sub_idx),
                    radii=0.025),
        static=True,
    )

    # Obstacles.
    n_ring_pts = 32
    for obs_idx, (obs_xy, obs_r) in enumerate(obstacles):
        rings = []
        for z in np.linspace(0, 4, 5):
            ring = []
            for k in range(n_ring_pts + 1):
                ang = 2 * math.pi * k / n_ring_pts
                ring.append([
                    float(obs_xy[0]) + obs_r * math.cos(ang),
                    float(obs_xy[1]) + obs_r * math.sin(ang),
                    float(z),
                ])
            rings.append(ring)
        rr.log(
            f"/world/obstacles/cyl_{obs_idx:02d}",
            rr.LineStrips3D(strips=rings, colors=[(255, 80, 80)] * len(rings),
                            radii=0.04),
            static=True,
        )

    # Gates (synthetic).
    GATE_HALF = 0.75
    frame_local = np.array([
        [0.0,  GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF,  GATE_HALF],
        [0.0, -GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF, -GATE_HALF],
        [0.0,  GATE_HALF,  GATE_HALF],
    ])
    track_for_viz = waypoints_to_track(
        gates_env, yaw_lookahead=1, yaw_smoothing_alpha=1.0, closed_loop=True,
    )
    for i, gate in enumerate(track_for_viz.gates):
        rotmat = quat_to_rotmat(gate.orientation)
        frame_world = (rotmat @ frame_local.T).T + gate.position
        rr.log(
            f"/world/gates/gate_{i:02d}",
            rr.LineStrips3D(strips=[frame_world.tolist()],
                            colors=[(90, 160, 255)], radii=0.025),
            static=True,
        )

    # Trajectories (full polylines, static).
    rr.log(
        "/world/path/gt",
        rr.LineStrips3D(strips=[gt_trajectory[:actual_steps].tolist()],
                        colors=[(140, 220, 250)], radii=0.03,
                        labels=["GT trajectory"]),
        static=True,
    )
    rr.log(
        "/world/path/localized",
        rr.LineStrips3D(strips=[est_trajectory[:actual_steps].tolist()],
                        colors=[(140, 230, 160)], radii=0.025,
                        labels=["ICP-localized estimate"]),
        static=True,
    )

    # Time-varying logs.
    for t in range(actual_steps):
        rr.set_time("sim_time", duration=t * args.dt)
        rr.log("/world/drone_gt",
               rr.Points3D(positions=[gt_trajectory[t].tolist()],
                           colors=[(140, 220, 250)], radii=0.16))
        rr.log("/world/drone_est",
               rr.Points3D(positions=[est_trajectory[t].tolist()],
                           colors=[(140, 230, 160)], radii=0.12))
        rr.log("/scalars/pose_error_cm", rr.Scalars(pose_err_per_step[t] * 100))
        rr.log("/scalars/icp_fitness", rr.Scalars(icp_fitness_per_step[t]))
        a = actions_log[t]
        rr.log("/scalars/action/thrust", rr.Scalars(float(a[0])))
        rr.log("/scalars/action/yaw_rate", rr.Scalars(float(a[3])))

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(
                name="World — map cloud + gates + obstacles + GT(blue) vs localized(green)",
                origin="/world",
                contents=["/world/**", "/scalars/**:exclude"],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="Pose error (cm)",
                                   origin="/scalars/pose_error_cm"),
                rrb.TimeSeriesView(name="ICP fitness",
                                   origin="/scalars/icp_fitness"),
                rrb.TimeSeriesView(name="Action — thrust + yaw_rate",
                                   origin="/scalars/action"),
            ),
            row_shares=[3, 1],
        ),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)

    pb.disconnect(cid)
    print(f"\n  Demo .rrd → {args.out.resolve()}")
    print(f"  Open with: rerun {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
