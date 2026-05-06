"""Sanity-check the ICP localizer end-to-end on the warehouse mesh.

Pipeline:
    1. Sample warehouse mesh surface → map point cloud (50 k points, with
       normals).
    2. Place the drone at a known ground-truth pose and render the
       PyBullet depth image (TINY renderer).
    3. Add Gaussian noise to the pose to simulate VIO drift.
    4. Run point-to-plane ICP using the noisy pose as the initial guess.
    5. Report refined pose error vs ground truth.

The sweep covers a few drift magnitudes so we can see how big the
"initial guess" can be before ICP fails to converge to the truth.

Usage:
    python -m scripts.perception.test_icp_localizer
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pybullet as pb

from perception.localization.icp_localizer import (
    CameraIntrinsics, IcpLocalizer, sample_mesh_surface,
    world_pose_to_matrix, matrix_to_pose,
)
from sim.pybullet.warehouse_loader import WarehouseScene


WAREHOUSE_ASSETS = Path("sim/assets/warehouse_fab_v1")
MESH_PATH = WAREHOUSE_ASSETS / "warehouse.obj"

# Camera convention from reference_rerun_drone_fpv.md.
R_BODY_TO_CAM = np.array([
    [0.0,  0.0,  1.0],
    [-1.0, 0.0,  0.0],
    [0.0, -1.0,  0.0],
])


def render_depth(
    cid: int, position: np.ndarray, yaw_rad: float,
    width: int, height: int, vfov_deg: float,
    near: float = 0.1, far: float = 100.0,
) -> np.ndarray:
    """Render a Z-buffer depth image from PyBullet at the given pose.

    Returns the raw [0, 1] z-buffer; the localizer un-linearises it.
    """
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    eye = position
    target = position + 5.0 * forward
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


def yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)])


def quat_yaw(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return math.atan2(2 * (qw * qz + qx * qy),
                      1 - 2 * (qy * qy + qz * qz))


def add_pose_noise(
    pos: np.ndarray, yaw: float, *,
    pos_sigma_m: float, yaw_sigma_rad: float, rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Perturb (position, yaw) with Gaussian noise — simulates VIO drift."""
    noisy_pos = pos + rng.normal(0, pos_sigma_m, size=3)
    noisy_yaw = yaw + float(rng.normal(0, yaw_sigma_rad))
    return noisy_pos, noisy_yaw


def main() -> int:
    print("Loading warehouse mesh + sampling surface points...")
    t0 = time.perf_counter()
    map_pcd = sample_mesh_surface(MESH_PATH, n_points=50_000)
    print(f"  {len(map_pcd.points)} map points, normals: {map_pcd.has_normals()} "
          f"({time.perf_counter() - t0:.1f} s)")

    cid = pb.connect(pb.DIRECT)
    handles = WarehouseScene(asset_dir=WAREHOUSE_ASSETS).load_into(cid)
    aabb_min, aabb_max = pb.getAABB(handles.warehouse_body_id, physicsClientId=cid)
    print(f"  warehouse AABB: {tuple(round(v, 2) for v in aabb_min)} → "
          f"{tuple(round(v, 2) for v in aabb_max)}")

    width, height, vfov = 512, 384, 70.0
    intrinsics = CameraIntrinsics.from_vfov(vfov, width, height)
    print(f"  camera intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
          f"({width}×{height})")

    localizer = IcpLocalizer(
        map_pcd=map_pcd,
        intrinsics=intrinsics,
        R_body_to_cam=R_BODY_TO_CAM,
        max_iterations=50,
        max_correspondence_distance_m=0.5,
        min_inlier_fraction=0.30,
        body_cloud_voxel_size_m=0.05,
        depth_stride=2,
    )

    # Ground-truth pose: middle of warehouse, ~1.5 m above floor.
    cx = 0.5 * (aabb_min[0] + aabb_max[0])
    cy = 0.5 * (aabb_min[1] + aabb_max[1])
    z_floor = aabb_min[2]
    gt_pos = np.array([cx, cy, z_floor + 1.5])
    gt_yaw = math.radians(30.0)
    gt_quat = yaw_to_quat_wxyz(gt_yaw)
    T_world_body_gt = world_pose_to_matrix(gt_pos, gt_quat)
    print(f"\nGround-truth pose: pos={gt_pos.round(3).tolist()}, yaw={math.degrees(gt_yaw):.1f}°")

    # Render GT depth.
    print("\nRendering GT depth from PyBullet...")
    depth = render_depth(cid, gt_pos, gt_yaw, width, height, vfov)
    n_finite = int(np.sum(np.isfinite(depth)))
    print(f"  depth shape {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}], "
          f"{n_finite} finite pixels")

    # Sweep noise levels.
    rng = np.random.default_rng(42)
    print(f"\n{'pos σ':>7} {'yaw σ':>7}  {'init pos err':>13} {'init yaw err':>13}  "
          f"{'final pos err':>14} {'final yaw err':>14}  {'fit':>5} {'rmse':>6}")
    for pos_sigma in (0.05, 0.10, 0.20, 0.50, 1.0, 2.0):
        for yaw_sigma_deg in (1.0, 5.0, 15.0):
            yaw_sigma = math.radians(yaw_sigma_deg)
            noisy_pos, noisy_yaw = add_pose_noise(
                gt_pos, gt_yaw, pos_sigma_m=pos_sigma,
                yaw_sigma_rad=yaw_sigma, rng=rng,
            )
            init_pos_err = float(np.linalg.norm(noisy_pos - gt_pos))
            init_yaw_err_deg = math.degrees(abs(noisy_yaw - gt_yaw))

            T_init = world_pose_to_matrix(noisy_pos, yaw_to_quat_wxyz(noisy_yaw))
            t0 = time.perf_counter()
            result = localizer.localize(
                depth, T_init,
                is_zbuffer=True, near=0.1, far=100.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            ref_pos, ref_quat = matrix_to_pose(result.refined_T_world_body)
            final_pos_err = float(np.linalg.norm(ref_pos - gt_pos))
            final_yaw_err_deg = abs(math.degrees(quat_yaw(ref_quat) - gt_yaw))

            print(
                f"{pos_sigma:>7.2f} {yaw_sigma_deg:>5.0f}°  "
                f"{init_pos_err:>11.3f} m {init_yaw_err_deg:>11.2f}°  "
                f"{final_pos_err:>11.3f} m {final_yaw_err_deg:>11.2f}°  "
                f"{result.fitness:>5.2f} {result.inlier_rmse_m:>5.3f}  "
                f"({elapsed_ms:.0f} ms, success={result.success})"
            )

    pb.disconnect(cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
