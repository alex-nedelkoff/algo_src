"""ICP-based pose recovery against a static map.

The standard recipe for live pose against a known 3-D map:

    1. Pre-extract the map's surface as a point cloud + normals (one shot).
    2. Per frame, project the drone's depth image into a body-frame
       point cloud.
    3. Use VIO / IMU integration as the *initial guess* for the drone's
       world-frame pose (close to truth but drifting).
    4. Transform the body cloud into world frame via the initial guess.
    5. Run point-to-plane ICP to refine the pose against the map cloud.
    6. Publish the refined pose.

Point-to-plane is preferred over point-to-point because we have surface
normals on the map and indoor scenes are dominated by planar walls /
ceilings; convergence is ~5-10× faster.

This module is independent of the depth source (PyBullet GT, DA V2,
VDA — any callable that returns an (H, W) depth array works) and
independent of how the map cloud was obtained (mesh sampling, TSDF
zero-crossing, real LiDAR scan).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from numpy.typing import NDArray


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_vfov(cls, vfov_deg: float, width: int, height: int) -> "CameraIntrinsics":
        """Build from a vertical FOV (PyBullet's computeProjectionMatrixFOV
        uses vertical FOV; horizontal is implied by the aspect ratio).
        """
        fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
        fx = fy   # square pixels
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height)


def depth_to_pointcloud(
    depth: NDArray[np.float32],
    intrinsics: CameraIntrinsics,
    *,
    near: float = 0.1,
    far: float = 100.0,
    is_zbuffer: bool = False,
    max_range_m: float = 30.0,
    stride: int = 2,
) -> NDArray[np.float64]:
    """Project a depth image into a 3-D point cloud in the camera frame.

    Args:
        depth: (H, W) depth array. If ``is_zbuffer=False``, the values are
            already in metres along the camera Z axis. If True, they're
            normalized GL Z-buffer values and we de-linearise them.
        intrinsics: camera intrinsics in pixels.
        near, far: clipping planes (only used when is_zbuffer=True).
        is_zbuffer: PyBullet's getCameraImage returns Z-buffer values
            in [0, 1]; this flag converts them to metric depth.
        max_range_m: discard points beyond this distance — they're
            usually background "infinity" or ill-defined.
        stride: sample every Nth pixel in row & column for speed.

    Returns:
        (N, 3) point cloud in camera frame (X right, Y down, Z forward).
    """
    h, w = depth.shape
    if (h, w) != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth shape {depth.shape} doesn't match intrinsics "
            f"({intrinsics.height}, {intrinsics.width})"
        )

    if is_zbuffer:
        # PyBullet (and GL) Z-buffer is non-linear: z_buf = (1/z_cam - 1/n) /
        # (1/f - 1/n). Solving for z_cam gives the standard formula below.
        z_buf = depth.astype(np.float64)
        z_cam = (far * near) / (far - z_buf * (far - near))
    else:
        z_cam = depth.astype(np.float64)

    # Sub-sample on a regular grid for speed.
    z_cam = z_cam[::stride, ::stride]
    h_s, w_s = z_cam.shape

    # Pixel grid in the original image coords (so cx/cy still apply).
    u = np.arange(0, w, stride, dtype=np.float64)[:w_s]
    v = np.arange(0, h, stride, dtype=np.float64)[:h_s]
    uu, vv = np.meshgrid(u, v, indexing="xy")

    # Pinhole projection: x = (u - cx) * z / fx, y = (v - cy) * z / fy.
    x = (uu - intrinsics.cx) * z_cam / intrinsics.fx
    y = (vv - intrinsics.cy) * z_cam / intrinsics.fy
    z = z_cam

    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Mask: positive depth, finite, within range.
    valid = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 2] > near + 1e-6)
        & (np.linalg.norm(pts, axis=1) < max_range_m)
    )
    return pts[valid]


def sample_mesh_surface(
    mesh_path: Path | str,
    n_points: int = 50_000,
    *,
    estimate_normals: bool = True,
) -> o3d.geometry.PointCloud:
    """Sample a uniform point cloud on a mesh's surface, with normals."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError(f"{mesh_path} loaded with 0 triangles")
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    if estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30),
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def points_to_o3d(points: NDArray[np.float64]) -> o3d.geometry.PointCloud:
    """Wrap an (N, 3) array in an open3d PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def cam_to_body_transform(R_body_to_cam: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build a 4×4 transform that maps camera-frame points → body-frame.

    The matrix ``R_body_to_cam`` from ``reference_rerun_drone_fpv.md``,
    verified empirically: ``R_body_to_cam @ [0, 0, 1]_cam = [1, 0, 0]_body``
    (camera look direction → drone forward axis). So as a standard
    rotation matrix, ``R_body_to_cam`` transforms vectors from the
    camera frame to the body frame.

    No translation: in our setup the camera origin is the drone body
    origin (FPV).
    """
    T = np.eye(4)
    T[:3, :3] = R_body_to_cam
    return T


def world_pose_to_matrix(
    position: NDArray[np.float64],
    quat_wxyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert (position, quaternion) to a 4×4 SE(3) world→body transform."""
    qw, qx, qy, qz = quat_wxyz
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def matrix_to_pose(T: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of ``world_pose_to_matrix``: (position, quat_wxyz)."""
    pos = T[:3, 3].copy()
    R = T[:3, :3]
    # R → quaternion via the standard branch-free formula.
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return pos, np.array([qw, qx, qy, qz])


@dataclass
class ICPResult:
    success: bool
    refined_T_world_body: NDArray[np.float64]   # 4×4
    fitness: float                              # fraction of source points with
                                                 # a correspondence
    inlier_rmse_m: float
    correspondence_count: int
    n_iterations: int


class IcpLocalizer:
    """Live pose estimation by ICP-aligning each depth frame against a
    pre-built map point cloud.

    Usage::

        loc = IcpLocalizer(map_pcd, intrinsics, R_body_to_cam=R)

        # per frame:
        result = loc.localize(
            depth_image=depth,                 # (H, W), metres
            initial_T_world_body=imu_pose,     # 4×4 from VIO/IMU
        )
        if result.success:
            corrected_pose = result.refined_T_world_body
    """

    def __init__(
        self,
        map_pcd: o3d.geometry.PointCloud,
        intrinsics: CameraIntrinsics,
        R_body_to_cam: NDArray[np.float64],
        *,
        max_iterations: int = 50,
        max_correspondence_distance_m: float = 0.5,
        min_inlier_fraction: float = 0.30,
        body_cloud_voxel_size_m: float = 0.05,
        depth_stride: int = 2,
    ) -> None:
        self.map_pcd = map_pcd
        if not map_pcd.has_normals():
            raise ValueError("map_pcd must have normals (point-to-plane ICP)")
        self.intrinsics = intrinsics
        self.T_body_cam = cam_to_body_transform(R_body_to_cam)
        self.max_iter = max_iterations
        self.max_corr_dist = max_correspondence_distance_m
        self.min_inlier_frac = min_inlier_fraction
        self.body_voxel = body_cloud_voxel_size_m
        self.depth_stride = depth_stride

    def localize(
        self,
        depth_image: NDArray[np.float32],
        initial_T_world_body: NDArray[np.float64],
        *,
        is_zbuffer: bool = False,
        near: float = 0.1,
        far: float = 100.0,
    ) -> ICPResult:
        """Refine the pose against the map using one depth frame."""
        cam_pts = depth_to_pointcloud(
            depth_image, self.intrinsics, near=near, far=far,
            is_zbuffer=is_zbuffer, stride=self.depth_stride,
        )
        if cam_pts.shape[0] < 100:
            return ICPResult(False, initial_T_world_body, 0.0, 0.0, 0, 0)

        # T_body_cam maps cam→body, so apply directly (no inverse).
        body_pcd = points_to_o3d(cam_pts).transform(self.T_body_cam)
        body_pcd = body_pcd.voxel_down_sample(self.body_voxel)

        reg = o3d.pipelines.registration.registration_icp(
            source=body_pcd,
            target=self.map_pcd,
            max_correspondence_distance=self.max_corr_dist,
            init=initial_T_world_body,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iter,
            ),
        )
        success = reg.fitness >= self.min_inlier_frac
        return ICPResult(
            success=success,
            refined_T_world_body=reg.transformation if success else initial_T_world_body,
            fitness=float(reg.fitness),
            inlier_rmse_m=float(reg.inlier_rmse),
            correspondence_count=len(reg.correspondence_set),
            n_iterations=self.max_iter,
        )
