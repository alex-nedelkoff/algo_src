"""Vectorized pairwise covisibility computation between frames.

Algorithm:
  1. Subsample S random pixels from frame i where depth > 0
  2. Unproject to 3D:  pts_cam = depth * K_inv @ [u, v, 1]^T
  3. To world frame:   pts_world = T_w_ci @ pts_cam
  4. To frame j cam:   pts_cam_j = inv(T_w_cj) @ pts_world
  5. Project to j:     uv_j = K @ pts_cam_j / z
  6. Check valid:      in_bounds & z > 0
  7. Depth check:      |z_reprojected - depth_j[v', u']| < threshold
  8. Score:            overlap = count(valid) / S

Optimizations:
  - Pre-compute all world-frame point clouds once
  - Pre-compute all T_w_c inverses once
  - Pose-distance pruning: skip pairs too far apart
  - Row-level progress logging
"""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


def unproject_pixels(
    depth: np.ndarray,
    K: np.ndarray,
    num_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample valid pixels and unproject to 3D camera-frame points.

    Args:
        depth: (H, W) depth map.
        K: (3, 3) camera intrinsic matrix.
        num_samples: Number of pixels to subsample.
        rng: Optional random generator for reproducibility.

    Returns:
        pts_cam: (S, 3) 3D points in camera frame.
        pixel_coords: (S, 2) pixel coordinates [u, v].
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = depth.shape
    valid_mask = depth > 0
    valid_coords = np.argwhere(valid_mask)  # (N_valid, 2) as [row, col] = [v, u]

    if len(valid_coords) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    n = min(num_samples, len(valid_coords))
    indices = rng.choice(len(valid_coords), size=n, replace=False)
    sampled = valid_coords[indices]  # (S, 2) as [v, u]

    v, u = sampled[:, 0], sampled[:, 1]
    z = depth[v, u].astype(np.float64)

    # Unproject: pts = z * K_inv @ [u, v, 1]^T
    K_inv = np.linalg.inv(K.astype(np.float64))
    pixels_h = np.stack([u, v, np.ones_like(u)], axis=1).astype(np.float64)  # (S, 3)
    rays = (K_inv @ pixels_h.T).T  # (S, 3)
    pts_cam = rays * z[:, None]

    pixel_coords = np.stack([u, v], axis=1).astype(np.float64)
    return pts_cam, pixel_coords


def _overlap_directed(
    pts_world_h: np.ndarray,
    T_cj_w: np.ndarray,
    depth_j: np.ndarray,
    K: np.ndarray,
    depth_threshold: float,
) -> float:
    """Compute directed overlap: project world points into frame j.

    Args:
        pts_world_h: (4, S) homogeneous world-frame points.
        T_cj_w: (4, 4) world-to-camera-j transform.
        depth_j: (H, W) depth map.
        K: (3, 3) intrinsics.
        depth_threshold: Relative depth tolerance.
    """
    S = pts_world_h.shape[1]
    if S == 0:
        return 0.0

    H, W = depth_j.shape

    # To camera j frame
    pts_cam_j = T_cj_w @ pts_world_h  # (4, S)
    z_j = pts_cam_j[2, :]

    # Check z > 0
    valid = z_j > 0

    # Project to image j
    uv_h = K @ pts_cam_j[:3, :]  # (3, S)
    # Avoid divide-by-zero for invalid z
    z_safe = np.where(valid, z_j, 1.0)
    u_j = uv_h[0, :] / z_safe
    v_j = uv_h[1, :] / z_safe

    # In-bounds check
    u_int = np.round(u_j).astype(np.int64)
    v_int = np.round(v_j).astype(np.int64)
    valid &= (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

    # Depth consistency
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        d_actual = depth_j[v_int[valid_idx], u_int[valid_idx]].astype(np.float64)
        d_reproj = z_j[valid_idx]
        depth_ok = np.abs(d_reproj - d_actual) < depth_threshold * d_actual
        depth_ok &= d_actual > 0
        valid[valid_idx] = depth_ok

    return float(np.sum(valid)) / S


# Keep the original API for backward compat and tests
def compute_pairwise_overlap(
    pts_cam_i: np.ndarray,
    T_w_ci: np.ndarray,
    depth_j: np.ndarray,
    T_w_cj: np.ndarray,
    K: np.ndarray,
    depth_threshold: float = 0.1,
) -> float:
    """Compute directed overlap score from frame i to frame j."""
    S = len(pts_cam_i)
    if S == 0:
        return 0.0

    K64 = K.astype(np.float64)
    pts_h = np.hstack([pts_cam_i, np.ones((S, 1))]).T  # (4, S)
    pts_world = T_w_ci.astype(np.float64) @ pts_h
    T_cj_w = np.linalg.inv(T_w_cj.astype(np.float64))

    return _overlap_directed(pts_world, T_cj_w, depth_j, K64, depth_threshold)


def compute_scene_covisibility(
    depths: list[np.ndarray],
    poses: np.ndarray,
    K: np.ndarray,
    num_samples: int = 10_000,
    depth_threshold: float = 0.1,
    seed: int = 42,
    max_pose_distance: float | None = None,
) -> np.ndarray:
    """Compute full N x N symmetric covisibility matrix for a scene.

    Pre-computes world-frame points and pose inverses to avoid redundant work.

    Args:
        depths: List of N depth maps, each (H, W).
        poses: (N, 4, 4) camera-to-world transforms.
        K: (3, 3) camera intrinsic matrix.
        num_samples: Pixels to subsample per frame.
        depth_threshold: Relative depth tolerance.
        seed: Random seed for reproducibility.
        max_pose_distance: If set, skip pairs with translation distance above
                          this threshold (they'll have zero overlap anyway).

    Returns:
        overlap: (N, N) symmetric overlap matrix with 1.0 on diagonal.
    """
    N = len(depths)
    assert poses.shape[0] == N, f"Got {N} depths but {poses.shape[0]} poses"

    rng = np.random.default_rng(seed)
    K64 = K.astype(np.float64)
    overlap = np.eye(N, dtype=np.float64)

    total_pairs = N * (N - 1) // 2
    log.info("  Covisibility: %d frames, %d pairs to compute", N, total_pairs)

    # Pre-compute world-frame point clouds (avoids re-transforming per pair)
    log.info("  Pre-computing world-frame point clouds ...")
    all_pts_world_h = []  # each (4, S)
    poses64 = poses.astype(np.float64)
    for i in range(N):
        pts_cam, _ = unproject_pixels(depths[i], K, num_samples, rng)
        S = len(pts_cam)
        if S > 0:
            pts_h = np.vstack([pts_cam.T, np.ones((1, S))])  # (4, S)
            pts_world = poses64[i] @ pts_h
        else:
            pts_world = np.empty((4, 0), dtype=np.float64)
        all_pts_world_h.append(pts_world)

    # Pre-compute all pose inverses
    all_T_inv = np.linalg.inv(poses64)  # (N, 4, 4)

    # Pre-compute pose distances for pruning
    positions = poses64[:, :3, 3]  # (N, 3)
    if max_pose_distance is not None:
        # Pairwise distance matrix
        diff = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
        pose_dists = np.linalg.norm(diff, axis=2)  # (N, N)

    # Compute upper triangle with progress
    t0 = time.time()
    pairs_done = 0
    pairs_skipped = 0

    for i in range(N):
        row_start = time.time()
        for j in range(i + 1, N):
            # Pose-distance pruning
            if max_pose_distance is not None and pose_dists[i, j] > max_pose_distance:
                pairs_skipped += 1
                pairs_done += 1
                continue

            # Directed overlap i→j
            score_ij = _overlap_directed(
                all_pts_world_h[i], all_T_inv[j], depths[j], K64, depth_threshold
            )
            # Directed overlap j→i
            score_ji = _overlap_directed(
                all_pts_world_h[j], all_T_inv[i], depths[i], K64, depth_threshold
            )
            score = (score_ij + score_ji) / 2.0
            overlap[i, j] = score
            overlap[j, i] = score
            pairs_done += 1

        # Log progress every 100 rows
        if (i + 1) % 100 == 0 or i == N - 1:
            elapsed = time.time() - t0
            rate = pairs_done / elapsed if elapsed > 0 else 0
            eta = (total_pairs - pairs_done) / rate if rate > 0 else 0
            log.info(
                "  Row %d/%d | %d/%d pairs (%.0f/s) | skipped %d | ETA %.0fs",
                i + 1, N, pairs_done, total_pairs, rate, pairs_skipped, eta,
            )

    elapsed = time.time() - t0
    log.info(
        "  Covisibility done: %d pairs in %.1fs (%.0f pairs/s), %d skipped",
        pairs_done, elapsed, pairs_done / elapsed if elapsed > 0 else 0, pairs_skipped,
    )

    return overlap
