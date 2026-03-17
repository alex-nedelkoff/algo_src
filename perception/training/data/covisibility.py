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

Each pair's computation is cache-friendly (4×S fits in L2).
Parallelized across CPU cores via fork — workers inherit shared data,
no pickling overhead.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time

import numpy as np

log = logging.getLogger(__name__)

# Module-level shared state for worker processes (inherited via fork, not pickled)
_shared = {}


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

    Operates on small arrays (4×S) that fit in CPU L2 cache.
    """
    S = pts_world_h.shape[1]
    if S == 0:
        return 0.0

    H, W = depth_j.shape

    pts_cam_j = T_cj_w @ pts_world_h  # (4, S)
    z_j = pts_cam_j[2, :]

    valid = z_j > 0

    uv_h = K @ pts_cam_j[:3, :]  # (3, S)
    z_safe = np.where(valid, z_j, 1.0)
    u_j = uv_h[0, :] / z_safe
    v_j = uv_h[1, :] / z_safe

    u_int = np.round(u_j).astype(np.int64)
    v_int = np.round(v_j).astype(np.int64)
    valid &= (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        d_actual = depth_j[v_int[valid_idx], u_int[valid_idx]].astype(np.float64)
        d_reproj = z_j[valid_idx]
        depth_ok = np.abs(d_reproj - d_actual) < depth_threshold * d_actual
        depth_ok &= d_actual > 0
        valid[valid_idx] = depth_ok

    return float(np.sum(valid)) / S


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
    pts_h = np.hstack([pts_cam_i, np.ones((S, 1))]).T
    pts_world = T_w_ci.astype(np.float64) @ pts_h
    T_cj_w = np.linalg.inv(T_w_cj.astype(np.float64))
    return _overlap_directed(pts_world, T_cj_w, depth_j, K64, depth_threshold)


def _compute_row(i: int) -> tuple[int, np.ndarray]:
    """Compute overlap[i, j] for all j > i. Reads from module-level _shared."""
    pts_list = _shared["pts"]
    T_inv = _shared["T_inv"]
    depths = _shared["depths"]
    K = _shared["K"]
    dt = _shared["depth_threshold"]
    N = _shared["N"]

    row = np.zeros(N, dtype=np.float64)
    pts_i = pts_list[i]
    T_inv_i = T_inv[i]

    for j in range(i + 1, N):
        score_ij = _overlap_directed(pts_i, T_inv[j], depths[j], K, dt)
        score_ji = _overlap_directed(pts_list[j], T_inv_i, depths[i], K, dt)
        row[j] = (score_ij + score_ji) / 2.0

    return i, row


def compute_scene_covisibility(
    depths: list[np.ndarray],
    poses: np.ndarray,
    K: np.ndarray,
    num_samples: int = 10_000,
    depth_threshold: float = 0.1,
    seed: int = 42,
    max_pose_distance: float | None = None,
    n_workers: int | None = None,
) -> np.ndarray:
    """Compute full N x N symmetric covisibility matrix for a scene.

    Parallelized across CPU cores. Workers inherit shared data via fork
    (no pickling). Per-pair computation is cache-friendly (4×S in L2).

    Args:
        depths: List of N depth maps, each (H, W).
        poses: (N, 4, 4) camera-to-world transforms.
        K: (3, 3) camera intrinsic matrix.
        num_samples: Pixels to subsample per frame.
        depth_threshold: Relative depth tolerance.
        seed: Random seed for reproducibility.
        max_pose_distance: If set, skip pairs with translation distance above
                          this threshold (they'll have zero overlap anyway).
        n_workers: Number of parallel workers. Defaults to CPU count.

    Returns:
        overlap: (N, N) symmetric overlap matrix with 1.0 on diagonal.
    """
    global _shared

    N = len(depths)
    assert poses.shape[0] == N, f"Got {N} depths but {poses.shape[0]} poses"

    rng = np.random.default_rng(seed)
    K64 = K.astype(np.float64)
    overlap = np.eye(N, dtype=np.float64)

    total_pairs = N * (N - 1) // 2
    if n_workers is None:
        n_workers = min(mp.cpu_count(), N)

    log.info("  Covisibility: %d frames, %d pairs, %d workers", N, total_pairs, n_workers)

    # Pre-compute world-frame point clouds
    log.info("  Pre-computing world-frame point clouds ...")
    poses64 = poses.astype(np.float64)
    all_pts_world_h = []
    for i in range(N):
        pts_cam, _ = unproject_pixels(depths[i], K, num_samples, rng)
        S = len(pts_cam)
        if S > 0:
            pts_h = np.vstack([pts_cam.T, np.ones((1, S))])
            pts_world = poses64[i] @ pts_h
        else:
            pts_world = np.empty((4, 0), dtype=np.float64)
        all_pts_world_h.append(pts_world)

    all_T_inv = np.linalg.inv(poses64)  # (N, 4, 4)

    # Set up shared state (inherited by forked workers, not pickled)
    _shared = {
        "pts": all_pts_world_h,
        "T_inv": all_T_inv,
        "depths": depths,
        "K": K64,
        "depth_threshold": depth_threshold,
        "N": N,
    }

    t0 = time.time()

    if n_workers <= 1:
        # Single-threaded
        for i in range(N):
            _, row = _compute_row(i)
            overlap[i, i + 1:] = row[i + 1:]
            overlap[i + 1:, i] = row[i + 1:]

            if (i + 1) % 100 == 0 or i == N - 1:
                elapsed = time.time() - t0
                pairs_done = sum(N - k - 1 for k in range(i + 1))
                rate = pairs_done / elapsed if elapsed > 0 else 0
                eta = (total_pairs - pairs_done) / rate if rate > 0 else 0
                log.info(
                    "  Row %d/%d | %d/%d pairs (%.0f/s) | ETA %.0fs",
                    i + 1, N, pairs_done, total_pairs, rate, eta,
                )
    else:
        # Parallel via fork — workers read _shared directly
        log.info("  Starting parallel computation (%d workers) ...", n_workers)
        ctx = mp.get_context("fork")
        rows_done = 0

        with ctx.Pool(n_workers) as pool:
            for i, row in pool.imap_unordered(_compute_row, range(N)):
                overlap[i, i + 1:] = row[i + 1:]
                overlap[i + 1:, i] = row[i + 1:]
                rows_done += 1

                if rows_done % 100 == 0 or rows_done == N:
                    elapsed = time.time() - t0
                    rate = total_pairs * rows_done / N / elapsed if elapsed > 0 else 0
                    log.info(
                        "  %d/%d rows (%.0f est pairs/s) | %.0fs elapsed",
                        rows_done, N, rate, elapsed,
                    )

    _shared = {}  # release references

    elapsed = time.time() - t0
    log.info(
        "  Covisibility done: %d pairs in %.1fs (%.0f pairs/s)",
        total_pairs, elapsed, total_pairs / elapsed if elapsed > 0 else 0,
    )

    return overlap
