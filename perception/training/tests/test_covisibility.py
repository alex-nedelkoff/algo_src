"""Unit tests for covisibility computation."""

import numpy as np
import pytest

from perception.training.data.covisibility import (
    compute_pairwise_overlap,
    compute_scene_covisibility,
    unproject_pixels,
)


def _make_K():
    """Simple pinhole intrinsics."""
    return np.array([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0],
    ])


def _make_flat_depth(H=100, W=100, z=5.0):
    """Flat depth map at z meters."""
    return np.full((H, W), z, dtype=np.float32)


class TestUnprojectPixels:
    def test_output_shape(self):
        depth = _make_flat_depth()
        K = _make_K()
        pts, coords = unproject_pixels(depth, K, num_samples=50)
        assert pts.shape == (50, 3)
        assert coords.shape == (50, 2)

    def test_z_matches_depth(self):
        depth = _make_flat_depth(z=3.0)
        K = _make_K()
        pts, _ = unproject_pixels(depth, K, num_samples=100)
        np.testing.assert_allclose(pts[:, 2], 3.0, atol=1e-10)

    def test_empty_depth(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        K = _make_K()
        pts, coords = unproject_pixels(depth, K, num_samples=50)
        assert pts.shape == (0, 3)
        assert coords.shape == (0, 2)


class TestPairwiseOverlap:
    def test_identical_frames_full_overlap(self):
        """Same pose + same depth → overlap should be ~1.0."""
        K = _make_K()
        depth = _make_flat_depth(z=5.0)
        T = np.eye(4)

        pts, _ = unproject_pixels(depth, K, num_samples=1000, rng=np.random.default_rng(0))
        score = compute_pairwise_overlap(pts, T, depth, T, K, depth_threshold=0.1)
        assert score > 0.95

    def test_no_overlap_far_apart(self):
        """Cameras far apart looking in same direction → no overlap."""
        K = _make_K()
        depth = _make_flat_depth(z=5.0)

        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[0, 3] = 1000.0  # huge translation

        pts, _ = unproject_pixels(depth, K, num_samples=500, rng=np.random.default_rng(0))
        score = compute_pairwise_overlap(pts, T1, depth, T2, K, depth_threshold=0.1)
        assert score < 0.05


class TestSceneCovisibility:
    def test_diagonal_is_one(self):
        K = _make_K()
        depths = [_make_flat_depth() for _ in range(3)]
        poses = np.stack([np.eye(4)] * 3)

        overlap = compute_scene_covisibility(depths, poses, K, num_samples=100)
        np.testing.assert_array_equal(np.diag(overlap), 1.0)

    def test_symmetric(self):
        K = _make_K()
        depths = [_make_flat_depth() for _ in range(3)]
        poses = np.stack([np.eye(4)] * 3)

        overlap = compute_scene_covisibility(depths, poses, K, num_samples=100)
        np.testing.assert_array_almost_equal(overlap, overlap.T)

    def test_shape(self):
        K = _make_K()
        N = 5
        depths = [_make_flat_depth() for _ in range(N)]
        poses = np.stack([np.eye(4)] * N)

        overlap = compute_scene_covisibility(depths, poses, K, num_samples=100)
        assert overlap.shape == (N, N)
