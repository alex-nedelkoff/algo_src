"""Tests for silhouette IoU computation."""
import numpy as np

from scripts.compare_pybullet_vs_fixtures import silhouette_iou


def test_iou_perfect_match_is_one():
    a = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    assert silhouette_iou(a, a.copy()) == 1.0


def test_iou_disjoint_is_zero():
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[0, 1], [0, 0]], dtype=bool)
    assert silhouette_iou(a, b) == 0.0


def test_iou_half_overlap():
    a = np.array([[1, 1, 0, 0]], dtype=bool)
    b = np.array([[0, 1, 1, 0]], dtype=bool)
    # Intersection = 1 px, Union = 3 px → IoU = 1/3
    assert abs(silhouette_iou(a, b) - 1 / 3) < 1e-9


def test_iou_both_empty_is_one():
    a = np.zeros((2, 2), dtype=bool)
    assert silhouette_iou(a, a.copy()) == 1.0
