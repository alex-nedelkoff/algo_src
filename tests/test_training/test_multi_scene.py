"""Tests for multi-scene track regeneration."""
import numpy as np
import pytest

from training.multi_scene_callback import assign_scene_groups


class TestMultiScene:
    def test_groups_cover_all_envs(self):
        groups = assign_scene_groups(n_envs=100, n_scenes=10)
        assert len(groups) == 10
        all_envs = set()
        for g in groups:
            all_envs.update(g)
        assert all_envs == set(range(100))

    def test_groups_are_equal_size(self):
        groups = assign_scene_groups(n_envs=100, n_scenes=10)
        for g in groups:
            assert len(g) == 10

    def test_uneven_division(self):
        groups = assign_scene_groups(n_envs=7, n_scenes=3)
        total = sum(len(g) for g in groups)
        assert total == 7
