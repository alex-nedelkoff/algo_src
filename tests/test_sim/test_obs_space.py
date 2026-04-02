"""Tests for expanded observation space with gate geometry and arena extent."""
from __future__ import annotations

import numpy as np
import pytest

from sim.envs.gate_race_env import _compute_obs_dim


class TestObsDim:
    def test_default_1_lookahead_no_history(self):
        # 20 base + 6*1 gate + 1 arena = 27
        assert _compute_obs_dim(1, 0) == 27

    def test_2_lookahead_no_history(self):
        # 20 base + 6*2 gate + 1 arena = 33
        assert _compute_obs_dim(2, 0) == 33

    def test_2_lookahead_7_history(self):
        # 20 base + 6*2 gate + 1 arena + 4*7 history = 61
        assert _compute_obs_dim(2, 7) == 61

    def test_1_lookahead_4_history(self):
        # 20 base + 6*1 gate + 1 arena + 4*4 history = 43
        assert _compute_obs_dim(1, 4) == 43
