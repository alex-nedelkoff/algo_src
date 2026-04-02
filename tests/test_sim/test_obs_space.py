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


from sim.envs.gate_race_env import GateRaceEnv
from sim.tracks import Track, _yaw_to_quat
from sim.types import GateState
import math


def _make_simple_track(n_gates=4, gate_width=1.5, gate_height=1.5):
    """Create a simple square track for testing."""
    positions = [
        [3.0, 0.0, 2.0],
        [3.0, 3.0, 2.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 2.0],
    ][:n_gates]
    gates = []
    for i, pos in enumerate(positions):
        next_pos = positions[(i + 1) % len(positions)]
        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]
        yaw = math.atan2(dy, dx)
        g = GateState(position=np.array(pos), orientation=_yaw_to_quat(yaw))
        if hasattr(g, 'dimensions'):
            g.dimensions = {"width": gate_width, "height": gate_height}
        if hasattr(g, 'shape'):
            g.shape = "rectangle"
        gates.append(g)
    return Track(gates)


class TestObsGateGeometry:
    def test_obs_shape_2_lookahead(self):
        track = _make_simple_track()
        env = GateRaceEnv(
            track=track, n_envs=1, n_lookahead_gates=2,
            arena_bounds=10.0,
        )
        obs, _ = env.reset()
        assert obs.shape == (33,), f"Expected (33,), got {obs.shape}"
        env.close()

    def test_gate_width_height_in_obs(self):
        track = _make_simple_track(gate_width=2.0, gate_height=1.0)
        env = GateRaceEnv(
            track=track, n_envs=1, n_lookahead_gates=2,
            arena_bounds=10.0,
        )
        obs, _ = env.reset()
        # Lookahead gate 1: width at offset 20+4=24, height at 20+5=25
        assert abs(obs[24] - 2.0) < 0.01, f"gate_width={obs[24]}, expected 2.0"
        assert abs(obs[25] - 1.0) < 0.01, f"gate_height={obs[25]}, expected 1.0"
        env.close()

    def test_arena_extent_in_obs(self):
        track = _make_simple_track()
        env = GateRaceEnv(
            track=track, n_envs=1, n_lookahead_gates=2,
            arena_bounds=8.0,
        )
        obs, _ = env.reset()
        # arena_extent at offset 20 + 6*2 = 32
        assert abs(obs[32] - 0.8) < 0.01, f"arena_extent={obs[32]}, expected 0.8"
        env.close()

    def test_default_gate_dims_from_passage_radius(self):
        track = _make_simple_track()
        env = GateRaceEnv(
            track=track, n_envs=1, n_lookahead_gates=1,
            gate_passage_radius=0.75, arena_bounds=10.0,
        )
        obs, _ = env.reset()
        # For 1 lookahead: gate dims at offset 24, 25
        assert abs(obs[24] - 1.5) < 0.01, f"gate_width={obs[24]}, expected 1.5"
        assert abs(obs[25] - 1.5) < 0.01, f"gate_height={obs[25]}, expected 1.5"
        env.close()
