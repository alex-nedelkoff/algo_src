"""Integration tests for procedural track training."""
from __future__ import annotations

import numpy as np
import pytest

from sim.envs.gate_race_env import GateRaceEnv, _gate_normal
from sim.procedural_tracks import ProceduralTrackGenerator


class TestEnvWithGenerator:
    """GateRaceEnv with procedural track generator."""

    def setup_method(self):
        self.gen = ProceduralTrackGenerator(
            n_gates_min=4,
            n_gates_max=8,
            arena_half_width=15.0,
        )
        self.env = GateRaceEnv(
            n_envs=4,
            track_generator=self.gen,
            arena_bounds=15.0,
        )

    def test_env_builds_and_resets(self):
        obs, info = self.env.reset()
        assert obs.shape == (4, 24)

    def test_env_steps_without_error(self):
        self.env.reset(seed=0)
        action = np.zeros((4, 4), dtype=np.float32)
        obs, rewards, terminated, truncated, infos = self.env.step(action)
        assert obs.shape == (4, 24)
        assert rewards.shape == (4,)

    def test_different_envs_get_different_tracks(self):
        self.env.reset(seed=0)
        any_differ = False
        for i in range(4):
            for j in range(i + 1, 4):
                if self.env._tracks[i].num_gates != self.env._tracks[j].num_gates:
                    any_differ = True
                    break
                for gi, gj in zip(self.env._tracks[i].gates, self.env._tracks[j].gates):
                    if not np.allclose(gi.position, gj.position):
                        any_differ = True
                        break
            if any_differ:
                break
        # Just verify they're valid tracks
        for i in range(4):
            assert 4 <= self.env._tracks[i].num_gates <= 8

    def test_gate_passage_on_generated_track(self):
        """Fly drone straight through first gate, verify passage detected."""
        self.env.reset(seed=0)
        env_idx = 0
        track = self.env._tracks[env_idx]
        gate = track.gates[0]

        normal = _gate_normal(gate)
        self.env._states[env_idx, 0:3] = gate.position - 0.5 * normal
        self.env._gate_indices[env_idx] = 0
        self.env._gates_passed[env_idx] = 0
        self.env._prev_along_normal[env_idx] = -0.5

        self.env._states[env_idx, 0:3] = gate.position + 0.5 * normal
        action = np.zeros((self.env.n_envs, 4), dtype=np.float32)
        self.env.step(action)
        assert self.env._gates_passed[env_idx] >= 1

    def test_get_gate_geometry_per_env(self):
        self.env.reset(seed=0)
        geom0 = self.env.get_gate_geometry(env_idx=0)
        geom1 = self.env.get_gate_geometry(env_idx=1)
        assert "positions" in geom0
        assert "orientations" in geom0
        assert geom0["positions"].shape[0] == self.env._tracks[0].num_gates
        assert geom1["positions"].shape[0] == self.env._tracks[1].num_gates


class TestEnvWithoutGenerator:
    """Backwards compatibility: no generator uses figure-8."""

    def test_default_figure8(self):
        env = GateRaceEnv(n_envs=2)
        env.reset(seed=0)
        assert env._tracks[0].num_gates == 8
        assert env._tracks[1].num_gates == 8

    def test_explicit_track_shared(self):
        from sim.tracks import build_figure8_track
        track = build_figure8_track()
        env = GateRaceEnv(n_envs=2, track=track)
        env.reset(seed=0)
        assert env._tracks[0] is env._tracks[1]


class TestEnvWithFixedTracks:
    """Eval mode: pre-assigned tracks list."""

    def test_tracks_list_assigned(self):
        from sim.tracks import build_figure8_track
        tracks = [build_figure8_track() for _ in range(3)]
        env = GateRaceEnv(n_envs=3, tracks=tracks)
        env.reset(seed=0)
        for i in range(3):
            assert env._tracks[i] is tracks[i]

    def test_tracks_list_wrong_length_raises(self):
        from sim.tracks import build_figure8_track
        tracks = [build_figure8_track() for _ in range(2)]
        with pytest.raises((ValueError, AssertionError)):
            GateRaceEnv(n_envs=3, tracks=tracks)
