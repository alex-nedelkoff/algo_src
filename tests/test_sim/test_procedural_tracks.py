"""Tests for procedural track generation."""
from __future__ import annotations

import numpy as np
import pytest

from sim.procedural_tracks import ProceduralTrackGenerator
from sim.tracks import Track


class TestGeneratorConstraints:
    """Generated tracks satisfy all configured constraints."""

    def setup_method(self):
        self.gen = ProceduralTrackGenerator(
            n_gates_min=4,
            n_gates_max=8,
            gate_spacing_min=2.0,
            gate_spacing_max=6.0,
            turn_angle_min=-90.0,
            turn_angle_max=90.0,
            elevation_min=1.0,
            elevation_max=4.0,
            elevation_delta_max=0.8,
            arena_half_width=15.0,
            closure_max_angle=90.0,
        )
        self.rng = np.random.default_rng(42)

    def test_returns_track(self):
        track = self.gen.generate(self.rng)
        assert isinstance(track, Track)

    def test_gate_count_in_range(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            assert 4 <= track.num_gates <= 8

    def test_positions_within_arena(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for gate in track.gates:
                assert abs(gate.position[0]) <= 15.0
                assert abs(gate.position[1]) <= 15.0

    def test_consecutive_spacing(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for i in range(track.num_gates):
                j = (i + 1) % track.num_gates
                dist = np.linalg.norm(
                    track.gates[j].position - track.gates[i].position
                )
                max_allowed = 6.0 * 2 if j == 0 else 6.0
                assert dist >= 2.0 - 0.01, f"Gates {i}->{j} too close: {dist}"
                assert dist <= max_allowed + 0.01, f"Gates {i}->{j} too far: {dist}"

    def test_elevation_in_range(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for gate in track.gates:
                assert 1.0 - 0.01 <= gate.position[2] <= 4.0 + 0.01

    def test_elevation_delta(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for i in range(track.num_gates):
                j = (i + 1) % track.num_gates
                dz = abs(track.gates[j].position[2] - track.gates[i].position[2])
                assert dz <= 0.8 + 0.01

    def test_orientations_are_unit_quaternions(self):
        track = self.gen.generate(self.rng)
        for gate in track.gates:
            assert abs(np.linalg.norm(gate.orientation) - 1.0) < 1e-6

    def test_gates_face_next_gate(self):
        track = self.gen.generate(self.rng)
        for i in range(track.num_gates):
            j = (i + 1) % track.num_gates
            q = track.gates[i].orientation
            w, x, y, z = q
            forward = np.array([
                1 - 2*(y*y + z*z),
                2*(x*y + w*z),
                2*(x*z - w*y),
            ])
            to_next = track.gates[j].position - track.gates[i].position
            to_next_xy = to_next[:2]
            forward_xy = forward[:2]
            if np.linalg.norm(to_next_xy) > 0.01:
                cos_angle = (
                    np.dot(forward_xy, to_next_xy)
                    / (np.linalg.norm(forward_xy) * np.linalg.norm(to_next_xy))
                )
                assert cos_angle > 0.5, f"Gate {i} not facing gate {j}"

    def test_seeded_reproducibility(self):
        t1 = self.gen.generate(np.random.default_rng(99))
        t2 = self.gen.generate(np.random.default_rng(99))
        assert t1.num_gates == t2.num_gates
        for g1, g2 in zip(t1.gates, t2.gates):
            np.testing.assert_array_equal(g1.position, g2.position)

    def test_min_separation_between_all_gates(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for i in range(track.num_gates):
                for j in range(i + 2, track.num_gates):
                    if j == track.num_gates - 1 and i == 0:
                        continue
                    dist = np.linalg.norm(
                        track.gates[j].position - track.gates[i].position
                    )
                    assert dist >= 2.0 - 0.01, (
                        f"Non-consecutive gates {i},{j} too close: {dist}"
                    )


class TestGeneratorValidation:
    def test_n_gates_min_zero(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(n_gates_min=0)

    def test_n_gates_min_two(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(n_gates_min=2, n_gates_max=4)

    def test_n_gates_min_gt_max(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(n_gates_min=10, n_gates_max=5)

    def test_spacing_min_gt_max(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(gate_spacing_min=10.0, gate_spacing_max=5.0)

    def test_elevation_min_gt_max(self):
        with pytest.raises(ValueError):
            ProceduralTrackGenerator(elevation_min=5.0, elevation_max=2.0)


class TestGeneratorFallback:
    def test_impossible_params_fallback(self):
        gen = ProceduralTrackGenerator(
            n_gates_min=12,
            n_gates_max=12,
            gate_spacing_min=7.0,
            gate_spacing_max=8.0,
            turn_angle_min=-10.0,
            turn_angle_max=10.0,
            arena_half_width=5.0,
            closure_max_angle=10.0,
            closure_max_retries=3,
        )
        rng = np.random.default_rng(42)
        track = gen.generate(rng)
        assert track.num_gates == 8  # figure-8 fallback
