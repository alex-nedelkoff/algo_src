"""Tests for ComposedTrackGenerator."""
from __future__ import annotations

import numpy as np
import pytest

from sim.composed_tracks import ComposedTrackGenerator, PerturbationConfig
from sim.tracks import Track


class TestPerturbationConfig:
    def test_default_disabled(self):
        pc = PerturbationConfig()
        assert pc.enabled is False

    def test_fields(self):
        pc = PerturbationConfig(enabled=True, position_noise=0.1)
        assert pc.enabled is True
        assert pc.position_noise == 0.1


class TestComposedTrackGenerator:
    @pytest.fixture
    def gen(self):
        return ComposedTrackGenerator(
            segment_weights={
                "straight": 0.2, "turn_90": 0.15, "turn_180": 0.1,
                "slalom": 0.2, "chicane": 0.1, "climb": 0.08,
                "dive": 0.07, "orbit": 0.1,
            },
            n_segments_min=2,
            n_segments_max=4,
            n_gates_max=14,
            arena_half_width=10.0,
            elevation_min=0.5,
            elevation_max=4.0,
            closure_max_retries=50,
        )

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_generates_track(self, gen, rng):
        track = gen.generate(rng)
        assert isinstance(track, Track)
        assert track.num_gates >= 3

    def test_gate_count_respects_max(self, gen):
        for seed in range(20):
            track = gen.generate(np.random.default_rng(seed))
            assert track.num_gates <= 14, f"seed {seed}: {track.num_gates} gates > 14"

    def test_gates_within_arena(self, gen):
        for seed in range(20):
            track = gen.generate(np.random.default_rng(seed))
            for gate in track.gates:
                assert abs(gate.position[0]) <= 10.5, f"gate X out of bounds: {gate.position}"
                assert abs(gate.position[1]) <= 10.5, f"gate Y out of bounds: {gate.position}"

    def test_deterministic_with_seed(self, gen):
        t1 = gen.generate(np.random.default_rng(99))
        t2 = gen.generate(np.random.default_rng(99))
        assert t1.num_gates == t2.num_gates
        for g1, g2 in zip(t1.gates, t2.gates):
            np.testing.assert_array_almost_equal(g1.position, g2.position)

    def test_perturbation_disabled_by_default(self, rng):
        gen = ComposedTrackGenerator(
            segment_weights={"straight": 0.5, "turn_90": 0.5},
            perturbation=PerturbationConfig(enabled=False),
        )
        track = gen.generate(rng)
        assert isinstance(track, Track)

    def test_closure_valid(self, gen):
        """First and last gate should be reachable from each other."""
        for seed in range(20):
            track = gen.generate(np.random.default_rng(seed))
            first = track.gates[0].position
            last = track.gates[-1].position
            dist = float(np.linalg.norm(first - last))
            assert dist < 15.0, f"seed {seed}: closure dist {dist:.1f}m"
