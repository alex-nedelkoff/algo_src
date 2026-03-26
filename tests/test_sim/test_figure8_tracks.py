"""Tests for the randomized figure-eight track generator."""
from __future__ import annotations

import numpy as np
import pytest

from sim.figure8_tracks import Figure8TrackGenerator
from sim.tracks import Track


class TestFigure8GeneratorBasicContract:
    """Generator returns a valid Track satisfying the figure-eight spec."""

    def setup_method(self):
        self.gen = Figure8TrackGenerator()
        self.rng = np.random.default_rng(42)

    def test_returns_track_instance(self):
        track = self.gen.generate(self.rng)
        assert isinstance(track, Track)

    def test_gate_count_in_range(self):
        """6-10 gates: 2 crossing gates + 2 loops of 2-4 gates each."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            assert 6 <= track.num_gates <= 10, (
                f"seed {seed}: expected 6-10 gates, got {track.num_gates}"
            )

    def test_no_two_gates_closer_than_02m(self):
        """All gate pairs must be at least 0.2 m apart."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            n = track.num_gates
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(
                        track.gates[i].position - track.gates[j].position
                    )
                    assert dist >= 0.2 - 1e-6, (
                        f"seed {seed}: gates {i} and {j} too close: {dist:.4f} m"
                    )

    def test_track_closes(self):
        """Last gate should be reasonably near the first gate (within 10 m)."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            first = track.gates[0].position
            last = track.gates[-1].position
            dist = np.linalg.norm(last - first)
            assert dist <= 10.0, (
                f"seed {seed}: last gate {dist:.2f} m from first — track does not close"
            )

    def test_elevation_within_bounds(self):
        """All gates must be between 1.0 m and 3.5 m elevation."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            for i, gate in enumerate(track.gates):
                z = gate.position[2]
                assert 1.0 - 1e-6 <= z <= 3.5 + 1e-6, (
                    f"seed {seed}: gate {i} elevation {z:.3f} out of [1.0, 3.5]"
                )

    def test_orientations_are_unit_quaternions(self):
        track = self.gen.generate(self.rng)
        for i, gate in enumerate(track.gates):
            norm = np.linalg.norm(gate.orientation)
            assert abs(norm - 1.0) < 1e-6, (
                f"gate {i} orientation not unit: norm={norm}"
            )


class TestFigure8GeneratorDeterminism:
    """Reproducibility and seed sensitivity checks."""

    def setup_method(self):
        self.gen = Figure8TrackGenerator()

    def test_same_seed_produces_identical_track(self):
        t1 = self.gen.generate(np.random.default_rng(7))
        t2 = self.gen.generate(np.random.default_rng(7))
        assert t1.num_gates == t2.num_gates
        for i, (g1, g2) in enumerate(zip(t1.gates, t2.gates)):
            np.testing.assert_array_almost_equal(
                g1.position, g2.position, decimal=10,
                err_msg=f"gate {i} position differs with same seed"
            )
            np.testing.assert_array_almost_equal(
                g1.orientation, g2.orientation, decimal=10,
                err_msg=f"gate {i} orientation differs with same seed"
            )

    def test_different_seeds_produce_different_tracks(self):
        tracks = [self.gen.generate(np.random.default_rng(s)) for s in range(10)]
        # At least some pairs must differ in gate count or positions
        all_same = all(
            t.num_gates == tracks[0].num_gates
            and np.allclose(t.gates[0].position, tracks[0].gates[0].position)
            for t in tracks[1:]
        )
        assert not all_same, "All seeds produced identical tracks — generator is not random"


class TestFigure8GeneratorGateOrientations:
    """Gate orientations should roughly point from each gate toward the next."""

    def setup_method(self):
        self.gen = Figure8TrackGenerator()

    def test_gates_face_next_gate(self):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            track = self.gen.generate(rng)
            n = track.num_gates
            for i in range(n):
                j = (i + 1) % n
                q = track.gates[i].orientation
                w, x, y, z = q
                # Forward vector from quaternion (local x-axis in world frame)
                forward = np.array([
                    1 - 2 * (y * y + z * z),
                    2 * (x * y + w * z),
                    2 * (x * z - w * y),
                ])
                to_next = track.gates[j].position - track.gates[i].position
                to_next_xy = to_next[:2]
                forward_xy = forward[:2]
                if np.linalg.norm(to_next_xy) > 0.1:
                    cos_angle = np.dot(forward_xy, to_next_xy) / (
                        np.linalg.norm(forward_xy) * np.linalg.norm(to_next_xy)
                    )
                    assert cos_angle > 0.5, (
                        f"seed {seed}: gate {i} does not face gate {j}: "
                        f"cos_angle={cos_angle:.3f}"
                    )
