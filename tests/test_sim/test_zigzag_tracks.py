import numpy as np
import pytest
from sim.tracks import Track
from sim.types import GateState


def _gate(x, y, z=1.5):
    return GateState(position=np.array([x, y, z]))


class TestTrackChainingAttrs:
    def test_default_track_not_chainable(self):
        track = Track([_gate(0, 0), _gate(1, 0)])
        assert track.chainable is False
        assert track.exit_pos is None
        assert track.exit_heading is None
        assert track.generator is None

    def test_chainable_track(self):
        track = Track(
            [_gate(0, 0), _gate(1, 0)],
            chainable=True,
            exit_pos=np.array([2.0, 0.0, 1.5]),
            exit_heading=0.0,
        )
        assert track.chainable is True
        np.testing.assert_array_almost_equal(track.exit_pos, [2.0, 0.0, 1.5])
        assert track.exit_heading == 0.0


from sim.zigzag_tracks import ZigzagTrackGenerator


class TestZigzagTrackGenerator:
    def setup_method(self):
        self.gen = ZigzagTrackGenerator(
            n_gates_min=5,
            n_gates_max=8,
            turn_angle_min=100.0,
            turn_angle_max=160.0,
            gate_spacing_min=1.0,
            gate_spacing_max=4.0,
            elevation_min=0.5,
            elevation_max=4.0,
            elevation_delta_max=1.5,
        )

    def test_generates_valid_track(self):
        rng = np.random.default_rng(42)
        track = self.gen.generate(rng)
        assert 5 <= track.num_gates <= 8
        assert track.chainable is True
        assert track.exit_pos is not None
        assert track.exit_heading is not None
        assert track.generator is self.gen

    def test_alternating_turns(self):
        rng = np.random.default_rng(42)
        track = self.gen.generate(rng)
        positions = [g.position for g in track.gates]
        headings = []
        for i in range(len(positions) - 1):
            dx = positions[i + 1][0] - positions[i][0]
            dy = positions[i + 1][1] - positions[i][1]
            headings.append(np.arctan2(dy, dx))
        turns = [headings[i + 1] - headings[i] for i in range(len(headings) - 1)]
        turns = [(t + np.pi) % (2 * np.pi) - np.pi for t in turns]
        significant = [(i, t) for i, t in enumerate(turns) if abs(t) > 0.1]
        for i in range(1, len(significant)):
            prev_sign = np.sign(significant[i - 1][1])
            curr_sign = np.sign(significant[i][1])
            assert prev_sign != curr_sign, (
                f"Turn {significant[i][0]} has same sign as {significant[i-1][0]}"
            )

    def test_chaining_from_exit(self):
        rng = np.random.default_rng(42)
        track1 = self.gen.generate(rng)
        track2 = self.gen.generate(rng, start_pos=track1.exit_pos, start_heading=track1.exit_heading)
        first_gate_t2 = track2.gates[0].position
        dist = np.linalg.norm(first_gate_t2 - track1.exit_pos)
        assert dist < self.gen.gate_spacing_max * 1.5, f"First gate of chained segment too far: {dist:.1f}m"

    def test_gate_spacing_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            track = self.gen.generate(rng)
            positions = [g.position for g in track.gates]
            for i in range(len(positions) - 1):
                dist = np.linalg.norm(positions[i + 1] - positions[i])
                assert dist >= self.gen.gate_spacing_min * 0.9, f"Gate {i} spacing too small: {dist:.2f}"
                assert dist <= self.gen.gate_spacing_max * 1.2, f"Gate {i} spacing too large: {dist:.2f}"

    def test_elevation_bounds(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            track = self.gen.generate(rng)
            for i, g in enumerate(track.gates):
                assert g.position[2] >= 0.5, f"Gate {i} below elevation_min"
                assert g.position[2] <= 4.0, f"Gate {i} above elevation_max"
