"""Tests for track segment primitives."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.track_segments import SegmentResult, segment_straight


class TestSegmentResult:
    def test_fields(self):
        gates = [(np.array([0.0, 0.0, 2.0]), 0.0)]
        sr = SegmentResult(gates=gates, exit_pos=np.array([5.0, 0.0, 2.0]), exit_heading=0.0)
        assert len(sr.gates) == 1
        assert sr.exit_heading == 0.0


class TestStraight:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_produces_1_or_2_gates(self, rng):
        result = segment_straight(
            start_pos=np.array([0.0, 0.0, 2.0]),
            start_heading=0.0,
            rng=rng,
            elevation_min=0.5,
            elevation_max=4.0,
            arena_half_width=10.0,
        )
        assert 1 <= len(result.gates) <= 2

    def test_gates_along_heading(self, rng):
        result = segment_straight(
            start_pos=np.array([0.0, 0.0, 2.0]),
            start_heading=0.0,
            rng=rng,
            elevation_min=0.5,
            elevation_max=4.0,
            arena_half_width=10.0,
        )
        for pos, heading in result.gates:
            assert pos[0] > 0.0
            assert abs(heading) < 0.1

    def test_exit_ahead_of_last_gate(self, rng):
        result = segment_straight(
            start_pos=np.array([0.0, 0.0, 2.0]),
            start_heading=0.0,
            rng=rng,
            elevation_min=0.5,
            elevation_max=4.0,
            arena_half_width=10.0,
        )
        last_gate_pos = result.gates[-1][0]
        assert result.exit_pos[0] >= last_gate_pos[0]
        assert abs(result.exit_heading) < 0.1

    def test_deterministic_with_seed(self):
        kwargs = dict(
            start_pos=np.array([0.0, 0.0, 2.0]),
            start_heading=0.0,
            elevation_min=0.5,
            elevation_max=4.0,
            arena_half_width=10.0,
        )
        r1 = segment_straight(rng=np.random.default_rng(99), **kwargs)
        r2 = segment_straight(rng=np.random.default_rng(99), **kwargs)
        assert len(r1.gates) == len(r2.gates)
        for (p1, h1), (p2, h2) in zip(r1.gates, r2.gates):
            np.testing.assert_array_equal(p1, p2)
            assert h1 == h2
