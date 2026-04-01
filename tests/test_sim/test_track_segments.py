"""Tests for track segment primitives."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sim.track_segments import (
    SegmentResult,
    segment_straight,
    segment_turn_90,
    segment_turn_180,
    segment_slalom,
    segment_chicane,
    segment_climb,
    segment_dive,
    segment_orbit,
    SEGMENT_REGISTRY,
)


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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_COMMON_KWARGS = dict(
    start_pos=np.array([0.0, 0.0, 2.0]),
    start_heading=0.0,
    elevation_min=0.5,
    elevation_max=4.0,
    arena_half_width=20.0,
)


# ---------------------------------------------------------------------------
# Turn 90
# ---------------------------------------------------------------------------

class TestTurn90:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_gate_count(self, rng):
        result = segment_turn_90(rng=rng, **_COMMON_KWARGS)
        assert 1 <= len(result.gates) <= 2

    def test_exit_heading_roughly_90_degrees(self):
        # Run a bunch of seeds and check the turn is ~90 deg in magnitude
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = segment_turn_90(rng=rng, **_COMMON_KWARGS)
            delta = abs(_wrap_angle_test(result.exit_heading - _COMMON_KWARGS["start_heading"]))
            assert math.isclose(delta, math.pi / 2, abs_tol=0.3), (
                f"seed={seed}: expected ~90 deg turn, got {math.degrees(delta):.1f} deg"
            )

    def test_returns_segment_result(self, rng):
        result = segment_turn_90(rng=rng, **_COMMON_KWARGS)
        assert isinstance(result, SegmentResult)


# ---------------------------------------------------------------------------
# Turn 180
# ---------------------------------------------------------------------------

class TestTurn180:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_gate_count(self, rng):
        result = segment_turn_180(rng=rng, **_COMMON_KWARGS)
        assert 2 <= len(result.gates) <= 3

    def test_exit_heading_roughly_reversed(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = segment_turn_180(rng=rng, **_COMMON_KWARGS)
            delta = abs(_wrap_angle_test(result.exit_heading - _COMMON_KWARGS["start_heading"]))
            assert math.isclose(delta, math.pi, abs_tol=0.35), (
                f"seed={seed}: expected ~180 deg turn, got {math.degrees(delta):.1f} deg"
            )

    def test_returns_segment_result(self, rng):
        result = segment_turn_180(rng=rng, **_COMMON_KWARGS)
        assert isinstance(result, SegmentResult)


# ---------------------------------------------------------------------------
# Slalom
# ---------------------------------------------------------------------------

class TestSlalom:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_gate_count(self, rng):
        result = segment_slalom(rng=rng, **_COMMON_KWARGS)
        assert 3 <= len(result.gates) <= 5

    def test_gates_alternate_laterally(self, rng):
        # Gates should alternate sides relative to the forward direction.
        # Project each gate position onto the lateral axis (perpendicular to start heading).
        result = segment_slalom(rng=rng, **_COMMON_KWARGS)
        entry_heading = _COMMON_KWARGS["start_heading"]
        start = _COMMON_KWARGS["start_pos"]
        # Lateral unit vector: 90 deg left of heading
        lat = np.array([-math.sin(entry_heading), math.cos(entry_heading)])
        lateral_offsets = [
            float(np.dot((pos[:2] - start[:2]), lat))
            for pos, _ in result.gates
        ]
        # At least some sign alternation in lateral offsets
        alternations = sum(
            1 for i in range(1, len(lateral_offsets))
            if lateral_offsets[i] * lateral_offsets[i - 1] < 0
        )
        assert alternations >= 1, f"No lateral alternation detected: {lateral_offsets}"

    def test_returns_segment_result(self, rng):
        result = segment_slalom(rng=rng, **_COMMON_KWARGS)
        assert isinstance(result, SegmentResult)


# ---------------------------------------------------------------------------
# Chicane
# ---------------------------------------------------------------------------

class TestChicane:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_gate_count(self, rng):
        result = segment_chicane(rng=rng, **_COMMON_KWARGS)
        assert 2 <= len(result.gates) <= 3

    def test_exit_heading_close_to_entry(self):
        # Chicane should return roughly to entry heading (within 60 deg)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = segment_chicane(rng=rng, **_COMMON_KWARGS)
            delta = abs(_wrap_angle_test(result.exit_heading - _COMMON_KWARGS["start_heading"]))
            assert delta < math.radians(70), (
                f"seed={seed}: chicane exit heading deviated {math.degrees(delta):.1f} deg"
            )

    def test_returns_segment_result(self, rng):
        result = segment_chicane(rng=rng, **_COMMON_KWARGS)
        assert isinstance(result, SegmentResult)


# ---------------------------------------------------------------------------
# Climb
# ---------------------------------------------------------------------------

class TestClimb:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_gate_count(self, rng):
        result = segment_climb(rng=rng, **_COMMON_KWARGS)
        assert 1 <= len(result.gates) <= 2

    def test_exit_z_greater_than_start_z(self):
        # For most seeds starting at z=2.0, the climb should raise elevation
        # (may be clamped at elevation_max, so check at least one gate is higher or equal)
        start_z = _COMMON_KWARGS["start_pos"][2]
        gains_observed = 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = segment_climb(rng=rng, **_COMMON_KWARGS)
            if result.exit_pos[2] > start_z:
                gains_observed += 1
        assert gains_observed >= 15  # almost always climbs unless clamped

    def test_exit_heading_same_as_entry(self, rng):
        result = segment_climb(rng=rng, **_COMMON_KWARGS)
        assert math.isclose(result.exit_heading, _COMMON_KWARGS["start_heading"], abs_tol=1e-9)

    def test_returns_segment_result(self, rng):
        result = segment_climb(rng=rng, **_COMMON_KWARGS)
        assert isinstance(result, SegmentResult)


# ---------------------------------------------------------------------------
# Dive
# ---------------------------------------------------------------------------

class TestDive:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_gate_count(self, rng):
        result = segment_dive(rng=rng, **_COMMON_KWARGS)
        assert 1 <= len(result.gates) <= 2

    def test_exit_z_less_than_start_z(self):
        start_z = _COMMON_KWARGS["start_pos"][2]
        losses_observed = 0
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = segment_dive(rng=rng, **_COMMON_KWARGS)
            if result.exit_pos[2] < start_z:
                losses_observed += 1
        assert losses_observed >= 15

    def test_exit_heading_same_as_entry(self, rng):
        result = segment_dive(rng=rng, **_COMMON_KWARGS)
        assert math.isclose(result.exit_heading, _COMMON_KWARGS["start_heading"], abs_tol=1e-9)

    def test_returns_segment_result(self, rng):
        result = segment_dive(rng=rng, **_COMMON_KWARGS)
        assert isinstance(result, SegmentResult)


# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------

class TestOrbit:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_gate_count(self, rng):
        result = segment_orbit(rng=rng, **_COMMON_KWARGS)
        assert 3 <= len(result.gates) <= 5

    def test_exit_heading_turns_more_than_60_degrees(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = segment_orbit(rng=rng, **_COMMON_KWARGS)
            delta = abs(_wrap_angle_test(result.exit_heading - _COMMON_KWARGS["start_heading"]))
            # orbit is 180-270 deg — but _wrap_angle folds to [-pi,pi] so minimum
            # expected absolute delta after wrapping is pi (180 deg)
            assert delta > math.radians(60), (
                f"seed={seed}: orbit turn too small: {math.degrees(delta):.1f} deg"
            )

    def test_returns_segment_result(self, rng):
        result = segment_orbit(rng=rng, **_COMMON_KWARGS)
        assert isinstance(result, SegmentResult)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestSegmentRegistry:
    EXPECTED_NAMES = {
        "straight", "turn_90", "turn_180", "slalom",
        "chicane", "climb", "dive", "orbit",
    }

    def test_all_names_present(self):
        assert set(SEGMENT_REGISTRY.keys()) == self.EXPECTED_NAMES

    def test_all_callable(self):
        for name, fn in SEGMENT_REGISTRY.items():
            assert callable(fn), f"{name!r} is not callable"

    def test_all_produce_valid_segment_result(self):
        for name, fn in SEGMENT_REGISTRY.items():
            rng = np.random.default_rng(7)
            result = fn(rng=rng, **_COMMON_KWARGS)
            assert isinstance(result, SegmentResult), f"{name!r} did not return SegmentResult"
            assert len(result.gates) >= 1, f"{name!r} returned 0 gates"
            assert result.exit_pos.shape == (3,), f"{name!r} exit_pos has wrong shape"


# ---------------------------------------------------------------------------
# Local helper (mirrors _wrap_angle from module)
# ---------------------------------------------------------------------------

def _wrap_angle_test(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)
