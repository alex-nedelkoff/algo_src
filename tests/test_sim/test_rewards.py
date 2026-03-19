"""Tests for sim.rewards module.

Contains hand-computed test cases for monorace_reward, delta-based progress
reward, body rate penalty, gate offset penalty, thresholded action smoothness,
and composite reward tests.

Aligned with MonoRace paper reward structure (arXiv:2601.15222, COR-44).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.rewards import (
    action_smoothness_penalty,
    body_rate_penalty,
    gate_offset_penalty,
    gate_progress_reward,
    heading_alignment_reward,
    monorace_reward,
    spline_proximity_reward,
)
from sim.types import Action, GateState, QuadState


class TestGateProgressReward:
    """Tests for delta-based gate_progress_reward."""

    def test_getting_closer(self) -> None:
        """Positive reward when drone moves closer to gate."""
        # prev_dist=10, curr_dist=9.8 => delta = 0.2, clip = 30*0.01 = 0.3
        reward = gate_progress_reward(prev_dist=10.0, curr_dist=9.8, v_max=30.0, dt=0.01)
        assert reward == pytest.approx(0.2)

    def test_moving_away(self) -> None:
        """Negative reward when drone moves further from gate."""
        # prev_dist=8, curr_dist=10 => delta = -2.0
        reward = gate_progress_reward(prev_dist=8.0, curr_dist=10.0, v_max=30.0, dt=0.01)
        assert reward == pytest.approx(-2.0)

    def test_no_movement(self) -> None:
        """Zero reward when distance unchanged."""
        reward = gate_progress_reward(prev_dist=5.0, curr_dist=5.0, v_max=30.0, dt=0.01)
        assert reward == pytest.approx(0.0)

    def test_clipped_by_vmax_dt(self) -> None:
        """Delta is clipped to v_max * dt to prevent artificial bonuses.

        v_max=30, dt=0.01 => max_delta = 0.3
        prev_dist=10, curr_dist=5 => raw_delta = 5.0, clipped to 0.3
        """
        reward = gate_progress_reward(prev_dist=10.0, curr_dist=5.0, v_max=30.0, dt=0.01)
        assert reward == pytest.approx(0.3)

    def test_negative_not_clipped(self) -> None:
        """Negative delta (moving away) is not clipped by v_max*dt.

        Only positive delta is clipped via min(delta, v_max*dt).
        """
        reward = gate_progress_reward(prev_dist=5.0, curr_dist=15.0, v_max=30.0, dt=0.01)
        assert reward == pytest.approx(-10.0)

    def test_exact_vmax_dt(self) -> None:
        """Delta exactly at v_max*dt is not clipped."""
        # v_max=10, dt=0.01 => max_delta = 0.1
        reward = gate_progress_reward(prev_dist=5.1, curr_dist=5.0, v_max=10.0, dt=0.01)
        assert reward == pytest.approx(0.1)

    def test_m23_vmax(self) -> None:
        """M23 policy uses v_max=10 for conservative clipping."""
        # v_max=10, dt=0.01 => max_delta = 0.1
        reward = gate_progress_reward(prev_dist=10.0, curr_dist=5.0, v_max=10.0, dt=0.01)
        assert reward == pytest.approx(0.1)


class TestBodyRatePenalty:
    """Tests for body_rate_penalty."""

    def test_zero_omega(self) -> None:
        """No penalty when angular velocity is zero."""
        state = QuadState(omega=np.array([0.0, 0.0, 0.0]))
        assert body_rate_penalty(state) == pytest.approx(0.0)

    def test_nonzero_omega(self) -> None:
        """Penalty equals negative squared L2 norm of omega.

        omega = [1, 2, 3] => ||omega||^2 = 1 + 4 + 9 = 14
        penalty = -14.0
        """
        state = QuadState(omega=np.array([1.0, 2.0, 3.0]))
        assert body_rate_penalty(state) == pytest.approx(-14.0)

    def test_single_axis_rotation(self) -> None:
        """Penalty for rotation about a single axis.

        omega = [0, 0, 5] => ||omega||^2 = 25
        penalty = -25.0
        """
        state = QuadState(omega=np.array([0.0, 0.0, 5.0]))
        assert body_rate_penalty(state) == pytest.approx(-25.0)

    def test_squared_not_linear(self) -> None:
        """Verify penalty uses squared norm, not linear norm.

        omega = [3, 4, 0]
        Squared norm: 9 + 16 = 25 => penalty = -25.0
        Linear norm would be sqrt(25) = 5.0 => penalty = -5.0
        """
        state = QuadState(omega=np.array([3.0, 4.0, 0.0]))
        assert body_rate_penalty(state) == pytest.approx(-25.0)
        # Confirm it is NOT the linear norm
        assert body_rate_penalty(state) != pytest.approx(-5.0)

    def test_always_non_positive(self) -> None:
        """Body rate penalty should always be <= 0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            omega = rng.uniform(-10, 10, size=3)
            state = QuadState(omega=omega)
            assert body_rate_penalty(state) <= 0.0


class TestGateOffsetPenalty:
    """Tests for gate_offset_penalty (simple Euclidean distance)."""

    def test_at_gate_center(self) -> None:
        """Zero penalty when drone is at gate center."""
        state = QuadState(pos=np.array([5.0, 0.0, 2.0]))
        gate = GateState(position=np.array([5.0, 0.0, 2.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(0.0)

    def test_3d_distance(self) -> None:
        """Penalty is full 3D Euclidean distance.

        Drone at [0,0,0], gate at [3,4,0] => distance = 5.0
        """
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        gate = GateState(position=np.array([3.0, 4.0, 0.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(-5.0)

    def test_z_offset(self) -> None:
        """Penalty includes vertical offset.

        Drone at [0,0,0], gate at [0,0,7] => distance = 7.0
        """
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        gate = GateState(position=np.array([0.0, 0.0, 7.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(-7.0)

    def test_full_3d(self) -> None:
        """Full 3D distance with all components nonzero.

        Drone at [1,2,3], gate at [4,6,3] => dist = sqrt(9+16+0) = 5.0
        """
        state = QuadState(pos=np.array([1.0, 2.0, 3.0]))
        gate = GateState(position=np.array([4.0, 6.0, 3.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(-5.0)

    def test_always_non_positive(self) -> None:
        """Offset penalty should always be <= 0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            pos = rng.uniform(-10, 10, size=3)
            gate_pos = rng.uniform(-10, 10, size=3)
            state = QuadState(pos=pos)
            gate = GateState(position=gate_pos)
            assert gate_offset_penalty(state, gate) <= 0.0


class TestActionSmoothness:
    """Tests for action_smoothness_penalty (thresholded L1)."""

    def test_no_prev_action_returns_zero(self) -> None:
        """First step with no previous action should return 0 penalty."""
        action = Action(values=np.array([100.0, 200.0, 300.0, 400.0]))
        assert action_smoothness_penalty(action) == pytest.approx(0.0)

    def test_same_action_zero_penalty(self) -> None:
        """Identical consecutive actions should have zero penalty."""
        action = Action(values=np.array([100.0, 100.0, 100.0, 100.0]))
        prev = Action(values=np.array([100.0, 100.0, 100.0, 100.0]))
        assert action_smoothness_penalty(action, prev) == pytest.approx(0.0)

    def test_below_threshold_zero_penalty(self) -> None:
        """Changes below threshold should incur zero penalty (dead zone).

        delta = [0.3, 0.1, 0.4, 0.2], threshold = 0.5
        All deltas < 0.5 => max(delta_i - 0.5, 0) = 0 for all i
        """
        action = Action(values=np.array([1.3, 2.1, 3.4, 4.2]))
        prev = Action(values=np.array([1.0, 2.0, 3.0, 4.0]))
        assert action_smoothness_penalty(action, prev, threshold=0.5) == pytest.approx(0.0)

    def test_above_threshold(self) -> None:
        """Changes above threshold should incur penalty for excess only.

        delta = |[2.0, 2.0, 2.0, 2.0]|, threshold = 0.5
        excess per motor = 2.0 - 0.5 = 1.5
        total = -(1.5 * 4) = -6.0
        """
        action = Action(values=np.array([3.0, 3.0, 3.0, 3.0]))
        prev = Action(values=np.array([1.0, 1.0, 1.0, 1.0]))
        assert action_smoothness_penalty(action, prev, threshold=0.5) == pytest.approx(-6.0)

    def test_mixed_per_motor(self) -> None:
        """Mixed case: some motors below threshold, some above.

        action = [1.0, 2.0, 3.0, 4.0]
        prev   = [1.0, 1.5, 1.0, 3.0]
        delta  = [0.0, 0.5, 2.0, 1.0]
        threshold = 0.5
        excess = [0, 0, 1.5, 0.5]  (0.5 - 0.5 = 0 exactly at threshold)
        total = -(0 + 0 + 1.5 + 0.5) = -2.0
        """
        action = Action(values=np.array([1.0, 2.0, 3.0, 4.0]))
        prev = Action(values=np.array([1.0, 1.5, 1.0, 3.0]))
        assert action_smoothness_penalty(action, prev, threshold=0.5) == pytest.approx(-2.0)

    def test_custom_threshold(self) -> None:
        """Custom threshold changes the dead zone.

        delta = [1.0, 1.0, 1.0, 1.0], threshold = 0.8
        excess = 0.2 each => total = -(0.2 * 4) = -0.8
        """
        action = Action(values=np.array([2.0, 2.0, 2.0, 2.0]))
        prev = Action(values=np.array([1.0, 1.0, 1.0, 1.0]))
        assert action_smoothness_penalty(action, prev, threshold=0.8) == pytest.approx(-0.8)

    def test_negative_delta_uses_absolute(self) -> None:
        """Penalty uses absolute delta (direction doesn't matter).

        action = [0.0, 0.0, 0.0, 0.0]
        prev   = [2.0, 2.0, 2.0, 2.0]
        |delta| = [2.0, 2.0, 2.0, 2.0], threshold = 0.5
        excess = 1.5 each => total = -(1.5 * 4) = -6.0
        """
        action = Action(values=np.array([0.0, 0.0, 0.0, 0.0]))
        prev = Action(values=np.array([2.0, 2.0, 2.0, 2.0]))
        assert action_smoothness_penalty(action, prev, threshold=0.5) == pytest.approx(-6.0)


class TestMonoraceReward:
    """Hand-computed test cases for the composite monorace_reward.

    Default weights (M23-aligned):
        gate_progress=1.0, gate_passage=1.5, gate_offset=1.5,
        body_rate=0.001, action_smoothness=0.0, crash_penalty=10.0

    Note: gate_passage, gate_offset, and crash_penalty are handled by the
    environment as discrete events. monorace_reward computes only
    gate_progress + body_rate + action_smoothness.
    """

    def test_case_1_stationary_zero_omega(self) -> None:
        """Case 1: Stationary drone, zero angular velocity, no prev_gate_dist.

        With default M23 weights:
        gate_progress:     1.0 * (skipped, prev_gate_dist=None)  = 0.0
        body_rate:         0.001 * (-0.0)                        = 0.0
        action_smoothness: 0.0 * (anything)                      = 0.0
        Total: 0.0
        """
        state = QuadState(
            pos=np.array([0.0, 0.0, 0.0]),
            omega=np.array([0.0, 0.0, 0.0]),
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([3.0, 4.0, 0.0]))

        result = monorace_reward(state, action, gate)
        assert result.total == pytest.approx(0.0)

    def test_case_2_with_delta_progress(self) -> None:
        """Case 2: Delta-based progress with body rate penalty.

        Drone at [1,0,0], gate at [4,0,0].
        prev_gate_dist=5.0, curr_dist=3.0, delta=2.0, v_max=30, dt=0.01
        clipped delta = min(2.0, 0.3) = 0.3

        omega = [0,0,0] => body_rate = 0

        Weights: gate_progress=1.0, body_rate=0.001, action_smoothness=0.0
        gate_progress: 1.0 * 0.3 = 0.3
        body_rate:     0.001 * 0.0 = 0.0
        Total: 0.3
        """
        state = QuadState(
            pos=np.array([1.0, 0.0, 0.0]),
            omega=np.array([0.0, 0.0, 0.0]),
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([4.0, 0.0, 0.0]))

        result = monorace_reward(
            state, action, gate,
            prev_gate_dist=5.0, v_max=30.0, dt=0.01,
        )
        assert result.total == pytest.approx(0.3)

    def test_case_3_body_rate_and_progress(self) -> None:
        """Case 3: Combined progress and body rate penalty.

        Drone at [0,0,0], gate at [0,0,5].
        prev_gate_dist=6.0, curr_dist=5.0, delta=1.0, v_max=30, dt=0.01
        clipped delta = min(1.0, 0.3) = 0.3

        omega = [10, 0, 0] => ||omega||^2 = 100

        Weights: gate_progress=2.0, body_rate=0.01, action_smoothness=0.0
        gate_progress: 2.0 * 0.3    =  0.6
        body_rate:     0.01 * (-100) = -1.0
        Total: 0.6 - 1.0 = -0.4
        """
        state = QuadState(
            pos=np.array([0.0, 0.0, 0.0]),
            omega=np.array([10.0, 0.0, 0.0]),
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([0.0, 0.0, 5.0]))
        weights = {
            "gate_progress": 2.0,
            "body_rate": 0.01,
            "action_smoothness": 0.0,
        }

        result = monorace_reward(
            state, action, gate, weights=weights,
            prev_gate_dist=6.0, v_max=30.0, dt=0.01,
        )
        assert result.total == pytest.approx(-0.4)

    def test_progress_skipped_when_prev_dist_none(self) -> None:
        """Progress reward is 0 when prev_gate_dist is None, even with weight > 0."""
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([5.0, 0.0, 0.0]))
        weights = {
            "gate_progress": 1.0,
            "body_rate": 0.0,
            "action_smoothness": 0.0,
        }

        result = monorace_reward(
            state, action, gate, weights=weights,
            prev_gate_dist=None,
        )
        assert result.total == pytest.approx(0.0)

    def test_action_smoothness_in_composite(self) -> None:
        """Action smoothness penalty integrates correctly in composite.

        action = [3.0, 3.0, 3.0, 3.0], prev = [1.0, 1.0, 1.0, 1.0]
        delta = [2.0, 2.0, 2.0, 2.0], threshold = 0.5
        excess = 1.5 each => penalty = -(1.5 * 4) = -6.0

        Weights: gate_progress=0.0, body_rate=0.0, action_smoothness=0.1
        action_smoothness: 0.1 * (-6.0) = -0.6
        Total: -0.6
        """
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        action = Action(values=np.array([3.0, 3.0, 3.0, 3.0]))
        prev_action = Action(values=np.array([1.0, 1.0, 1.0, 1.0]))
        gate = GateState(position=np.array([5.0, 0.0, 0.0]))
        weights = {
            "gate_progress": 0.0,
            "body_rate": 0.0,
            "action_smoothness": 0.1,
        }

        result = monorace_reward(
            state, action, gate, weights=weights,
            prev_action=prev_action,
            action_smoothness_threshold=0.5,
        )
        assert result.total == pytest.approx(-0.6)

    def test_first_step_no_prev_action_zero_smoothness(self) -> None:
        """First step with no prev_action should contribute 0 smoothness penalty."""
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        action = Action(values=np.array([100.0, 100.0, 100.0, 100.0]))
        gate = GateState(position=np.array([5.0, 0.0, 0.0]))
        weights = {
            "gate_progress": 0.0,
            "body_rate": 0.0,
            "action_smoothness": 1.0,  # high weight to catch any bug
        }

        result = monorace_reward(
            state, action, gate, weights=weights,
            prev_action=None,
        )
        assert result.total == pytest.approx(0.0)

    def test_custom_smoothness_threshold(self) -> None:
        """Custom action_smoothness_threshold is passed through.

        action = [2.0, 2.0, 2.0, 2.0], prev = [1.0, 1.0, 1.0, 1.0]
        delta = [1.0, 1.0, 1.0, 1.0], threshold = 0.8
        excess = 0.2 each => penalty = -(0.2 * 4) = -0.8

        Weights: action_smoothness=1.0
        action_smoothness: 1.0 * (-0.8) = -0.8
        """
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        action = Action(values=np.array([2.0, 2.0, 2.0, 2.0]))
        prev_action = Action(values=np.array([1.0, 1.0, 1.0, 1.0]))
        gate = GateState(position=np.array([5.0, 0.0, 0.0]))
        weights = {
            "gate_progress": 0.0,
            "body_rate": 0.0,
            "action_smoothness": 1.0,
        }

        result = monorace_reward(
            state, action, gate, weights=weights,
            prev_action=prev_action,
            action_smoothness_threshold=0.8,
        )
        assert result.total == pytest.approx(-0.8)

    def test_m23_defaults_all_components(self) -> None:
        """Full M23 defaults with all active components.

        Drone at [2,0,0], gate at [5,0,0].
        prev_gate_dist=4.0, curr_dist=3.0, delta=1.0, v_max=30, dt=0.01
        clipped delta = min(1.0, 0.3) = 0.3

        omega = [1, 1, 1] => ||omega||^2 = 3

        M23 defaults: gate_progress=1.0, body_rate=0.001, action_smoothness=0.0

        gate_progress: 1.0 * 0.3     = 0.3
        body_rate:     0.001 * (-3.0) = -0.003
        action_smoothness: 0.0 * x   = 0.0
        Total: 0.3 - 0.003 = 0.297
        """
        state = QuadState(
            pos=np.array([2.0, 0.0, 0.0]),
            omega=np.array([1.0, 1.0, 1.0]),
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([5.0, 0.0, 0.0]))

        result = monorace_reward(
            state, action, gate,
            prev_gate_dist=4.0, v_max=30.0, dt=0.01,
        )
        assert result.total == pytest.approx(0.297)


class TestSplineProximityReward:
    def test_on_spline_gives_max_reward(self):
        assert spline_proximity_reward(0.0) == pytest.approx(1.0)

    def test_far_from_spline_gives_low_reward(self):
        r = spline_proximity_reward(10.0)
        assert r < 0.02

    def test_moderate_distance(self):
        assert spline_proximity_reward(1.0) == pytest.approx(0.5)

    def test_negative_distance_handled(self):
        assert spline_proximity_reward(-1.0) == pytest.approx(0.5)


class TestHeadingAlignmentReward:
    def test_aligned_gives_max(self):
        assert heading_alignment_reward(0.0) == pytest.approx(1.0)

    def test_perpendicular_gives_low(self):
        r = heading_alignment_reward(np.pi / 2)
        assert 0.2 < r < 0.4

    def test_opposite_gives_lowest(self):
        r = heading_alignment_reward(np.pi)
        assert r < 0.15


class TestSpeedBonusReward:
    def test_zero_speed_gives_zero(self):
        from sim.rewards import speed_bonus_reward
        assert speed_bonus_reward(0.0, v_target=5.0) == pytest.approx(0.0)

    def test_at_target_gives_one(self):
        from sim.rewards import speed_bonus_reward
        assert speed_bonus_reward(5.0, v_target=5.0) == pytest.approx(1.0)

    def test_above_target_capped(self):
        from sim.rewards import speed_bonus_reward
        assert speed_bonus_reward(10.0, v_target=5.0) == pytest.approx(1.0)

    def test_half_speed_gives_half(self):
        from sim.rewards import speed_bonus_reward
        assert speed_bonus_reward(2.5, v_target=5.0) == pytest.approx(0.5)

    def test_negative_speed_gives_zero(self):
        from sim.rewards import speed_bonus_reward
        assert speed_bonus_reward(-1.0, v_target=5.0) == pytest.approx(0.0)

    def test_zero_target_gives_zero(self):
        from sim.rewards import speed_bonus_reward
        assert speed_bonus_reward(5.0, v_target=0.0) == pytest.approx(0.0)


class TestBoundaryPenalty:
    def test_center_of_arena_zero_penalty(self):
        from sim.rewards import boundary_penalty
        assert boundary_penalty(np.array([0.0, 0.0]), arena_bounds=10.0, margin=3.0) == pytest.approx(0.0)

    def test_at_margin_starts_penalty(self):
        from sim.rewards import boundary_penalty
        # 8m from center, 2m from wall, within 3m margin → penalty
        p = boundary_penalty(np.array([8.0, 0.0]), arena_bounds=10.0, margin=3.0)
        assert p < 0.0

    def test_at_wall_max_penalty(self):
        from sim.rewards import boundary_penalty
        p = boundary_penalty(np.array([10.0, 0.0]), arena_bounds=10.0, margin=3.0)
        assert p == pytest.approx(-1.0)

    def test_outside_wall_capped(self):
        from sim.rewards import boundary_penalty
        p = boundary_penalty(np.array([12.0, 0.0]), arena_bounds=10.0, margin=3.0)
        assert p == pytest.approx(-1.0)

    def test_y_axis_also_penalized(self):
        from sim.rewards import boundary_penalty
        p = boundary_penalty(np.array([0.0, 9.0]), arena_bounds=10.0, margin=3.0)
        assert p < 0.0


class TestGateApproachReward:
    def test_aligned_velocity_max_reward(self):
        from sim.rewards import gate_approach_reward
        r = gate_approach_reward(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert r == pytest.approx(1.0, abs=0.01)

    def test_perpendicular_velocity_zero(self):
        from sim.rewards import gate_approach_reward
        r = gate_approach_reward(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert r == pytest.approx(0.0, abs=0.01)

    def test_opposite_velocity_zero(self):
        from sim.rewards import gate_approach_reward
        r = gate_approach_reward(np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert r == pytest.approx(0.0)

    def test_zero_velocity_zero_reward(self):
        from sim.rewards import gate_approach_reward
        r = gate_approach_reward(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert r == pytest.approx(0.0)

    def test_diagonal_approach(self):
        from sim.rewards import gate_approach_reward
        r = gate_approach_reward(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        assert 0.6 < r < 0.8


class TestGateCenteringReward:
    def test_centered_at_gate_zero_penalty(self):
        from sim.rewards import gate_centering_reward
        r = gate_centering_reward(lateral_offset=0.0, dist_to_plane=0.0, gate_radius=1.5)
        assert r == pytest.approx(0.0)

    def test_off_center_at_gate_max_penalty(self):
        from sim.rewards import gate_centering_reward
        r = gate_centering_reward(lateral_offset=1.5, dist_to_plane=0.0, gate_radius=1.5)
        assert r == pytest.approx(-1.0)

    def test_off_center_far_from_gate_small_penalty(self):
        from sim.rewards import gate_centering_reward
        r = gate_centering_reward(lateral_offset=1.5, dist_to_plane=5.0, gate_radius=1.5)
        assert abs(r) < 0.1

    def test_half_offset_at_gate(self):
        from sim.rewards import gate_centering_reward
        r = gate_centering_reward(lateral_offset=0.75, dist_to_plane=0.0, gate_radius=1.5)
        assert r == pytest.approx(-0.5)

    def test_penalty_increases_as_approaching(self):
        from sim.rewards import gate_centering_reward
        far = gate_centering_reward(lateral_offset=1.0, dist_to_plane=3.0, gate_radius=1.5)
        near = gate_centering_reward(lateral_offset=1.0, dist_to_plane=0.5, gate_radius=1.5)
        assert near < far

    def test_zero_radius_returns_zero(self):
        from sim.rewards import gate_centering_reward
        r = gate_centering_reward(lateral_offset=1.0, dist_to_plane=0.0, gate_radius=0.0)
        assert r == pytest.approx(0.0)
