"""Tests for sim.rewards module.

Contains hand-computed test cases for monorace_reward, delta-based progress
reward, gate offset penalty, and other component tests.

Aligned with MonoRace paper reward structure (COR-44).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.rewards import (
    action_smoothness_penalty,
    attitude_penalty,
    gate_offset_penalty,
    gate_progress_reward,
    monorace_reward,
    speed_bonus,
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


class TestGateOffsetPenalty:
    """Tests for gate_offset_penalty."""

    def test_at_gate_center(self) -> None:
        """Zero penalty when drone is at gate center."""
        state = QuadState(pos=np.array([5.0, 0.0, 2.0]))
        gate = GateState(position=np.array([5.0, 0.0, 2.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(0.0)

    def test_on_gate_normal_axis(self) -> None:
        """Zero lateral offset when drone is directly in front of gate.

        Gate at origin with default orientation (identity quat, normal=[1,0,0]).
        Drone at [3,0,0] is directly along the normal, so lateral offset = 0.
        """
        state = QuadState(pos=np.array([3.0, 0.0, 0.0]))
        gate = GateState(position=np.array([0.0, 0.0, 0.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(0.0)

    def test_lateral_offset_y(self) -> None:
        """Penalty for lateral offset in y direction.

        Gate at origin with identity orientation (normal = [1,0,0]).
        Drone at [0, 2, 0] -> rel_pos = [0, 2, 0].
        Along normal = 0, lateral = [0, 2, 0], offset = 2.0.
        """
        state = QuadState(pos=np.array([0.0, 2.0, 0.0]))
        gate = GateState(position=np.array([0.0, 0.0, 0.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(-2.0)

    def test_lateral_offset_z(self) -> None:
        """Penalty for lateral offset in z direction.

        Gate at origin with identity orientation (normal = [1,0,0]).
        Drone at [0, 0, 3] -> lateral = [0, 0, 3], offset = 3.0.
        """
        state = QuadState(pos=np.array([0.0, 0.0, 3.0]))
        gate = GateState(position=np.array([0.0, 0.0, 0.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(-3.0)

    def test_mixed_along_normal_and_lateral(self) -> None:
        """Offset ignores the component along the gate normal.

        Gate at origin, normal = [1, 0, 0].
        Drone at [5, 3, 4] -> along_normal = 5, lateral = [0, 3, 4].
        Lateral offset = sqrt(9 + 16) = 5.0.
        """
        state = QuadState(pos=np.array([5.0, 3.0, 4.0]))
        gate = GateState(position=np.array([0.0, 0.0, 0.0]))
        assert gate_offset_penalty(state, gate) == pytest.approx(-5.0)

    def test_rotated_gate(self) -> None:
        """Offset with rotated gate normal.

        Gate with 90-degree yaw rotation around z-axis.
        Quaternion for 90deg about z: [cos(45deg), 0, 0, sin(45deg)]
        = [sqrt(0.5), 0, 0, sqrt(0.5)].
        This rotates [1,0,0] to [0,1,0], so gate normal = [0,1,0].

        Drone at [3, 0, 0] relative to gate at origin:
        along_normal = dot([3,0,0], [0,1,0]) = 0
        lateral = [3, 0, 0], offset = 3.0
        """
        q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        state = QuadState(pos=np.array([3.0, 0.0, 0.0]))
        gate = GateState(position=np.array([0.0, 0.0, 0.0]), orientation=q)
        assert gate_offset_penalty(state, gate) == pytest.approx(-3.0)

    def test_always_non_positive(self) -> None:
        """Offset penalty should always be <= 0."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            pos = rng.uniform(-10, 10, size=3)
            gate_pos = rng.uniform(-10, 10, size=3)
            state = QuadState(pos=pos)
            gate = GateState(position=gate_pos)
            assert gate_offset_penalty(state, gate) <= 0.0


class TestAttitudePenalty:
    """Tests for attitude_penalty."""

    def test_upright(self) -> None:
        """No penalty when perfectly upright (identity quaternion)."""
        state = QuadState(quat=np.array([1.0, 0.0, 0.0, 0.0]))
        assert attitude_penalty(state) == pytest.approx(0.0)

    def test_tilted(self) -> None:
        """Penalty when tilted. q=[0, 1, 0, 0] => w=0, penalty=-(1-0)=-1."""
        state = QuadState(quat=np.array([0.0, 1.0, 0.0, 0.0]))
        assert attitude_penalty(state) == pytest.approx(-1.0)


class TestMonoraceReward:
    """Hand-computed test cases for the composite monorace_reward.

    Default weights (M16-aligned): gate_progress=0.0, gate_offset=2.0,
    attitude_penalty=0.1, speed_bonus=0.05, action_smoothness=0.01
    """

    def test_case_1_m16_no_progress(self) -> None:
        """Case 1: M16 defaults — no progress reward, offset penalty active.

        Drone at origin, stationary, upright, zero action, gate at [3,4,0].
        gate_progress:    0.0 (weight=0, skipped)
        gate_offset:      2.0 * (-4.0)   = -8.0  (gate normal along x; lateral = y-component = 4.0)
        attitude_penalty: 0.1 * (0.0)    =   0.0
        speed_bonus:      0.05 * (0.0)   =   0.0
        action_smooth:    0.01 * (0.0)   =   0.0
        Total: -8.0
        """
        state = QuadState(
            pos=np.array([0.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([3.0, 4.0, 0.0]))

        reward = monorace_reward(state, action, gate)
        assert reward == pytest.approx(-8.0)

    def test_case_2_with_delta_progress(self) -> None:
        """Case 2: Delta-based progress with custom weights (M23-style).

        Drone at [1,0,0], gate at [4,0,0].
        prev_gate_dist=5.0, curr_dist=3.0, delta=2.0, v_max=30, dt=0.01
        clipped delta = min(2.0, 0.3) = 0.3

        Weights: gate_progress=1.0, gate_offset=0.0, attitude_penalty=0.0,
                 speed_bonus=0.0, action_smoothness=0.0

        gate_progress: 1.0 * 0.3 = 0.3
        Total: 0.3
        """
        state = QuadState(
            pos=np.array([1.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([4.0, 0.0, 0.0]))
        weights = {
            "gate_progress": 1.0,
            "gate_offset": 0.0,
            "attitude_penalty": 0.0,
            "speed_bonus": 0.0,
            "action_smoothness": 0.0,
        }

        reward = monorace_reward(
            state, action, gate, weights=weights,
            prev_gate_dist=5.0, v_max=30.0, dt=0.01,
        )
        assert reward == pytest.approx(0.3)

    def test_case_3_tilted_with_offset_and_progress(self) -> None:
        """Case 3: Tilted drone with both progress and offset.

        State: pos=[0,0,0], vel=[1,0,0], quat=[sqrt(0.5), sqrt(0.5), 0, 0]
        Action: [0,0,0,0], Gate at [0,0,5] (identity orientation, normal=[1,0,0])

        Offset: rel_pos = [0,0,-5]. Gate normal = [1,0,0].
          along_normal = 0. lateral = [0,0,-5]. offset = 5.0.

        prev_gate_dist=6.0, curr_dist=5.0, delta=1.0, v_max=30, dt=0.01
        clipped delta = min(1.0, 0.3) = 0.3

        w_quat = sqrt(0.5), so w^2 = 0.5

        Weights: gate_progress=2.0, gate_offset=1.0, attitude_penalty=1.0,
                 speed_bonus=0.0, action_smoothness=0.0

        gate_progress:    2.0 * 0.3    =  0.6
        gate_offset:      1.0 * (-5.0) = -5.0
        attitude_penalty: 1.0 * (-0.5) = -0.5
        Total: 0.6 - 5.0 - 0.5 = -4.9
        """
        q = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
        state = QuadState(
            pos=np.array([0.0, 0.0, 0.0]),
            vel=np.array([1.0, 0.0, 0.0]),
            quat=q,
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([0.0, 0.0, 5.0]))
        weights = {
            "gate_progress": 2.0,
            "gate_offset": 1.0,
            "attitude_penalty": 1.0,
            "speed_bonus": 0.0,
            "action_smoothness": 0.0,
        }

        reward = monorace_reward(
            state, action, gate, weights=weights,
            prev_gate_dist=6.0, v_max=30.0, dt=0.01,
        )
        assert reward == pytest.approx(-4.9)

    def test_progress_skipped_when_prev_dist_none(self) -> None:
        """Progress reward is 0 when prev_gate_dist is None, even with weight > 0."""
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([5.0, 0.0, 0.0]))
        weights = {
            "gate_progress": 1.0,
            "gate_offset": 0.0,
            "attitude_penalty": 0.0,
            "speed_bonus": 0.0,
            "action_smoothness": 0.0,
        }

        reward = monorace_reward(
            state, action, gate, weights=weights,
            prev_gate_dist=None,
        )
        assert reward == pytest.approx(0.0)

    def test_m16_defaults_zero_progress(self) -> None:
        """M16 default weights have gate_progress=0, so prev_gate_dist is irrelevant."""
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([0.0, 0.0, 0.0]))

        # With and without prev_gate_dist should be the same
        r1 = monorace_reward(state, action, gate, prev_gate_dist=10.0)
        r2 = monorace_reward(state, action, gate, prev_gate_dist=None)
        assert r1 == pytest.approx(r2)


class TestActionSmoothness:
    """Tests for action_smoothness_penalty."""

    def test_zero_action(self) -> None:
        """Zero action should have zero penalty."""
        action = Action(values=np.zeros(4))
        assert action_smoothness_penalty(action) == pytest.approx(0.0)

    def test_with_prev_action(self) -> None:
        """Penalty with prev_action should measure delta."""
        action = Action(values=np.array([100.0, 100.0, 100.0, 100.0]))
        prev = Action(values=np.array([100.0, 100.0, 100.0, 100.0]))
        # Same action => zero delta => zero penalty
        assert action_smoothness_penalty(action, prev) == pytest.approx(0.0)
