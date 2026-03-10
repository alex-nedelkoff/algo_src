"""Tests for sim.rewards module.

Contains 3 hand-computed test cases for monorace_reward plus component tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.rewards import (
    action_smoothness_penalty,
    attitude_penalty,
    gate_progress_reward,
    monorace_reward,
)
from sim.types import Action, GateState, QuadState


class TestGateProgressReward:
    """Tests for gate_progress_reward."""

    def test_at_gate(self) -> None:
        """Reward should be 0 when drone is at the gate."""
        state = QuadState(pos=np.array([5.0, 0.0, 2.0]))
        gate = GateState(position=np.array([5.0, 0.0, 2.0]))
        assert gate_progress_reward(state, gate) == pytest.approx(0.0)

    def test_distance_penalty(self) -> None:
        """Reward should be -distance when away from gate."""
        state = QuadState(pos=np.array([0.0, 0.0, 0.0]))
        gate = GateState(position=np.array([3.0, 4.0, 0.0]))
        # Distance = sqrt(9 + 16) = 5.0
        assert gate_progress_reward(state, gate) == pytest.approx(-5.0)


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

    Using default weights: gate_progress=1.0, attitude_penalty=0.1,
    speed_bonus=0.05, action_smoothness=0.01
    """

    def test_case_1_origin_stationary_upright(self) -> None:
        """Case 1: Drone at origin, stationary, upright, zero action, gate at [3,4,0].

        gate_progress:    1.0 * (-5.0)  = -5.0
        attitude_penalty: 0.1 * (0.0)   =  0.0   (w=1, upright)
        speed_bonus:      0.05 * (0.0)  =  0.0   (stationary)
        action_smooth:    0.01 * (0.0)  =  0.0   (zero action)
        Total: -5.0
        """
        state = QuadState(
            pos=np.array([0.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        action = Action(values=np.zeros(4))
        gate = GateState(position=np.array([3.0, 4.0, 0.0]))

        reward = monorace_reward(state, action, gate)
        assert reward == pytest.approx(-5.0)

    def test_case_2_moving_toward_gate(self) -> None:
        """Case 2: Drone at [1,0,0], vel=[2,0,0], upright, action=[100]*4, gate at [4,0,0].

        gate_progress:    1.0 * (-3.0)     = -3.0     (dist=3)
        attitude_penalty: 0.1 * (0.0)      =  0.0     (w=1)
        speed_bonus:      0.05 * (2.0)     =  0.1     (speed=2)
        action_smooth:    0.01 * (-40000)  = -400.0   (sum(100^2 * 4)=40000)
        Total: -3.0 + 0.0 + 0.1 + (-400.0) = -402.9
        """
        state = QuadState(
            pos=np.array([1.0, 0.0, 0.0]),
            vel=np.array([2.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        action = Action(values=np.array([100.0, 100.0, 100.0, 100.0]))
        gate = GateState(position=np.array([4.0, 0.0, 0.0]))

        reward = monorace_reward(state, action, gate)
        assert reward == pytest.approx(-402.9)

    def test_case_3_tilted_with_custom_weights(self) -> None:
        """Case 3: Tilted drone, custom weights.

        State: pos=[0,0,0], vel=[1,0,0], quat=[sqrt(0.5), sqrt(0.5), 0, 0]
        Action: [0,0,0,0], Gate at [0,0,5]

        w_quat = sqrt(0.5), so w^2 = 0.5
        Weights: gate_progress=2.0, attitude_penalty=1.0, speed_bonus=0.0, action_smoothness=0.0

        gate_progress:    2.0 * (-5.0)   = -10.0  (dist to [0,0,5] = 5)
        attitude_penalty: 1.0 * (-0.5)   = -0.5   (1 - 0.5 = 0.5)
        speed_bonus:      0.0 * (1.0)    =  0.0
        action_smooth:    0.0 * (0.0)    =  0.0
        Total: -10.5
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
            "attitude_penalty": 1.0,
            "speed_bonus": 0.0,
            "action_smoothness": 0.0,
        }

        reward = monorace_reward(state, action, gate, weights=weights)
        assert reward == pytest.approx(-10.5)


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
