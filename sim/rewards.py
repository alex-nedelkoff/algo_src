"""Pure reward functions for drone racing RL training.

All reward functions are pure functions that take state + weights dict, not classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.types import Action, GateState, QuadState

# Default reward weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "gate_progress": 1.0,
    "attitude_penalty": 0.1,
    "speed_bonus": 0.05,
    "action_smoothness": 0.01,
}


def gate_progress_reward(state: QuadState, gate_state: GateState) -> float:
    """Reward for making progress toward the next gate.

    Computed as the negative distance to the gate center. Closer is better.

    Args:
        state: Current quadrotor state.
        gate_state: Target gate state.

    Returns:
        Negative Euclidean distance to gate center.
    """
    distance = float(np.linalg.norm(state.pos - gate_state.position))
    return -distance


def attitude_penalty(state: QuadState) -> float:
    """Penalty for deviation from upright orientation.

    Uses the quaternion w component (scalar part) to measure deviation
    from identity rotation. Penalty is 0 when perfectly upright (w=1),
    and increases as the drone tilts.

    Args:
        state: Current quadrotor state.

    Returns:
        Negative penalty value (0 when upright, negative otherwise).
    """
    # w=1 means identity quaternion (upright). Penalty = -(1 - w^2)
    w = state.quat[0]
    return -(1.0 - w * w)


def speed_bonus(state: QuadState) -> float:
    """Bonus for maintaining forward speed.

    Args:
        state: Current quadrotor state.

    Returns:
        Speed magnitude (always non-negative).
    """
    return float(np.linalg.norm(state.vel))


def action_smoothness_penalty(action: Action, prev_action: Action | None = None) -> float:
    """Penalty for large or jerky actions.

    If prev_action is provided, penalizes the change between consecutive actions.
    Otherwise, penalizes the magnitude of the current action.

    Args:
        action: Current action.
        prev_action: Previous action, if available.

    Returns:
        Negative penalty value.
    """
    if prev_action is not None:
        delta = action.values - prev_action.values
        return -float(np.sum(delta**2))
    return -float(np.sum(action.values**2))


def monorace_reward(
    state: QuadState,
    action: Action,
    gate_state: GateState,
    weights: dict[str, float] | None = None,
    prev_action: Action | None = None,
) -> float:
    """Composite reward function for single-drone racing.

    Combines gate progress, attitude penalty, speed bonus, and action smoothness
    into a single scalar reward using configurable weights.

    Args:
        state: Current quadrotor state.
        action: Action taken.
        gate_state: Current target gate state.
        weights: Weight dictionary. Keys: gate_progress, attitude_penalty,
            speed_bonus, action_smoothness. Defaults to DEFAULT_WEIGHTS.
        prev_action: Previous action for smoothness computation.

    Returns:
        Weighted sum of reward components.
    """
    w: dict[str, Any] = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    reward = 0.0
    reward += w["gate_progress"] * gate_progress_reward(state, gate_state)
    reward += w["attitude_penalty"] * attitude_penalty(state)
    reward += w["speed_bonus"] * speed_bonus(state)
    reward += w["action_smoothness"] * action_smoothness_penalty(action, prev_action)

    return reward
