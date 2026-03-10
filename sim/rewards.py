"""Pure reward functions for drone racing RL training.

Based on the MonoRace paper (arXiv:2601.15222) reward structure:
    r = r_prog + r_gate - p_rate - p_offset - p_delta_u - p_u - p_crash

All reward functions are pure functions that take state + weights dict, not classes.
Default weights align with the M23 configuration from the paper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.types import Action, GateState, QuadState

# Default reward weights — aligned with M23 (MonoRace paper)
DEFAULT_WEIGHTS: dict[str, float] = {
    "gate_progress": 1.0,      # M23: lambda_prog=1
    "gate_passage": 1.5,       # M23: lambda_gate=1.5
    "gate_offset": 1.5,        # M23: lambda_offset=1.5
    "body_rate": 0.001,        # Both M16/M23: lambda_rate=0.001
    "action_smoothness": 0.0,  # M23: disabled
    "crash_penalty": 10.0,     # Both M16/M23: lambda_crash=10
}


def gate_progress_reward(
    prev_dist: float,
    curr_dist: float,
    v_max: float,
    dt: float,
) -> float:
    """Delta-based progress reward toward the next gate.

    Rewards the drone for getting closer to the gate compared to the previous
    timestep, clipped by the maximum possible displacement (v_max * dt) to
    prevent artificial bonuses from teleportation or gate switching.

    Follows MonoRace paper: r_prog = min(||p_{k-1} - p_g|| - ||p_k - p_g||, v_max*dt)

    Args:
        prev_dist: Distance to gate at previous timestep.
        curr_dist: Distance to gate at current timestep.
        v_max: Maximum expected velocity for clipping (m/s).
        dt: Simulation timestep (s).

    Returns:
        Clipped delta distance (positive = getting closer).
    """
    delta = prev_dist - curr_dist  # positive when getting closer
    return min(delta, v_max * dt)


def body_rate_penalty(state: QuadState) -> float:
    """Penalize high angular velocities (body rates).

    Paper: p_rate = lambda_rate * ||Omega||^2

    Args:
        state: Current quadrotor state.

    Returns:
        Negative squared L2 norm of angular velocity.
    """
    omega = state.omega  # angular velocity vector (3,)
    return -float(np.dot(omega, omega))  # -||Omega||^2


def gate_offset_penalty(state: QuadState, gate_state: GateState) -> float:
    """Penalize distance from gate center at moment of gate passage.

    Paper: p_offset = lambda_offset * ||p - p_gate||
    Applied ONLY at gate passage (env handles the timing), not continuously.

    Args:
        state: Current quadrotor state.
        gate_state: Target gate state.

    Returns:
        Negative Euclidean distance from gate center.
    """
    return -float(np.linalg.norm(state.pos - gate_state.position))


def action_smoothness_penalty(
    action: Action, prev_action: Action | None = None, threshold: float = 0.5
) -> float:
    """Thresholded L1 action smoothness penalty.

    Paper: p_delta_u = SUM max(|delta_u_i| - threshold, 0)
    Dead-zone: small corrections below threshold incur zero penalty.

    Args:
        action: Current action.
        prev_action: Previous action, if available.
        threshold: Dead-zone threshold per motor channel.

    Returns:
        Negative penalty value (0 on first step with no prev_action).
    """
    if prev_action is not None:
        delta = np.abs(action.values - prev_action.values)
        return -float(np.sum(np.maximum(delta - threshold, 0.0)))
    return 0.0  # no penalty on first step (no previous action to compare)


def monorace_reward(
    state: QuadState,
    action: Action,
    gate_state: GateState,
    weights: dict[str, float] | None = None,
    prev_action: Action | None = None,
    prev_gate_dist: float | None = None,
    v_max: float = 30.0,
    dt: float = 0.01,
    action_smoothness_threshold: float = 0.5,
) -> float:
    """Composite reward function for single-drone racing.

    Based on the MonoRace paper (arXiv:2601.15222):
        r = r_prog - p_rate - p_delta_u

    Gate passage reward (r_gate) and gate offset penalty (p_offset) are applied
    as discrete events in the environment, not here.

    Args:
        state: Current quadrotor state.
        action: Action taken.
        gate_state: Current target gate state.
        weights: Weight dictionary. Defaults to DEFAULT_WEIGHTS.
        prev_action: Previous action for smoothness computation.
        prev_gate_dist: Distance to gate at previous timestep. Required when
            gate_progress weight > 0. If None, progress reward is skipped.
        v_max: Maximum velocity for delta-progress clipping (m/s).
        dt: Simulation timestep (s).
        action_smoothness_threshold: Dead-zone threshold for action smoothness.

    Returns:
        Weighted sum of reward components.
    """
    w: dict[str, Any] = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    reward = 0.0

    # Delta-based gate progress reward
    progress_weight = w.get("gate_progress", 0.0)
    if progress_weight != 0.0 and prev_gate_dist is not None:
        curr_dist = float(np.linalg.norm(state.pos - gate_state.position))
        reward += progress_weight * gate_progress_reward(
            prev_gate_dist, curr_dist, v_max, dt
        )

    # Body rate penalty
    reward += w["body_rate"] * body_rate_penalty(state)

    # Action smoothness penalty (thresholded L1)
    reward += w["action_smoothness"] * action_smoothness_penalty(
        action, prev_action, threshold=action_smoothness_threshold
    )

    return reward
