"""Pure reward functions for drone racing RL training.

All reward functions are pure functions that take state + weights dict, not classes.

Reward structure aligned with MonoRace paper (MAVLab, arXiv:2601.15222):
  r = r_prog + r_gate - p_offset - p_attitude - p_smoothness + r_speed

M16 (fastest, 28.2 m/s): lambda_gate=30, lambda_prog=0, lambda_offset=2
M23 (safest, 88.4%):      lambda_gate=1.5, lambda_prog=1, lambda_offset=1.5
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.types import Action, GateState, QuadState

# Default reward weights (aligned with M16 fastest policy)
DEFAULT_WEIGHTS: dict[str, float] = {
    "gate_progress": 0.0,
    "gate_offset": 2.0,
    "attitude_penalty": 0.1,
    "speed_bonus": 0.05,
    "action_smoothness": 0.01,
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


def gate_offset_penalty(state: QuadState, gate_state: GateState) -> float:
    """Penalize off-center position relative to a gate.

    Projects the drone position onto the gate plane and computes the lateral
    offset from the gate center. The gate normal is derived from the gate
    orientation quaternion (forward = rotated x-axis).

    Follows MonoRace paper: p_offset = lambda_offset * ||p - p_gate||
    Applied as a continuous penalty (not only at passage) so the agent
    learns to approach gates centered.

    Args:
        state: Current quadrotor state.
        gate_state: Target gate state (position + orientation).

    Returns:
        Negative lateral offset magnitude (always <= 0).
    """
    # Vector from gate center to drone
    rel_pos = state.pos - gate_state.position

    # Gate normal from orientation quaternion (gate faces along its local x-axis)
    # Rotate [1, 0, 0] by gate quaternion to get the gate normal direction
    q = gate_state.orientation  # [w, x, y, z]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Quaternion rotation of [1, 0, 0]:
    # R * [1, 0, 0] = [1 - 2(qy^2 + qz^2), 2(qx*qy + qw*qz), 2(qx*qz - qw*qy)]
    normal = np.array([
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qw * qz),
        2.0 * (qx * qz - qw * qy),
    ])

    # Project rel_pos onto the gate plane by removing the normal component
    along_normal = np.dot(rel_pos, normal)
    lateral = rel_pos - along_normal * normal

    # Return negative offset magnitude
    return -float(np.linalg.norm(lateral))


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
    prev_gate_dist: float | None = None,
    v_max: float = 30.0,
    dt: float = 0.01,
) -> float:
    """Composite reward function for single-drone racing.

    Combines delta-based gate progress, gate offset penalty, attitude penalty,
    speed bonus, and action smoothness into a single scalar reward using
    configurable weights. Aligned with MonoRace paper (arXiv:2601.15222).

    Args:
        state: Current quadrotor state.
        action: Action taken.
        gate_state: Current target gate state.
        weights: Weight dictionary. Keys: gate_progress, gate_offset,
            attitude_penalty, speed_bonus, action_smoothness.
            Defaults to DEFAULT_WEIGHTS.
        prev_action: Previous action for smoothness computation.
        prev_gate_dist: Distance to gate at previous timestep. Required when
            gate_progress weight > 0. If None, progress reward is skipped.
        v_max: Maximum velocity for delta-progress clipping (m/s).
        dt: Simulation timestep (s).

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

    # Gate offset penalty
    offset_weight = w.get("gate_offset", 0.0)
    if offset_weight != 0.0:
        reward += offset_weight * gate_offset_penalty(state, gate_state)

    reward += w["attitude_penalty"] * attitude_penalty(state)
    reward += w["speed_bonus"] * speed_bonus(state)
    reward += w["action_smoothness"] * action_smoothness_penalty(action, prev_action)

    return reward
