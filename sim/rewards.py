"""Pure reward functions for drone racing RL training.

Based on the MonoRace paper (arXiv:2601.15222) reward structure:
    r = r_prog + r_gate - p_rate - p_offset - p_delta_u - p_u - p_crash

All reward functions are pure functions that take state + weights dict, not classes.
Default weights align with the M23 configuration from the paper.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from sim.types import Action, GateState, QuadState

# Default reward weights — aligned with M23 (MonoRace paper)
class RewardResult(NamedTuple):
    """Composite reward with component breakdown.

    Attributes:
        total: Weighted sum of all components.
        components: Per-component weighted values (sign-included).
    """

    total: float
    components: dict[str, float]


DEFAULT_WEIGHTS: dict[str, float] = {
    "gate_progress": 1.0,      # M23: lambda_prog=1
    "gate_passage": 1.5,       # M23: lambda_gate=1.5
    "gate_offset": 1.5,        # M23: lambda_offset=1.5
    "body_rate": 0.001,        # Both M16/M23: lambda_rate=0.001
    "action_smoothness": 0.0,  # M23: disabled
    "crash_penalty": 10.0,     # Both M16/M23: lambda_crash=10
    "spline_proximity": 0.0,   # disabled by default (backwards compat)
    "heading_alignment": 0.0,  # disabled by default
    "speed_bonus": 0.0,        # disabled by default
    "boundary_penalty": 0.0,   # disabled by default
    "gate_approach": 0.0,      # disabled by default
    "gate_centering": 0.0,     # disabled by default
    "sideslip": 0.0,           # TRANSFER-06 lever: penalize body-y velocity^2
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


def sideslip_penalty(state: QuadState) -> float:
    """Penalize tail-first sideslip: -(body-y velocity)^2.

    TRANSFER-05/06: the live RL policy flies camera-forward (tail-first, |beta|~170)
    and at v10+ the sideslip curves it laterally off the gate aperture; the analytic
    true-frame law (vq_waypoint2) bounds sideslip and threads cleanly. Teaching the
    policy to bound body-y velocity is the missing lever. Squared to bite hard above
    a few m/s lateral.

    Args:
        state: Current quadrotor state.

    Returns:
        Negative squared body-y velocity (rad of energy in the sideslip mode).
    """
    # body velocity = R^T @ vel_world (z-y-x convention quat; use rotmat)
    q = state.quat  # [w,x,y,z]
    w, x, y, z = q[0], q[1], q[2], q[3]
    # body-y row of R^T (= column of R, index 1)
    rxy = 2.0 * (x * y - w * z)
    ryy = 1.0 - 2.0 * (x * x + z * z)
    rzy = 2.0 * (y * z + w * x)
    vby = rxy * state.vel[0] + ryy * state.vel[1] + rzy * state.vel[2]
    return -float(vby * vby)


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
) -> RewardResult:
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
        RewardResult with total and per-component breakdown.
    """
    w: dict[str, Any] = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    components: dict[str, float] = {}

    # Delta-based gate progress reward
    progress_val = 0.0
    progress_weight = w.get("gate_progress", 0.0)
    if progress_weight != 0.0 and prev_gate_dist is not None:
        curr_dist = float(np.linalg.norm(state.pos - gate_state.position))
        progress_val = progress_weight * gate_progress_reward(
            prev_gate_dist, curr_dist, v_max, dt
        )
    components["progress"] = progress_val

    # Body rate penalty
    body_rate_val = w["body_rate"] * body_rate_penalty(state)
    components["body_rate"] = body_rate_val

    # Action smoothness penalty (thresholded L1)
    action_smooth_val = w["action_smoothness"] * action_smoothness_penalty(
        action, prev_action, threshold=action_smoothness_threshold
    )
    components["action_smooth"] = action_smooth_val

    # Sideslip penalty (TRANSFER-06; weight 0 = disabled = default)
    sideslip_val = w.get("sideslip", 0.0) * sideslip_penalty(state)
    components["sideslip"] = sideslip_val

    total = progress_val + body_rate_val + action_smooth_val + sideslip_val
    return RewardResult(total=total, components=components)


def spline_proximity_reward(distance: float) -> float:
    """Reward for proximity to the racing spline.

    TraD-RL Eq. 14 adapted: r = 1 / (1 + d^2)

    Args:
        distance: Euclidean distance to nearest spline point (meters).

    Returns:
        Reward in (0, 1].
    """
    d = abs(distance)
    return 1.0 / (1.0 + d * d)


def heading_alignment_reward(yaw_error: float) -> float:
    """Reward for aligning heading with the spline tangent direction.

    CRL paper adapted: r = 1 / (1 + theta^2)

    Args:
        yaw_error: Angle between drone heading and spline tangent (radians).

    Returns:
        Reward in (0, 1].
    """
    return 1.0 / (1.0 + yaw_error * yaw_error)


def speed_bonus_reward(speed: float, v_target: float) -> float:
    """Reward for flying fast, linearly scaled up to target speed.

    r = min(max(speed, 0), v_target) / v_target

    Gives continuous gradient for acceleration up to v_target,
    then caps at 1.0 (no penalty for exceeding target).

    Args:
        speed: Current speed in m/s (scalar, magnitude of velocity).
        v_target: Target speed in m/s. Reward = 1.0 at this speed.

    Returns:
        Reward in [0, 1].
    """
    if v_target <= 0.0:
        return 0.0
    return min(max(speed, 0.0), v_target) / v_target


def boundary_penalty(
    pos_xy: NDArray[np.float64],
    arena_bounds: float,
    margin: float = 3.0,
) -> float:
    """Penalty for proximity to arena boundaries.

    Quadratic penalty that activates within `margin` meters of the arena edge.
    Returns 0 when safely inside, -1.0 at the wall.

    r = -max(0, 1 - d_wall / margin)^2

    Args:
        pos_xy: [x, y] position (world frame).
        arena_bounds: Half-width of the square arena in meters.
        margin: Distance from wall where penalty starts.

    Returns:
        Penalty in [-1, 0].
    """
    d_wall = arena_bounds - max(abs(float(pos_xy[0])), abs(float(pos_xy[1])))
    if d_wall > margin:
        return 0.0
    penetration = min(1.0, max(0.0, 1.0 - d_wall / margin))
    return -(penetration * penetration)


def spline_speed_reward(
    velocity: NDArray[np.float64],
    spline_tangent: NDArray[np.float64],
    v_max: float,
) -> float:
    """Reward for speed projected along the spline tangent direction.

    r = dot(velocity, tangent_unit) / v_max

    Positive when flying along the racing line, zero when perpendicular,
    negative when flying backwards. Naturally encourages sprinting on
    straights and braking for turns (tangent curves away from velocity).

    Args:
        velocity: [vx, vy, vz] drone velocity in world frame.
        spline_tangent: Unit tangent vector of the nearest spline point.
        v_max: Normalizing speed (m/s). Reward ~1.0 at this speed.

    Returns:
        Reward, typically in [-1, 1] but unbounded if speed > v_max.
    """
    if v_max <= 0.0:
        return 0.0
    tangent_norm = np.linalg.norm(spline_tangent)
    if tangent_norm < 1e-6:
        return 0.0
    tangent_unit = spline_tangent / tangent_norm
    return float(np.dot(velocity, tangent_unit)) / v_max


def gate_approach_reward(
    velocity: NDArray[np.float64],
    gate_normal: NDArray[np.float64],
) -> float:
    """Reward for approaching a gate with velocity aligned to its normal.

    r = max(0, cos(angle between velocity and gate_normal))

    Positive when flying through the gate (aligned), zero when perpendicular
    or flying away.

    Args:
        velocity: [vx, vy, vz] drone velocity in world frame.
        gate_normal: [nx, ny, nz] gate forward-facing normal vector.

    Returns:
        Reward in [0, 1].
    """
    speed = np.linalg.norm(velocity)
    if speed < 1e-6:
        return 0.0
    cos_angle = np.dot(velocity, gate_normal) / (speed * max(np.linalg.norm(gate_normal), 1e-6))
    return max(0.0, float(cos_angle))


def gate_centering_reward(
    lateral_offset: float,
    dist_to_plane: float,
    gate_radius: float,
) -> float:
    """Continuous centering reward that activates near gate plane.

    Penalizes lateral offset from gate center, with strength increasing
    as the drone approaches the gate plane. Inspired by Song et al. (2021).

    r = -(lateral_offset / gate_radius) * (1 / (1 + dist_to_plane^2))

    Args:
        lateral_offset: Distance from gate center perpendicular to normal (m).
        dist_to_plane: Distance to gate plane along normal (m).
        gate_radius: Gate passage radius (m). Normalizes penalty.

    Returns:
        Penalty in [-1, 0]. Zero when centered or far from gate.
    """
    if gate_radius <= 0.0:
        return 0.0
    proximity = 1.0 / (1.0 + dist_to_plane * dist_to_plane)
    return -(abs(lateral_offset) / gate_radius) * proximity
