"""Gate racing Gymnasium environment.

Wraps NumpyQuadDynamics with a Track from sim.tracks to provide a
Gymnasium-compatible RL training environment for single-drone racing.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from sim.rewards import monorace_reward

try:
    from sim.rewards import gate_offset_penalty
except ImportError:  # pragma: no cover – available once rewards.py is updated
    def gate_offset_penalty(state, gate_state):  # type: ignore[misc]
        """Fallback: negative Euclidean distance to gate center."""
        return -float(np.linalg.norm(state.pos - gate_state.position))
from sim.tracks import Track
from sim.types import Action, GateState, QuadState
from sim.dynamics.numpy_quad import (
    GRAVITY,
    MOTOR,
    OMEGA,
    POS,
    QUAT,
    VEL,
    NumpyQuadDynamics,
)
from sim.dynamics.params import VehicleParams


# Observation dimension breakdown (MonoRace paper spec):
#   gate_rel_pos(3) + vel(3) + roll_pitch(2) + yaw_rel(1) +
#   body_rates(3) + motor_speeds(4) + next_gate_rel_pos(3) +
#   next_gate_yaw_rel(1) + prev_action(4) = 24
OBS_DIM = 24

# Default termination thresholds
DEFAULT_CEILING = 10.0
DEFAULT_MAX_STEPS = 1200


def _gate_normal(gate_state: GateState) -> NDArray[np.float64]:
    """Compute the forward-facing normal vector of a gate.

    The gate's forward direction is its local x-axis, rotated by the gate's
    orientation quaternion.

    Args:
        gate_state: Gate with orientation quaternion [w, x, y, z].

    Returns:
        Unit normal vector (3,) in world frame.
    """
    q = gate_state.orientation  # [w, x, y, z]
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Quaternion rotation of [1, 0, 0]:
    # R * [1, 0, 0] = [1 - 2(qy^2 + qz^2), 2(qx*qy + qw*qz), 2(qx*qz - qw*qy)]
    normal = np.array([
        1.0 - 2.0 * (qy * qy + qz * qz),
        2.0 * (qx * qy + qw * qz),
        2.0 * (qx * qz - qw * qy),
    ])
    return normal


def _quat_to_yaw(q: NDArray[np.float64]) -> float:
    """Extract yaw angle from a quaternion [w, x, y, z].

    Uses standard ZYX Euler extraction for the yaw component.

    Args:
        q: Quaternion (4,) in [w, x, y, z] convention.

    Returns:
        Yaw angle in radians.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def _quat_to_euler(q: NDArray[np.float64]) -> tuple[float, float, float]:
    """Extract roll, pitch, yaw from a quaternion [w, x, y, z].

    Uses standard ZYX Euler convention.

    Args:
        q: Quaternion (4,) in [w, x, y, z] convention.

    Returns:
        Tuple of (roll, pitch, yaw) in radians.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = float(np.arcsin(sinp))
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Convert ZYX Euler angles to quaternion [w, x, y, z].

    Inverse of ``_quat_to_euler``.

    Args:
        roll: Roll angle in radians.
        pitch: Pitch angle in radians.
        yaw: Yaw angle in radians.

    Returns:
        Quaternion (4,) in [w, x, y, z] convention.
    """
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _wrap_angle(angle: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Wrap angle(s) to [-pi, pi].

    Args:
        angle: Angle or array of angles in radians.

    Returns:
        Wrapped angle(s).
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rotate_xy(
    xy: NDArray[np.float64],
    cos_yaw: float,
    sin_yaw: float,
) -> NDArray[np.float64]:
    """Rotate 2D vector(s) by a gate yaw angle.

    Applies a 2D rotation matrix [[cos, sin], [-sin, cos]] to transform
    world-frame XY offsets into gate-yaw-relative frame.

    Args:
        xy: Shape (..., 2) array of XY coordinates.
        cos_yaw: Cosine of the gate yaw angle.
        sin_yaw: Sine of the gate yaw angle.

    Returns:
        Rotated XY array with same shape as input.
    """
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([cos_yaw * x + sin_yaw * y,
                     -sin_yaw * x + cos_yaw * y], axis=-1)


class GateRaceEnv(gym.Env):
    """Gymnasium environment for quadrotor gate racing.

    Observation (24-dim, MonoRace paper spec):
        [0:3]   position to current gate (gate-yaw-relative frame)
        [3:6]   velocity (gate-yaw-relative frame)
        [6:8]   roll, pitch (world frame Euler angles)
        [8]     yaw relative to gate (drone_yaw - gate_yaw, wrapped [-pi, pi])
        [9:12]  body angular rates p, q, r (body frame)
        [12:16] motor speeds (normalized to [-1, 1]: (w / w_max) * 2 - 1)
        [16:19] position from current gate to next gate (gate-yaw-relative frame)
        [19]    relative yaw to next gate (next_yaw - cur_yaw, wrapped [-pi, pi])
        [20:24] previous action / motor commands (normalized [-1, 1])

    Action (4-dim):
        Normalized motor commands in [-1, 1], mapped to motor speeds via a
        nonlinear ESC (Electronic Speed Controller) curve per MonoRace paper.

    Supports vectorized operation with n_envs parallel episodes.

    Gate passage uses plane-crossing detection: the drone must cross the gate
    plane from the front (negative-normal side) to the back (positive-normal
    side) while within ``gate_passage_radius`` of the gate center laterally.

    Args:
        track: Track object defining the gate sequence.
        params: Vehicle parameters. Uses defaults if None.
        n_envs: Number of parallel environments.
        dt: Simulation timestep in seconds.
        max_steps: Maximum steps per episode before timeout.
        ceiling: Maximum altitude (z) before crash.
        reward_weights: Weights for the monorace_reward function.
        gate_passage_radius: Lateral distance threshold for gate passage detection.
        v_max: Maximum velocity for delta-progress reward clipping (m/s).
        action_smoothness_threshold: Threshold for action smoothness penalty.
        esc_nonlinearity: ESC curve parameter k in [0, 1]. k=0 gives sqrt
            response, k=1 gives linear. Default 0.5 per MonoRace paper.
        omega_min: Minimum motor speed in rad/s (idle speed). Default 0.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track: Track | None = None,
        params: VehicleParams | None = None,
        n_envs: int = 1,
        dt: float = 0.01,
        max_steps: int = DEFAULT_MAX_STEPS,
        ceiling: float = DEFAULT_CEILING,
        reward_weights: dict[str, float] | None = None,
        gate_passage_radius: float = 1.0,
        v_max: float = 30.0,
        action_smoothness_threshold: float = 0.5,
        esc_nonlinearity: float = 0.5,
        omega_min: float = 0.0,
        random_gate_start: bool = False,
        start_behind_dist: float = 1.0,
        start_vel_std: float = 0.5,
        start_att_std: float = 0.1,
        gate_collision: bool = False,
    ) -> None:
        super().__init__()

        self.n_envs = n_envs
        self.dt = dt
        self.max_steps = max_steps
        self.ceiling = ceiling
        self.reward_weights = reward_weights
        self.gate_passage_radius = gate_passage_radius
        self.v_max = v_max
        self.action_smoothness_threshold = action_smoothness_threshold
        self.esc_nonlinearity = esc_nonlinearity
        self.omega_min = omega_min
        self.random_gate_start = random_gate_start
        self.start_behind_dist = start_behind_dist
        self.start_vel_std = start_vel_std
        self.start_att_std = start_att_std
        self.gate_collision = gate_collision

        # Default track: simple 3-gate circuit
        if track is None:
            track = Track([
                GateState(position=np.array([5.0, 0.0, 2.0])),
                GateState(position=np.array([10.0, 5.0, 2.0])),
                GateState(position=np.array([5.0, 10.0, 2.0])),
            ])
        self.track = track

        # Dynamics
        self.params = params or VehicleParams()
        self.dynamics = NumpyQuadDynamics(params=self.params, dt=dt)

        # Gymnasium spaces — normalized action space [-1, 1] per MonoRace paper
        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self._states: NDArray[np.float64] = np.zeros((n_envs, 17))
        self._step_counts = np.zeros(n_envs, dtype=np.int64)
        self._gate_indices = np.zeros(n_envs, dtype=np.int64)
        self._prev_gate_dists = np.zeros(n_envs, dtype=np.float64)
        self._prev_along_normal = np.zeros(n_envs, dtype=np.float64)
        self._prev_actions: NDArray[np.float64] = np.zeros(
            (n_envs, 4), dtype=np.float64
        )

    def _randomize_start(self, env_indices: NDArray[np.intp]) -> None:
        """Place envs at random gates with perturbation.

        For each env in ``env_indices``, picks a random gate, places the drone
        ``start_behind_dist`` meters behind the gate along its negative normal,
        adds small velocity and attitude perturbations, and sets motors to hover.

        Args:
            env_indices: Array of env indices to randomize.
        """
        rng = self.np_random
        hover_omega = np.sqrt(
            self.params.mass * GRAVITY / (4.0 * self.params.k_thrust)
        )

        for idx in env_indices:
            # Pick a random gate
            gate_idx = int(rng.integers(0, self.track.num_gates))
            self._gate_indices[idx] = gate_idx
            gate = self.track.gates[gate_idx]
            normal = _gate_normal(gate)

            # Position: behind gate along negative normal + small lateral noise
            self._states[idx, POS] = (
                gate.position
                - self.start_behind_dist * normal
                + rng.normal(0, 0.1, size=3)
            )

            # Velocity perturbation
            self._states[idx, VEL] = rng.normal(0, self.start_vel_std, size=3)

            # Attitude: face the gate (yaw from gate normal) + perturbation
            gate_yaw = float(np.arctan2(normal[1], normal[0]))
            roll = rng.normal(0, self.start_att_std)
            pitch = rng.normal(0, self.start_att_std)
            yaw = gate_yaw + rng.normal(0, self.start_att_std)
            self._states[idx, QUAT] = _euler_to_quat(roll, pitch, yaw)

            # Zero angular rates, set motors to hover
            self._states[idx, OMEGA] = 0.0
            self._states[idx, MOTOR] = hover_omega

    def _update_gate_tracking(self, env_indices: NDArray[np.intp]) -> None:
        """Recompute prev_gate_dists and prev_along_normal for given envs.

        Args:
            env_indices: Array of env indices to update.
        """
        for idx in env_indices:
            gate_idx = int(self._gate_indices[idx])
            gate = self.track.gates[gate_idx % self.track.num_gates]
            rel_pos = self._states[idx, POS] - gate.position
            self._prev_gate_dists[idx] = float(np.linalg.norm(rel_pos))
            normal = _gate_normal(gate)
            self._prev_along_normal[idx] = float(np.dot(rel_pos, normal))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset all environments.

        Args:
            seed: Random seed.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        self._states = self.dynamics.reset(self.n_envs)
        self._step_counts[:] = 0
        self._gate_indices[:] = 0
        self._prev_actions = np.zeros((self.n_envs, 4), dtype=np.float64)

        all_indices = np.arange(self.n_envs)

        if self.random_gate_start:
            self._randomize_start(all_indices)

        self._update_gate_tracking(all_indices)

        self.track.reset()
        obs = self._compute_obs()
        return obs, {}

    def _esc_to_omega(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map normalized action [-1, 1] to motor speed [rad/s] via ESC curve.

        MonoRace ESC model:
            U = (u + 1) / 2                          # [-1,1] -> [0,1]
            omega = (w_max - w_min) * sqrt(k*U^2 + (1-k)*U) + w_min

        Args:
            u: Normalized actions, shape (N, 4), values in [-1, 1].

        Returns:
            Motor speeds in rad/s, shape (N, 4).
        """
        k = self.esc_nonlinearity
        w_max = self.params.max_omega
        w_min = self.omega_min
        U = np.clip((u + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] -> [0,1]
        return (w_max - w_min) * np.sqrt(k * U**2 + (1.0 - k) * U) + w_min

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]:
        """Step all environments forward.

        Args:
            action: Normalized motor commands (n_envs, 4) or (4,) in [-1, 1].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            action = action[None, :]  # (1, 4)

        # Map normalized [-1, 1] action through nonlinear ESC curve to rad/s
        action_rads = self._esc_to_omega(action)

        # Step dynamics
        self._states = self.dynamics.step(self._states, action_rads)
        self._step_counts += 1

        # Compute rewards, termination, truncation
        rewards = np.zeros(self.n_envs, dtype=np.float64)
        terminated = np.zeros(self.n_envs, dtype=np.bool_)
        truncated = np.zeros(self.n_envs, dtype=np.bool_)

        for i in range(self.n_envs):
            # Check termination conditions
            z = self._states[i, 2]
            quat = self._states[i, QUAT]
            quat_norm = np.linalg.norm(quat)

            # Configurable crash penalty
            crash_penalty = (
                self.reward_weights.get("crash_penalty", 10.0)
                if self.reward_weights
                else 10.0
            )

            # Ground crash
            if z <= 0.0:
                terminated[i] = True
                rewards[i] = -crash_penalty
                continue

            # Ceiling crash
            if z > self.ceiling:
                terminated[i] = True
                rewards[i] = -crash_penalty
                continue

            # Quaternion divergence
            if quat_norm < 0.5 or quat_norm > 1.5:
                terminated[i] = True
                rewards[i] = -crash_penalty
                continue

            # Timeout truncation
            if self._step_counts[i] >= self.max_steps:
                truncated[i] = True

            # Current target gate
            gate_idx = int(self._gate_indices[i])
            gate = self.track.gates[gate_idx % self.track.num_gates]

            state_obj = QuadState.from_vector(self._states[i])
            action_obj = Action(values=action[i] if action.shape[0] > 1 else action[0])

            prev_vals = self._prev_actions[i] if self._prev_actions.shape[0] > 1 else self._prev_actions[0]
            prev_action_obj = Action(values=prev_vals)

            # Current distance to gate (for delta-based progress reward)
            curr_dist = float(np.linalg.norm(
                self._states[i, POS] - gate.position
            ))

            # Continuous reward via monorace_reward (no gate_offset -- that's
            # applied discretely at passage only).
            # Pass new M23 params; fall back to old signature if rewards.py
            # hasn't been updated yet by the other agent.
            try:
                rewards[i] = monorace_reward(
                    state=state_obj,
                    action=action_obj,
                    gate_state=gate,
                    weights=self.reward_weights,
                    prev_action=prev_action_obj,
                    prev_gate_dist=self._prev_gate_dists[i],
                    v_max=self.v_max,
                    dt=self.dt,
                    action_smoothness_threshold=self.action_smoothness_threshold,
                )
            except TypeError:
                # Old monorace_reward signature (pre-M23 update)
                rewards[i] = monorace_reward(
                    state=state_obj,
                    action=action_obj,
                    gate_state=gate,
                    weights=self.reward_weights,
                    prev_action=prev_action_obj,
                )

            # --- Plane-crossing gate passage detection ---
            normal = _gate_normal(gate)
            rel_pos = self._states[i, POS] - gate.position
            curr_along_normal = float(np.dot(rel_pos, normal))
            prev_along_normal = self._prev_along_normal[i]

            # Plane crossing: sign change from negative (approaching) to
            # positive (passed through front-to-back)
            if prev_along_normal <= 0 and curr_along_normal > 0:
                # Check lateral distance from gate center
                lateral = rel_pos - curr_along_normal * normal
                lateral_dist = float(np.linalg.norm(lateral))
                if lateral_dist <= self.gate_passage_radius:
                    # Gate passed! Increment gate index
                    self._gate_indices[i] += 1
                    if self._gate_indices[i] >= self.track.num_gates:
                        self._gate_indices[i] = 0

                    # Gate passage bonus
                    passage_weight = (
                        self.reward_weights.get("gate_passage", 1.5)
                        if self.reward_weights
                        else 1.5
                    )
                    rewards[i] += passage_weight

                    # Gate offset penalty (discrete, at passage only)
                    offset_weight = (
                        (self.reward_weights or {}).get("gate_offset", 1.5)
                    )
                    rewards[i] += offset_weight * gate_offset_penalty(
                        state_obj, gate
                    )

                    # CRITICAL: recompute curr_dist against NEW target gate
                    new_gate = self.track.gates[
                        int(self._gate_indices[i]) % self.track.num_gates
                    ]
                    curr_dist = float(np.linalg.norm(
                        self._states[i, POS] - new_gate.position
                    ))

                    # Recompute along-normal for the new gate
                    new_normal = _gate_normal(new_gate)
                    new_rel = self._states[i, POS] - new_gate.position
                    curr_along_normal = float(np.dot(new_rel, new_normal))
                elif self.gate_collision:
                    # Crossed gate plane but outside opening -> gate collision
                    terminated[i] = True
                    rewards[i] = -crash_penalty
                    continue

            # Update tracking state for next step
            self._prev_along_normal[i] = curr_along_normal
            self._prev_gate_dists[i] = curr_dist

        self._prev_actions = action.copy()

        # Compute obs BEFORE auto-reset so we capture terminal observations.
        # Always keep batch dimension so VecEnvAdapter can index terminal_obs[i] safely.
        pre_reset_obs = self._compute_obs_batched()

        # Auto-reset terminated/truncated envs
        done = terminated | truncated
        if np.any(done):
            done_indices = np.where(done)[0]
            reset_states = self.dynamics.reset(int(np.sum(done)))
            self._states[done] = reset_states
            self._step_counts[done] = 0
            self._gate_indices[done] = 0
            self._prev_actions[done] = 0.0

            if self.random_gate_start:
                self._randomize_start(done_indices)

            self._update_gate_tracking(done_indices)

        obs = self._compute_obs()

        # Copy to avoid aliasing between returned obs and terminal_obs
        info: dict[str, Any] = {"terminal_obs": pre_reset_obs.copy()}

        return obs, rewards, terminated, truncated, info

    def _compute_obs_batched(self) -> NDArray[np.float32]:
        """Compute observation vectors for all environments.

        Always returns shape ``(n_envs, OBS_DIM)`` regardless of ``n_envs``.
        Use ``_compute_obs`` when you need the single-env squeeze behaviour.

        MonoRace paper observation layout (24-dim):
            [0:3]   position drone -> current gate  (gate-yaw-relative frame)
            [3:6]   velocity                        (gate-yaw-relative frame)
            [6:8]   roll, pitch                     (world-frame Euler angles)
            [8]     yaw relative to gate             (drone_yaw - gate_yaw, wrapped)
            [9:12]  body angular rates (p, q, r)     (body frame)
            [12:16] motor speeds                     (normalized [-1, 1])
            [16:19] position current gate -> next gate (gate-yaw-relative frame)
            [19]    relative yaw to next gate        (next_yaw - cur_yaw, wrapped)
            [20:24] previous action                  (normalized [-1, 1])

        Gate-yaw-relative frame: XY rotated by negative gate yaw, Z unchanged.

        Returns:
            Observations (n_envs, OBS_DIM).
        """
        obs = np.zeros((self.n_envs, OBS_DIM), dtype=np.float32)
        max_omega = self.params.max_omega

        for i in range(self.n_envs):
            state = self._states[i]
            gate_idx = int(self._gate_indices[i])
            gate = self.track.gates[gate_idx % self.track.num_gates]
            next_gate = self.track.gates[(gate_idx + 1) % self.track.num_gates]

            # Gate yaw from its orientation quaternion
            gate_yaw = _quat_to_yaw(gate.orientation)
            cos_yaw = np.cos(gate_yaw)
            sin_yaw = np.sin(gate_yaw)

            # --- [0:3] Position drone -> current gate (gate-yaw frame) ---
            dpos = state[POS] - gate.position  # world frame
            obs[i, 0:2] = _rotate_xy(dpos[:2], cos_yaw, sin_yaw)
            obs[i, 2] = dpos[2]  # Z plain offset

            # --- [3:6] Velocity (gate-yaw frame) ---
            vel_world = state[VEL]
            obs[i, 3:5] = _rotate_xy(vel_world[:2], cos_yaw, sin_yaw)
            obs[i, 5] = vel_world[2]  # Z unchanged

            # --- [6:8] Roll, Pitch (world frame Euler) ---
            drone_quat = state[QUAT]
            roll, pitch, drone_yaw = _quat_to_euler(drone_quat)
            obs[i, 6] = roll
            obs[i, 7] = pitch

            # --- [8] Yaw relative to gate ---
            obs[i, 8] = _wrap_angle(drone_yaw - gate_yaw)

            # --- [9:12] Body angular rates ---
            obs[i, 9:12] = state[OMEGA]

            # --- [12:16] Motor speeds normalized to [-1, 1] ---
            obs[i, 12:16] = (state[MOTOR] / max(max_omega, 1e-10)) * 2.0 - 1.0

            # --- [16:19] Position current gate -> next gate (gate-yaw frame) ---
            dnext = next_gate.position - gate.position  # world frame
            obs[i, 16:18] = _rotate_xy(dnext[:2], cos_yaw, sin_yaw)
            obs[i, 18] = dnext[2]

            # --- [19] Relative yaw to next gate ---
            next_gate_yaw = _quat_to_yaw(next_gate.orientation)
            obs[i, 19] = _wrap_angle(next_gate_yaw - gate_yaw)

            # --- [20:24] Previous action (normalized [-1, 1]) ---
            # _prev_actions already stores normalized [-1, 1] values
            obs[i, 20:24] = self._prev_actions[i]

        return obs

    def _compute_obs(self) -> NDArray[np.float32]:
        """Compute observation vectors, squeezing for single env.

        Returns:
            Observations (n_envs, OBS_DIM) or (OBS_DIM,) for single env.
        """
        obs = self._compute_obs_batched()
        if self.n_envs == 1:
            return obs[0]
        return obs
