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

from sim.rewards import gate_offset_penalty, monorace_reward, spline_proximity_reward, heading_alignment_reward, speed_bonus_reward, boundary_penalty, gate_approach_reward, gate_centering_reward
from sim.spline import GateSpline
from sim.dynamics.trpy_mixer import TRPYMixer
from sim.types import ActionMode

# Reward component column indices for the (n_envs, 8) array
RC_PROGRESS = 0
RC_BODY_RATE = 1
RC_ACTION_SMOOTH = 2
RC_GATE_PASSAGE = 3
RC_GATE_OFFSET = 4
RC_CRASH_PENALTY = 5
RC_SPLINE_PROXIMITY = 6
RC_HEADING_ALIGNMENT = 7
RC_SPEED_BONUS = 8
RC_BOUNDARY_PENALTY = 9
RC_GATE_APPROACH = 10
RC_GATE_CENTERING = 11
NUM_REWARD_COMPONENTS = 12
REWARD_COMPONENT_NAMES = [
    "progress", "body_rate", "action_smooth",
    "gate_passage", "gate_offset", "crash_penalty",
    "spline_proximity", "heading_alignment", "speed_bonus",
    "boundary_penalty", "gate_approach", "gate_centering",
]
from sim.tracks import Track
from sim.procedural_tracks import ProceduralTrackGenerator
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
from sim.domain_randomization import DomainRandomizer
from sim.dynamics.params import VehicleParams


# Default observation dimension (n_lookahead_gates=1):
#   gate_rel_pos(3) + vel(3) + roll_pitch(2) + yaw_rel(1) +
#   body_rates(3) + motor_speeds(4) + prev_action(4) +
#   lookahead_gates(4 * n_lookahead_gates) = 20 + 4*N
OBS_DIM = 24  # Keep for backwards compat (N=1 default)


def _compute_obs_dim(n_lookahead_gates: int) -> int:
    """Compute observation dimension: 20 base + 4 per lookahead gate."""
    return 20 + 4 * n_lookahead_gates

# Default termination thresholds
DEFAULT_CEILING = 10.0
DEFAULT_MAX_STEPS = 1200

# Termination reason codes
TERM_NONE = 0
TERM_GROUND = 1          # z <= 0
TERM_CEILING = 2         # z > ceiling
TERM_QUAT = 3            # quat norm out of [0.5, 1.5]
TERM_NAN = 4             # non-finite state
TERM_ARENA_OOB = 5       # |x| or |y| > arena_bounds
TERM_BODY_RATE = 6       # per-axis omega > max_body_rate
TERM_GATE_COLLISION = 7  # crossed gate plane outside opening
TERM_TIMEOUT = 8         # step >= max_steps (truncation)

GATE_RACE_TERM_NAMES = {
    0: "none",
    1: "ground",
    2: "ceiling",
    3: "quat",
    4: "nan",
    5: "arena_oob",
    6: "body_rate",
    7: "gate_collision",
    8: "timeout",
}


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

    Observation (20 + 4*N dims, where N = n_lookahead_gates, default N=1 → 24 dims):
        [0:3]       position to current gate (gate-yaw-relative frame)
        [3:6]       velocity (gate-yaw-relative frame)
        [6:8]       roll, pitch (world frame Euler angles)
        [8]         yaw relative to gate (drone_yaw - gate_yaw, wrapped [-pi, pi])
        [9:12]      body angular rates p, q, r (body frame)
        [12:16]     motor speeds (normalized to [-1, 1]: (w / w_max) * 2 - 1)
        [16:20]     previous action / motor commands (normalized [-1, 1])
        [20:20+4*N] lookahead gates (4 dims each: rel_pos(3) + yaw_delta(1))

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
        max_body_rate: Maximum body angular rate (rad/s) before crash.
            Default 17.45 (1000 deg/s, MonoRace M23).
        max_velocity: Maximum velocity for state clamping (m/s). Default 50.
            Used only for float32 overflow protection, not termination.
        arena_bounds: Half-width of lateral arena bounds (m). Default 20.
            Drone terminates if |x| or |y| exceeds this. MonoRace uses 5m
            but that's for a fixed indoor track; 20m suits our gate layout.
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
        start_omega_std: float = 0.0,
        gate_collision: bool = False,
        domain_randomizer: DomainRandomizer | None = None,
        max_body_rate: float = 17.45,
        max_velocity: float = 50.0,
        arena_bounds: float = 20.0,
        track_generator: ProceduralTrackGenerator | None = None,
        tracks: list[Track] | None = None,
        n_lookahead_gates: int = 1,
        action_mode: ActionMode | str = ActionMode.MOTOR_RPM,
    ) -> None:
        super().__init__()

        if isinstance(action_mode, str):
            action_mode = ActionMode(action_mode)
        self.action_mode = action_mode

        if n_lookahead_gates < 1:
            raise ValueError(f"n_lookahead_gates must be >= 1, got {n_lookahead_gates}")
        self._n_lookahead_gates = n_lookahead_gates
        self._obs_dim = _compute_obs_dim(n_lookahead_gates)

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
        self.start_omega_std = start_omega_std
        self.gate_collision = gate_collision
        self._domain_randomizer = domain_randomizer
        self.max_body_rate = max_body_rate
        self.max_velocity = max_velocity
        self.arena_bounds = arena_bounds

        # Track initialization: explicit list > single track > default figure-8
        if tracks is not None:
            assert len(tracks) == n_envs, (
                f"tracks list length ({len(tracks)}) must match n_envs ({n_envs})"
            )
            self._tracks: list[Track] = list(tracks)
        elif track is not None:
            self._tracks = [track] * n_envs
        else:
            from sim.tracks import build_figure8_track
            self._tracks = [build_figure8_track()] * n_envs
        self.track_generator: ProceduralTrackGenerator | None = track_generator

        # Build splines for reward shaping (one per env track)
        self._splines: list[GateSpline | None] = []
        for t in self._tracks:
            positions = np.array([g.position for g in t.gates])
            self._splines.append(GateSpline(positions) if len(positions) >= 2 else None)

        # Dynamics
        self.params = params or VehicleParams()
        self.dynamics = NumpyQuadDynamics(params=self.params, dt=dt)
        if action_mode == ActionMode.TRPY:
            self._trpy_mixer = TRPYMixer(self.params)

        # Gymnasium spaces — normalized action space [-1, 1] per MonoRace paper
        obs_high = np.full(self._obs_dim, np.inf, dtype=np.float32)
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

        # Per-episode gate/lap tracking
        self._gates_passed = np.zeros(n_envs, dtype=np.int64)
        self._laps_completed = np.zeros(n_envs, dtype=np.int64)
        self._start_gate_indices = np.zeros(n_envs, dtype=np.int64)
        self._episode_rewards = np.zeros(n_envs, dtype=np.float64)
        self._termination_reasons = np.zeros(n_envs, dtype=np.int8)

        # Per-step reward component array (zeroed each step)
        self._step_reward_components = np.zeros(
            (n_envs, NUM_REWARD_COMPONENTS), dtype=np.float64
        )
        # Per-episode accumulated reward components
        self._episode_reward_components = np.zeros(
            (n_envs, NUM_REWARD_COMPONENTS), dtype=np.float64
        )
        # Per-episode speed tracking (for avg_speed metric)
        self._episode_speed_sum = np.zeros(n_envs, dtype=np.float64)
        self._episode_speed_count = np.zeros(n_envs, dtype=np.int64)
        # Per-episode first gate step tracking
        self._first_gate_step = np.full(n_envs, -1, dtype=np.int64)  # -1 = no gate passed

    @property
    def track(self) -> Track:
        """First env's track (backwards compatibility)."""
        return self._tracks[0]

    def _apply_domain_rand(self, env_indices: NDArray[np.intp]) -> None:
        """Draw fresh randomized physics for specified envs."""
        if self._domain_randomizer is None:
            return
        for idx in env_indices:
            rp = self._domain_randomizer.apply(self.params, rng=self.np_random)
            self.dynamics.update_params(
                env_indices=np.array([idx]),
                mass=np.array([rp.mass]),
                inertia=rp.inertia[None],
                k_thrust=np.array([rp.k_thrust]),
                k_torque=np.array([rp.k_torque]),
                arm_length=np.array([rp.arm_length]),
                tau_motor=np.array([rp.tau_motor]),
                max_rpm=np.array([rp.max_rpm]),
                drag_coeff=rp.drag_coeff[None],
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

        for idx in env_indices:
            # Pick a random gate
            gate_idx = int(rng.integers(0, self._tracks[idx].num_gates))
            self._gate_indices[idx] = gate_idx
            self._start_gate_indices[idx] = gate_idx
            gate = self._tracks[idx].gates[gate_idx]
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

            # Angular rate perturbation (M23: random initial body rates)
            self._states[idx, OMEGA] = rng.normal(0, self.start_omega_std, size=3)
            hover_omega_i = np.sqrt(
                self.dynamics._mass[idx] * GRAVITY / (4.0 * self.dynamics._k_thrust[idx])
            )
            self._states[idx, MOTOR] = hover_omega_i

    def _update_gate_tracking(self, env_indices: NDArray[np.intp]) -> None:
        """Recompute prev_gate_dists and prev_along_normal for given envs.

        Args:
            env_indices: Array of env indices to update.
        """
        for idx in env_indices:
            gate_idx = int(self._gate_indices[idx])
            gate = self._tracks[idx].gates[gate_idx % self._tracks[idx].num_gates]
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
        self._prev_actions = np.zeros((self.n_envs, 4), dtype=np.float64)
        self._gates_passed[:] = 0
        self._laps_completed[:] = 0
        self._episode_rewards[:] = 0.0
        self._termination_reasons[:] = TERM_NONE
        self._step_reward_components[:] = 0.0
        self._episode_reward_components[:] = 0.0
        self._episode_speed_sum[:] = 0.0
        self._episode_speed_count[:] = 0
        self._first_gate_step[:] = -1

        all_indices = np.arange(self.n_envs)

        if self._domain_randomizer is not None:
            self._apply_domain_rand(all_indices)

        if self.track_generator is not None:
            for i in range(self.n_envs):
                self._tracks[i] = self.track_generator.generate(self.np_random)

        if self.random_gate_start:
            self._randomize_start(all_indices)

        self._update_gate_tracking(all_indices)

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
        n = u.shape[0]
        if self.dynamics._max_omega.size >= n:
            w_max = self.dynamics._max_omega[:n, None]  # (N, 1) per-env
        else:
            w_max = self.params.max_omega  # scalar fallback
        w_min = self.omega_min
        U = np.clip((u + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] -> [0,1]
        return (w_max - w_min) * np.sqrt(k * U**2 + (1.0 - k) * U) + w_min

    def _trpy_to_omega(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map normalized TRPY action [-1, 1] to motor speed [rad/s].
        u[0] in [-1,1] → thrust in [0, 2*m*g]
        u[1:4] in [-1,1] → body rates in [-max_body_rate, max_body_rate]
        """
        physical = np.empty_like(u)
        max_thrust = self.params.mass * GRAVITY * 2.0
        physical[:, 0] = (u[:, 0] + 1.0) / 2.0 * max_thrust
        physical[:, 1:4] = u[:, 1:4] * self.max_body_rate
        return self._trpy_mixer.mix_batch(physical)

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

        # Map normalized [-1, 1] action to motor speeds in rad/s
        if self.action_mode == ActionMode.TRPY:
            action_rads = self._trpy_to_omega(action)
        else:
            action_rads = self._esc_to_omega(action)

        # Step dynamics
        self._states = self.dynamics.step(self._states, action_rads)
        self._step_counts += 1

        # Clamp state to prevent overflow propagating to observations (float32)
        np.clip(self._states[:, VEL], -self.max_velocity, self.max_velocity, out=self._states[:, VEL])
        np.clip(self._states[:, OMEGA], -self.max_body_rate, self.max_body_rate, out=self._states[:, OMEGA])

        # Compute rewards, termination, truncation
        rewards = np.zeros(self.n_envs, dtype=np.float64)
        terminated = np.zeros(self.n_envs, dtype=np.bool_)
        truncated = np.zeros(self.n_envs, dtype=np.bool_)

        # Zero per-step reward components
        self._step_reward_components[:] = 0.0

        # Accumulate speed for avg_speed metric
        speed = np.linalg.norm(self._states[:, VEL], axis=1)
        self._episode_speed_sum += speed
        self._episode_speed_count += 1

        # Pre-compute speed bonus (vectorized — speed already computed above)
        speed_bonus_weight = (self.reward_weights or {}).get("speed_bonus", 0.0)
        if speed_bonus_weight != 0.0:
            v_target = max(self.v_max, 1e-6)
            _speed_bonuses = speed_bonus_weight * np.clip(speed, 0.0, v_target) / v_target
        else:
            _speed_bonuses = np.zeros(self.n_envs, dtype=np.float64)

        # Pre-compute boundary penalty (vectorized)
        boundary_weight = (self.reward_weights or {}).get("boundary_penalty", 0.0)
        if boundary_weight != 0.0:
            _boundary_penalties = np.array([
                boundary_weight * boundary_penalty(
                    self._states[i, :2], self.arena_bounds, margin=3.0
                ) for i in range(self.n_envs)
            ], dtype=np.float64)
        else:
            _boundary_penalties = np.zeros(self.n_envs, dtype=np.float64)

        # Pre-compute spline rewards
        spline_weight = (self.reward_weights or {}).get("spline_proximity", 0.0)
        heading_weight = (self.reward_weights or {}).get("heading_alignment", 0.0)
        _spline_rewards = np.zeros(self.n_envs, dtype=np.float64)
        _heading_rewards = np.zeros(self.n_envs, dtype=np.float64)

        if (spline_weight != 0.0 or heading_weight != 0.0) and any(s is not None for s in self._splines):
            for i in range(self.n_envs):
                if self._splines[i] is None:
                    continue
                if spline_weight != 0.0:
                    d = self._splines[i].distance_to_nearest(self._states[i, POS])
                    _spline_rewards[i] = spline_weight * spline_proximity_reward(d)
                if heading_weight != 0.0:
                    _, tangent = self._splines[i].nearest_point_and_tangent(self._states[i, POS])
                    drone_yaw = _quat_to_euler(self._states[i, QUAT])[2]
                    tangent_xy = tangent[:2]
                    tangent_xy_norm = np.linalg.norm(tangent_xy)
                    if tangent_xy_norm > 1e-6:
                        cos_a = np.clip(
                            (np.cos(drone_yaw) * tangent_xy[0] + np.sin(drone_yaw) * tangent_xy[1])
                            / tangent_xy_norm, -1.0, 1.0
                        )
                        yaw_error = np.arccos(cos_a)
                    else:
                        yaw_error = 0.0
                    _heading_rewards[i] = heading_weight * heading_alignment_reward(yaw_error)

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
                self._step_reward_components[i, RC_CRASH_PENALTY] = -crash_penalty
                self._termination_reasons[i] = TERM_GROUND
                continue

            # Ceiling crash
            if z > self.ceiling:
                terminated[i] = True
                rewards[i] = -crash_penalty
                self._step_reward_components[i, RC_CRASH_PENALTY] = -crash_penalty
                self._termination_reasons[i] = TERM_CEILING
                continue

            # Quaternion divergence
            if quat_norm < 0.5 or quat_norm > 1.5:
                terminated[i] = True
                rewards[i] = -crash_penalty
                self._step_reward_components[i, RC_CRASH_PENALTY] = -crash_penalty
                self._termination_reasons[i] = TERM_QUAT
                continue

            # State divergence: NaN/inf (quaternion sim safety)
            state_i = self._states[i]
            if not np.all(np.isfinite(state_i)):
                terminated[i] = True
                rewards[i] = -crash_penalty
                self._step_reward_components[i, RC_CRASH_PENALTY] = -crash_penalty
                self._termination_reasons[i] = TERM_NAN
                continue

            # Out-of-bounds (lateral arena limits, MonoRace-style)
            if (abs(state_i[0]) > self.arena_bounds
                    or abs(state_i[1]) > self.arena_bounds):
                terminated[i] = True
                rewards[i] = -crash_penalty
                self._step_reward_components[i, RC_CRASH_PENALTY] = -crash_penalty
                self._termination_reasons[i] = TERM_ARENA_OOB
                continue

            # Excessive body rate per axis (MonoRace: 1000 deg/s ≈ 17.45 rad/s)
            if np.any(np.abs(state_i[OMEGA]) > self.max_body_rate):
                terminated[i] = True
                rewards[i] = -crash_penalty
                self._step_reward_components[i, RC_CRASH_PENALTY] = -crash_penalty
                self._termination_reasons[i] = TERM_BODY_RATE
                continue

            # Timeout truncation
            if self._step_counts[i] >= self.max_steps:
                truncated[i] = True
                self._termination_reasons[i] = TERM_TIMEOUT

            # Current target gate
            gate_idx = int(self._gate_indices[i])
            gate = self._tracks[i].gates[gate_idx % self._tracks[i].num_gates]

            state_obj = QuadState.from_vector(self._states[i])
            action_obj = Action(values=action[i] if action.shape[0] > 1 else action[0])

            prev_vals = self._prev_actions[i] if self._prev_actions.shape[0] > 1 else self._prev_actions[0]
            prev_action_obj = Action(values=prev_vals)

            # Current distance to gate (for delta-based progress reward)
            curr_dist = float(np.linalg.norm(
                self._states[i, POS] - gate.position
            ))

            # Continuous reward via monorace_reward (returns RewardResult)
            reward_result = monorace_reward(
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
            rewards[i] = reward_result.total
            self._step_reward_components[i, RC_PROGRESS] = reward_result.components["progress"]
            self._step_reward_components[i, RC_BODY_RATE] = reward_result.components["body_rate"]
            self._step_reward_components[i, RC_ACTION_SMOOTH] = reward_result.components["action_smooth"]

            # Spline proximity reward (pre-computed)
            if spline_weight != 0.0:
                rewards[i] += _spline_rewards[i]
                self._step_reward_components[i, RC_SPLINE_PROXIMITY] = _spline_rewards[i]

            # Heading alignment reward (pre-computed)
            if heading_weight != 0.0:
                rewards[i] += _heading_rewards[i]
                self._step_reward_components[i, RC_HEADING_ALIGNMENT] = _heading_rewards[i]

            # Speed bonus (pre-computed)
            if speed_bonus_weight != 0.0:
                rewards[i] += _speed_bonuses[i]
                self._step_reward_components[i, RC_SPEED_BONUS] = _speed_bonuses[i]

            # Boundary penalty (pre-computed)
            if boundary_weight != 0.0:
                rewards[i] += _boundary_penalties[i]
                self._step_reward_components[i, RC_BOUNDARY_PENALTY] = _boundary_penalties[i]

            # Gate approach reward (needs per-env gate normal + velocity)
            approach_weight = (self.reward_weights or {}).get("gate_approach", 0.0)
            if approach_weight != 0.0:
                gate_for_approach = self._tracks[i].gates[int(self._gate_indices[i]) % self._tracks[i].num_gates]
                approach_normal = _gate_normal(gate_for_approach)
                approach_val = approach_weight * gate_approach_reward(self._states[i, VEL], approach_normal)
                rewards[i] += approach_val
                self._step_reward_components[i, RC_GATE_APPROACH] = approach_val

            # Gate centering reward (continuous, distance-attenuated)
            centering_weight = (self.reward_weights or {}).get("gate_centering", 0.0)
            if centering_weight != 0.0:
                centering_normal = _gate_normal(gate)
                centering_rel = self._states[i, POS] - gate.position
                dist_to_plane = float(np.dot(centering_rel, centering_normal))
                lateral_vec = centering_rel - dist_to_plane * centering_normal
                lateral_offset = float(np.linalg.norm(lateral_vec))
                centering_val = centering_weight * gate_centering_reward(
                    lateral_offset, dist_to_plane, self.gate_passage_radius
                )
                rewards[i] += centering_val
                self._step_reward_components[i, RC_GATE_CENTERING] = centering_val

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
                    # Gate passed! Increment gate index (wrapping)
                    self._gate_indices[i] = (self._gate_indices[i] + 1) % self._tracks[i].num_gates
                    self._gates_passed[i] += 1
                    # Lap complete when we've passed num_gates gates (full circuit)
                    if self._gates_passed[i] > 0 and self._gates_passed[i] % self._tracks[i].num_gates == 0:
                        self._laps_completed[i] += 1

                    # Track first gate step
                    if self._first_gate_step[i] < 0:
                        self._first_gate_step[i] = self._step_counts[i]

                    # Gate passage bonus
                    passage_weight = (
                        self.reward_weights.get("gate_passage", 1.5)
                        if self.reward_weights
                        else 1.5
                    )
                    rewards[i] += passage_weight
                    self._step_reward_components[i, RC_GATE_PASSAGE] = passage_weight

                    # Gate offset penalty (discrete, at passage only)
                    offset_weight = (
                        (self.reward_weights or {}).get("gate_offset", 1.5)
                    )
                    offset_val = offset_weight * gate_offset_penalty(
                        state_obj, gate
                    )
                    rewards[i] += offset_val
                    self._step_reward_components[i, RC_GATE_OFFSET] = offset_val

                    # CRITICAL: recompute curr_dist against NEW target gate
                    new_gate = self._tracks[i].gates[
                        int(self._gate_indices[i]) % self._tracks[i].num_gates
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
                    self._step_reward_components[i, RC_CRASH_PENALTY] = -crash_penalty
                    self._termination_reasons[i] = TERM_GATE_COLLISION
                    continue

            # Update tracking state for next step
            self._prev_along_normal[i] = curr_along_normal
            self._prev_gate_dists[i] = curr_dist

        self._prev_actions = action.copy()
        self._episode_rewards += rewards
        self._episode_reward_components += self._step_reward_components

        # Compute obs BEFORE auto-reset so we capture terminal observations.
        # Always keep batch dimension so VecEnvAdapter can index terminal_obs[i] safely.
        pre_reset_obs = self._compute_obs_batched()

        # Auto-reset terminated/truncated envs
        done = terminated | truncated

        # Snapshot episode metrics BEFORE auto-reset clears counters
        ep_info: dict[str, Any] | None = None
        if np.any(done):
            # Compute avg_speed per env
            avg_speed = np.where(
                self._episode_speed_count > 0,
                self._episode_speed_sum / self._episode_speed_count,
                0.0,
            )

            ep_info = {
                "r": self._episode_rewards.copy(),
                "l": self._step_counts.copy(),
                "effective_dt": self.dt,
                "gates_passed": self._gates_passed.copy(),
                "laps_completed": self._laps_completed.copy(),
                "termination": np.array([
                    GATE_RACE_TERM_NAMES[int(c)]
                    for c in self._termination_reasons
                ]),
                "success": self._termination_reasons == TERM_TIMEOUT,
                "success_criterion": "survived_full_episode",
                "avg_speed": avg_speed.copy(),
                "first_gate_step": self._first_gate_step.copy(),
                "reward_components": np.array([
                    {name: float(self._episode_reward_components[i, j])
                     for j, name in enumerate(REWARD_COMPONENT_NAMES)}
                    for i in range(self.n_envs)
                ], dtype=object),
                "n_gates": np.array([self._tracks[i].num_gates for i in range(self.n_envs)]),
                "track_id": np.array([
                    hash(tuple(tuple(g.position) for g in self._tracks[i].gates))
                    for i in range(self.n_envs)
                ]),
            }

            done_indices = np.where(done)[0]

            # Randomize params FIRST, then generate states with correct hover omega
            if self._domain_randomizer is not None:
                self._apply_domain_rand(done_indices)

            # Generate reset states using per-env params (not dynamics.reset()!)
            reset_states = self.dynamics.make_reset_states(
                n_envs=int(np.sum(done)),
                env_indices=done_indices,
            )
            self._states[done] = reset_states
            self._step_counts[done] = 0
            # _gate_indices and _start_gate_indices set by make_reset_states
            self._prev_actions[done] = 0.0
            self._gates_passed[done] = 0
            self._laps_completed[done] = 0
            self._episode_rewards[done] = 0.0
            self._termination_reasons[done] = TERM_NONE
            self._episode_reward_components[done] = 0.0
            self._episode_speed_sum[done] = 0.0
            self._episode_speed_count[done] = 0
            self._first_gate_step[done] = -1

            if self.track_generator is not None:
                for idx in done_indices:
                    self._tracks[idx] = self.track_generator.generate(self.np_random)
                    # Rebuild spline for regenerated track
                    positions = np.array([g.position for g in self._tracks[idx].gates])
                    self._splines[idx] = GateSpline(positions) if len(positions) >= 2 else None

            if self.random_gate_start:
                self._randomize_start(done_indices)

            self._update_gate_tracking(done_indices)

        obs = self._compute_obs()

        # Copy to avoid aliasing between returned obs and terminal_obs
        info: dict[str, Any] = {"terminal_obs": pre_reset_obs.copy()}

        # Episode metrics for done envs (SB3 auto-logs these)
        if ep_info is not None:
            info["episode"] = ep_info

        return obs, rewards, terminated, truncated, info

    def _compute_obs_batched(self) -> NDArray[np.float32]:
        """Compute observation vectors for all environments.

        Observation layout (20 + 4*N dims, where N = n_lookahead_gates):
            [0:3]   position drone -> current gate  (gate-yaw-relative frame)
            [3:6]   velocity                        (gate-yaw-relative frame)
            [6:8]   roll, pitch                     (world-frame Euler angles)
            [8]     yaw relative to gate             (drone_yaw - gate_yaw, wrapped)
            [9:12]  body angular rates (p, q, r)     (body frame)
            [12:16] motor speeds                     (normalized [-1, 1])
            [16:20] previous action                  (normalized [-1, 1])
            [20:20+4*N] lookahead gates              (4 dims each: rel_pos(3) + yaw_delta(1))

        Gate-yaw-relative frame: XY rotated by negative gate yaw, Z unchanged.

        Returns:
            Observations (n_envs, obs_dim).
        """
        obs = np.zeros((self.n_envs, self._obs_dim), dtype=np.float32)

        for i in range(self.n_envs):
            state = self._states[i]
            gate_idx = int(self._gate_indices[i])
            track = self._tracks[i]
            n_gates = track.num_gates
            gate = track.gates[gate_idx % n_gates]

            # Gate yaw from its orientation quaternion
            gate_yaw = _quat_to_yaw(gate.orientation)
            cos_yaw = np.cos(gate_yaw)
            sin_yaw = np.sin(gate_yaw)

            # --- [0:3] Position drone -> current gate (gate-yaw frame) ---
            dpos = state[POS] - gate.position
            obs[i, 0:2] = _rotate_xy(dpos[:2], cos_yaw, sin_yaw)
            obs[i, 2] = dpos[2]

            # --- [3:6] Velocity (gate-yaw frame) ---
            vel_world = state[VEL]
            obs[i, 3:5] = _rotate_xy(vel_world[:2], cos_yaw, sin_yaw)
            obs[i, 5] = vel_world[2]

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
            if self.dynamics._max_omega.size > i:
                max_omega_i = self.dynamics._max_omega[i]
            else:
                max_omega_i = self.params.max_omega
            obs[i, 12:16] = (state[MOTOR] / max(max_omega_i, 1e-10)) * 2.0 - 1.0

            # --- [16:20] Previous action (normalized [-1, 1]) ---
            obs[i, 16:20] = self._prev_actions[i]

            # --- [20:20+4*N] Lookahead gates ---
            for k in range(1, self._n_lookahead_gates + 1):
                lookahead_gate = track.gates[(gate_idx + k) % n_gates]
                # Relative position: lookahead gate - current gate, in current gate yaw frame
                dg = lookahead_gate.position - gate.position
                offset = 20 + 4 * (k - 1)
                obs[i, offset:offset + 2] = _rotate_xy(dg[:2], cos_yaw, sin_yaw)
                obs[i, offset + 2] = dg[2]
                # Yaw delta: lookahead gate yaw - current gate yaw
                lookahead_yaw = _quat_to_yaw(lookahead_gate.orientation)
                obs[i, offset + 3] = _wrap_angle(lookahead_yaw - gate_yaw)

        return obs

    def _compute_obs(self) -> NDArray[np.float32]:
        """Compute observation vectors, squeezing for single env.

        Returns:
            Observations (n_envs, obs_dim) or (obs_dim,) for single env.
        """
        obs = self._compute_obs_batched()
        if self.n_envs == 1:
            return obs[0]
        return obs

    # --- TrajectoryProvider protocol ---

    def get_state(self, env_idx: int) -> dict[str, np.ndarray]:
        """Return current state for one environment."""
        state = self._states[env_idx]
        return {
            "position": state[POS].copy(),
            "quaternion": state[QUAT].copy(),
            "velocity": state[VEL].copy(),
            "body_rates": state[OMEGA].copy(),
            "motor_rpms": state[MOTOR].copy(),
        }

    def get_gate_geometry(self, env_idx: int = 0) -> dict[str, np.ndarray]:
        """Return gate geometry for a specific environment's track."""
        track = self._tracks[env_idx]
        n_gates = track.num_gates
        positions = np.zeros((n_gates, 3), dtype=np.float64)
        orientations = np.zeros((n_gates, 4), dtype=np.float64)
        half_extents = np.zeros((n_gates, 2), dtype=np.float64)
        radius = self.gate_passage_radius
        for g in range(n_gates):
            gate = track.gates[g]
            positions[g] = gate.position
            orientations[g] = gate.orientation
            half_extents[g] = [radius, radius]
        return {
            "positions": positions,
            "orientations": orientations,
            "half_extents": half_extents,
        }

    def get_step_reward_components(self, env_idx: int) -> tuple[list[str], np.ndarray]:
        """Return per-step reward component breakdown."""
        return REWARD_COMPONENT_NAMES, self._step_reward_components[env_idx].copy()

    def set_reward_weights(self, updates: dict[str, float]) -> None:
        """Update reward weights at runtime (for curriculum learning)."""
        if self.reward_weights is None:
            from sim.rewards import DEFAULT_WEIGHTS
            self.reward_weights = dict(DEFAULT_WEIGHTS)
        self.reward_weights.update(updates)

    def set_v_max(self, v_max: float) -> None:
        """Update v_max at runtime (for curriculum learning)."""
        self.v_max = v_max
