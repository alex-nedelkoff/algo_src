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
    quat_to_rotmat_batch,
)
from sim.dynamics.params import VehicleParams


# Observation dimension breakdown:
#   gate_rel_pos(3) + vel(3) + omega(3) + motor_speeds(4) +
#   next_gate_rel_pos(3) + gate_progress(1) + padding(7) = 24
OBS_DIM = 24

# Default termination thresholds
DEFAULT_CEILING = 10.0
DEFAULT_MAX_STEPS = 1200


class GateRaceEnv(gym.Env):
    """Gymnasium environment for quadrotor gate racing.

    Observation (24-dim):
        [0:3]   relative position to current gate (body frame)
        [3:6]   velocity (world frame)
        [6:9]   angular rate (body frame)
        [9:13]  motor speeds (rad/s, normalized by max_omega)
        [13:16] relative position to next gate (body frame)
        [16]    gate progress scalar (current_gate_idx / num_gates)
        [17:24] padding zeros

    Action (4-dim):
        Motor RPM commands, continuous, clipped to [0, max_rpm].

    Supports vectorized operation with n_envs parallel episodes.

    Args:
        track: Track object defining the gate sequence.
        params: Vehicle parameters. Uses defaults if None.
        n_envs: Number of parallel environments.
        dt: Simulation timestep in seconds.
        max_steps: Maximum steps per episode before timeout.
        ceiling: Maximum altitude (z) before crash.
        reward_weights: Weights for the monorace_reward function.
        gate_passage_radius: Distance threshold for gate passage detection.
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
    ) -> None:
        super().__init__()

        self.n_envs = n_envs
        self.max_steps = max_steps
        self.ceiling = ceiling
        self.reward_weights = reward_weights
        self.gate_passage_radius = gate_passage_radius

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

        max_rpm = self.params.max_rpm
        max_omega = self.params.max_omega

        # Gymnasium spaces
        obs_high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.full(4, max_rpm, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self._states: NDArray[np.float64] = np.zeros((n_envs, 17))
        self._step_counts = np.zeros(n_envs, dtype=np.int64)
        self._gate_indices = np.zeros(n_envs, dtype=np.int64)
        self._prev_gate_dists = np.zeros(n_envs, dtype=np.float64)
        self._prev_actions: NDArray[np.float64] | None = None

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
        self._prev_actions = None

        # Compute initial distances to gate
        for i in range(self.n_envs):
            gate_pos = self.track.gates[0].position
            self._prev_gate_dists[i] = np.linalg.norm(
                self._states[i, POS] - gate_pos
            )

        self.track.reset()
        obs = self._compute_obs()
        return obs, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], dict[str, Any]]:
        """Step all environments forward.

        Args:
            action: Motor RPM commands (n_envs, 4) or (4,) for single env.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            action = action[None, :]  # (1, 4)

        # Convert RPM to rad/s for dynamics
        action_rads = action * 2.0 * np.pi / 60.0

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

            # Ground crash
            if z <= 0.0:
                terminated[i] = True
                rewards[i] = -10.0
                continue

            # Ceiling crash
            if z > self.ceiling:
                terminated[i] = True
                rewards[i] = -10.0
                continue

            # Quaternion divergence
            if quat_norm < 0.5 or quat_norm > 1.5:
                terminated[i] = True
                rewards[i] = -10.0
                continue

            # Timeout truncation
            if self._step_counts[i] >= self.max_steps:
                truncated[i] = True

            # Compute reward using monorace_reward
            gate_idx = int(self._gate_indices[i])
            gate = self.track.gates[gate_idx % self.track.num_gates]

            state_obj = QuadState.from_vector(self._states[i])
            action_obj = Action(values=action[i] if action.shape[0] > 1 else action[0])

            prev_action_obj = None
            if self._prev_actions is not None:
                prev_vals = self._prev_actions[i] if self._prev_actions.shape[0] > 1 else self._prev_actions[0]
                prev_action_obj = Action(values=prev_vals)

            rewards[i] = monorace_reward(
                state=state_obj,
                action=action_obj,
                gate_state=gate,
                weights=self.reward_weights,
                prev_action=prev_action_obj,
            )

            # Gate passage detection (per-env, distance-based)
            gate_pos = self.track.gates[gate_idx % self.track.num_gates].position
            dist_to_gate = np.linalg.norm(self._states[i, POS] - gate_pos)

            if dist_to_gate <= self.gate_passage_radius:
                self._gate_indices[i] += 1
                # Handle lap completion
                if self._gate_indices[i] >= self.track.num_gates:
                    self._gate_indices[i] = 0
                # Add gate passage bonus
                rewards[i] += self.reward_weights.get("gate_passage", 30.0) if self.reward_weights else 30.0

        self._prev_actions = action.copy()

        # Auto-reset terminated/truncated envs
        done = terminated | truncated
        if np.any(done):
            reset_states = self.dynamics.reset(int(np.sum(done)))
            self._states[done] = reset_states
            self._step_counts[done] = 0
            self._gate_indices[done] = 0

        obs = self._compute_obs()

        # For single env, squeeze the batch dimension
        if self.n_envs == 1:
            return obs, rewards, terminated, truncated, {}
        return obs, rewards, terminated, truncated, {}

    def _compute_obs(self) -> NDArray[np.float32]:
        """Compute observation vectors for all environments.

        Returns:
            Observations (n_envs, OBS_DIM) or (OBS_DIM,) for single env.
        """
        obs = np.zeros((self.n_envs, OBS_DIM), dtype=np.float32)
        max_omega = self.params.max_omega

        for i in range(self.n_envs):
            state = self._states[i]
            gate_idx = int(self._gate_indices[i])
            gate = self.track.gates[gate_idx % self.track.num_gates]
            next_gate = self.track.gates[(gate_idx + 1) % self.track.num_gates]

            # Rotation matrix (body to world)
            quat = state[QUAT].reshape(1, 4)
            R = quat_to_rotmat_batch(quat)[0]  # (3, 3)
            R_inv = R.T  # world to body

            # Relative gate position in body frame
            gate_rel_world = gate.position - state[POS]
            gate_rel_body = R_inv @ gate_rel_world

            # Next gate relative position in body frame
            next_gate_rel_world = next_gate.position - state[POS]
            next_gate_rel_body = R_inv @ next_gate_rel_world

            # Gate progress
            progress = gate_idx / max(self.track.num_gates, 1)

            # Build observation
            obs[i, 0:3] = gate_rel_body
            obs[i, 3:6] = state[VEL]
            obs[i, 6:9] = state[OMEGA]
            obs[i, 9:13] = state[MOTOR] / max(max_omega, 1e-10)  # normalized
            obs[i, 13:16] = next_gate_rel_body
            obs[i, 16] = progress
            # [17:24] remain zero (padding)

        if self.n_envs == 1:
            return obs[0]
        return obs
