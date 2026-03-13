"""Vectorized quadrotor racing environment with rate-control action space.

Port of the playground's QuadRaceEnv adapted for algo_src conventions.
Uses SymPy-compiled dynamics (MAVLab sysid), first-order motor model,
and the MAVLab MonoRace reward formulation.

Internal state layout (n_envs, 16):
  [0:3]   position (x, y, z) — NED
  [3:6]   velocity (vx, vy, vz)
  [6:9]   Euler angles (phi, theta, psi)
  [9:12]  angular velocity (p, q, r)
  [12:16] motor speeds (w1, w2, w3, w4)

Action space (4): [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
  All in [0, 1]. Rate commands centered at 0.5, mapped to +/-max_rate.

Returns the same (obs, rewards, terminated, truncated, info) format as
GateRaceEnv for compatibility with VecEnvAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.dynamics.sympy_quad import NOMINAL_PARAMS, build_dynamics_fn
from sim.motor_model import motor_step
from sim.envs.playground_obs import gate_relative_obs
from sim.rewards_mavlab import compute_reward_components, MAVLAB_REWARD_COMPONENT_NAMES


# ---------------------------------------------------------------------------
# Inline domain randomization (matches playground's simple uniform approach)
# ---------------------------------------------------------------------------

def _randomize_params(
    nominal: Dict[str, float],
    percentage: float,
    n_envs: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Per-episode domain randomization: U[nominal*(1-p), nominal*(1+p)]."""
    params = {}
    for key, val in nominal.items():
        lo = val * (1.0 - percentage)
        hi = val * (1.0 + percentage)
        params[key] = rng.uniform(lo, hi, size=n_envs)
    return params


# ---------------------------------------------------------------------------
# Inline figure-8 track definition
# ---------------------------------------------------------------------------

def make_figure8_track(
    gate_width: float = 0.55,
    gate_height: float = 0.55,
    gate_z: float = -1.5,
) -> dict:
    """Figure-8 track with 8 gates within a 10m x 10m arena."""
    gates = [
        {"pos": np.array([2.0, 2.0, gate_z]),   "yaw": np.pi},
        {"pos": np.array([-0.5, 1.0, gate_z]),  "yaw": -np.pi * 0.75},
        {"pos": np.array([-2.0, -2.0, gate_z]), "yaw": 0.0},
        {"pos": np.array([-3.5, 0.0, gate_z]),  "yaw": np.pi / 2},
        {"pos": np.array([-2.0, 2.0, gate_z]),  "yaw": 0.0},
        {"pos": np.array([0.5, 1.0, gate_z]),   "yaw": -np.pi * 0.25},
        {"pos": np.array([2.0, -2.0, gate_z]),  "yaw": np.pi},
        {"pos": np.array([3.5, 0.0, gate_z]),   "yaw": np.pi / 2},
    ]
    return {
        "gates": gates,
        "gate_width": gate_width,
        "gate_height": gate_height,
        "n_gates": len(gates),
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HOVER_W = 992.4   # initial motor speed at hover equilibrium
_ARENA_RADIUS = 8.0        # OOB if |x| or |y| > this
_Z_FLOOR = 0.1             # crash if z > this (NED: positive = down)
_Z_CEIL = -6.0             # crash if z < this (too high)
_MAX_TILT = np.pi / 2      # crash if |roll| or |pitch| exceeds this

# Observation / action dims
OBS_DIM = 24
ACT_DIM = 4

# Termination reason codes (compatible with GateRaceEnv)
TERM_NONE = 0
TERM_CRASH = 1
TERM_TIMEOUT = 2

RATE_CTRL_TERM_NAMES = {
    0: "none",
    1: "crash",
    2: "timeout",
}


class RateCtrlEnv:
    """Vectorized quadrotor gate-racing environment with rate-control actions.

    Accepts the same constructor-parameter style as GateRaceEnv and returns
    the same (obs, rewards, terminated, truncated, info) tuple from step()
    for VecEnvAdapter compatibility.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments.
    seed : int
        Random seed.
    dt : float
        Physics timestep in seconds (500 Hz default).
    action_repeat : int
        Number of physics substeps per RL step.
    dr_percentage : float
        Domain randomization range as fraction of nominal (0 = no DR).
    reward_preset : str
        Reward preset name ("baseline", "M16", or "M23").
    init_vel_range : float
        Uniform range for initial velocity perturbation (m/s).
    init_rp_deg : float
        Uniform range for initial roll/pitch perturbation (degrees).
    init_rate_range : float
        Uniform range for initial angular rate perturbation (rad/s).
    max_steps : int
        Episode horizon in RL steps.
    track : dict, optional
        Track definition dict with keys 'gates', 'gate_width', 'gate_height',
        'n_gates'. If None, uses the default figure-8 track.
    """

    def __init__(
        self,
        n_envs: int = 1,
        seed: int = 0,
        dt: float = 0.002,
        action_repeat: int = 5,
        dr_percentage: float = 0.0,
        reward_preset: str = "M23",
        init_vel_range: float = 0.5,
        init_rp_deg: float = 20.0,
        init_rate_range: float = 0.1,
        max_steps: int = 1200,
        track: Optional[dict] = None,
    ) -> None:
        self.n_envs = n_envs
        self.dt = dt
        self.action_repeat = action_repeat
        self.dr_percentage = dr_percentage
        self.reward_preset = reward_preset
        self.init_vel_range = init_vel_range
        self.init_rp_deg = init_rp_deg
        self.init_rate_range = init_rate_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Build compiled dynamics function (cached)
        self._dynamics_fn = build_dynamics_fn()

        # Track
        self._track = track if track is not None else make_figure8_track()
        self._n_gates = self._track["n_gates"]
        self._gate_positions = np.array(
            [g["pos"] for g in self._track["gates"]]
        )  # (n_gates, 3)
        self._gate_yaws = np.array(
            [g["yaw"] for g in self._track["gates"]]
        )  # (n_gates,)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(ACT_DIM,), dtype=np.float64,
        )

        # Allocate persistent arrays
        self._state = np.zeros((n_envs, 16), dtype=np.float64)
        self._prev_action = np.zeros((n_envs, 4), dtype=np.float64)
        self._gate_idx = np.zeros(n_envs, dtype=np.int32)
        self._step_count = np.zeros(n_envs, dtype=np.int32)
        self._d2g = np.zeros(n_envs, dtype=np.float64)
        self._prev_signed_dist = np.zeros(n_envs, dtype=np.float64)
        self._params: Dict[str, np.ndarray] = {}

        # Per-episode accumulators for info dict
        self._ep_reward = np.zeros(n_envs, dtype=np.float64)
        self._ep_reward_components = np.zeros((n_envs, len(MAVLAB_REWARD_COMPONENT_NAMES)), dtype=np.float64)
        self._ep_gates_passed = np.zeros(n_envs, dtype=np.int32)
        self._ep_laps = np.zeros(n_envs, dtype=np.int32)
        self._ep_speed_sum = np.zeros(n_envs, dtype=np.float64)
        self._ep_speed_count = np.zeros(n_envs, dtype=np.int32)
        self._ep_first_gate_step = np.full(n_envs, -1, dtype=np.int32)
        # Per-step reward components (for TrajectoryProvider)
        self._step_reward_components = np.zeros(
            (n_envs, len(MAVLAB_REWARD_COMPONENT_NAMES)), dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environments.

        Parameters
        ----------
        seed : optional new RNG seed.
        options : unused, for Gymnasium compatibility.
        env_mask : bool array (n_envs,), optional
            If given, only reset envs where mask is True.
            If None, reset all.

        Returns
        -------
        obs : (n_envs, 24)
        info : empty dict (Gymnasium convention)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if env_mask is None:
            env_mask = np.ones(self.n_envs, dtype=bool)

        n_reset = int(env_mask.sum())
        if n_reset == 0:
            return self._get_obs(), {}

        # Domain randomization
        if self.dr_percentage > 0:
            new_params = _randomize_params(
                NOMINAL_PARAMS, self.dr_percentage, n_reset, self.rng
            )
        else:
            new_params = {
                k: np.full(n_reset, v) for k, v in NOMINAL_PARAMS.items()
            }

        # Merge into full param arrays
        if not self._params:
            self._params = {
                k: np.zeros(self.n_envs) for k in NOMINAL_PARAMS
            }
        for k in NOMINAL_PARAMS:
            self._params[k][env_mask] = new_params[k]

        # Reset state: spawn 1m behind a random target gate
        gate_idx = self.rng.integers(0, self._n_gates, size=n_reset)
        self._gate_idx[env_mask] = gate_idx

        gate_pos = self._gate_positions[gate_idx]  # (n_reset, 3)
        gate_yaw = self._gate_yaws[gate_idx]       # (n_reset,)

        # Spawn 1m behind gate in its forward direction
        spawn_offset_x = -np.cos(gate_yaw)
        spawn_offset_y = -np.sin(gate_yaw)

        self._state[env_mask] = 0.0
        self._state[env_mask, 0] = gate_pos[:, 0] + spawn_offset_x
        self._state[env_mask, 1] = gate_pos[:, 1] + spawn_offset_y
        self._state[env_mask, 2] = gate_pos[:, 2]

        # Random velocity perturbation
        vr = self.init_vel_range
        self._state[env_mask, 3:6] = self.rng.uniform(-vr, vr, size=(n_reset, 3))

        # Random attitude perturbation
        rp_rad = np.radians(self.init_rp_deg)
        self._state[env_mask, 6] = self.rng.uniform(-rp_rad, rp_rad, size=n_reset)
        self._state[env_mask, 7] = self.rng.uniform(-rp_rad, rp_rad, size=n_reset)
        self._state[env_mask, 8] = self.rng.uniform(-np.pi, np.pi, size=n_reset)

        # Random angular rate perturbation
        rr = self.init_rate_range
        self._state[env_mask, 9:12] = self.rng.uniform(-rr, rr, size=(n_reset, 3))

        # Initial motor speeds near hover
        self._state[env_mask, 12:16] = _DEFAULT_HOVER_W

        # Init prev_action
        self._prev_action[env_mask, 0] = 0.228  # hover thrust
        self._prev_action[env_mask, 1:] = 0.5   # centered roll/pitch/yaw
        self._step_count[env_mask] = 0

        # Reset episode accumulators
        self._ep_reward[env_mask] = 0.0
        self._ep_reward_components[env_mask] = 0.0
        self._ep_gates_passed[env_mask] = 0
        self._ep_laps[env_mask] = 0
        self._ep_speed_sum[env_mask] = 0.0
        self._ep_speed_count[env_mask] = 0
        self._ep_first_gate_step[env_mask] = -1
        self._step_reward_components[env_mask] = 0.0

        # Initialize signed distance for plane-crossing detection
        cur_gate_pos = self._gate_positions[self._gate_idx[env_mask]]
        cur_gate_yaw = self._gate_yaws[self._gate_idx[env_mask]]
        dp = self._state[env_mask, :3] - cur_gate_pos
        normal_x = np.cos(cur_gate_yaw)
        normal_y = np.sin(cur_gate_yaw)
        self._prev_signed_dist[env_mask] = dp[:, 0] * normal_x + dp[:, 1] * normal_y

        # Compute initial distance-to-gate
        self._d2g[env_mask] = self._compute_d2g(env_mask)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Take one environment step.

        Parameters
        ----------
        actions : (n_envs, 4), motor commands in [0, 1].

        Returns
        -------
        obs : (n_envs, 24)
        rewards : (n_envs,)
        terminated : (n_envs,) bool — crash termination
        truncated : (n_envs,) bool — timeout truncation
        info : dict with batched arrays (VecEnvAdapter-compatible)
        """
        actions = np.asarray(actions, dtype=np.float64)
        actions = np.clip(actions, 0.0, 1.0)

        # RL outputs: [thrust, roll_rate_cmd, pitch_rate_cmd, yaw_rate_cmd]
        _MAX_RATE = 6.0  # rad/s max commanded rate
        thrust_cmd = actions[:, 0:1]  # (N, 1)
        p_des = (actions[:, 1] - 0.5) * 2 * _MAX_RATE
        q_des = (actions[:, 2] - 0.5) * 2 * _MAX_RATE
        r_des = (actions[:, 3] - 0.5) * 2 * _MAX_RATE

        # PD gains
        Kp_pq = 2e-5
        Kp_r = 5e-5

        # Physics substeps
        for _substep in range(self.action_repeat):
            p_cur = self._state[:, 9]
            q_cur = self._state[:, 10]
            r_cur = self._state[:, 11]

            p_err = p_des - p_cur
            q_err = q_des - q_cur
            r_err = r_des - r_cur

            dp = Kp_pq * p_err
            dq = Kp_pq * q_err
            dr = Kp_r * r_err

            motor_cmds = np.column_stack([
                thrust_cmd[:, 0] + (-dp - dq - dr),  # M1
                thrust_cmd[:, 0] + (-dp + dq + dr),  # M2
                thrust_cmd[:, 0] + (+dp - dq + dr),  # M3
                thrust_cmd[:, 0] + (+dp + dq - dr),  # M4
            ])
            motor_cmds = np.clip(motor_cmds, 0.0, 1.0)

            # Motor model
            w_old = self._state[:, 12:16].copy()
            w_new = motor_step(w_old, motor_cmds, self.dt)
            dW = (w_new - w_old) / self.dt

            self._state[:, 12:16] = w_new

            # Dynamics
            state_T = self._state.T
            dW_T = dW.T
            deriv = self._dynamics_fn(state_T, dW_T, self._params)

            # Forward Euler
            self._state[:, :12] += deriv[:12, :].T * self.dt

        # Increment step counter
        self._step_count += 1

        # Gate passage detection
        gate_passed = self._check_gate_passage()

        # Distance to gate
        d2g_old = self._d2g.copy()
        self._d2g = self._compute_d2g()
        d2g_new = self._d2g.copy()

        # Crash / termination detection
        crashed = self._check_crash()
        timed_out = self._step_count >= self.max_steps
        terminated = crashed
        truncated = timed_out & ~crashed  # only truncated if not already crashed

        dones = terminated | truncated

        # Compute reward
        omega = self._state[:, 9:12]
        delta_action = actions - self._prev_action

        offset = np.zeros(self.n_envs)
        theta_cam = np.zeros(self.n_envs)

        rewards, step_components = compute_reward_components(
            d2g_old=d2g_old,
            d2g_new=d2g_new,
            omega=omega,
            delta_action=delta_action,
            gate_passed=gate_passed,
            crashed=crashed,
            offset=offset,
            theta_cam=theta_cam,
            preset=self.reward_preset,
            dt=self.dt * self.action_repeat,
        )

        self._prev_action = actions.copy()
        self._step_reward_components = step_components.copy()

        # Update episode accumulators
        self._ep_reward += rewards
        self._ep_reward_components += step_components
        self._ep_gates_passed += gate_passed.astype(np.int32)
        # Track lap completion
        lap_completed = gate_passed & (self._gate_idx == 0)
        self._ep_laps += lap_completed.astype(np.int32)

        # Speed tracking (velocity magnitude)
        vel = self._state[:, 3:6]
        speed = np.linalg.norm(vel, axis=-1)
        self._ep_speed_sum += speed
        self._ep_speed_count += 1

        # First gate step tracking
        just_passed = gate_passed & (self._ep_first_gate_step < 0)
        self._ep_first_gate_step[just_passed] = self._step_count[just_passed]

        # Build info dict (batched, VecEnvAdapter-compatible format)
        # Capture terminal obs BEFORE auto-reset
        terminal_obs = self._get_obs()

        # Build termination reason array
        term_reason = np.full(self.n_envs, TERM_NONE, dtype=np.int32)
        term_reason[crashed] = TERM_CRASH
        term_reason[timed_out & ~crashed] = TERM_TIMEOUT

        # Compute avg speed per env
        avg_speed = np.where(
            self._ep_speed_count > 0,
            self._ep_speed_sum / self._ep_speed_count,
            0.0,
        )

        info: dict[str, Any] = {
            "terminal_obs": terminal_obs,
            "episode": {
                "r": self._ep_reward.copy(),
                "l": self._step_count.copy(),
                "effective_dt": self.dt * self.action_repeat,
                "gates_passed": self._ep_gates_passed.copy(),
                "laps_completed": self._ep_laps.copy(),
                "termination": np.array([
                    RATE_CTRL_TERM_NAMES[int(c)] for c in term_reason
                ]),
                "success": term_reason == TERM_TIMEOUT,
                "success_criterion": "survived_full_episode",
                "avg_speed": avg_speed.copy(),
                "first_gate_step": self._ep_first_gate_step.copy(),
                "reward_components": np.array([
                    {name: float(self._ep_reward_components[i, j])
                     for j, name in enumerate(MAVLAB_REWARD_COMPONENT_NAMES)}
                    for i in range(self.n_envs)
                ], dtype=object),
            },
        }

        # Auto-reset done environments
        if np.any(dones):
            self.reset(env_mask=dones)

        # Return observation (post-reset for done envs)
        obs = self._get_obs()

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Compute 24D gate-relative observations for all envs."""
        gate_pos = self._gate_positions[self._gate_idx]
        gate_yaw = self._gate_yaws[self._gate_idx]
        next_idx = (self._gate_idx + 1) % self._n_gates
        next_gate_pos = self._gate_positions[next_idx]
        next_gate_yaw = self._gate_yaws[next_idx]

        return gate_relative_obs(
            self._state, gate_pos, gate_yaw, next_gate_pos, next_gate_yaw
        )

    def _get_obs_single(self, idx: int) -> np.ndarray:
        """Compute observation for a single env. Returns (24,)."""
        gi = self._gate_idx[idx]
        ni = (gi + 1) % self._n_gates
        return gate_relative_obs(
            self._state[idx],
            self._gate_positions[gi],
            self._gate_yaws[gi],
            self._gate_positions[ni],
            self._gate_yaws[ni],
        )

    def _compute_d2g(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Euclidean distance from each env's drone to its current gate."""
        if mask is None:
            pos = self._state[:, :3]
            gate_pos = self._gate_positions[self._gate_idx]
        else:
            pos = self._state[mask, :3]
            gate_pos = self._gate_positions[self._gate_idx[mask]]
        return np.linalg.norm(pos - gate_pos, axis=1)

    def _check_gate_passage(self) -> np.ndarray:
        """Plane-crossing gate passage detection.

        A gate is passed when the drone crosses the gate plane from behind
        (negative signed distance) to in front (positive) AND is within the
        gate opening dimensions.
        """
        gate_pos = self._gate_positions[self._gate_idx]
        gate_yaw = self._gate_yaws[self._gate_idx]

        normal_x = np.cos(gate_yaw)
        normal_y = np.sin(gate_yaw)

        dp = self._state[:, :3] - gate_pos
        signed_dist = dp[:, 0] * normal_x + dp[:, 1] * normal_y

        crossed = (self._prev_signed_dist < 0) & (signed_dist >= 0)

        gate_right_x = -np.sin(gate_yaw)
        gate_right_y = np.cos(gate_yaw)
        lateral = np.abs(dp[:, 0] * gate_right_x + dp[:, 1] * gate_right_y)
        vertical = np.abs(dp[:, 2])

        within_opening = (
            (lateral < self._track["gate_width"] / 2)
            & (vertical < self._track["gate_height"] / 2)
        )

        passed = crossed & within_opening

        # Update previous signed distance
        self._prev_signed_dist = signed_dist

        if np.any(passed):
            self._gate_idx[passed] = (self._gate_idx[passed] + 1) % self._n_gates
            new_gate_pos = self._gate_positions[self._gate_idx[passed]]
            new_gate_yaw = self._gate_yaws[self._gate_idx[passed]]
            new_dp = self._state[passed, :3] - new_gate_pos
            new_normal_x = np.cos(new_gate_yaw)
            new_normal_y = np.sin(new_gate_yaw)
            self._prev_signed_dist[passed] = (
                new_dp[:, 0] * new_normal_x + new_dp[:, 1] * new_normal_y
            )

        return passed

    def _check_crash(self) -> np.ndarray:
        """Check for out-of-bounds or extreme attitude."""
        pos = self._state[:, :3]
        euler = self._state[:, 6:9]

        oob_xy = (np.abs(pos[:, 0]) > _ARENA_RADIUS) | (
            np.abs(pos[:, 1]) > _ARENA_RADIUS
        )
        oob_z = (pos[:, 2] > _Z_FLOOR) | (pos[:, 2] < _Z_CEIL)
        tilt = (np.abs(euler[:, 0]) > _MAX_TILT) | (
            np.abs(euler[:, 1]) > _MAX_TILT
        )

        has_nan = np.any(~np.isfinite(self._state), axis=1)

        return oob_xy | oob_z | tilt | has_nan

    # --- TrajectoryProvider protocol ---

    def get_state(self, env_idx: int) -> dict[str, np.ndarray]:
        """Return current state for one environment."""
        state = self._state[env_idx]
        # Convert Euler angles to quaternion (w, x, y, z)
        phi, theta, psi = state[6], state[7], state[8]
        cr, sr = np.cos(phi / 2), np.sin(phi / 2)
        cp, sp = np.cos(theta / 2), np.sin(theta / 2)
        cy, sy = np.cos(psi / 2), np.sin(psi / 2)
        quat = np.array([
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy,  # z
        ])
        return {
            "position": state[0:3].copy(),
            "quaternion": quat,
            "velocity": state[3:6].copy(),
            "body_rates": state[9:12].copy(),
            "motor_rpms": state[12:16].copy(),
        }

    def get_gate_geometry(self) -> dict[str, np.ndarray]:
        """Return gate geometry for the track."""
        n_gates = self._n_gates
        positions = self._gate_positions.copy()
        yaws = self._gate_yaws
        orientations = np.zeros((n_gates, 4), dtype=np.float64)
        # Yaw-only quaternion: [cos(yaw/2), 0, 0, sin(yaw/2)]
        orientations[:, 0] = np.cos(yaws / 2)
        orientations[:, 3] = np.sin(yaws / 2)
        hw = self._track["gate_width"] / 2
        hh = self._track["gate_height"] / 2
        half_extents = np.full((n_gates, 2), [hw, hh], dtype=np.float64)
        return {
            "positions": positions,
            "orientations": orientations,
            "half_extents": half_extents,
        }

    def get_step_reward_components(self, env_idx: int) -> tuple[list[str], np.ndarray]:
        """Return per-step reward component breakdown."""
        return list(MAVLAB_REWARD_COMPONENT_NAMES), self._step_reward_components[env_idx].copy()
