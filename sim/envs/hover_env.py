"""Hover environment for sanity-checking dynamics and basic control.

Simplified environment where the goal is to hover at a target height (default 1m).
Obs = full 17-dim state, reward = -distance_to_target_height.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from sim.dynamics.numpy_quad import NumpyQuadDynamics
from sim.dynamics.params import VehicleParams

STATE_DIM = 17


class HoverEnv(gym.Env):
    """Gymnasium environment for quadrotor hovering.

    A minimal environment for sanity-checking the dynamics model.
    The drone starts at the target height and the goal is to stay there.

    Observation (17-dim): full state vector
        [pos(3), vel(3), quat(4), omega(3), motor_speeds(4)]

    Action (4-dim): motor RPM commands, continuous, clipped to [0, max_rpm].

    Reward: -|z - target_height| (negative distance to target altitude).

    Args:
        params: Vehicle parameters. Uses defaults if None.
        target_height: Target hover altitude in meters.
        dt: Simulation timestep in seconds.
        max_steps: Maximum steps per episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: VehicleParams | None = None,
        target_height: float = 1.0,
        dt: float = 0.01,
        max_steps: int = 1000,
    ) -> None:
        super().__init__()

        self.target_height = target_height
        self.max_steps = max_steps

        self.params = params or VehicleParams()
        self.dynamics = NumpyQuadDynamics(params=self.params, dt=dt)

        max_rpm = self.params.max_rpm

        # Gymnasium spaces
        obs_high = np.full(STATE_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.full(4, max_rpm, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (single env)
        self._state: NDArray[np.float64] = np.zeros((1, 17))
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        self._state = self.dynamics.reset(1)
        # Set initial height to target
        self._state[0, 2] = self.target_height
        self._step_count = 0

        obs = self._state[0].astype(np.float32)
        return obs, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Step the environment forward.

        Args:
            action: Motor RPM commands (4,).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            action = action[None, :]  # (1, 4)

        # Convert RPM to rad/s
        action_rads = action * 2.0 * np.pi / 60.0

        self._state = self.dynamics.step(self._state, action_rads)
        self._step_count += 1

        z = self._state[0, 2]
        reward = -abs(z - self.target_height)

        # Termination: ground crash
        terminated = bool(z <= 0.0)
        # Truncation: timeout
        truncated = bool(self._step_count >= self.max_steps)

        if terminated:
            reward = -10.0

        obs = self._state[0].astype(np.float32)
        return obs, float(reward), terminated, truncated, {}
