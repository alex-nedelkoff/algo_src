"""Observation transforms for RL training.

Transforms raw simulation state into policy-consumable observation vectors.
Supports both privileged (full state, for asymmetric critic) and sensor-like
(noisy, delayed, for actor) observations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ObservationTransform(ABC):
    """Base class for observation transforms.

    Subclasses implement transform() to convert raw state + track info
    into a fixed-size observation vector suitable for a neural network.
    """

    @abstractmethod
    def transform(
        self,
        state: NDArray[np.float64],
        track_info: dict[str, Any],
    ) -> NDArray[np.float64]:
        """Transform a single environment's state into an observation.

        Args:
            state: Raw state vector (17,).
            track_info: Dictionary with track-related info such as
                'gate_pos', 'next_gate_pos', 'gate_progress', etc.

        Returns:
            Observation vector of fixed size.
        """

    @abstractmethod
    def obs_size(self) -> int:
        """Return the dimensionality of the observation vector."""


class PrivilegedObservation(ObservationTransform):
    """Full-state observation for asymmetric critic training.

    Returns the complete 17-dim state vector without any noise or delay.
    This is used for the value function (critic) in asymmetric actor-critic
    setups where the critic has access to privileged information.
    """

    def transform(
        self,
        state: NDArray[np.float64],
        track_info: dict[str, Any],
    ) -> NDArray[np.float64]:
        """Return the full state vector as observation.

        Args:
            state: Raw state vector (17,).
            track_info: Unused for privileged observation.

        Returns:
            Full state vector (17,).
        """
        return state.copy()

    def obs_size(self) -> int:
        """Return observation dimensionality (17)."""
        return 17


class SensorObservation(ObservationTransform):
    """Noisy, delayed sensor observation for actor training.

    Simulates realistic sensor limitations:
    - Gaussian noise on position and velocity
    - Delay buffer for simulating sensor latency
    - No direct access to motor speeds (must be estimated)

    Args:
        pos_noise_std: Standard deviation of position noise (m).
        vel_noise_std: Standard deviation of velocity noise (m/s).
        omega_noise_std: Standard deviation of angular rate noise (rad/s).
        delay_steps: Number of timesteps of sensor delay.
        rng: Random number generator.
    """

    def __init__(
        self,
        pos_noise_std: float = 0.01,
        vel_noise_std: float = 0.05,
        omega_noise_std: float = 0.02,
        delay_steps: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.omega_noise_std = omega_noise_std
        self.delay_steps = delay_steps
        self._rng = rng or np.random.default_rng()
        self._buffer: list[NDArray[np.float64]] = []

    def transform(
        self,
        state: NDArray[np.float64],
        track_info: dict[str, Any],
    ) -> NDArray[np.float64]:
        """Return a noisy, delayed subset of the state.

        Observation layout (13,):
            [0:3]  noisy position
            [3:6]  noisy velocity
            [6:10] quaternion (no noise -- from IMU integration)
            [10:13] noisy angular rate

        Args:
            state: Raw state vector (17,).
            track_info: Unused for sensor observation.

        Returns:
            Noisy observation vector (13,).
        """
        noisy_state = state[:13].copy()

        # Add noise
        noisy_state[0:3] += self._rng.normal(0, self.pos_noise_std, 3)
        noisy_state[3:6] += self._rng.normal(0, self.vel_noise_std, 3)
        noisy_state[10:13] += self._rng.normal(0, self.omega_noise_std, 3)

        # Delay buffer
        self._buffer.append(noisy_state)
        if len(self._buffer) > self.delay_steps:
            return self._buffer.pop(0)
        # If not enough history yet, return the current (noisy) reading
        return noisy_state

    def obs_size(self) -> int:
        """Return observation dimensionality (13)."""
        return 13

    def reset(self) -> None:
        """Clear the delay buffer on episode reset."""
        self._buffer.clear()
