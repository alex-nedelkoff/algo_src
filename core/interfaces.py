"""Abstract base classes defining the core interfaces for the drone racing stack."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from core.types import Action, GateState, Observation, QuadState


class Dynamics(ABC):
    """Abstract dynamics model for quadrotor simulation."""

    @abstractmethod
    def reset(self, state: QuadState | None = None) -> QuadState:
        """Reset the simulation to an initial state.

        Args:
            state: Optional initial state. If None, use a default.

        Returns:
            The initial QuadState after reset.
        """

    @abstractmethod
    def step(self, state: QuadState, action: Action, dt: float) -> QuadState:
        """Advance the simulation by one timestep.

        Args:
            state: Current quadrotor state.
            action: Control action to apply.
            dt: Timestep duration in seconds.

        Returns:
            The new QuadState after applying the action.
        """


class Policy(ABC):
    """Abstract policy that maps observations to actions."""

    @abstractmethod
    def predict(self, observation: Observation) -> Action:
        """Predict an action given an observation.

        Args:
            observation: Current observation including state and sensor data.

        Returns:
            The action to execute.
        """


class Detector(ABC):
    """Abstract gate detector for the perception pipeline."""

    @abstractmethod
    def detect(self, image: NDArray[np.uint8]) -> list[GateState]:
        """Detect gates in an image.

        Args:
            image: Input image as a uint8 numpy array (H, W, C).

        Returns:
            List of detected gate states with estimated positions and orientations.
        """


class RewardFn(ABC):
    """Abstract reward function for RL training."""

    @abstractmethod
    def compute(
        self,
        state: QuadState,
        action: Action,
        next_state: QuadState,
        gate_state: GateState,
    ) -> float:
        """Compute the reward for a transition.

        Args:
            state: State before the action.
            action: Action taken.
            next_state: State after the action.
            gate_state: Current target gate state.

        Returns:
            Scalar reward value.
        """
