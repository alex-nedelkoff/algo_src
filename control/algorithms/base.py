"""Abstract base class for RL training algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gymnasium as gym


class Algorithm(ABC):
    """Abstract RL training algorithm."""

    @abstractmethod
    def train(self, env: gym.Env, total_timesteps: int, **kwargs: Any) -> None:
        """Train the policy on the given environment."""

    @abstractmethod
    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """Predict an action for the given observation."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save a training checkpoint."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load a training checkpoint."""

    @abstractmethod
    def export_onnx(self, path: str | Path) -> Path:
        """Export the trained policy to ONNX format."""
