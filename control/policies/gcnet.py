"""GCNet — Guidance and Control Network for direct motor RPM output.

A compact MLP policy (~10K params) that maps observations directly to 4 motor
RPM commands, bypassing PID controllers entirely.  Based on the MonoRace M23
architecture (3×64 ReLU) that achieves 500 Hz inference on an STM32.

This module requires ``torch`` and ``stable-baselines3``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

__all__ = ["GCNet", "GCNetExtractor", "TORCH_AVAILABLE"]

TORCH_AVAILABLE = _TORCH_AVAILABLE


def _check_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "GCNet requires PyTorch.  Install with:  pip install 'algo-src[control]'"
        )


if _TORCH_AVAILABLE:

    class GCNet(nn.Module):
        """Guidance & Control Network: compact MLP for motor RPM output.

        Default architecture: obs_dim -> 64 -> 64 -> 64 -> action_dim (~10K params).
        Matches MonoRace M23 policy (3×64, ReLU).
        All hidden layers use ReLU activation.

        Args:
            obs_dim: Observation dimension (default 24 for GateRaceEnv).
            action_dim: Action dimension (default 4 motor RPMs).
            hidden_dims: Hidden layer sizes.
        """

        def __init__(
            self,
            obs_dim: int = 24,
            action_dim: int = 4,
            hidden_dims: tuple[int, ...] = (64, 64, 64),
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = obs_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class _GCNetFeatures(nn.Module):
        """Hidden layers of GCNet used as SB3 feature extractor."""

        def __init__(
            self,
            obs_dim: int,
            hidden_dims: tuple[int, ...],
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = obs_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.net = nn.Sequential(*layers)
            self.output_dim = hidden_dims[-1]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    if _SB3_AVAILABLE:

        class GCNetExtractor(BaseFeaturesExtractor):
            """SB3-compatible feature extractor using GCNet hidden layers.

            Extracts features through the GCNet hidden layers, leaving the
            action/value heads to SB3's ActorCriticPolicy.

            Args:
                observation_space: Gymnasium observation space.
                hidden_dims: Hidden layer sizes for the GCNet backbone.
            """

            def __init__(
                self,
                observation_space: spaces.Box,
                hidden_dims: tuple[int, ...] = (64, 64, 64),
            ) -> None:
                features_dim = hidden_dims[-1]
                super().__init__(observation_space, features_dim=features_dim)
                obs_dim = observation_space.shape[0]
                self._backbone = _GCNetFeatures(obs_dim, hidden_dims)

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                return self._backbone(observations)

    else:

        class GCNetExtractor:  # type: ignore[no-redef]
            """Stub — stable-baselines3 is not installed."""

            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                raise ImportError(
                    "GCNetExtractor requires stable-baselines3.  "
                    "Install with:  pip install 'algo-src[control]'"
                )

else:

    class GCNet:  # type: ignore[no-redef]
        """Stub — PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            _check_torch()

    class GCNetExtractor:  # type: ignore[no-redef]
        """Stub — PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            _check_torch()
