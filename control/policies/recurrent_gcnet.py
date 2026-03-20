"""Recurrent GCNet — MLP encoder for sb3-contrib RecurrentPPO.

Provides a feature extractor that feeds into sb3-contrib's LSTM layer.
The LSTM hidden state management is handled entirely by RecurrentPPO —
this module only needs to encode observations into a feature vector.

Architecture:
    obs -> MLP [hidden_dims, ReLU] -> features (features_dim)
    features -> LSTM (managed by RecurrentPPO) -> Actor/Critic heads
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


if _AVAILABLE:

    class RecurrentGCNetExtractor(BaseFeaturesExtractor):
        """MLP encoder for RecurrentPPO.

        Two-layer MLP that encodes observations into a fixed-size feature
        vector. The LSTM layer sits after this extractor (managed by
        sb3-contrib's RecurrentPPO, not by this class).

        Args:
            observation_space: Gymnasium Box observation space.
            hidden_dims: MLP hidden layer sizes. Default (64, 64).
        """

        def __init__(
            self,
            observation_space: spaces.Box,
            hidden_dims: tuple[int, ...] = (64, 64),
        ) -> None:
            features_dim = hidden_dims[-1]
            super().__init__(observation_space, features_dim=features_dim)

            obs_dim = observation_space.shape[0]
            layers: list[nn.Module] = []
            in_dim = obs_dim
            for h in hidden_dims:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self._encoder = nn.Sequential(*layers)

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self._encoder(observations)

else:

    class RecurrentGCNetExtractor:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "RecurrentGCNetExtractor requires torch + stable-baselines3. "
                "Install with: pip install 'algo-src[control]'"
            )
