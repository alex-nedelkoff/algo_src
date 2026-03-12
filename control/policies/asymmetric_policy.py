"""Asymmetric actor-critic feature extractors for SB3.

Splits a 50D observation into actor features (noisy obs + metadata)
and critic features (clean obs).

Observation layout:
  [0:24]  noisy gate-relative obs  -> Actor
  [24:48] clean gate-relative obs  -> Critic
  [48:50] perception metadata      -> Actor
"""
from __future__ import annotations

from typing import Any, Union

import gymnasium
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device


class ActorExtractor(BaseFeaturesExtractor):
    """Extracts actor features: noisy obs [0:24] + metadata [48:50] = 26D."""

    def __init__(self, observation_space: gymnasium.spaces.Box) -> None:
        super().__init__(observation_space, features_dim=26)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [observations[:, :24], observations[:, 48:50]], dim=1
        )


class CriticExtractor(BaseFeaturesExtractor):
    """Extracts critic features: clean obs [24:48] = 24D."""

    def __init__(self, observation_space: gymnasium.spaces.Box) -> None:
        super().__init__(observation_space, features_dim=24)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, 24:48]


class AsymmetricMlpExtractor(nn.Module):
    """MlpExtractor variant that supports different input dims for pi and vf.

    When ``pi_feature_dim != vf_feature_dim``, the policy and value networks
    are built with their own input dimensions.  Otherwise behaves identically
    to :class:`stable_baselines3.common.torch_layers.MlpExtractor`.
    """

    def __init__(
        self,
        pi_feature_dim: int,
        vf_feature_dim: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # Build policy network
        policy_net: list[nn.Module] = []
        last_pi = pi_feature_dim
        for dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_pi, dim))
            policy_net.append(activation_fn())
            last_pi = dim

        # Build value network
        value_net: list[nn.Module] = []
        last_vf = vf_feature_dim
        for dim in vf_layers_dims:
            value_net.append(nn.Linear(last_vf, dim))
            value_net.append(activation_fn())
            last_vf = dim

        self.latent_dim_pi = last_pi
        self.latent_dim_vf = last_vf

        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class AsymmetricPolicy(ActorCriticPolicy):
    """MlpPolicy that uses ActorExtractor for pi and CriticExtractor for vf.

    Pass this as the policy class to PPO instead of ``"MlpPolicy"``::

        model = PPO(AsymmetricPolicy, env, ...)
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        lr_schedule: Schedule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Force the extractors before the parent __init__ builds the network.
        kwargs["features_extractor_class"] = ActorExtractor
        kwargs["features_extractor_kwargs"] = {}
        kwargs["share_features_extractor"] = False
        super().__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )

    def _setup_extractors(self) -> None:
        """Replace the vf extractor with CriticExtractor after parent init."""
        self.vf_features_extractor = CriticExtractor(
            self.observation_space
        ).to(self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        # Set up extractors first so dims are known
        self._setup_extractors()
        # Now build with correct dims
        super()._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        """Build MLP with different input dims for pi (26D) and vf (24D)."""
        pi_dim = self.pi_features_extractor.features_dim
        vf_dim = self.vf_features_extractor.features_dim
        self.mlp_extractor = AsymmetricMlpExtractor(
            pi_feature_dim=pi_dim,
            vf_feature_dim=vf_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
