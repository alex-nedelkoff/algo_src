"""Mixture of Experts policy for drone racing.

4 expert networks with a top-2 router. Each expert produces TRPY actions.
The router selects the 2 most relevant experts per step and blends their
outputs. A separate critic estimates state value.

Spec: docs/superpowers/specs/2026-03-23-moe-policy-design.md
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule


class Router(nn.Module):
    """Routes observations to experts via softmax weights."""

    def __init__(self, obs_dim: int, n_experts: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(obs), dim=-1)


class ExpertNetwork(nn.Module):
    """Single expert: obs -> action."""

    def __init__(self, obs_dim: int, action_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def top_k_selection(
    weights: torch.Tensor, k: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k experts and renormalize weights."""
    top_weights, top_indices = torch.topk(weights, k, dim=-1)
    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    return top_weights, top_indices


def load_balance_loss(
    selected_indices: torch.Tensor, n_experts: int = 4
) -> torch.Tensor:
    """Compute load balance loss: L = sum((f_i - 1/N)^2)."""
    counts = torch.zeros(n_experts, device=selected_indices.device)
    for i in range(n_experts):
        counts[i] = (selected_indices == i).float().sum()
    total_selections = selected_indices.numel()
    fractions = counts / max(total_selections, 1)
    target = 1.0 / n_experts
    return ((fractions - target) ** 2).sum()


class MoEPolicy(ActorCriticPolicy):
    """Mixture of Experts policy for SB3 PPO.

    4 expert networks + router with top-2 selection.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        n_experts: int = 4,
        expert_hidden_dim: int = 128,
        router_hidden_dim: int = 64,
        top_k: int = 2,
        balance_coef: float = 0.01,
        **kwargs: Any,
    ) -> None:
        self.n_experts = n_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.router_hidden_dim = router_hidden_dim
        self.top_k = top_k
        self.balance_coef = balance_coef
        self._last_balance_loss = None
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        self.router = Router(obs_dim, self.n_experts, self.router_hidden_dim)

        self.experts = nn.ModuleList([
            ExpertNetwork(obs_dim, action_dim, self.expert_hidden_dim)
            for _ in range(self.n_experts)
        ])

        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, self.expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.expert_hidden_dim, self.expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.expert_hidden_dim, 1),
        )

        self.action_dist = DiagGaussianDistribution(action_dim)
        self.log_std = nn.Parameter(
            torch.zeros(action_dim) * self.log_std_init,
            requires_grad=True,
        )

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_moe_action_mean(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_weights = self.router(obs)
        selected_weights, selected_indices = top_k_selection(
            router_weights, self.top_k
        )

        all_expert_actions = torch.stack(
            [expert(obs) for expert in self.experts], dim=1
        )

        action_dim = all_expert_actions.shape[2]
        idx_expanded = selected_indices.unsqueeze(-1).expand(-1, -1, action_dim)
        selected_actions = torch.gather(all_expert_actions, 1, idx_expanded)

        weights_expanded = selected_weights.unsqueeze(-1)
        action_mean = (selected_actions * weights_expanded).sum(dim=1)

        return action_mean, selected_weights, selected_indices

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, _, selected_indices = self._get_moe_action_mean(obs)
        values = self.value_net(obs)

        distribution = self.action_dist.proba_distribution(
            action_mean, self.log_std.expand_as(action_mean)
        )
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        self._last_balance_loss = (
            self.balance_coef * load_balance_loss(selected_indices, self.n_experts)
        )

        return actions, values.squeeze(-1), log_prob

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        action_mean, _, _ = self._get_moe_action_mean(observation)
        distribution = self.action_dist.proba_distribution(
            action_mean, self.log_std.expand_as(action_mean)
        )
        return distribution.get_actions(deterministic=deterministic)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict state values (for bootstrapping)."""
        return self.value_net(obs).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        action_mean, _, selected_indices = self._get_moe_action_mean(obs)
        values = self.value_net(obs)

        distribution = self.action_dist.proba_distribution(
            action_mean, self.log_std.expand_as(action_mean)
        )
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        self._last_balance_loss = (
            self.balance_coef * load_balance_loss(selected_indices, self.n_experts)
        )

        return values.squeeze(-1), log_prob, entropy
