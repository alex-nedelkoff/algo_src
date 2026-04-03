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
        return torch.tanh(self.net(obs))  # Tanh keeps means in [-1, 1]


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
    Optional GRU layer for temporal context (gru_hidden_dim > 0).
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
        gru_hidden_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        self.n_experts = n_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.router_hidden_dim = router_hidden_dim
        self.top_k = top_k
        self.balance_coef = balance_coef
        self.gru_hidden_dim = gru_hidden_dim
        self._last_balance_loss = None
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        # Optional GRU for temporal context
        if self.gru_hidden_dim > 0:
            self.gru = nn.GRU(
                input_size=obs_dim,
                hidden_size=self.gru_hidden_dim,
                batch_first=True,
            )
            self._gru_hidden: Optional[torch.Tensor] = None
            enriched_dim = obs_dim + self.gru_hidden_dim
        else:
            self.gru = None
            enriched_dim = obs_dim

        self.router = Router(enriched_dim, self.n_experts, self.router_hidden_dim)

        self.experts = nn.ModuleList([
            ExpertNetwork(enriched_dim, action_dim, self.expert_hidden_dim)
            for _ in range(self.n_experts)
        ])

        self.value_net = nn.Sequential(
            nn.Linear(enriched_dim, self.expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.expert_hidden_dim, self.expert_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.expert_hidden_dim, 1),
        )

        self.action_dist = DiagGaussianDistribution(action_dim)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), self.log_std_init),
            requires_grad=True,
        )
        # Cap for log_std: prevent runaway exploration
        self._log_std_max = 0.0  # std <= 1.0
        self._log_std_min = -3.0  # std >= 0.05

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _clamped_log_std(self) -> torch.Tensor:
        """Return log_std clamped to valid range."""
        return torch.clamp(self.log_std, self._log_std_min, self._log_std_max)

    def _enrich_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Concatenate GRU context to obs if GRU is enabled.

        During prediction (single step): updates hidden state and returns
        [obs, gru_output]. During evaluation (full rollout batch): runs GRU
        without persistent state (each sample is independent in the buffer).
        """
        if self.gru is None:
            return obs

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Shape: (batch, 1, obs_dim) for single-step GRU
        gru_input = obs.unsqueeze(1)

        if self.training or self._gru_hidden is None or self._gru_hidden.shape[1] != obs.shape[0]:
            # During training (evaluate_actions on rollout buffer) or batch size mismatch:
            # run without persistent hidden state
            gru_out, _ = self.gru(gru_input)
        else:
            # During prediction (collect_rollouts): use persistent hidden state
            gru_out, self._gru_hidden = self.gru(gru_input, self._gru_hidden)

        gru_context = gru_out.squeeze(1)  # (batch, gru_hidden_dim)
        return torch.cat([obs, gru_context], dim=-1)

    def reset_gru_hidden(self, env_indices: list[int] | None = None) -> None:
        """Reset GRU hidden state for specified envs (or all if None)."""
        if self.gru is None:
            return
        if env_indices is None or self._gru_hidden is None:
            self._gru_hidden = None
        else:
            for idx in env_indices:
                if idx < self._gru_hidden.shape[1]:
                    self._gru_hidden[0, idx] = 0.0

    def _get_moe_action_mean(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute MoE blended action mean. Expects raw obs (will enrich)."""
        return self._get_moe_action_mean_from_enriched(self._enrich_obs(obs))

    def _get_moe_action_mean_from_enriched(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute MoE blended action mean from already-enriched obs."""
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
        enriched = self._enrich_obs(obs)
        action_mean, _, selected_indices = self._get_moe_action_mean_from_enriched(enriched)
        values = self.value_net(enriched)
        # Store enriched for evaluate_actions consistency

        distribution = self.action_dist.proba_distribution(
            action_mean, self._clamped_log_std().expand_as(action_mean)
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
            action_mean, self._clamped_log_std().expand_as(action_mean)
        )
        return distribution.get_actions(deterministic=deterministic)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict state values (for bootstrapping)."""
        enriched = self._enrich_obs(obs)
        return self.value_net(enriched).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        enriched = self._enrich_obs(obs)
        action_mean, _, selected_indices = self._get_moe_action_mean_from_enriched(enriched)
        values = self.value_net(enriched)

        distribution = self.action_dist.proba_distribution(
            action_mean, self._clamped_log_std().expand_as(action_mean)
        )
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        self._last_balance_loss = (
            self.balance_coef * load_balance_loss(selected_indices, self.n_experts)
        )

        return values.squeeze(-1), log_prob, entropy
