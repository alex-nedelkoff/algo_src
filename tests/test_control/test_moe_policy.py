"""Tests for Mixture of Experts policy."""
import numpy as np
import pytest
import torch
from gymnasium import spaces


class TestRouter:
    def test_output_shape(self):
        from control.policies.moe_policy import Router
        router = Router(obs_dim=28, n_experts=4, hidden_dim=64)
        obs = torch.randn(8, 28)
        weights = router(obs)
        assert weights.shape == (8, 4)

    def test_weights_sum_to_one(self):
        from control.policies.moe_policy import Router
        router = Router(obs_dim=28, n_experts=4, hidden_dim=64)
        obs = torch.randn(8, 28)
        weights = router(obs)
        sums = weights.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=1e-5)

    def test_weights_non_negative(self):
        from control.policies.moe_policy import Router
        router = Router(obs_dim=28, n_experts=4, hidden_dim=64)
        obs = torch.randn(8, 28)
        weights = router(obs)
        assert (weights >= 0).all()


class TestExpertNetwork:
    def test_output_shape(self):
        from control.policies.moe_policy import ExpertNetwork
        expert = ExpertNetwork(obs_dim=28, action_dim=4, hidden_dim=128)
        obs = torch.randn(8, 28)
        actions = expert(obs)
        assert actions.shape == (8, 4)

    def test_different_experts_different_outputs(self):
        from control.policies.moe_policy import ExpertNetwork
        e1 = ExpertNetwork(obs_dim=28, action_dim=4, hidden_dim=128)
        e2 = ExpertNetwork(obs_dim=28, action_dim=4, hidden_dim=128)
        obs = torch.randn(1, 28)
        assert not torch.allclose(e1(obs), e2(obs))


class TestTopKSelection:
    def test_top2_selects_two(self):
        from control.policies.moe_policy import top_k_selection
        weights = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
        selected_weights, selected_indices = top_k_selection(weights, k=2)
        assert selected_indices.shape == (1, 2)
        assert selected_weights.shape == (1, 2)

    def test_top2_renormalized(self):
        from control.policies.moe_policy import top_k_selection
        weights = torch.tensor([[0.5, 0.3, 0.1, 0.1]])
        selected_weights, _ = top_k_selection(weights, k=2)
        torch.testing.assert_close(selected_weights.sum(dim=1), torch.ones(1), atol=1e-5, rtol=1e-5)

    def test_top2_picks_highest(self):
        from control.policies.moe_policy import top_k_selection
        weights = torch.tensor([[0.1, 0.5, 0.3, 0.1]])
        _, indices = top_k_selection(weights, k=2)
        assert 1 in indices[0]
        assert 2 in indices[0]


class TestLoadBalanceLoss:
    def test_uniform_usage_zero_loss(self):
        from control.policies.moe_policy import load_balance_loss
        indices = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
        loss = load_balance_loss(indices, n_experts=4)
        assert loss.item() < 0.01

    def test_imbalanced_usage_nonzero_loss(self):
        from control.policies.moe_policy import load_balance_loss
        indices = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        loss = load_balance_loss(indices, n_experts=4)
        assert loss.item() > 0.01
