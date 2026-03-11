"""Tests for GCNet policy network."""

from __future__ import annotations

import pytest
import torch

from control.policies.gcnet import GCNet, GCNetExtractor


class TestGCNetForwardPass:
    """Test GCNet forward pass produces correct output shapes."""

    def test_default_output_shape(self) -> None:
        model = GCNet()
        x = torch.randn(32, 24)
        out = model(x)
        assert out.shape == (32, 4)

    def test_single_sample(self) -> None:
        model = GCNet()
        x = torch.randn(1, 24)
        out = model(x)
        assert out.shape == (1, 4)

    def test_custom_dims(self) -> None:
        model = GCNet(obs_dim=17, action_dim=4, hidden_dims=(64, 64))
        x = torch.randn(8, 17)
        out = model(x)
        assert out.shape == (8, 4)

    def test_output_is_finite(self) -> None:
        model = GCNet()
        x = torch.randn(16, 24)
        out = model(x)
        assert torch.isfinite(out).all()


class TestGCNetParamCount:
    """Test GCNet parameter count is in the 25K-35K range."""

    def test_default_param_count_in_range(self) -> None:
        model = GCNet()
        n_params = sum(p.numel() for p in model.parameters())
        assert 25_000 <= n_params <= 35_000, f"Param count {n_params} not in [25K, 35K]"

    def test_param_count_exact(self) -> None:
        model = GCNet(obs_dim=24, action_dim=4, hidden_dims=(128, 128, 64))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 28_228


class TestGCNetAsymmetric:
    """Test asymmetric actor-critic variant."""

    def test_actor_standard_obs(self) -> None:
        actor = GCNet(obs_dim=24, action_dim=4)
        x = torch.randn(8, 24)
        out = actor(x)
        assert out.shape == (8, 4)

    def test_critic_privileged_obs(self) -> None:
        critic = GCNet(obs_dim=48, action_dim=1, hidden_dims=(128, 128, 64))
        x = torch.randn(8, 48)
        out = critic(x)
        assert out.shape == (8, 1)

    def test_asymmetric_different_input_dims(self) -> None:
        actor = GCNet(obs_dim=24, action_dim=4)
        critic = GCNet(obs_dim=48, action_dim=1)
        obs_actor = torch.randn(4, 24)
        obs_critic = torch.randn(4, 48)
        assert actor(obs_actor).shape == (4, 4)
        assert critic(obs_critic).shape == (4, 1)


class TestGCNetGradients:
    """Test gradient flow through all layers."""

    def test_gradients_flow(self) -> None:
        model = GCNet()
        x = torch.randn(4, 24)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_all_layers_have_params(self) -> None:
        model = GCNet()
        param_names = [n for n, _ in model.named_parameters()]
        # 3 hidden + 1 output = 4 linear layers, each with weight + bias
        assert len(param_names) == 8


class TestGCNetArchitectureLayers:
    """Test GCNet internal layer structure matches expectations."""

    def test_default_layer_sizes(self) -> None:
        model = GCNet()
        linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        sizes = [(l.in_features, l.out_features) for l in linears]
        assert sizes == [(24, 128), (128, 128), (128, 64), (64, 4)]

    def test_all_hidden_activations_are_relu(self) -> None:
        model = GCNet()
        activations = [m for m in model.net if not isinstance(m, torch.nn.Linear)]
        assert len(activations) == 3  # one per hidden layer, none after output
        assert all(isinstance(a, torch.nn.ReLU) for a in activations)

    def test_no_activation_after_output(self) -> None:
        model = GCNet()
        last_module = list(model.net)[-1]
        assert isinstance(last_module, torch.nn.Linear)

    def test_custom_hidden_dims(self) -> None:
        model = GCNet(obs_dim=20, action_dim=4, hidden_dims=(64, 64, 64))
        linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        sizes = [(l.in_features, l.out_features) for l in linears]
        assert sizes == [(20, 64), (64, 64), (64, 64), (64, 4)]


class TestGCNetExtractor:
    """Test GCNetExtractor for SB3 integration."""

    def test_features_dim(self) -> None:
        from gymnasium import spaces
        import numpy as np

        obs_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
        extractor = GCNetExtractor(obs_space)
        assert extractor.features_dim == 64

    def test_forward_shape(self) -> None:
        from gymnasium import spaces
        import numpy as np

        obs_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
        extractor = GCNetExtractor(obs_space)
        x = torch.randn(8, 24)
        features = extractor(x)
        assert features.shape == (8, 64)

    def test_extractor_layer_sizes(self) -> None:
        """Extractor should have hidden layers only (no action head)."""
        from gymnasium import spaces
        import numpy as np

        obs_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float32)
        extractor = GCNetExtractor(obs_space)
        linears = [m for m in extractor._backbone.net if isinstance(m, torch.nn.Linear)]
        sizes = [(l.in_features, l.out_features) for l in linears]
        assert sizes == [(24, 128), (128, 128), (128, 64)]
