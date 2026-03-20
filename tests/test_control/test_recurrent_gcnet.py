"""Tests for recurrent GCNet policy components."""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from control.policies.recurrent_gcnet import RecurrentGCNetExtractor


class TestRecurrentGCNetExtractor:
    @pytest.fixture
    def obs_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

    def test_output_shape(self, obs_space):
        ext = RecurrentGCNetExtractor(obs_space)
        obs = torch.randn(4, 32)
        out = ext(obs)
        assert out.shape == (4, 64)

    def test_features_dim(self, obs_space):
        ext = RecurrentGCNetExtractor(obs_space)
        assert ext.features_dim == 64

    def test_custom_hidden_dims(self, obs_space):
        ext = RecurrentGCNetExtractor(obs_space, hidden_dims=(128, 128))
        obs = torch.randn(2, 32)
        out = ext(obs)
        assert out.shape == (2, 128)
        assert ext.features_dim == 128

    def test_different_obs_dim(self):
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        ext = RecurrentGCNetExtractor(obs_space, hidden_dims=(64, 64))
        obs = torch.randn(3, 28)
        out = ext(obs)
        assert out.shape == (3, 64)
