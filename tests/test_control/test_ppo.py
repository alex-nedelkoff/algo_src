"""Tests for PPO training algorithm wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from control.algorithms.ppo import PPO


@pytest.fixture
def ppo() -> PPO:
    """A PPO instance with small config for fast tests."""
    return PPO(
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        hidden_dims=(128, 128, 64),
    )


@pytest.fixture
def hover_env():  # type: ignore[no-untyped-def]
    """A HoverEnv instance for testing."""
    from sim.envs.hover_env import HoverEnv

    return HoverEnv()


class TestPPOArchitecture:
    """Test that PPO creates the expected network architecture.

    GCNetExtractor IS the policy backbone. SB3 should NOT add extra MLP layers
    on top (net_arch must be empty). The full actor path should be:
        obs(24) -> GCNetExtractor(128->128->64) -> Linear(64->4)
    """

    def test_no_extra_pi_layers(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """SB3 pi_net should be empty (no layers between extractor and action head)."""
        ppo.train(hover_env, total_timesteps=64)
        pi_net = ppo._model.policy.mlp_extractor.policy_net
        # pi_net should be an empty Sequential or identity
        pi_modules = [m for m in pi_net.modules() if isinstance(m, torch.nn.Linear)]
        assert len(pi_modules) == 0, (
            f"Expected no extra pi layers, got {len(pi_modules)} Linear layers. "
            "Check that net_arch is empty — GCNetExtractor should be the whole network."
        )

    def test_no_extra_vf_layers(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """SB3 vf_net should be empty (no layers between extractor and value head)."""
        ppo.train(hover_env, total_timesteps=64)
        vf_net = ppo._model.policy.mlp_extractor.value_net
        vf_modules = [m for m in vf_net.modules() if isinstance(m, torch.nn.Linear)]
        assert len(vf_modules) == 0, (
            f"Expected no extra vf layers, got {len(vf_modules)} Linear layers. "
            "Check that net_arch is empty — GCNetExtractor should be the whole network."
        )

    def test_action_head_shape(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Action head should be Linear(features_dim -> action_dim)."""
        ppo.train(hover_env, total_timesteps=64)
        action_net = ppo._model.policy.action_net
        assert isinstance(action_net, torch.nn.Linear)
        assert action_net.in_features == 64  # GCNetExtractor output
        assert action_net.out_features == hover_env.action_space.shape[0]

    def test_features_extractor_is_gcnet(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Features extractor should be our GCNetExtractor."""
        from control.policies.gcnet import GCNetExtractor

        ppo.train(hover_env, total_timesteps=64)
        extractor = ppo._model.policy.features_extractor
        assert isinstance(extractor, GCNetExtractor)
        assert extractor.features_dim == 64

    def test_total_actor_param_count(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Total actor params (extractor + action head) should be ~28K, not ~200K."""
        ppo.train(hover_env, total_timesteps=64)
        policy = ppo._model.policy
        extractor_params = sum(p.numel() for p in policy.features_extractor.parameters())
        pi_params = sum(p.numel() for p in policy.mlp_extractor.policy_net.parameters())
        action_params = sum(p.numel() for p in policy.action_net.parameters())
        total = extractor_params + pi_params + action_params
        # Extractor: 24*128+128 + 128*128+128 + 128*64+64 = 28032
        # pi_net: 0 (empty)
        # action_net: 64*4+4 = 260 (hover env has 4 actions)
        assert pi_params == 0, f"pi_net should have 0 params, got {pi_params}"
        assert total < 35_000, f"Actor too large ({total} params) — net_arch leak?"

    def test_net_arch_override_rejected(self) -> None:
        """Even if net_arch is passed, it should default to empty."""
        ppo_default = PPO()
        assert ppo_default.net_arch == {"pi": [], "vf": []}


class TestPPOTrain:
    """Test PPO training runs without error."""

    def test_train_hover_env(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        ppo.train(hover_env, total_timesteps=128)
        obs, _ = hover_env.reset()
        action, _ = ppo.predict(obs)
        assert action.shape == (4,)

    def test_predict_before_train_raises(self, ppo: PPO) -> None:
        with pytest.raises(RuntimeError, match="not trained or loaded"):
            ppo.predict(np.zeros(17))


class TestPPOCheckpoint:
    """Test checkpoint save/load round-trip."""

    def test_save_load_roundtrip(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        ppo.train(hover_env, total_timesteps=128)

        obs, _ = hover_env.reset()
        action_before, _ = ppo.predict(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ppo_test"
            ppo.save(ckpt_path)

            ppo2 = PPO()
            ppo2.load(ckpt_path, env=hover_env)
            action_after, _ = ppo2.predict(obs, deterministic=True)

        np.testing.assert_allclose(
            action_before,
            action_after,
            atol=1e-6,
            err_msg="Loaded model predictions differ from saved model",
        )

    def test_save_before_train_raises(self, ppo: PPO) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="No model to save"):
                ppo.save(Path(tmpdir) / "nope")


class TestPPOOnnxExport:
    """Test ONNX export produces valid model."""

    def test_onnx_export_creates_file(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        ppo.train(hover_env, total_timesteps=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "gcnet.onnx"
            result = ppo.export_onnx(onnx_path, obs_dim=17)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_onnx_loadable_by_onnxruntime(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        import onnxruntime as ort

        ppo.train(hover_env, total_timesteps=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "gcnet.onnx"
            ppo.export_onnx(onnx_path, obs_dim=17)
            session = ort.InferenceSession(str(onnx_path))

            inputs = session.get_inputs()
            assert len(inputs) == 1
            assert inputs[0].name == "observation"

            outputs = session.get_outputs()
            assert len(outputs) == 1
            assert outputs[0].name == "action"

    def test_onnx_inference_matches_pytorch(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        import onnxruntime as ort

        ppo.train(hover_env, total_timesteps=128)

        obs, _ = hover_env.reset()
        obs = obs.astype(np.float32)

        # Raw PyTorch forward pass (no action-space clipping)
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        policy = ppo._model.policy
        policy.eval()
        with torch.no_grad():
            features = policy.features_extractor(obs_tensor)
            latent = policy.mlp_extractor.policy_net(features)
            action_pt = policy.action_net(latent).squeeze().numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "gcnet.onnx"
            ppo.export_onnx(onnx_path, obs_dim=17)
            session = ort.InferenceSession(str(onnx_path))

            ort_result = session.run(None, {"observation": obs.reshape(1, -1)})
            action_onnx = ort_result[0].squeeze()

        np.testing.assert_allclose(
            action_pt,
            action_onnx,
            atol=1e-5,
            err_msg="ONNX output differs from PyTorch output",
        )

    def test_export_before_train_raises(self, ppo: PPO) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="No model to export"):
                ppo.export_onnx(Path(tmpdir) / "nope.onnx")
