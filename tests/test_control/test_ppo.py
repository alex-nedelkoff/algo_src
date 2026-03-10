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
