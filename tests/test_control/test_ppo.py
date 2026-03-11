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
    """A PPO instance with M23 defaults for fast tests."""
    return PPO(
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
    )


@pytest.fixture
def hover_env():  # type: ignore[no-untyped-def]
    """A HoverEnv instance for testing."""
    from sim.envs.hover_env import HoverEnv

    return HoverEnv()


class TestPPOArchitectureM23:
    """Test that PPO creates MonoRace M23 architecture by default.

    M23 style: separate 3×64 ReLU policy and value networks, no shared backbone.
    The full actor path should be:
        obs(17/24) -> FlattenExtractor(identity) -> pi_net(3×64 ReLU) -> Linear(64->4)
    """

    def test_pi_net_has_three_hidden_layers(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Policy net should have 3 hidden layers (3×64)."""
        ppo.train(hover_env, total_timesteps=64)
        pi_net = ppo._model.policy.mlp_extractor.policy_net
        pi_linears = [m for m in pi_net.modules() if isinstance(m, torch.nn.Linear)]
        assert len(pi_linears) == 3, (
            f"Expected 3 policy layers (M23 3×64), got {len(pi_linears)}"
        )

    def test_vf_net_has_three_hidden_layers(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Value net should have 3 hidden layers (3×64)."""
        ppo.train(hover_env, total_timesteps=64)
        vf_net = ppo._model.policy.mlp_extractor.value_net
        vf_linears = [m for m in vf_net.modules() if isinstance(m, torch.nn.Linear)]
        assert len(vf_linears) == 3, (
            f"Expected 3 value layers (M23 3×64), got {len(vf_linears)}"
        )

    def test_pi_layer_sizes(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Policy net layers should be obs_dim→64→64→64."""
        ppo.train(hover_env, total_timesteps=64)
        pi_net = ppo._model.policy.mlp_extractor.policy_net
        linears = [m for m in pi_net.modules() if isinstance(m, torch.nn.Linear)]
        sizes = [(l.in_features, l.out_features) for l in linears]
        obs_dim = hover_env.observation_space.shape[0]
        assert sizes == [(obs_dim, 64), (64, 64), (64, 64)]

    def test_vf_layer_sizes(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Value net layers should be obs_dim→64→64→64."""
        ppo.train(hover_env, total_timesteps=64)
        vf_net = ppo._model.policy.mlp_extractor.value_net
        linears = [m for m in vf_net.modules() if isinstance(m, torch.nn.Linear)]
        sizes = [(l.in_features, l.out_features) for l in linears]
        obs_dim = hover_env.observation_space.shape[0]
        assert sizes == [(obs_dim, 64), (64, 64), (64, 64)]

    def test_policy_and_value_are_independent(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Policy and value nets should NOT share parameters."""
        ppo.train(hover_env, total_timesteps=64)
        policy = ppo._model.policy
        pi_params = set(id(p) for p in policy.mlp_extractor.policy_net.parameters())
        vf_params = set(id(p) for p in policy.mlp_extractor.value_net.parameters())
        assert pi_params.isdisjoint(vf_params), "Policy and value nets share parameters"

    def test_activations_are_relu(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Hidden activations should be ReLU (M23 style)."""
        ppo.train(hover_env, total_timesteps=64)
        pi_net = ppo._model.policy.mlp_extractor.policy_net
        activations = [m for m in pi_net.modules() if isinstance(m, torch.nn.ReLU)]
        assert len(activations) == 3, f"Expected 3 ReLU activations, got {len(activations)}"

    def test_action_head_shape(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Action head should be Linear(64 -> action_dim)."""
        ppo.train(hover_env, total_timesteps=64)
        action_net = ppo._model.policy.action_net
        assert isinstance(action_net, torch.nn.Linear)
        assert action_net.in_features == 64
        assert action_net.out_features == hover_env.action_space.shape[0]

    def test_no_custom_extractor_by_default(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Default config should NOT use GCNetExtractor (uses FlattenExtractor)."""
        from control.policies.gcnet import GCNetExtractor

        ppo.train(hover_env, total_timesteps=64)
        extractor = ppo._model.policy.features_extractor
        assert not isinstance(extractor, GCNetExtractor)

    def test_total_actor_param_count(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        """Total actor params (pi_net + action head) should be ~10K for M23."""
        ppo.train(hover_env, total_timesteps=64)
        policy = ppo._model.policy
        pi_params = sum(p.numel() for p in policy.mlp_extractor.policy_net.parameters())
        action_params = sum(p.numel() for p in policy.action_net.parameters())
        total = pi_params + action_params
        # obs_dim=17: 17*64+64 + 64*64+64 + 64*64+64 + 64*4+4 = ~10K
        assert total < 15_000, f"Actor too large ({total} params) for M23 architecture"
        assert total > 5_000, f"Actor too small ({total} params)"

    def test_net_arch_default_is_m23(self) -> None:
        """Default net_arch should be M23: separate 3×64."""
        ppo_default = PPO()
        assert ppo_default.net_arch == {"pi": [64, 64, 64], "vf": [64, 64, 64]}


class TestPPOSharedExtractorMode:
    """Test legacy shared-extractor mode when hidden_dims is provided."""

    def test_gcnet_extractor_used_when_hidden_dims_set(self, hover_env) -> None:  # type: ignore[no-untyped-def]
        from control.policies.gcnet import GCNetExtractor

        ppo = PPO(
            n_steps=64, batch_size=32, n_epochs=2,
            hidden_dims=(128, 128, 64),
            net_arch={"pi": [], "vf": []},
        )
        ppo.train(hover_env, total_timesteps=64)
        extractor = ppo._model.policy.features_extractor
        assert isinstance(extractor, GCNetExtractor)


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
            result = ppo.export_onnx(onnx_path)
            assert result.exists()
            assert result.stat().st_size > 0

    def test_onnx_loadable_by_onnxruntime(self, ppo: PPO, hover_env) -> None:  # type: ignore[no-untyped-def]
        import onnxruntime as ort

        ppo.train(hover_env, total_timesteps=128)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "gcnet.onnx"
            ppo.export_onnx(onnx_path)
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
            ppo.export_onnx(onnx_path)
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
