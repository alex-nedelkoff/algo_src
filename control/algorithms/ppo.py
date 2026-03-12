"""PPO training algorithm wrapping Stable Baselines3.

Injects the GCNetExtractor as a custom feature extractor so we get SB3's
battle-tested PPO implementation with our own policy architecture.

Hydra target: ``control.algorithms.ppo.PPO``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import torch
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3.common.callbacks import BaseCallback

    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from control.algorithms.base import Algorithm
from control.policies.gcnet import GCNetExtractor, _TORCH_AVAILABLE

__all__ = ["PPO"]


def _check_deps() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PPO requires PyTorch.  Install with:  pip install 'algo-src[control]'")
    if not _SB3_AVAILABLE:
        raise ImportError(
            "PPO requires stable-baselines3.  Install with:  pip install 'algo-src[control]'"
        )


class PPO(Algorithm):
    """PPO trainer wrapping SB3 with GCNetExtractor.

    Configurable via Hydra (``configs/control/ppo.yaml``).
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 1000,
        batch_size: int = 5000,
        n_epochs: int = 10,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.005,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_type: str = "MlpPolicy",
        net_arch: dict[str, list[int]] | None = None,
        activation_fn: str = "relu",
        use_sde: bool = False,
        hidden_dims: tuple[int, ...] = (),
        asymmetric_critic: bool = False,
        critic_obs_dim: int = 48,
        actor_obs_dim: int = 24,
        tensorboard_log: str | None = None,
        **kwargs: Any,
    ) -> None:
        _check_deps()

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.policy_type = policy_type
        self.activation_fn = activation_fn
        self.use_sde = use_sde
        self.hidden_dims = hidden_dims
        self.asymmetric_critic = asymmetric_critic
        self.critic_obs_dim = critic_obs_dim
        self.actor_obs_dim = actor_obs_dim
        self.tensorboard_log = tensorboard_log
        self.log_std_init: float = kwargs.pop("log_std_init", 0.0)
        self.extra_policy_kwargs: dict[str, Any] = kwargs.pop("extra_policy_kwargs", {})

        # MonoRace M23 default: separate 3×64 policy and value networks.
        # When hidden_dims is empty, SB3's FlattenExtractor is used (identity for Box)
        # and net_arch defines the full architecture for each head independently.
        self.net_arch: dict[str, list[int]] | list[dict[str, list[int]]] = net_arch or {
            "pi": [64, 64, 64],
            "vf": [64, 64, 64],
        }

        self._model: SB3_PPO | None = None

    def _build_policy_kwargs(self) -> dict[str, Any]:
        """Build SB3 policy_kwargs.

        When ``hidden_dims`` is empty (default, M23-style), architecture is
        defined entirely via ``net_arch`` with separate policy/value networks.
        When ``hidden_dims`` is set, GCNetExtractor is injected as a shared
        feature backbone (legacy shared-extractor mode).
        """
        activation_map = {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
        }
        act_fn = activation_map.get(self.activation_fn, torch.nn.ReLU)

        kwargs: dict[str, Any] = {
            "net_arch": self.net_arch,
            "activation_fn": act_fn,
            "log_std_init": self.log_std_init,
        }

        if self.hidden_dims:
            kwargs["features_extractor_class"] = GCNetExtractor
            kwargs["features_extractor_kwargs"] = {"hidden_dims": self.hidden_dims}

        # Merge any extra policy kwargs (e.g. custom features_extractor_class)
        kwargs.update(self.extra_policy_kwargs)

        return kwargs

    def _create_model(self, env: gym.Env) -> SB3_PPO:
        """Instantiate the SB3 PPO model."""
        return SB3_PPO(
            policy=self.policy_type,
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            use_sde=self.use_sde,
            policy_kwargs=self._build_policy_kwargs(),
            tensorboard_log=self.tensorboard_log,
            verbose=1,
        )

    def train(
        self,
        env: gym.Env,
        total_timesteps: int = 1000,
        callbacks: list[BaseCallback] | None = None,
        **kwargs: Any,
    ) -> None:
        """Train PPO on the given environment."""
        if self._model is None:
            self._model = self._create_model(env)
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            **kwargs,
        )

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        """Predict action for an observation."""
        if self._model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")
        return self._model.predict(observation, deterministic=deterministic)

    def save(self, path: str | Path) -> None:
        """Save the trained model checkpoint."""
        if self._model is None:
            raise RuntimeError("No model to save. Call train() first.")
        self._model.save(str(path))

    def load(self, path: str | Path, env: gym.Env | None = None) -> None:
        """Load a model checkpoint."""
        _check_deps()
        self._model = SB3_PPO.load(str(path), env=env)

    def export_onnx(self, path: str | Path, obs_dim: int | None = None) -> Path:
        """Export the trained actor network to ONNX.

        Chains features_extractor → policy_net → action_net into a single
        ONNX model for deployment on Jetson Orin NX.

        Works for both shared-extractor and separate-net architectures.
        """
        if self._model is None:
            raise RuntimeError("No model to export. Call train() or load() first.")

        import onnx

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        policy = self._model.policy
        policy.eval()

        if obs_dim is None:
            obs_dim = policy.observation_space.shape[0]

        # Build a sequential actor: features_extractor -> pi_net -> action_net
        extractor = policy.features_extractor
        pi_net = policy.mlp_extractor.policy_net
        action_net = policy.action_net

        class _ActorForExport(torch.nn.Module):
            def __init__(self, extractor, pi_net, action_net):  # type: ignore[no-untyped-def]
                super().__init__()
                self.extractor = extractor
                self.pi_net = pi_net
                self.action_net = action_net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.extractor(x)
                latent = self.pi_net(features)
                return self.action_net(latent)

        actor = _ActorForExport(extractor, pi_net, action_net)
        actor.eval()

        dummy_input = torch.randn(1, obs_dim)
        torch.onnx.export(
            actor,
            dummy_input,
            str(path),
            opset_version=17,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={
                "observation": {0: "batch"},
                "action": {0: "batch"},
            },
        )

        # Validate
        model = onnx.load(str(path))
        onnx.checker.check_model(model)

        return path
