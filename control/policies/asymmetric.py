"""Asymmetric actor-critic: actor sees observable obs, critic also sees privileged.

The privileged channel (true per-env plant params from VQMatchedDynamics.randomize:
drag / weathervane slope / disturbance magnitude) gives the value function a cleaner
advantage signal in live-unsafe states WITHOUT leaking unobservable info into the
deployed actor. The critic is discarded at deploy, so the residual actor flies on
observable obs alone -> asymmetric adds zero deploy cost. (COR-127, HANDOFF 06-14 #2.)

Design: we fully bypass SB3's shared mlp_extractor. The actor MLP maps obs["policy"] ->
action mean (paired with the inherited self.log_std / self.action_dist); the critic MLP
maps concat(obs["policy"], obs["privileged"]) -> value. The inherited features_extractor /
mlp_extractor / action_net / value_net are left inert (no grad path through them). This is
robust to SB3 version differences in how Dict obs get flattened.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy


def _mlp(in_dim: int, out_dim: int, hidden=(256, 256)) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pol_dim = int(np.prod(self.observation_space["policy"].shape))
        priv_dim = int(np.prod(self.observation_space["privileged"].shape))
        act_dim = int(np.prod(self.action_space.shape))
        self.actor_mlp = _mlp(pol_dim, act_dim).to(self.device)
        self.critic_mlp = _mlp(pol_dim + priv_dim, 1).to(self.device)
        # near-zero initial action mean -> policy starts at the FF anchor (residual learning)
        with torch.no_grad():
            self.actor_mlp[-1].weight.mul_(0.01)
            self.actor_mlp[-1].bias.zero_()
        self.optimizer.add_param_group(
            {"params": list(self.actor_mlp.parameters()) + list(self.critic_mlp.parameters())})

    # --- routed heads (obs is a dict of tensors by the time these are called) ---
    def _actor_dist(self, obs):
        mean = self.actor_mlp(obs["policy"].float())
        return self.action_dist.proba_distribution(mean, self.log_std)

    def _value(self, obs):
        x = torch.cat([obs["policy"].float(), obs["privileged"].float()], dim=1)
        return self.critic_mlp(x)

    def forward(self, obs, deterministic: bool = False):
        dist = self._actor_dist(obs)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self._value(obs), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        dist = self._actor_dist(obs)
        return self._value(obs), dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        return self._value(obs)

    def get_distribution(self, obs):
        return self._actor_dist(obs)

    def _predict(self, obs, deterministic: bool = False):
        return self._actor_dist(obs).get_actions(deterministic=deterministic)


if __name__ == "__main__":   # pod self-test (SB3 not importable on the Mac)
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class _Dummy(gym.Env):
        observation_space = spaces.Dict({
            "policy": spaces.Box(-1, 1, (20,), dtype=np.float32),
            "privileged": spaces.Box(-1, 1, (11,), dtype=np.float32)})
        action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, True, {}

    m = PPO(AsymmetricActorCriticPolicy, _Dummy(), n_steps=64, batch_size=32,
            policy_kwargs=dict(net_arch=[64, 64], log_std_init=-2.5), device="cpu")
    m.learn(256)
    # deploy path: actor must run without the privileged key needing meaningful values
    obs = {"policy": np.zeros((1, 20), np.float32), "privileged": np.zeros((1, 11), np.float32)}
    a, _ = m.predict(obs, deterministic=True)
    assert a.shape == (1, 4)
    print("OK: asymmetric PPO trained 256 steps; predict ->", a.round(3))
