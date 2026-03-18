"""α-RPO: Attenuated Residual Policy Optimization for SB3.

Uses a VecEnv action wrapper (not a callback) to fuse base policy actions
with learned policy actions BEFORE they reach the environment.

A companion SB3 callback handles the sync trick: alpha is updated AFTER
rollout collection, BEFORE optimization.

Reference: Trumpp et al. (2026), arXiv:2603.12960
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

GRAVITY = 9.81


class AlphaSchedule:
    """Linear attenuation schedule. alpha=0 → base only, alpha=1 → learned only."""

    def __init__(self, k_end_fraction: float = 0.25, total_steps: int = 20_000_000) -> None:
        self.k_end = int(total_steps * k_end_fraction)

    def get_alpha(self, timestep: int) -> float:
        if self.k_end <= 0:
            return 1.0
        return min(1.0, timestep / self.k_end)


def fuse_actions(base_actions, learned_actions, alpha):
    """Fuse: μ = (1-α)*base + α*learned."""
    return (1.0 - alpha) * base_actions + alpha * learned_actions


def normalize_base_trpy(physical, mass, max_body_rate):
    """Normalize physical TRPY to [-1, 1]."""
    max_thrust = mass * GRAVITY * 2.0
    normalized = np.empty_like(physical)
    normalized[:, 0] = physical[:, 0] / max_thrust * 2.0 - 1.0
    normalized[:, 1:4] = physical[:, 1:4] / max_body_rate
    return np.clip(normalized, -1.0, 1.0)


class ARPOActionWrapper:
    """VecEnv wrapper that fuses base policy with learned policy actions.

    Intercepts step() to compute base_action from pre-step privileged state,
    blend with learned_action, and forward fused action to real env.
    When alpha reaches 1.0, becomes a no-op passthrough.
    """

    def __init__(self, env, base_policy, alpha_schedule):
        self._env = env
        self.base_policy = base_policy
        self.alpha_schedule = alpha_schedule
        self.alpha = 0.0

        # Unwrap to get the GateRaceEnv
        self._gate_env = env
        while hasattr(self._gate_env, "env"):
            self._gate_env = self._gate_env.env

    def step(self, learned_actions):
        if self.alpha >= 1.0:
            return self._env.step(learned_actions)

        ge = self._gate_env
        n = ge.n_envs
        pos = ge._states[:n, 0:3]
        vel = ge._states[:n, 3:6]
        quat = ge._states[:n, 6:10]
        omega = ge._states[:n, 10:13]
        gate_idx = ge._gate_indices[:n]

        base_physical = self.base_policy.get_action_batch(pos, vel, quat, omega, gate_idx)
        base_norm = normalize_base_trpy(base_physical, ge.params.mass, ge.max_body_rate)

        fused = fuse_actions(base_norm, np.asarray(learned_actions, dtype=np.float64), self.alpha)
        return self._env.step(fused.astype(np.float32))

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self._env, name)


try:
    from stable_baselines3.common.callbacks import BaseCallback

    class ARPOAlphaCallback(BaseCallback):
        """SB3 callback for sync trick: update alpha AFTER collection, BEFORE optimization."""

        def __init__(self, wrapper, alpha_schedule, verbose=1):
            super().__init__(verbose)
            self.wrapper = wrapper
            self.alpha_schedule = alpha_schedule

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            new_alpha = self.alpha_schedule.get_alpha(self.num_timesteps)
            self.wrapper.alpha = new_alpha
            if self.verbose and self.num_timesteps % 500_000 < self.model.n_steps * self.model.n_envs:
                log.info("α-RPO: step=%d, alpha=%.3f", self.num_timesteps, new_alpha)

except ImportError:
    pass
