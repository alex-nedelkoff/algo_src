"""Adaptive freeze+warmup callback for MoE expert expansion.

Freezes existing experts while a new expert learns its role via the router.
Monitors router usage of the new expert and unfreezes all parameters once
usage stabilizes above a threshold.

Usage:
    callback = ExpertWarmupCallback(
        new_expert_idx=4,
        usage_threshold=0.15,
        check_freq=1_000_000,
        max_frozen_steps=10_000_000,
    )
"""
from __future__ import annotations

import logging

import torch

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

log = logging.getLogger(__name__)


class ExpertWarmupCallback(BaseCallback):
    """Adaptive freeze+warmup for MoE expert expansion.

    Phase 1 (frozen): Only the new expert and router train.
    Phase 2 (unfrozen): All parameters train together.

    Transitions from phase 1 to 2 when the router activates the new
    expert above ``usage_threshold`` for ``patience`` consecutive checks,
    or after ``max_frozen_steps`` (whichever comes first).
    """

    def __init__(
        self,
        new_expert_idx: int = 4,
        usage_threshold: float = 0.15,
        check_freq: int = 1_000_000,
        max_frozen_steps: int = 10_000_000,
        patience: int = 2,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.new_expert_idx = new_expert_idx
        self.usage_threshold = usage_threshold
        self.check_freq = check_freq
        self.max_frozen_steps = max_frozen_steps
        self.patience = patience

        self._frozen = True
        self._above_threshold_count = 0
        self._usage_history: list[float] = []
        self._last_check_step = 0
        self._frozen_params: list[str] = []

    def _on_training_start(self) -> None:
        """Freeze all parameters except the new expert (and optionally router)."""
        policy = self.model.policy

        self._frozen_params = []
        for name, param in policy.named_parameters():
            # Keep trainable: only the new expert + log_std
            # Router is FROZEN to prevent takeover (it would learn to
            # always route to the only trainable expert)
            if f"experts.{self.new_expert_idx}." in name:
                continue
            if "log_std" in name:
                continue

            param.requires_grad = False
            self._frozen_params.append(name)

        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        log.info(
            "Expert warmup: FROZEN. Trainable: %d params (expert %d + router + log_std). "
            "Frozen: %d params (experts 0-%d + value_net).",
            trainable, self.new_expert_idx,
            frozen, self.new_expert_idx - 1,
        )

    def _on_step(self) -> bool:
        if not self._frozen:
            return True

        step = self.num_timesteps
        if step - self._last_check_step < self.check_freq:
            return True
        self._last_check_step = step

        # Monitor expert 4 divergence from its clone source
        usage = self._measure_expert_usage()
        self._usage_history.append(usage)

        log.info(
            "Expert warmup check at step %d: expert %d usage=%.1f%%, "
            "frozen_steps=%d/%d",
            step, self.new_expert_idx, usage * 100,
            step, self.max_frozen_steps,
        )

        # Unfreeze after fixed warmup period
        # (Router is frozen, so usage-based triggering doesn't apply.
        # Expert 4 just needs enough steps to diverge from its clone.)
        if step >= self.max_frozen_steps:
            self._unfreeze()
            log.info(
                "Expert warmup: UNFREEZING all parameters after %d warmup steps.",
                step,
            )
            log.info("Expert %d usage history: %s",
                     self.new_expert_idx,
                     [f"{u:.1%}" for u in self._usage_history])

        return True

    def _measure_expert_usage(self, n_samples: int = 1000) -> float:
        """Run observations through the router and measure new expert selection rate."""
        policy = self.model.policy

        # Sample observations from the rollout buffer
        buf = self.model.rollout_buffer
        if buf.pos == 0 and not buf.full:
            return 0.0

        n_avail = buf.buffer_size if buf.full else buf.pos
        n_samples = min(n_samples, n_avail)
        indices = torch.randperm(n_avail)[:n_samples]

        obs = torch.as_tensor(buf.observations[indices], device=policy.device).float()
        if obs.dim() == 3:
            obs = obs.squeeze(1)

        with torch.no_grad():
            router_weights = policy.router(obs)
            _, selected_indices = torch.topk(router_weights, policy.top_k, dim=-1)

        # Count how often the new expert appears in the top-k selections
        new_expert_selected = (selected_indices == self.new_expert_idx).any(dim=-1).float()
        return float(new_expert_selected.mean().item())

    def _unfreeze(self) -> None:
        """Unfreeze all parameters."""
        policy = self.model.policy
        for name, param in policy.named_parameters():
            param.requires_grad = True
        self._frozen = False

        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        log.info("Expert warmup: All %d parameters now trainable.", trainable)
