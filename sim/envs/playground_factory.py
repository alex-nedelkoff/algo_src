"""Playground (SymPy-dynamics) environment factory.

Concrete ``EnvFactory`` implementation that builds ``RateCtrlEnv`` +
``VecEnvAdapter`` using the SymPy-compiled MAVLab quadrotor dynamics
from the playground sim layer.  Selected via Hydra
``_target_: sim.envs.playground_factory.PlaygroundEnvFactory`` in a
``sim/playground.yaml`` config group.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from sim.envs.rate_ctrl_env import RateCtrlEnv, make_figure8_track
from sim.envs.vec_env_adapter import VecEnvAdapter

log = logging.getLogger(__name__)


class PlaygroundEnvFactory:
    """Factory that builds RateCtrlEnv-backed VecEnv instances.

    All sim-level parameters are captured at construction time (typically
    via Hydra recursive instantiation of the ``sim`` config group).  The
    ``make_*`` methods receive training-level knobs that vary between
    training and evaluation.

    Args:
        n_envs: Number of parallel environments for training.
        dt: Physics timestep in seconds (default 0.002 = 500 Hz).
        action_repeat: Physics substeps per RL step (default 5 = 100 Hz RL).
        reward_preset: Reward preset name ("baseline", "M16", or "M23").
        dr_percentage: Domain randomization range as fraction of nominal.
        init_vel_range: Uniform range for initial velocity perturbation (m/s).
        init_rp_deg: Uniform range for initial roll/pitch perturbation (deg).
        init_rate_range: Uniform range for initial angular rate perturbation (rad/s).
        max_steps: Episode horizon in RL steps.
        track: Optional track definition dict. If None, uses figure-8 default.
    """

    def __init__(
        self,
        n_envs: int = 100,
        dt: float = 0.002,
        action_repeat: int = 5,
        reward_preset: str = "M23",
        dr_percentage: float = 0.0,
        init_vel_range: float = 0.5,
        init_rp_deg: float = 20.0,
        init_rate_range: float = 0.1,
        max_steps: int = 1200,
        track: dict | DictConfig | None = None,
    ) -> None:
        self.n_envs = n_envs
        self.dt = dt
        self.action_repeat = action_repeat
        self.reward_preset = reward_preset
        self.dr_percentage = dr_percentage
        self.init_vel_range = init_vel_range
        self.init_rp_deg = init_rp_deg
        self.init_rate_range = init_rate_range
        self.max_steps = max_steps
        self.track = _resolve_track(track)

    # ------------------------------------------------------------------
    # EnvFactory protocol
    # ------------------------------------------------------------------

    def make_vec_env(
        self,
        domain_rand_cfg: DictConfig,
        reward_cfg: DictConfig,
    ) -> VecEnvAdapter:
        """Build a training VecEnv with domain randomization."""
        dr_pct = self._resolve_dr_percentage(domain_rand_cfg)
        preset = self._resolve_reward_preset(reward_cfg)

        return self._build(
            n_envs=self.n_envs,
            dr_percentage=dr_pct,
            reward_preset=preset,
        )

    def make_eval_env(
        self,
        domain_rand_cfg: DictConfig,
        reward_cfg: DictConfig,
        n_envs: int,
    ) -> VecEnvAdapter:
        """Build an evaluation VecEnv (domain randomization disabled)."""
        preset = self._resolve_reward_preset(reward_cfg)

        return self._build(
            n_envs=n_envs,
            dr_percentage=0.0,  # DR disabled for eval
            reward_preset=preset,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(
        self,
        n_envs: int,
        dr_percentage: float,
        reward_preset: str,
    ) -> VecEnvAdapter:
        """Shared builder for both training and eval envs."""
        env = RateCtrlEnv(
            n_envs=n_envs,
            dt=self.dt,
            action_repeat=self.action_repeat,
            dr_percentage=dr_percentage,
            reward_preset=reward_preset,
            init_vel_range=self.init_vel_range,
            init_rp_deg=self.init_rp_deg,
            init_rate_range=self.init_rate_range,
            max_steps=self.max_steps,
            track=self.track,
        )

        log.info(
            "Built RateCtrlEnv: n_envs=%d, dt=%.4f, action_repeat=%d, "
            "max_steps=%d, dr=%.1f%%, preset=%s",
            n_envs,
            self.dt,
            self.action_repeat,
            self.max_steps,
            dr_percentage * 100,
            reward_preset,
        )

        return VecEnvAdapter(env)

    def _resolve_dr_percentage(self, domain_rand_cfg: DictConfig) -> float:
        """Extract DR percentage from config, respecting enable flag."""
        dr_dict: dict[str, Any] = OmegaConf.to_container(
            domain_rand_cfg, resolve=True
        )  # type: ignore[assignment]

        if not dr_dict.get("enabled", False):
            return 0.0

        # If a simple top-level percentage is provided, use it directly.
        if "percentage" in dr_dict:
            return dr_dict["percentage"]

        # Otherwise, compute from per-param config: use the median of all
        # param percentages as a single uniform DR range for the playground.
        params = dr_dict.get("params", {})
        if params:
            return float(np.median(list(params.values())))

        return self.dr_percentage

    def _resolve_reward_preset(self, reward_cfg: DictConfig) -> str:
        """Extract reward preset name from config."""
        return reward_cfg.get("preset", self.reward_preset)


def _resolve_track(track: dict | DictConfig | None) -> dict:
    """Convert track config to a plain dict, or use default."""
    if track is None:
        return make_figure8_track()

    if isinstance(track, DictConfig):
        track_dict: dict[str, Any] = OmegaConf.to_container(
            track, resolve=True
        )  # type: ignore[assignment]
        # Reconstruct gate positions as numpy arrays
        if "gates" in track_dict:
            for g in track_dict["gates"]:
                if isinstance(g.get("pos"), (list, tuple)):
                    g["pos"] = np.array(g["pos"], dtype=np.float64)
        return track_dict

    return track
