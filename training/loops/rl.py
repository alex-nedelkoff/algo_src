"""RL training loop for drone racing.

Orchestrates environment construction (via EnvFactory), perception
wrapping, PPO training, and checkpoint management.  Selected via
``loop: rl`` in Hydra config (``_target_: training.loops.rl.RLTrainingLoop``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig, OmegaConf

from control.algorithms.ppo import PPO

if TYPE_CHECKING:
    from artifacts.uploader import ArtifactUploader

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers (moved from control/__main__.py)
# ------------------------------------------------------------------


def build_ppo(cfg: DictConfig) -> PPO:
    """Build a PPO trainer from Hydra config."""
    ctrl = cfg.control
    net_arch = OmegaConf.to_container(ctrl.net_arch, resolve=True) if "net_arch" in ctrl else None

    # Collect extra policy_kwargs from config (e.g. features_extractor_class)
    extra_policy_kwargs: dict = {}
    if "policy_kwargs" in ctrl:
        raw = OmegaConf.to_container(ctrl.policy_kwargs, resolve=True)
        if isinstance(raw, dict):
            # Resolve any _target_ references (e.g. features_extractor_class)
            for k, v in raw.items():
                if isinstance(v, dict) and "_target_" in v:
                    extra_policy_kwargs[k] = hydra.utils.get_class(v["_target_"])
                else:
                    extra_policy_kwargs[k] = v

    return PPO(
        learning_rate=ctrl.learning_rate,
        n_steps=ctrl.n_steps,
        batch_size=ctrl.batch_size,
        n_epochs=ctrl.n_epochs,
        gamma=ctrl.gamma,
        gae_lambda=ctrl.gae_lambda,
        clip_range=ctrl.clip_range,
        ent_coef=ctrl.ent_coef,
        vf_coef=ctrl.get("vf_coef", 0.5),
        max_grad_norm=ctrl.get("max_grad_norm", 0.5),
        policy_type=ctrl.policy_type,
        net_arch=net_arch,
        activation_fn=ctrl.activation_fn,
        use_sde=ctrl.get("use_sde", False),
        log_std_init=ctrl.get("log_std_init", 0.0),
        extra_policy_kwargs=extra_policy_kwargs,
        action_bias_init=list(ctrl.action_bias_init) if ctrl.get("action_bias_init") else None,
        tensorboard_log=str(Path(cfg.output_dir) / "tb_logs") if cfg.get("output_dir") else None,
    )


def setup_callbacks(
    cfg: DictConfig,
    eval_env=None,
    uploader: ArtifactUploader | None = None,
    env_factory: Any = None,
) -> list:
    """Create SB3 training callbacks from config.

    SB3 callback frequencies count env.step() calls, not total timesteps.
    With n_envs parallel envs, divide by n_envs to get the intended
    timestep-based frequency.
    """
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    from training.callbacks import EvalRacingCallback, GateMetricsCallback
    from training.trajectory_recorder import TrajectoryRecorderCallback

    callbacks = []
    output_dir = Path(cfg.output_dir)
    n_envs = cfg.sim.n_envs

    callbacks.append(
        CheckpointCallback(
            save_freq=max(1, cfg.checkpoint_freq // n_envs),
            save_path=str(output_dir / "checkpoints"),
            name_prefix="ppo",
        )
    )

    if eval_env is not None:
        eval_env.enable_episode_buffer()
        eval_racing = EvalRacingCallback(eval_env)
        callbacks.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=cfg.n_eval_episodes,
                eval_freq=max(1, cfg.eval_freq // n_envs),
                best_model_save_path=str(output_dir / "best_model"),
                log_path=str(output_dir / "eval_logs"),
                callback_after_eval=eval_racing,
            )
        )

    # Gate racing metrics (gates/laps/termination breakdown)
    gate_metrics_freq = cfg.get("gate_metrics_freq", 100_000)
    callbacks.append(GateMetricsCallback(log_freq=max(1, gate_metrics_freq // n_envs)))

    # Trajectory recording for visualisation
    viz_freq = cfg.get("viz_freq", 1_000_000)
    n_viz_episodes = cfg.get("n_viz_episodes", 5)
    callbacks.append(
        TrajectoryRecorderCallback(
            viz_freq=viz_freq,
            n_envs=n_envs,
            save_path=str(output_dir / "checkpoints"),
            env_factory=env_factory,
            eval_reward_cfg=cfg.get("reward"),
            n_viz_episodes=n_viz_episodes,
            uploader=uploader,
        )
    )

    return callbacks


# ------------------------------------------------------------------
# RLTrainingLoop
# ------------------------------------------------------------------


class RLTrainingLoop:
    """RL training loop: env factory + perception wrapper + PPO.

    Instantiated by Hydra via ``cfg.loop._target_`` and driven by
    :meth:`run`, which receives the full config and an optional
    artifact uploader from ``training/__main__.py``.
    """

    def run(self, cfg: DictConfig, uploader: ArtifactUploader | None = None) -> None:
        """Execute the full RL training pipeline."""
        # 1. Build training environment via EnvFactory
        log.info("Building training env (n_envs=%d)...", cfg.sim.n_envs)
        env_factory = hydra.utils.instantiate(cfg.sim)
        train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)

        # 2. Wrap with perception (no-op if identity)
        perception_wrapper = hydra.utils.instantiate(cfg.perception)
        train_env = perception_wrapper.wrap(train_env)

        # 3. Build eval env (no domain rand, fewer envs)
        eval_dr_cfg = OmegaConf.create(OmegaConf.to_container(cfg.domain_rand, resolve=True))
        OmegaConf.update(eval_dr_cfg, "enabled", False)
        eval_env = env_factory.make_eval_env(eval_dr_cfg, cfg.reward, n_envs=cfg.n_eval_episodes)
        eval_env = perception_wrapper.wrap(eval_env)

        # 4. Build PPO trainer
        log.info("Building PPO trainer...")
        ppo = build_ppo(cfg)

        # 5. Resume from checkpoint if specified
        resume_path = cfg.get("resume_checkpoint", None)
        if resume_path:
            log.info("Resuming from checkpoint: %s", resume_path)
            ppo.load(resume_path, env=train_env)

        # 6. Setup callbacks
        callbacks = setup_callbacks(
            cfg, eval_env=eval_env, uploader=uploader, env_factory=env_factory
        )

        # 7. Train
        log.info("Starting training for %d timesteps...", cfg.total_timesteps)
        ppo.train(train_env, total_timesteps=cfg.total_timesteps, callbacks=callbacks)

        # 8. Save final model + close envs
        final_path = Path(cfg.output_dir) / "final_model"
        ppo.save(final_path)
        log.info("Final model saved to %s", final_path)

        train_env.close()
        eval_env.close()
