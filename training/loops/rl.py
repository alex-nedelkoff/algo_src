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
        # Recurrent (LSTM) support
        recurrent=ctrl.get("recurrent", False),
        lstm_hidden_size=ctrl.get("lstm_hidden_size", 128),
        n_lstm_layers=ctrl.get("n_lstm_layers", 1),
        # Mixture of Experts
        moe=ctrl.get("moe", False),
        n_experts=ctrl.get("n_experts", 4),
        expert_hidden_dim=ctrl.get("expert_hidden_dim", 128),
        top_k=ctrl.get("top_k", 2),
        balance_coef=ctrl.get("balance_coef", 0.01),
    )


def setup_callbacks(
    cfg: DictConfig,
    eval_env=None,
    uploader: ArtifactUploader | None = None,
    env_factory: Any = None,
    train_env=None,
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

    # Curriculum learning (if configured)
    curriculum_cfg = cfg.get("curriculum")
    if curriculum_cfg is not None and train_env is not None:
        from omegaconf import OmegaConf
        from training.curriculum_callback import CurriculumCallback, CurriculumSB3Callback

        curr_dict = OmegaConf.to_container(curriculum_cfg, resolve=True)
        curriculum = CurriculumCallback(curr_dict)
        if curriculum.enabled:
            unwrapped_env = train_env
            while hasattr(unwrapped_env, "env"):
                unwrapped_env = unwrapped_env.env
            callbacks.append(CurriculumSB3Callback(curriculum, unwrapped_env))

    # Multi-scene track regeneration
    ms_cfg = cfg.get("multi_scene")
    if ms_cfg is not None and ms_cfg.get("enabled", False) and train_env is not None:
        from training.multi_scene_callback import MultiSceneCallback

        unwrapped_env = train_env
        while hasattr(unwrapped_env, "env"):
            unwrapped_env = unwrapped_env.env
        callbacks.append(MultiSceneCallback(
            unwrapped_env, n_scenes=ms_cfg.get("n_scenes", 10)
        ))

    # aRPO alpha schedule callback (sync trick)
    arpo_cfg = cfg.get("arpo")
    if arpo_cfg is not None and arpo_cfg.get("enabled", False) and train_env is not None:
        from training.arpo import ARPOAlphaCallback, AlphaSchedule

        alpha_sched = AlphaSchedule(
            k_end_fraction=arpo_cfg.get("k_end_fraction", 0.25),
            total_steps=cfg.total_timesteps,
        )
        # train_env should be the ARPOActionWrapper at this point
        callbacks.append(ARPOAlphaCallback(train_env, alpha_sched))

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

        # Wire track generation config (top-level, not under sim)
        if "track_gen" in cfg:
            env_factory._track_gen_cfg = cfg.track_gen
            env_factory._seed = cfg.get("seed", 42)

        train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)

        # 2. Wrap with perception (no-op if identity)
        perception_wrapper = hydra.utils.instantiate(cfg.perception)
        train_env = perception_wrapper.wrap(train_env)

        # 2b. Wrap with EKF filtering if configured
        if "ekf" in cfg:
            from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper

            ekf_kwargs = OmegaConf.to_container(cfg.ekf, resolve=True)
            log.info("Applying EKF wrapper (corner_noise_k=%s)", ekf_kwargs.get("corner_noise_k"))
            train_env = EKFVecEnvWrapper(train_env, **ekf_kwargs)

        # 3. Build eval env (no domain rand, fewer envs)
        eval_dr_cfg = OmegaConf.create(OmegaConf.to_container(cfg.domain_rand, resolve=True))
        OmegaConf.update(eval_dr_cfg, "enabled", False)
        eval_env = env_factory.make_eval_env(eval_dr_cfg, cfg.reward, n_envs=cfg.n_eval_episodes)
        eval_env = perception_wrapper.wrap(eval_env)

        if "ekf" in cfg:
            from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper

            eval_ekf_kwargs = OmegaConf.to_container(cfg.ekf, resolve=True)
            eval_env = EKFVecEnvWrapper(eval_env, **eval_ekf_kwargs)

        # 3b. aRPO base policy wrapper (must wrap env BEFORE PPO sees it)
        arpo_cfg = cfg.get("arpo")
        if arpo_cfg is not None and arpo_cfg.get("enabled", False):
            import numpy as np_
            from control.base_policies.pd_waypoint_tracker import PDWaypointTracker
            from training.arpo import ARPOActionWrapper, AlphaSchedule

            unwrapped = train_env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env
            gate_positions = np_.array([g.position for g in unwrapped._tracks[0].gates])
            base_policy = PDWaypointTracker(gate_positions, mass=unwrapped.params.mass)

            alpha_sched = AlphaSchedule(
                k_end_fraction=arpo_cfg.get("k_end_fraction", 0.25),
                total_steps=cfg.total_timesteps,
            )
            train_env = ARPOActionWrapper(train_env, base_policy, alpha_sched)
            log.info("aRPO wrapper applied: k_end_fraction=%.2f", arpo_cfg.get("k_end_fraction", 0.25))

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
            cfg, eval_env=eval_env, uploader=uploader, env_factory=env_factory,
            train_env=train_env,
        )

        # 6b. Apply LR schedule if configured
        lr_schedule_cfg = cfg.get("lr_schedule")
        if lr_schedule_cfg is not None and ppo._model is not None:
            from control.algorithms.ppo import PPO as PPOWrapper
            schedule = PPOWrapper.cosine_lr_schedule(
                initial_lr=lr_schedule_cfg.get("initial_lr", 3e-4),
                final_lr=lr_schedule_cfg.get("final_lr", 5e-5),
            )
            ppo._model.learning_rate = schedule
            log.info(
                "LR cosine schedule: %.1e -> %.1e",
                lr_schedule_cfg.get("initial_lr", 3e-4),
                lr_schedule_cfg.get("final_lr", 5e-5),
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
