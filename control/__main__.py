"""Training entrypoint for drone racing RL.

Usage:
    python -m control                              # default config
    python -m control +experiment=monorace_baseline # experiment override
    python -m control sim.n_envs=50                # CLI override
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from control.algorithms.ppo import PPO
from sim.domain_randomization import DomainRandomizer
from sim.dynamics.params import VehicleParams
from sim.envs.gate_race_env import GateRaceEnv
from sim.envs.vec_env_adapter import VecEnvAdapter

log = logging.getLogger(__name__)


def build_env(cfg: DictConfig) -> VecEnvAdapter:
    """Build a VecEnvAdapter-wrapped GateRaceEnv from Hydra config."""
    sim_cfg = cfg.sim

    params = VehicleParams(
        mass=sim_cfg.params.mass,
        arm_length=sim_cfg.params.arm_length,
        k_thrust=sim_cfg.params.k_thrust,
        k_torque=sim_cfg.params.k_torque,
        tau_motor=sim_cfg.params.tau_motor,
        prop_radius=sim_cfg.params.prop_radius,
        inertia=sim_cfg.params.inertia,
        drag_coeff=sim_cfg.params.drag_coeff,
        max_rpm=sim_cfg.params.max_rpm,
    )

    # Domain randomization
    dr_cfg = OmegaConf.to_container(cfg.domain_rand, resolve=True)
    if dr_cfg.get("enabled", False):
        domain_randomizer = DomainRandomizer.from_config(dr_cfg)
    else:
        domain_randomizer = None

    reward_weights = OmegaConf.to_container(cfg.reward.weights, resolve=True)

    env = GateRaceEnv(
        params=params,
        n_envs=sim_cfg.n_envs,
        dt=sim_cfg.dt,
        max_steps=sim_cfg.max_steps,
        gate_passage_radius=sim_cfg.gate_passage_radius,
        reward_weights=reward_weights,
        v_max=cfg.reward.get("v_max", 30.0),
        action_smoothness_threshold=cfg.reward.get("action_smoothness_threshold", 0.5),
        random_gate_start=sim_cfg.get("random_gate_start", False),
        start_behind_dist=sim_cfg.get("start_behind_dist", 1.0),
        start_vel_std=sim_cfg.get("start_vel_std", 0.5),
        start_att_std=sim_cfg.get("start_att_std", 0.1),
        start_omega_std=sim_cfg.get("start_omega_std", 0.0),
        gate_collision=sim_cfg.get("gate_collision", False),
        domain_randomizer=domain_randomizer,
        esc_nonlinearity=sim_cfg.get("esc_nonlinearity", 0.5),
        max_body_rate=sim_cfg.get("max_body_rate", 17.45),
        max_velocity=sim_cfg.get("max_velocity", 50.0),
        arena_bounds=sim_cfg.get("arena_bounds", 20.0),
    )

    return VecEnvAdapter(env)


def build_ppo(cfg: DictConfig) -> PPO:
    """Build a PPO trainer from Hydra config."""
    ctrl = cfg.control
    net_arch = OmegaConf.to_container(ctrl.net_arch, resolve=True) if "net_arch" in ctrl else None

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
        tensorboard_log=str(Path(cfg.output_dir) / "tb_logs") if cfg.get("output_dir") else None,
    )


def _setup_callbacks(cfg: DictConfig, eval_env: VecEnvAdapter | None = None, uploader=None) -> list:
    """Create SB3 training callbacks from config.

    SB3 callback frequencies count env.step() calls, not total timesteps.
    With n_envs parallel envs, divide by n_envs to get the intended
    timestep-based frequency.
    """
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    from control.callbacks import GateMetricsCallback
    from control.trajectory_recorder import TrajectoryRecorderCallback

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
        callbacks.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=cfg.n_eval_episodes,
                eval_freq=max(1, cfg.eval_freq // n_envs),
                best_model_save_path=str(output_dir / "best_model"),
                log_path=str(output_dir / "eval_logs"),
            )
        )

    # Gate racing metrics (gates/laps/termination breakdown)
    callbacks.append(GateMetricsCallback(log_freq=max(1, 100_000 // n_envs)))

    # Trajectory recording for visualisation
    viz_freq = cfg.get("viz_freq", 1_000_000)
    n_viz_episodes = cfg.get("n_viz_episodes", 5)
    callbacks.append(
        TrajectoryRecorderCallback(
            viz_freq=viz_freq,
            n_envs=n_envs,
            save_path=str(output_dir / "checkpoints"),
            n_viz_episodes=n_viz_episodes,
            uploader=uploader,
        )
    )

    return callbacks


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Training entrypoint."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)

    log.info("Building training env (n_envs=%d)...", cfg.sim.n_envs)
    train_env = build_env(cfg)

    # Build eval env — same config but fewer envs
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.sim.n_envs = cfg.n_eval_episodes
    # Eval env uses nominal physics (no domain randomization)
    OmegaConf.update(eval_cfg, "domain_rand.enabled", False)
    eval_env = build_env(eval_cfg)

    log.info("Building PPO trainer...")
    ppo = build_ppo(cfg)

    # Resume from checkpoint if specified
    resume_path = cfg.get("resume_checkpoint", None)
    if resume_path:
        log.info("Resuming from checkpoint: %s", resume_path)
        ppo.load(resume_path, env=train_env)

    # W&B init (optional)
    uploader = None
    if cfg.logging.get("backend") == "wandb":
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.logging.project,
                entity=cfg.logging.get("entity"),
                tags=list(cfg.logging.get("tags", [])),
                group=cfg.logging.get("group"),
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
            )

            from artifacts.uploader import ArtifactUploader

            uploader = ArtifactUploader(
                run_id=wandb_run.id,
                wandb_entity=wandb_run.entity,
                wandb_project=wandb_run.project,
            )
        except ImportError:
            log.warning("wandb not installed, skipping W&B logging")

    callbacks = _setup_callbacks(cfg, eval_env=eval_env, uploader=uploader)

    log.info("Starting training for %d timesteps...", cfg.total_timesteps)
    ppo.train(train_env, total_timesteps=cfg.total_timesteps, callbacks=callbacks)

    if uploader is not None:
        uploader.close(timeout=60)

    final_path = Path(cfg.output_dir) / "final_model"
    ppo.save(final_path)
    log.info("Final model saved to %s", final_path)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
