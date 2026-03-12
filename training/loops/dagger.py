"""DAgger distillation training loop for playground Phase 4.

Orchestrates teacher-student distillation: loads a Phase 3 teacher policy,
collects (EKF_obs, teacher_action) pairs with decaying beta, trains a
StudentNetwork, and saves checkpoints.

Selected via ``loop: dagger`` in Hydra config
(``_target_: training.loops.dagger.DAggerTrainingLoop``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from artifacts.uploader import ArtifactUploader

log = logging.getLogger(__name__)


class DAggerTrainingLoop:
    """DAgger distillation training loop.

    Instantiated by Hydra via ``cfg.loop._target_`` and driven by
    :meth:`run`, which receives the full config and an optional
    artifact uploader from ``training/__main__.py``.
    """

    def run(self, cfg: DictConfig, uploader: ArtifactUploader | None = None) -> None:
        """Execute the DAgger distillation pipeline."""
        import torch
        from stable_baselines3 import PPO as SB3_PPO

        from control.algorithms.dagger import DAggerTrainer

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Build environment via EnvFactory
        log.info("Building training env (n_envs=%d)...", cfg.sim.n_envs)
        env_factory = hydra.utils.instantiate(cfg.sim)
        train_env = env_factory.make_vec_env(cfg.domain_rand, cfg.reward)

        # 2. Wrap with EKF filtering if configured
        if "ekf" in cfg:
            from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper

            train_env = EKFVecEnvWrapper(train_env, **OmegaConf.to_container(cfg.ekf, resolve=True))

        # 3. Load teacher model
        teacher_path = cfg.teacher_checkpoint
        log.info("Loading teacher from %s", teacher_path)
        teacher = SB3_PPO.load(str(teacher_path))

        # 4. Create student trainer
        obs_dim = train_env.observation_space.shape[0]
        act_dim = train_env.action_space.shape[0]
        dagger_cfg = cfg.get("dagger", {})
        trainer = DAggerTrainer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=list(dagger_cfg.get("hidden", [64, 64, 64])),
            lr=dagger_cfg.get("lr", 1e-3),
            device=dagger_cfg.get("device", "cpu"),
        )

        # 5. DAgger rounds
        n_rounds = dagger_cfg.get("n_rounds", 20)
        steps_per_round = dagger_cfg.get("steps_per_round", 50_000)
        n_train_epochs = dagger_cfg.get("n_train_epochs", 10)
        beta_start = dagger_cfg.get("beta_start", 1.0)
        beta_decay = dagger_cfg.get("beta_decay", 0.9)

        log.info(
            "Starting DAgger: %d rounds, %d steps/round, beta0=%.2f, decay=%.2f",
            n_rounds,
            steps_per_round,
            beta_start,
            beta_decay,
        )

        obs = train_env.reset()

        for round_idx in range(n_rounds):
            beta = beta_start * (beta_decay ** round_idx)
            log.info("Round %d/%d, beta=%.4f", round_idx + 1, n_rounds, beta)

            # Collect data
            round_obs_list: list[np.ndarray] = []
            round_act_list: list[np.ndarray] = []
            total_gates = 0
            n_episodes = 0

            for step in range(steps_per_round):
                # Teacher provides actions
                teacher_action, _ = teacher.predict(obs, deterministic=True)

                # Student provides actions
                student_action = trainer.predict(obs)

                # Mix: beta * teacher + (1 - beta) * student
                rng_vals = np.random.random(size=(obs.shape[0], 1))
                mixed_action = np.where(rng_vals < beta, teacher_action, student_action)

                # Record (obs, teacher_action) pairs
                round_obs_list.append(obs.copy())
                round_act_list.append(teacher_action.copy())

                # Step environment
                obs, rewards, dones, infos = train_env.step(mixed_action)

                for info in infos:
                    ep = info.get("episode")
                    if ep is not None:
                        total_gates += ep.get("gates_passed", 0)
                        n_episodes += 1

            # Add collected data to trainer buffer
            round_obs = np.concatenate(round_obs_list, axis=0)
            round_acts = np.concatenate(round_act_list, axis=0)
            trainer.add_data(round_obs, round_acts)

            # Train student
            for epoch in range(n_train_epochs):
                loss = trainer.train_epoch()

            # Evaluate student (brief rollout)
            eval_gates = 0
            eval_obs = train_env.reset()
            for _ in range(1000):
                eval_action = trainer.predict(eval_obs)
                eval_obs, _, eval_dones, eval_infos = train_env.step(eval_action)
                for info in eval_infos:
                    ep = info.get("episode")
                    if ep is not None:
                        eval_gates += ep.get("gates_passed", 0)

            log.info(
                "Round %d: loss=%.6f, collect_gates=%d, eval_gates=%d",
                round_idx + 1,
                loss,
                total_gates,
                eval_gates,
            )

            # Save checkpoint
            ckpt_path = output_dir / f"student_round{round_idx + 1}.pt"
            trainer.save(str(ckpt_path))

            if uploader is not None:
                uploader.submit(ckpt_path, round_idx + 1)

        # 6. Save final student
        final_path = output_dir / "student_final.pt"
        trainer.save(str(final_path))
        log.info("Final student saved to %s", final_path)

        if uploader is not None:
            uploader.submit(final_path, n_rounds)

        train_env.close()
