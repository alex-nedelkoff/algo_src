#!/usr/bin/env python
"""Evaluate a DAgger student (.pt) checkpoint on the playground environment.

Usage:
    python scripts/evaluate_student.py <student_path> [options]

Example:
    python scripts/evaluate_student.py \
        ../playground/runs/phase4_dagger_v3/student_final.pt \
        --n_episodes 100 --ekf --corner_noise_k 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def build_student(weights_path: str, obs_dim: int = 24, act_dim: int = 4) -> torch.nn.Module:
    """Load a student network from a .pt checkpoint."""
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, act_dim),
        torch.nn.Sigmoid(),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Handle "net." prefix from playground's StudentNetwork wrapper
    cleaned = {k.removeprefix("net."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model


def evaluate(
    student: torch.nn.Module,
    n_episodes: int = 100,
    max_steps: int = 6000,
    n_envs: int = 10,
    use_ekf: bool = False,
    corner_noise_k: float = 2.0,
    dr_pct: float = 0.0,
) -> dict:
    """Run student through playground env, return metrics."""
    from omegaconf import OmegaConf

    from sim.envs.playground_factory import PlaygroundEnvFactory

    factory = PlaygroundEnvFactory(n_envs=n_envs)

    # make_vec_env expects DictConfig objects
    dr_cfg = OmegaConf.create({"enabled": dr_pct > 0, "percentage": dr_pct})
    reward_cfg = OmegaConf.create({"preset": "M23"})

    env = factory.make_vec_env(dr_cfg, reward_cfg)

    if use_ekf:
        from sim.envs.ekf_env_wrapper import EKFVecEnvWrapper
        env = EKFVecEnvWrapper(
            env, corner_noise_k=corner_noise_k, provide_teacher_obs=False,
        )

    gates_passed = []
    steps_taken = []
    episodes_done = 0
    obs = env.reset()

    step = 0
    max_total_steps = max_steps * n_episodes // n_envs
    while episodes_done < n_episodes and step < max_total_steps:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            action = student(obs_t).numpy()

        obs, rewards, dones, infos = env.step(action)
        step += 1

        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                gates_passed.append(ep.get("gates_passed", 0))
                steps_taken.append(ep.get("l", 0))
                episodes_done += 1

    env.close()

    return {
        "n_episodes": len(gates_passed),
        "avg_gates": float(np.mean(gates_passed)) if gates_passed else 0.0,
        "std_gates": float(np.std(gates_passed)) if gates_passed else 0.0,
        "max_gates": int(np.max(gates_passed)) if gates_passed else 0,
        "avg_steps": float(np.mean(steps_taken)) if steps_taken else 0.0,
        "use_ekf": use_ekf,
        "corner_noise_k": corner_noise_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DAgger student checkpoint")
    parser.add_argument("student_path", type=str, help="Path to student .pt file")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_envs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--ekf", action="store_true", help="Use EKF wrapper")
    parser.add_argument("--corner_noise_k", type=float, default=2.0)
    parser.add_argument("--dr_pct", type=float, default=0.0, help="Domain rand percentage")
    args = parser.parse_args()

    print(f"Loading student from {args.student_path}")
    student = build_student(args.student_path)

    print(f"Evaluating ({args.n_episodes} episodes, ekf={args.ekf})...")
    results = evaluate(
        student,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        use_ekf=args.ekf,
        corner_noise_k=args.corner_noise_k,
        dr_pct=args.dr_pct,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
