"""Seed the autoresearch state with the current best baseline.

Usage:
    python -m autoresearch.seed_baseline [--trajectory PATH]

Without --trajectory, uses estimated descriptors from noble-wildflower-115 metrics.
With --trajectory, computes real descriptors from a .npz file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autoresearch.archive.map_elites import CellEntry, MapElitesArchive
from autoresearch.archive.serialization import load_archive, save_archive
from autoresearch.descriptors.compute import DescriptorVector, compute_descriptors
from autoresearch.runner import compute_fitness
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.serialization import load_tree, save_tree


# noble-wildflower-115 metrics
BASELINE = {
    "wandb_run_id": "noble-wildflower-115",
    "git_commit": "44d3f78",
    "hypothesis_id": "baseline_v1",
    "gates_per_ep": 2.07,
    "success_rate": 0.405,
    "avg_speed_ms": 3.0,
    "best_laps": 2,
    "total_steps": 20_000_000,
    "checkpoint": "outputs/2026-03-25/18-11-09/final_model",
    "config": "configs/experiment/mlp_fixed_long.yaml",
}

# Estimated descriptors from noble-wildflower-115 behavior:
# - actuator_utilization ~0.45: moderate throttle (3 m/s is gentle)
# - control_smoothness ~0.6: reasonably smooth but not perfect
# - aero_regime ~3.0: matches avg speed of 3.0 m/s
ESTIMATED_DESCRIPTORS = DescriptorVector(
    actuator_utilization=0.45,
    control_smoothness=0.6,
    aero_regime=3.0,
)


def seed(state_dir: Path, trajectory_path: Path | None = None) -> None:
    archive = load_archive(state_dir / "archive.json")
    tree = load_tree(state_dir / "tree.json")

    # Compute or use estimated descriptors
    if trajectory_path is not None:
        from autoresearch.analysis.trajectory import load_trajectory

        traj = load_trajectory(trajectory_path)
        descriptors = compute_descriptors(traj, max_rpm=31470.0, control_freq=100.0)
        print(f"Computed descriptors from trajectory: {descriptors}")
    else:
        descriptors = ESTIMATED_DESCRIPTORS
        print(f"Using estimated descriptors: {descriptors}")

    # Fitness = composite score: lower is better
    # We use negative of (gates/ep * success_rate * avg_speed) so lower = better
    # This rewards all three objectives simultaneously
    fitness = compute_fitness(
        gates_per_ep=BASELINE["gates_per_ep"],
        success_rate=BASELINE["success_rate"],
        avg_speed=BASELINE["avg_speed_ms"],
    )
    print(f"Baseline fitness: {fitness:.4f} (lower is better)")

    # Insert into archive
    entry = CellEntry(
        fitness=fitness,
        status="approved",
        wandb_run_id=BASELINE["wandb_run_id"],
        git_commit=BASELINE["git_commit"],
        descriptors=descriptors,
        constraint_results={"baseline": True},
        budget_spent=BASELINE["total_steps"],
        hypothesis_id=BASELINE["hypothesis_id"],
    )
    inserted = archive.try_insert(entry)
    cell = archive.descriptor_to_cell(descriptors)
    print(f"Archive cell {cell}: {'inserted' if inserted else 'already occupied with better'}")

    # Update tree baseline node with fitness and W&B run
    root = tree.root
    root.fitness = fitness
    root.wandb_run_id = BASELINE["wandb_run_id"]
    root.archive_cell = cell

    save_archive(archive, state_dir / "archive.json")
    save_tree(tree, state_dir / "tree.json")
    print(f"State saved to {state_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed autoresearch baseline")
    parser.add_argument(
        "--state-dir",
        default="autoresearch/state/",
        help="Path to state directory",
    )
    parser.add_argument(
        "--trajectory",
        default=None,
        help="Optional .npz trajectory file for real descriptor computation",
    )
    args = parser.parse_args(argv)
    seed(Path(args.state_dir), Path(args.trajectory) if args.trajectory else None)


if __name__ == "__main__":
    main()
