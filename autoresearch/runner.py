"""Durable monitoring process for training runs.

Runs independently of Claude Code session. Monitors a W&B training run,
refreshes coordination claims, and records results to state files on completion.

Usage:
    nohup python -m autoresearch.runner \
        --run-id <wandb_run_id> \
        --claim-id <hypothesis_id> \
        --state-dir autoresearch/state/ &
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("autoresearch.runner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner CLI arguments."""
    parser = argparse.ArgumentParser(description="Auto-research durable monitor")
    parser.add_argument("--run-id", required=True, help="W&B run ID to monitor")
    parser.add_argument("--claim-id", required=True, help="Hypothesis ID for claim refresh")
    parser.add_argument("--wandb-project", default="corvidx-drone-racing")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (defaults to default entity)")
    parser.add_argument("--state-dir", default="autoresearch/state/")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls")
    parser.add_argument("--claim-refresh-minutes", type=int, default=15)
    parser.add_argument("--max-rpm", type=float, default=31470.0)
    parser.add_argument("--control-freq", type=float, default=100.0)
    parser.add_argument("--budget", type=int, default=5_000_000)
    parser.add_argument("--baseline-fitness", type=float, default=None)
    parser.add_argument("--baseline-threshold", type=float, default=0.7)
    parser.add_argument("--trajectory-dir", default=None, help="Dir to look for .npz trajectory files on completion")
    return parser.parse_args(argv)


def refresh_claim(state_dir: Path, claim_id: str) -> None:
    """Refresh the timestamp on an active claim."""
    from autoresearch.coordination.claims import load_claims, save_claims

    claims_path = state_dir / "claims.json"
    claims = load_claims(claims_path)
    for c in claims:
        if c.hypothesis_id == claim_id and c.status == "running":
            c.timestamp = datetime.now(timezone.utc)
    save_claims(claims, claims_path)


@dataclass
class RunSnapshot:
    """Snapshot of a W&B run's current state."""

    state: str  # "running", "finished", "crashed", "failed"
    step: int
    summary: dict  # latest summary metrics
    config: dict


def poll_wandb_run(run_path: str) -> RunSnapshot:
    """Query W&B API for current run status and metrics.

    Args:
        run_path: Full run path "entity/project/run_id"

    Returns:
        RunSnapshot with current state and metrics.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    return RunSnapshot(
        state=run.state,
        step=run.lastHistoryStep,
        summary=dict(run.summary),
        config=dict(run.config),
    )


def resolve_run_path(args: argparse.Namespace) -> str:
    """Build W&B run path from args, auto-detecting entity if needed."""
    if args.wandb_entity:
        entity = args.wandb_entity
    else:
        import wandb

        api = wandb.Api()
        entity = api.default_entity
    return f"{entity}/{args.wandb_project}/{args.run_id}"


def extract_metrics(summary: dict) -> dict:
    """Extract relevant training metrics from W&B summary.

    Looks for eval metrics with common naming patterns from our training stack.
    """
    metrics = {}

    # Try eval/ prefixed keys first (our WandbCallback format), then unprefixed
    for key, target in [
        ("eval/gates_per_episode", "gates_per_ep"),
        ("eval/success_rate", "success_rate"),
        ("eval/avg_speed", "avg_speed"),
        ("eval/mean_reward", "mean_reward"),
        ("eval/best_laps", "best_laps"),
        ("eval/avg_lap_time", "avg_lap_time"),
    ]:
        if key in summary:
            metrics[target] = summary[key]
        elif key.replace("eval/", "") in summary:
            metrics[target] = summary[key.replace("eval/", "")]

    # Fallback: look for rollout/ keys (SB3 default)
    if "gates_per_ep" not in metrics and "rollout/ep_rew_mean" in summary:
        metrics["mean_reward"] = summary["rollout/ep_rew_mean"]

    return metrics


def compute_fitness(
    gates_per_ep: float,
    success_rate: float,
    avg_speed: float,
) -> float:
    """Composite fitness: negative product of objectives (lower = better).

    Rewards improvements in ANY of the three axes:
    - gates_per_ep: more gates completed per episode
    - success_rate: higher fraction of successful laps
    - avg_speed: faster average flight speed (m/s)
    """
    return -(gates_per_ep * success_rate * avg_speed)


def on_run_complete(args: argparse.Namespace, snapshot: RunSnapshot) -> None:
    """Handle run completion: update archive, tree, and release claim."""
    from autoresearch.archive.serialization import load_archive, save_archive
    from autoresearch.archive.map_elites import CellEntry
    from autoresearch.coordination.claims import load_claims, release_claim, save_claims
    from autoresearch.descriptors.compute import DescriptorVector
    from autoresearch.tree.serialization import load_tree, save_tree

    state_dir = Path(args.state_dir)
    metrics = extract_metrics(snapshot.summary)

    gates_per_ep = metrics.get("gates_per_ep", 0.0)
    success_rate = metrics.get("success_rate", 0.0)
    avg_speed = metrics.get("avg_speed", 0.0)

    logger.info(
        f"Run complete — gates/ep: {gates_per_ep:.2f}, "
        f"success: {success_rate:.1%}, speed: {avg_speed:.1f} m/s"
    )

    fitness = compute_fitness(gates_per_ep, success_rate, avg_speed)
    logger.info(f"Computed fitness: {fitness:.4f}")

    # Try to compute real descriptors from trajectory
    descriptors = None
    if args.trajectory_dir:
        descriptors = _try_compute_descriptors(
            Path(args.trajectory_dir), args.max_rpm, args.control_freq
        )

    if descriptors is None:
        # Estimate descriptors from summary metrics
        descriptors = _estimate_descriptors(metrics)
        logger.info(f"Using estimated descriptors: {descriptors}")
    else:
        logger.info(f"Computed descriptors from trajectory: {descriptors}")

    # Update archive
    archive = load_archive(state_dir / "archive.json")
    entry = CellEntry(
        fitness=fitness,
        status="candidate",
        wandb_run_id=args.run_id,
        git_commit="",  # filled by the launching skill
        descriptors=descriptors,
        constraint_results={},
        budget_spent=snapshot.step,
        hypothesis_id=args.claim_id,
    )
    inserted = archive.try_insert(entry)
    cell = archive.descriptor_to_cell(descriptors)
    logger.info(f"Archive cell {cell}: {'inserted' if inserted else 'not inserted (worse fitness)'}")
    save_archive(archive, state_dir / "archive.json")

    # Update tree
    tree = load_tree(state_dir / "tree.json")
    try:
        tree.complete_experiment(
            args.claim_id,
            fitness=fitness,
            wandb_run_id=args.run_id,
            archive_cell=cell if inserted else None,
        )
        save_tree(tree, state_dir / "tree.json")
        logger.info("Tree updated with completion")
    except KeyError:
        logger.warning(f"Hypothesis {args.claim_id} not found in tree — skipping tree update")

    # Release claim
    claims = load_claims(state_dir / "claims.json")
    claims = release_claim(claims, args.claim_id, "completed")
    save_claims(claims, state_dir / "claims.json")
    logger.info("Claim released")


def on_run_failed(args: argparse.Namespace, snapshot: RunSnapshot) -> None:
    """Handle run failure: update tree and release claim."""
    from autoresearch.coordination.claims import load_claims, release_claim, save_claims
    from autoresearch.tree.serialization import load_tree, save_tree

    state_dir = Path(args.state_dir)
    reason = f"W&B state: {snapshot.state}"
    logger.warning(f"Run failed: {reason}")

    tree = load_tree(state_dir / "tree.json")
    try:
        tree.fail_experiment(args.claim_id, reason)
        save_tree(tree, state_dir / "tree.json")
    except KeyError:
        logger.warning(f"Hypothesis {args.claim_id} not found in tree")

    claims = load_claims(state_dir / "claims.json")
    claims = release_claim(claims, args.claim_id, "failed")
    save_claims(claims, state_dir / "claims.json")
    logger.info("Claim released as failed")


def _try_compute_descriptors(
    traj_dir: Path, max_rpm: float, control_freq: float
) -> DescriptorVector | None:
    """Try to load the most recent .npz trajectory and compute descriptors."""
    from autoresearch.descriptors.compute import DescriptorVector, compute_descriptors

    npz_files = sorted(traj_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not npz_files:
        logger.info(f"No .npz files found in {traj_dir}")
        return None

    try:
        from autoresearch.analysis.trajectory import load_trajectory

        traj = load_trajectory(npz_files[0])
        return compute_descriptors(traj, max_rpm, control_freq)
    except Exception:
        logger.exception(f"Failed to compute descriptors from {npz_files[0]}")
        return None


def _estimate_descriptors(metrics: dict) -> DescriptorVector:
    """Estimate behavioral descriptors from summary metrics when no trajectory is available."""
    from autoresearch.descriptors.compute import DescriptorVector

    avg_speed = metrics.get("avg_speed", 3.0)
    # Rough heuristics: faster flight = higher actuator use and aero regime
    actuator_est = min(1.0, max(0.25, avg_speed / 15.0 + 0.25))
    smoothness_est = 0.5  # default mid-range without trajectory data
    aero_est = avg_speed
    return DescriptorVector(
        actuator_utilization=actuator_est,
        control_smoothness=smoothness_est,
        aero_regime=aero_est,
    )


def run_monitor(args: argparse.Namespace) -> None:
    """Main monitoring loop. Polls W&B, refreshes claims, checks early stop."""
    state_dir = Path(args.state_dir)
    last_claim_refresh = time.time()
    claim_refresh_interval = args.claim_refresh_minutes * 60

    run_path = resolve_run_path(args)
    logger.info(f"Monitoring W&B run {run_path} for claim {args.claim_id}")
    logger.info(f"Poll interval: {args.poll_interval}s, budget: {args.budget}")

    best_fitness = None

    while True:
        try:
            # Refresh claim periodically
            if time.time() - last_claim_refresh >= claim_refresh_interval:
                refresh_claim(state_dir, args.claim_id)
                last_claim_refresh = time.time()
                logger.info("Claim timestamp refreshed")

            # Poll W&B for run status
            try:
                snapshot = poll_wandb_run(run_path)
            except Exception:
                logger.exception("Failed to poll W&B — will retry next interval")
                time.sleep(args.poll_interval)
                continue

            # Handle terminal states
            if snapshot.state in ("finished",):
                logger.info(f"Run finished at step {snapshot.step}")
                on_run_complete(args, snapshot)
                break

            if snapshot.state in ("crashed", "failed"):
                on_run_failed(args, snapshot)
                break

            # Check early stopping signals
            metrics = extract_metrics(snapshot.summary)
            gates_per_ep = metrics.get("gates_per_ep", 0.0)
            success_rate = metrics.get("success_rate", 0.0)
            avg_speed = metrics.get("avg_speed", 0.0)
            current_fitness = compute_fitness(gates_per_ep, success_rate, avg_speed)

            if best_fitness is None or current_fitness < best_fitness:
                best_fitness = current_fitness

            logger.info(
                f"Step {snapshot.step}/{args.budget} — "
                f"fitness: {current_fitness:.4f}, best: {best_fitness:.4f}, "
                f"gates/ep: {gates_per_ep:.2f}, success: {success_rate:.1%}, "
                f"speed: {avg_speed:.1f} m/s"
            )

            # Early stopping check (only if we have a baseline to compare against)
            if args.baseline_fitness is not None and snapshot.step > 0:
                from autoresearch.analysis.early_stopping import check_early_stop, EarlyStopSignal
                from autoresearch.archive.serialization import load_archive

                archive = load_archive(state_dir / "archive.json")
                target_cell = archive.descriptor_to_cell(_estimate_descriptors(metrics))

                signal = check_early_stop(
                    current_fitness=best_fitness,
                    baseline_fitness=args.baseline_fitness,
                    steps_completed=snapshot.step,
                    budget=args.budget,
                    baseline_threshold=args.baseline_threshold,
                    archive=archive,
                    target_cell=target_cell,
                )

                if signal != EarlyStopSignal.CONTINUE:
                    logger.warning(f"Early stop triggered: {signal.value}")
                    on_run_failed(
                        args,
                        RunSnapshot(
                            state=f"early_stopped:{signal.value}",
                            step=snapshot.step,
                            summary=snapshot.summary,
                            config=snapshot.config,
                        ),
                    )
                    break

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            logger.info("Runner interrupted")
            break
        except Exception:
            logger.exception("Runner error, will retry")
            time.sleep(args.poll_interval)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m autoresearch.runner`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    run_monitor(args)


if __name__ == "__main__":
    main()
