"""Log benchmark results to W&B."""
from __future__ import annotations

import logging

import wandb

from benchmark.results import BenchmarkResults

log = logging.getLogger(__name__)


def log_benchmark_results(results: BenchmarkResults) -> None:
    """Log BenchmarkResults to the current W&B run."""
    if wandb.run is None:
        log.warning("No active W&B run; skipping benchmark logging.")
        return

    summary = {}

    # Per-environment metrics
    for env_r in results.per_environment:
        prefix = f"benchmark/{env_r.layout_name}/variant_{env_r.variant_index}"
        summary[f"{prefix}/lap_time"] = env_r.lap_time
        summary[f"{prefix}/laps_completed"] = env_r.laps_completed
        summary[f"{prefix}/gates_passed"] = env_r.gates_passed
        summary[f"{prefix}/crash"] = int(env_r.crash)
        summary[f"{prefix}/avg_speed"] = env_r.avg_speed
        summary[f"{prefix}/termination_reason"] = env_r.termination_reason
        if env_r.rerun_url:
            summary[f"{prefix}/rerun_url"] = env_r.rerun_url

    # Per-layout metrics
    for lr in results.per_layout:
        prefix = f"benchmark/{lr.layout_name}"
        summary[f"{prefix}/fastest_lap"] = lr.fastest_lap
        summary[f"{prefix}/avg_lap"] = lr.avg_lap
        summary[f"{prefix}/laps_completed"] = lr.laps_completed
        summary[f"{prefix}/gates_passed"] = lr.gates_passed
        summary[f"{prefix}/crash_rate"] = lr.crash_rate
        summary[f"{prefix}/avg_speed"] = lr.avg_speed

    # Aggregate metrics
    agg = results.aggregate
    summary["benchmark/aggregate/fastest_lap"] = agg.fastest_lap
    summary["benchmark/aggregate/avg_lap"] = agg.avg_lap
    summary["benchmark/aggregate/laps_completed"] = agg.laps_completed
    summary["benchmark/aggregate/gates_passed"] = agg.gates_passed
    summary["benchmark/aggregate/crash_rate"] = agg.crash_rate
    summary["benchmark/aggregate/avg_speed"] = agg.avg_speed

    # Metadata
    summary["benchmark/golden_set_version"] = results.golden_set_version
    summary["benchmark/golden_set_mode"] = results.golden_set_mode
    summary["benchmark/checkpoint_path"] = results.checkpoint_path

    wandb.run.summary.update(summary)

    # Add tags
    mode_tag = f"golden-{results.golden_set_mode}"
    existing_tags = list(wandb.run.tags or [])
    for tag in ["golden-benchmark", mode_tag]:
        if tag not in existing_tags:
            existing_tags.append(tag)
    wandb.run.tags = existing_tags

    log.info("Benchmark results logged to W&B (run: %s)", wandb.run.id)
