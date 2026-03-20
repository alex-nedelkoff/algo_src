"""Early stopping signal computation for training runs."""

from __future__ import annotations

from enum import Enum

from autoresearch.archive.map_elites import MapElitesArchive


class EarlyStopSignal(Enum):
    CONTINUE = "continue"
    POOR_PERFORMANCE = "poor_performance"
    ARCHIVE_REDUNDANT = "archive_redundant"


def check_early_stop(
    current_fitness: float,
    baseline_fitness: float | None,
    steps_completed: int,
    budget: int,
    baseline_threshold: float,
    archive: MapElitesArchive,
    target_cell: tuple[int, int, int],
) -> EarlyStopSignal:
    """Check whether to early-stop a training run.

    Args:
        current_fitness: Best fitness (lap time) observed so far. Lower is better.
        baseline_fitness: Best fitness in archive when experiment started. None for first experiment.
        steps_completed: Training steps completed so far.
        budget: Total step budget for this experiment.
        baseline_threshold: Performance ratio threshold (e.g., 0.7).
        archive: Current MAP-Elites archive.
        target_cell: Archive cell this experiment targets.
    """
    # First experiment (no baseline) always continues
    if baseline_fitness is None:
        return EarlyStopSignal.CONTINUE

    # Both stopping checks are only applied after 1/3 of budget has been spent
    if steps_completed >= budget / 3:
        # Performance check: for lap time (lower=better), reject if current is much worse than baseline.
        # "70% of baseline" means current should be <= baseline / threshold
        max_acceptable = baseline_fitness / baseline_threshold
        if current_fitness > max_acceptable:
            return EarlyStopSignal.POOR_PERFORMANCE

        # Archive redundancy: target cell already has a better incumbent
        incumbent = archive.get(target_cell)
        if incumbent is not None and current_fitness > incumbent.fitness:
            return EarlyStopSignal.ARCHIVE_REDUNDANT

    return EarlyStopSignal.CONTINUE
