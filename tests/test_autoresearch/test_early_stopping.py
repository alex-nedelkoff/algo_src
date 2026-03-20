"""Tests for early stopping signal computation."""

import pytest
from autoresearch.analysis.early_stopping import EarlyStopSignal, check_early_stop
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def archive_with_incumbent():
    archive = MapElitesArchive()
    archive.try_insert(CellEntry(
        fitness=5.0, status="approved", wandb_run_id="run_best",
        git_commit="abc", descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={}, budget_spent=5_000_000, hypothesis_id="best",
    ))
    return archive


class TestEarlyStop:
    def test_no_stop_when_performing_well(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=4.5, baseline_fitness=5.0,
            steps_completed=2_000_000, budget=5_000_000,
            baseline_threshold=0.7, archive=archive_with_incumbent, target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.CONTINUE

    def test_stop_poor_performance(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=20.0, baseline_fitness=5.0,
            steps_completed=2_000_000, budget=5_000_000,
            baseline_threshold=0.7, archive=archive_with_incumbent, target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.POOR_PERFORMANCE

    def test_no_stop_poor_performance_early(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=20.0, baseline_fitness=5.0,
            steps_completed=500_000, budget=5_000_000,
            baseline_threshold=0.7, archive=archive_with_incumbent, target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.CONTINUE

    def test_stop_archive_redundancy(self, archive_with_incumbent):
        # The incumbent is at cell (1,1,2) with fitness 5.0
        # Current fitness 6.0 is worse
        signal = check_early_stop(
            current_fitness=6.0, baseline_fitness=5.0,
            steps_completed=3_000_000, budget=5_000_000,
            baseline_threshold=0.7, archive=archive_with_incumbent, target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.ARCHIVE_REDUNDANT

    def test_no_stop_first_experiment(self):
        signal = check_early_stop(
            current_fitness=100.0, baseline_fitness=None,
            steps_completed=3_000_000, budget=5_000_000,
            baseline_threshold=0.7, archive=MapElitesArchive(), target_cell=(0, 0, 0),
        )
        assert signal == EarlyStopSignal.CONTINUE

    def test_empty_target_cell_no_redundancy(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=6.0, baseline_fitness=5.0,
            steps_completed=3_000_000, budget=5_000_000,
            baseline_threshold=0.7, archive=archive_with_incumbent, target_cell=(4, 4, 4),
        )
        assert signal == EarlyStopSignal.CONTINUE

    def test_better_than_incumbent_continues(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=3.0, baseline_fitness=5.0,
            steps_completed=3_000_000, budget=5_000_000,
            baseline_threshold=0.7, archive=archive_with_incumbent, target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.CONTINUE
