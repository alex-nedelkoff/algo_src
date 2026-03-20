"""Tests for branch selection algorithm."""

import pytest
import numpy as np

from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.selection import (
    score_branch, select_next_branch, behavioral_distance, BranchStats,
)
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def tree_with_branches():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment("a1", "baseline_v1", "algorithm", "Branch A exp 1", "completed")
    tree.complete_experiment("a1", fitness=5.0, wandb_run_id="run_a1")
    tree.add_experiment("a2", "a1", "algorithm", "Branch A exp 2", "completed")
    tree.complete_experiment("a2", fitness=4.0, wandb_run_id="run_a2")
    tree.add_experiment("b1", "baseline_v1", "algorithm", "Branch B exp 1", "completed")
    tree.complete_experiment("b1", fitness=6.0, wandb_run_id="run_b1")
    return tree


@pytest.fixture
def archive_with_entries():
    archive = MapElitesArchive()
    archive.try_insert(CellEntry(
        fitness=4.0, status="approved", wandb_run_id="run_a2",
        git_commit="abc", descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={}, budget_spent=5_000_000, hypothesis_id="a2",
    ))
    return archive


@pytest.fixture
def default_config():
    return {
        "exploit_weight": 1.0, "explore_weight": 1.0, "gap_weight": 0.5,
        "diversity_weight": 0.3, "random_restart_pct": 0.0,
        "min_branch_experiments": 3, "tie_threshold": 0.05,
    }


class TestBehavioralDistance:
    def test_identical_is_zero(self):
        a = DescriptorVector(0.5, 0.5, 10.0)
        b = DescriptorVector(0.5, 0.5, 10.0)
        ranges = [(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)]
        assert behavioral_distance(a, b, ranges) == pytest.approx(0.0)

    def test_max_distance(self):
        a = DescriptorVector(0.25, 0.0, 0.0)
        b = DescriptorVector(1.0, 1.0, 24.0)
        ranges = [(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)]
        assert behavioral_distance(a, b, ranges) == pytest.approx(np.sqrt(3.0), abs=0.01)

    def test_symmetry(self):
        a = DescriptorVector(0.5, 0.3, 10.0)
        b = DescriptorVector(0.8, 0.7, 5.0)
        ranges = [(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)]
        assert behavioral_distance(a, b, ranges) == pytest.approx(behavioral_distance(b, a, ranges))


class TestBranchStats:
    def test_compute_from_tree(self, tree_with_branches):
        stats = BranchStats.from_tree(tree_with_branches, "a1")
        assert stats.best_fitness == 4.0
        assert stats.visit_count == 2

    def test_single_experiment_branch(self, tree_with_branches):
        stats = BranchStats.from_tree(tree_with_branches, "b1")
        assert stats.best_fitness == 6.0
        assert stats.visit_count == 1


class TestScoreBranch:
    def test_better_fitness_scores_higher_exploit(self, archive_with_entries, tree_with_branches, default_config):
        default_config["min_branch_experiments"] = 0
        default_config["explore_weight"] = 0  # isolate exploit term
        default_config["gap_weight"] = 0
        default_config["diversity_weight"] = 0
        stats_a = BranchStats(best_fitness=4.0, visit_count=2)
        stats_b = BranchStats(best_fitness=6.0, visit_count=2)
        score_a = score_branch(stats_a, archive_with_entries, tree_with_branches, default_config)
        score_b = score_branch(stats_b, archive_with_entries, tree_with_branches, default_config)
        assert score_a > score_b  # lower fitness = higher exploit score

    def test_minimum_exploration_guarantee(self, archive_with_entries, tree_with_branches, default_config):
        stats = BranchStats(best_fitness=100.0, visit_count=1)
        score = score_branch(stats, archive_with_entries, tree_with_branches, default_config)
        assert score == float("inf")

    def test_gap_bonus_included(self, archive_with_entries, tree_with_branches, default_config):
        default_config["min_branch_experiments"] = 0
        default_config["exploit_weight"] = 0
        default_config["explore_weight"] = 0
        default_config["diversity_weight"] = 0
        stats = BranchStats(best_fitness=5.0, visit_count=3, empty_target_ratio=0.8)
        score = score_branch(stats, archive_with_entries, tree_with_branches, default_config)
        assert score == pytest.approx(0.5 * 0.8)  # gap_weight * empty_target_ratio

    def test_zero_fitness_no_crash(self, archive_with_entries, tree_with_branches, default_config):
        default_config["min_branch_experiments"] = 0
        stats = BranchStats(best_fitness=0.0, visit_count=5)
        score = score_branch(stats, archive_with_entries, tree_with_branches, default_config)
        assert score > 0 and score != float("inf")


class TestSelectNextBranch:
    def test_selects_a_branch(self, archive_with_entries, tree_with_branches, default_config):
        default_config["min_branch_experiments"] = 0
        result = select_next_branch(tree_with_branches, archive_with_entries, default_config, seed=42)
        assert result in ["a1", "b1"]

    def test_random_restart_returns_none(self, archive_with_entries, tree_with_branches, default_config):
        default_config["random_restart_pct"] = 1.0
        result = select_next_branch(tree_with_branches, archive_with_entries, default_config, seed=42)
        assert result is None

    def test_empty_tree_returns_none(self, archive_with_entries, default_config):
        tree = ResearchTree.create_with_baseline("baseline", "abc")
        result = select_next_branch(tree, archive_with_entries, default_config, seed=42)
        assert result is None
