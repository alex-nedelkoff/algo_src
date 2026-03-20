"""Tests for baseline promotion workflow."""

from pathlib import Path
import pytest
from autoresearch.promotion import (
    verify_promotion_candidate, generate_pr_body,
    update_state_after_promotion, PromotionCandidate, PromotionResult,
)
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.serialization import save_tree, load_tree
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def candidate():
    return PromotionCandidate(
        hypothesis_id="hyp_win", wandb_run_id="run_win", fitness=3.5,
        descriptors=DescriptorVector(0.6, 0.2, 12.0),
        branch="ar/exp-hyp_win", git_commit="win123",
    )

@pytest.fixture
def archive_with_winner():
    archive = MapElitesArchive()
    archive.try_insert(CellEntry(
        fitness=3.5, status="approved", wandb_run_id="run_win",
        git_commit="win123", descriptors=DescriptorVector(0.6, 0.2, 12.0),
        constraint_results={"passed": True}, budget_spent=15_000_000,
        hypothesis_id="hyp_win",
    ))
    return archive


def test_valid_candidate_passes(candidate, archive_with_winner):
    result = verify_promotion_candidate(candidate, archive_with_winner)
    assert result.verified is True

def test_candidate_not_in_archive(candidate):
    result = verify_promotion_candidate(candidate, MapElitesArchive())
    assert result.verified is False
    assert "not found" in result.reason.lower()

def test_candidate_not_approved(candidate):
    archive = MapElitesArchive()
    archive.try_insert(CellEntry(
        fitness=3.5, status="candidate", wandb_run_id="run_win",
        git_commit="win123", descriptors=DescriptorVector(0.6, 0.2, 12.0),
        constraint_results={}, budget_spent=15_000_000, hypothesis_id="hyp_win",
    ))
    result = verify_promotion_candidate(candidate, archive)
    assert result.verified is False
    assert "approved" in result.reason.lower()

def test_pr_body_contains_sections(candidate):
    body = generate_pr_body(candidate, old_baseline_fitness=5.0, wandb_url="https://wandb.ai/test/run")
    assert "## Summary" in body
    assert "3.500" in body
    assert "5.0" in body
    assert "wandb.ai" in body
    assert "improvement" in body.lower() or "better" in body.lower()

def test_state_update(candidate, archive_with_winner, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    save_archive(archive_with_winner, state_dir / "archive.json")
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment("hyp_win", "baseline_v1", "algorithm", "Winner", "completed")
    tree.complete_experiment("hyp_win", fitness=3.5, wandb_run_id="run_win")
    save_tree(tree, state_dir / "tree.json")

    update_state_after_promotion(
        state_dir=state_dir, hypothesis_id="hyp_win",
        merge_commit="merge_abc", new_baseline_name="baseline_v2", promoter="shaan",
    )

    assert (state_dir / "archive_v1.json").exists()
    new_archive = load_archive(state_dir / "archive.json")
    assert new_archive.n_occupied == 0
    new_tree = load_tree(state_dir / "tree.json")
    assert new_tree.root.hypothesis_id == "baseline_v2"
    assert new_tree.root.git_commit == "merge_abc"
    assert new_tree.get_node("baseline_v1") is not None

def test_promote_baseline_method():
    tree = ResearchTree.create_with_baseline("baseline", "abc")
    new_root = tree.promote_baseline("baseline_v2", "merge123", "Promoted", promoter="shaan")
    assert tree.root.hypothesis_id == "baseline_v2"
    assert tree.get_node("baseline_v1").hypothesis_id == "baseline_v1"
    assert new_root.parent_id == "baseline_v1"
