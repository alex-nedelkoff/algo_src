"""Tests for cross-pollination."""

import pytest
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.cross_pollination import (
    find_successful_patterns, propose_cross_pollinations, CrossPollinationProposal,
)


@pytest.fixture
def tree_with_successful_branches():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment("a1", "baseline_v1", "hyperparameter", "Increase gate_passage weight to 30", "completed")
    tree.complete_experiment("a1", fitness=4.0, wandb_run_id="run_a1")
    tree.add_experiment("a2", "a1", "hyperparameter", "Also reduce crash_penalty to 5", "completed")
    tree.complete_experiment("a2", fitness=3.5, wandb_run_id="run_a2")
    tree.add_experiment("b1", "baseline_v1", "hyperparameter", "Try larger network [128,128,128]", "completed")
    tree.complete_experiment("b1", fitness=7.0, wandb_run_id="run_b1")
    return tree


def test_finds_improving_branches(tree_with_successful_branches):
    patterns = find_successful_patterns(tree_with_successful_branches)
    assert len(patterns) >= 1
    assert patterns[0].branch_root_id == "a1"

def test_empty_tree_no_patterns():
    tree = ResearchTree.create_with_baseline("baseline", "abc")
    assert find_successful_patterns(tree) == []

def test_proposes_grafting(tree_with_successful_branches):
    proposals = propose_cross_pollinations(tree_with_successful_branches)
    assert len(proposals) >= 1
    for p in proposals:
        assert isinstance(p, CrossPollinationProposal)
        assert p.source_branch != p.target_branch
        assert p.inspired_by is not None

def test_no_proposals_when_no_success():
    tree = ResearchTree.create_with_baseline("baseline", "abc")
    tree.add_experiment("a1", "baseline_v1", "hyperparameter", "test", "failed")
    tree.fail_experiment("a1", "diverged")
    assert propose_cross_pollinations(tree) == []

def test_proposal_has_description(tree_with_successful_branches):
    proposals = propose_cross_pollinations(tree_with_successful_branches)
    if proposals:
        assert len(proposals[0].description) > 0
