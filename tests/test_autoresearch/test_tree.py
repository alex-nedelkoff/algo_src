"""Tests for research tree."""

from autoresearch.tree.research_tree import ResearchTree, TreeNode
from autoresearch.tree.serialization import save_tree, load_tree


def test_create_baseline_root():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    root = tree.root
    assert root.hypothesis_id == "baseline_v1"
    assert root.git_commit == "abc123"
    assert root.parent_id is None
    assert tree.total_experiments == 0


def test_add_experiment():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="running",
    )
    assert tree.total_experiments == 1
    node = tree.get_node("hyp_001")
    assert node.parent_id == "baseline_v1"
    assert node.status == "running"


def test_complete_experiment():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="running",
    )
    tree.complete_experiment("hyp_001", fitness=5.0, wandb_run_id="run_abc")
    node = tree.get_node("hyp_001")
    assert node.status == "completed"
    assert node.fitness == 5.0


def test_fail_experiment():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="running",
    )
    tree.fail_experiment("hyp_001", reason="NaN at step 50k")
    node = tree.get_node("hyp_001")
    assert node.status == "failed"
    assert "NaN" in node.failure_reason


def test_serialization_round_trip(tmp_path):
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id="hyp_001",
        parent_id="baseline_v1",
        scope="hyperparameter",
        description="Increase LR",
        status="completed",
    )
    path = tmp_path / "tree.json"
    save_tree(tree, path)
    loaded = load_tree(path)
    assert loaded.total_experiments == 1
    assert loaded.get_node("hyp_001").description == "Increase LR"
