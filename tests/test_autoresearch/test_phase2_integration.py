"""Integration test for Phase 2 features."""

from datetime import datetime, timezone

from autoresearch.constraints.diff_policy import DiffPolicy, FileChange
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.selection import select_next_branch, BranchStats
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector
from autoresearch.analysis.early_stopping import check_early_stop, EarlyStopSignal
from autoresearch.hypothesis.schema import Hypothesis, HydraOverride, FileDiff
from autoresearch.worktree import WorktreeManager, BRANCH_PREFIX
from autoresearch.runner import parse_args, refresh_claim
from autoresearch.coordination.claims import Claim, save_claims, load_claims
from autoresearch.config import AutoResearchConfig


def test_phase2_pipeline(tmp_path):
    """End-to-end Phase 2 flow: branch selection -> hypothesis -> diff check -> early stop."""

    # 1. Build a research tree with branches
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment("a1", "baseline_v1", "algorithm", "Try entropy bonus", "completed")
    tree.complete_experiment("a1", fitness=5.0, wandb_run_id="run_a1")
    tree.add_experiment("b1", "baseline_v1", "algorithm", "Try larger network", "completed")
    tree.complete_experiment("b1", fitness=6.0, wandb_run_id="run_b1")

    # 2. Archive with one entry
    archive = MapElitesArchive()
    archive.try_insert(CellEntry(
        fitness=5.0, status="approved", wandb_run_id="run_a1",
        git_commit="abc", descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={}, budget_spent=5_000_000, hypothesis_id="a1",
    ))

    # 3. Branch selection
    config = {
        "exploit_weight": 1.0, "explore_weight": 1.0, "gap_weight": 0.5,
        "diversity_weight": 0.3, "random_restart_pct": 0.0,
        "min_branch_experiments": 0, "tie_threshold": 0.05,
    }
    branch_id = select_next_branch(tree, archive, config, seed=42)
    assert branch_id in ["a1", "b1"]

    # 4. Create hypothesis with code changes
    hyp = Hypothesis.create(
        scope="algorithm",
        description="Add entropy bonus to PPO",
        changes=[
            HydraOverride("control.ent_coef", "0.01"),
            FileDiff("control/algorithms/ppo.py", "Add entropy coefficient"),
        ],
        rationale="Improve exploration via entropy regularization",
        parent_id=branch_id,
    )
    assert len(hyp.to_cli_overrides()) == 1  # Only HydraOverride produces CLI arg

    # 5. Diff policy validation
    policy = DiffPolicy()
    diff_result = policy.validate(
        [FileChange("control/algorithms/ppo.py", added=20, removed=5)],
        scope="algorithm",
    )
    assert diff_result.passed is True

    # Verify denylist blocks sim/dynamics
    bad_diff = policy.validate(
        [FileChange("sim/dynamics/numpy_quad.py", added=1, removed=0)],
        scope="system",
    )
    assert bad_diff.passed is False

    # 6. Early stopping — good performance, continue
    signal = check_early_stop(
        current_fitness=4.8, baseline_fitness=5.0,
        steps_completed=2_000_000, budget=15_000_000,
        baseline_threshold=0.7, archive=archive, target_cell=(1, 1, 2),
    )
    assert signal == EarlyStopSignal.CONTINUE

    # 7. Worktree manager branch naming
    manager = WorktreeManager(repo_root=tmp_path)
    assert manager.branch_name(hyp.id).startswith(BRANCH_PREFIX)

    # 8. Runner claim refresh
    claims_path = tmp_path / "claims.json"
    claim = Claim(
        hypothesis_id=hyp.id, researcher="test", branch=f"ar/exp-{hyp.id}",
        target_cells=[], scope="algorithm",
        timestamp=datetime.now(timezone.utc), wandb_run_id="run_test",
        status="running",
    )
    save_claims([claim], claims_path)
    refresh_claim(tmp_path, hyp.id)
    loaded = load_claims(claims_path)
    assert loaded[0].status == "running"

    # 9. Config has resource limits
    cfg = AutoResearchConfig()
    assert cfg.resource_limits["max_worktrees"] == 5


def test_tree_branch_aggregation():
    """Test the new tree methods added in Phase 2."""
    tree = ResearchTree.create_with_baseline("baseline", "abc")
    tree.add_experiment("a1", "baseline_v1", "algorithm", "Branch A", "completed")
    tree.complete_experiment("a1", fitness=5.0)
    tree.add_experiment("a2", "a1", "algorithm", "Refine A", "completed")
    tree.complete_experiment("a2", fitness=4.0)
    tree.add_experiment("b1", "baseline_v1", "algorithm", "Branch B", "completed")
    tree.complete_experiment("b1", fitness=6.0)

    # Level 1 branches
    branches = tree.get_level1_branches()
    assert set(branches) == {"a1", "b1"}

    # Descendants
    desc_a = tree.get_descendants("a1")
    assert len(desc_a) == 1
    assert desc_a[0].hypothesis_id == "a2"

    desc_b = tree.get_descendants("b1")
    assert len(desc_b) == 0

    # Branch stats
    stats_a = BranchStats.from_tree(tree, "a1")
    assert stats_a.best_fitness == 4.0
    assert stats_a.visit_count == 2
