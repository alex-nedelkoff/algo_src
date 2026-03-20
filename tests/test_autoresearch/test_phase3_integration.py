"""Integration test for Phase 3 features."""

import numpy as np
from pathlib import Path

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.analysis.trajectory_insights import analyze_trajectory
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.cross_pollination import find_successful_patterns, propose_cross_pollinations
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.subdivision import should_subdivide, subdivide_cell
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.promotion import (
    verify_promotion_candidate, generate_pr_body,
    update_state_after_promotion, PromotionCandidate,
)
from autoresearch.tree.serialization import save_tree, load_tree
from autoresearch.descriptors.compute import DescriptorVector


def test_trajectory_to_insights():
    T = 200
    traj = TrajectoryData(
        schema_version=2,
        positions=np.column_stack([np.linspace(0, 20, T), np.zeros(T), np.full(T, 2.0)]),
        quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)),
        velocities=np.tile([10.0, 0.0, 0.0], (T, 1)),
        body_rates=np.zeros((T, 3)),
        motor_rpms=np.full((T, 4), 20000.0),
        actions=np.zeros((T, 4)),
        rewards=np.ones(T),
        reward_components=np.column_stack([np.ones(T) * 5, np.zeros(T), np.zeros(T),
                                           np.ones(T) * 10, np.zeros(T), np.zeros(T)]),
        reward_component_names=["progress", "body_rate", "action_smooth",
                                "gate_passage", "gate_offset", "crash_penalty"],
        gate_events=np.array([[50, 0], [100, 1]], dtype=np.int64),
        gate_positions=np.array([[5, 0, 2], [10, 0, 2], [15, 0, 2]], dtype=np.float64),
        gate_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        gate_half_extents=np.full((3, 2), 0.5),
        dt=0.01,
    )
    insights = analyze_trajectory(traj)
    assert insights.mean_speed == 10.0
    assert insights.dominant_reward_component == "gate_passage"
    assert len(insights.gate_analyses) == 2
    assert "gate_passage" in insights.to_summary()


def test_cross_pollination_pipeline():
    tree = ResearchTree.create_with_baseline("baseline", "abc")
    tree.add_experiment("a1", "baseline_v1", "hyperparameter", "High gate weight", "completed")
    tree.complete_experiment("a1", fitness=4.0)
    tree.add_experiment("a2", "a1", "hyperparameter", "Refine gate weight", "completed")
    tree.complete_experiment("a2", fitness=3.5)
    tree.add_experiment("b1", "baseline_v1", "algorithm", "New network", "completed")
    tree.complete_experiment("b1", fitness=7.0)

    patterns = find_successful_patterns(tree)
    assert len(patterns) >= 1
    proposals = propose_cross_pollinations(tree)
    assert len(proposals) >= 1
    assert proposals[0].source_branch == "a1"
    assert proposals[0].target_branch == "b1"


def test_subdivision_pipeline():
    archive = MapElitesArchive()
    desc = DescriptorVector(0.5, 0.3, 10.0)
    for i in range(4):
        archive.try_insert(CellEntry(
            fitness=5.0 - i * 0.3, status="approved", wandb_run_id=f"run_{i}",
            git_commit=f"c{i}", descriptors=desc,
            constraint_results={}, budget_spent=5_000_000, hypothesis_id=f"h{i}",
        ))
    cell = archive.descriptor_to_cell(desc)
    assert archive.get_insertion_count(cell) == 4
    assert should_subdivide(archive, cell, min_insertions=3)
    sub = subdivide_cell(archive, cell)
    assert sub.n_occupied >= 1


def test_promotion_pipeline(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    archive = MapElitesArchive()
    desc = DescriptorVector(0.6, 0.2, 12.0)
    archive.try_insert(CellEntry(
        fitness=3.5, status="approved", wandb_run_id="run_win",
        git_commit="win123", descriptors=desc,
        constraint_results={}, budget_spent=15_000_000, hypothesis_id="hyp_win",
    ))
    save_archive(archive, state_dir / "archive.json")

    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment("hyp_win", "baseline_v1", "algorithm", "Winner", "completed")
    tree.complete_experiment("hyp_win", fitness=3.5, wandb_run_id="run_win")
    save_tree(tree, state_dir / "tree.json")

    candidate = PromotionCandidate(
        hypothesis_id="hyp_win", wandb_run_id="run_win", fitness=3.5,
        descriptors=desc, branch="ar/exp-hyp_win", git_commit="win123",
    )
    result = verify_promotion_candidate(candidate, archive)
    assert result.verified

    body = generate_pr_body(candidate, old_baseline_fitness=5.0,
                            wandb_url="https://wandb.ai/test/run")
    assert "3.500" in body

    update_state_after_promotion(
        state_dir=state_dir, hypothesis_id="hyp_win",
        merge_commit="merge_abc", new_baseline_name="baseline_v2",
        promoter="shaan",
    )

    assert (state_dir / "archive_v1.json").exists()
    new_archive = load_archive(state_dir / "archive.json")
    assert new_archive.n_occupied == 0
    new_tree = load_tree(state_dir / "tree.json")
    assert new_tree.root.hypothesis_id == "baseline_v2"
    assert new_tree.get_node("baseline_v1") is not None
