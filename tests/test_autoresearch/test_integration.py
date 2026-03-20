"""Integration test: full pipeline from trajectory to archive insertion."""

import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from autoresearch.analysis.trajectory import load_trajectory
from autoresearch.descriptors.compute import compute_descriptors
from autoresearch.constraints.validator import ConstraintValidator
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.hypothesis.schema import Hypothesis, HydraOverride
from autoresearch.coordination.claims import Claim, save_claims, load_claims, add_claim
from autoresearch.coordination.dedup import is_duplicate
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.serialization import save_tree, load_tree


def test_full_pipeline(sample_npz, tmp_path):
    """Simulate the complete MVP research loop (minus actual training)."""
    max_rpm = 31470.0
    control_freq = 100.0

    # 1. Load trajectory
    traj = load_trajectory(sample_npz)
    assert traj.n_timesteps == 100

    # 2. Compute descriptors
    desc = compute_descriptors(traj, max_rpm, control_freq)
    assert 0 <= desc.actuator_utilization <= 1.0
    assert 0 <= desc.control_smoothness <= 1.0
    assert desc.aero_regime >= 0.0

    # 3. Validate constraints
    validator = ConstraintValidator(max_rpm=max_rpm)
    metrics = {"success_rate": 0.9, "lap_times": [5.0] * 20}
    result = validator.validate(traj, metrics, dr_active=True)
    # Result may or may not pass depending on fixture data — we test that it runs

    # 4. Create hypothesis
    hyp = Hypothesis.create(
        scope="hyperparameter",
        description="Increase learning rate to 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="Baseline LR may be too conservative",
    )
    assert hyp.id is not None
    assert hyp.to_cli_overrides() == ["control.learning_rate=3e-4"]

    # 5. Check deduplication
    assert is_duplicate(hyp, set(), set()) is False
    assert is_duplicate(hyp, {hyp.id}, set()) is True

    # 6. Create claim
    claims_path = tmp_path / "claims.json"
    save_claims([], claims_path)
    claim = Claim(
        hypothesis_id=hyp.id,
        researcher="test@laptop",
        branch=f"ar/exp-{hyp.id}",
        target_cells=[],
        scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id="run_test",
        status="running",
    )
    claims = add_claim(load_claims(claims_path), claim)
    save_claims(claims, claims_path)
    assert len(load_claims(claims_path)) == 1

    # 7. Archive insertion
    archive_path = tmp_path / "archive.json"
    archive = MapElitesArchive()
    entry = CellEntry(
        fitness=5.2,
        status="candidate",
        wandb_run_id="run_test",
        git_commit="test123",
        descriptors=desc,
        constraint_results={"passed": result.passed},
        budget_spent=5_000_000,
        hypothesis_id=hyp.id,
    )
    archive.try_insert(entry)
    save_archive(archive, archive_path)
    loaded_archive = load_archive(archive_path)
    assert loaded_archive.n_occupied == 1

    # 8. Research tree update
    tree_path = tmp_path / "tree.json"
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        hypothesis_id=hyp.id,
        parent_id="baseline_v1",
        scope="hyperparameter",
        description=hyp.description,
        status="completed",
    )
    tree.complete_experiment(hyp.id, fitness=5.2, wandb_run_id="run_test")
    save_tree(tree, tree_path)
    loaded_tree = load_tree(tree_path)
    assert loaded_tree.total_experiments == 1
    assert loaded_tree.get_node(hyp.id).fitness == 5.2
