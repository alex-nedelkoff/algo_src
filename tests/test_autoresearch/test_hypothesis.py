"""Tests for hypothesis schema and content hashing."""

from autoresearch.hypothesis.schema import (
    Hypothesis, HydraOverride, FileDiff, compute_hypothesis_id,
)


def test_hydra_override_to_string():
    override = HydraOverride(key="control.learning_rate", value="1e-4")
    assert override.to_cli_arg() == "control.learning_rate=1e-4"


def test_hypothesis_id_deterministic():
    h1 = Hypothesis.create(
        scope="hyperparameter",
        description="Increase learning rate",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="Current LR may be too conservative",
    )
    h2 = Hypothesis.create(
        scope="hyperparameter",
        description="Different description",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="Different rationale",
    )
    assert h1.id == h2.id  # same scope + changes + targets = same ID


def test_different_changes_different_id():
    h1 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    h2 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 1e-3",
        changes=[HydraOverride("control.learning_rate", "1e-3")],
        rationale="test",
    )
    assert h1.id != h2.id


def test_different_targets_different_id():
    h1 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        target_cells=[(1, 2, 3)],
        rationale="test",
    )
    h2 = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        target_cells=[(4, 4, 4)],
        rationale="test",
    )
    assert h1.id != h2.id


def test_hypothesis_cli_overrides():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="Multi-param change",
        changes=[
            HydraOverride("control.learning_rate", "3e-4"),
            HydraOverride("reward.gate_passage", "2.0"),
        ],
        rationale="test",
    )
    args = h.to_cli_overrides()
    assert "control.learning_rate=3e-4" in args
    assert "reward.gate_passage=2.0" in args


def test_hypothesis_default_budget():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="test",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    assert h.estimated_budget == 5_000_000

    h2 = Hypothesis.create(
        scope="algorithm",
        description="test",
        changes=[HydraOverride("control.n_steps", "2000")],
        rationale="test",
    )
    assert h2.estimated_budget == 15_000_000


def test_file_diff_creation():
    diff = FileDiff(path="control/algorithms/ppo.py", description="Add entropy bonus")
    assert diff.path == "control/algorithms/ppo.py"


def test_hypothesis_with_file_diffs():
    h = Hypothesis.create(
        scope="algorithm",
        description="Add entropy bonus to PPO",
        changes=[
            HydraOverride("control.ent_coef", "0.01"),
            FileDiff("control/algorithms/ppo.py", "Add entropy coefficient parameter"),
        ],
        rationale="Entropy bonus may improve exploration",
    )
    assert len(h.changes) == 2
    overrides = h.to_cli_overrides()
    assert len(overrides) == 1  # Only HydraOverrides produce CLI args
    assert "control.ent_coef=0.01" in overrides
