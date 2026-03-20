"""Tests for experiment coordination and deduplication."""

from datetime import datetime, timezone, timedelta
from pathlib import Path

from autoresearch.coordination.claims import (
    Claim, load_claims, save_claims, add_claim, expire_claims, release_claim,
)
from autoresearch.coordination.dedup import is_duplicate
from autoresearch.hypothesis.schema import Hypothesis, HydraOverride


def test_add_and_load_claim(tmp_path):
    path = tmp_path / "claims.json"
    save_claims([], path)

    claim = Claim(
        hypothesis_id="hyp_001",
        researcher="shaan@laptop",
        branch="ar/exp-hyp001",
        target_cells=[(2, 1, 3)],
        scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id="run_abc",
        status="running",
    )
    claims = load_claims(path)
    claims = add_claim(claims, claim)
    save_claims(claims, path)

    loaded = load_claims(path)
    assert len(loaded) == 1
    assert loaded[0].hypothesis_id == "hyp_001"


def test_expire_old_claims():
    old = Claim(
        hypothesis_id="old", researcher="test", branch="ar/old",
        target_cells=[], scope="hyperparameter",
        timestamp=datetime.now(timezone.utc) - timedelta(hours=5),
        wandb_run_id=None, status="running",
    )
    fresh = Claim(
        hypothesis_id="fresh", researcher="test", branch="ar/fresh",
        target_cells=[], scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id=None, status="running",
    )
    result = expire_claims([old, fresh], timeout_hours=4)
    assert len(result) == 1
    assert result[0].hypothesis_id == "fresh"


def test_release_claim():
    claim = Claim(
        hypothesis_id="hyp_001", researcher="test", branch="ar/test",
        target_cells=[], scope="hyperparameter",
        timestamp=datetime.now(timezone.utc),
        wandb_run_id=None, status="running",
    )
    claims = release_claim([claim], "hyp_001", new_status="completed")
    assert claims[0].status == "completed"


def test_is_duplicate_exact_match():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    completed_ids = {h.id}
    active_ids: set[str] = set()
    assert is_duplicate(h, completed_ids, active_ids) is True


def test_is_not_duplicate():
    h = Hypothesis.create(
        scope="hyperparameter",
        description="LR 3e-4",
        changes=[HydraOverride("control.learning_rate", "3e-4")],
        rationale="test",
    )
    assert is_duplicate(h, set(), set()) is False
