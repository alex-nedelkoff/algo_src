"""Tests for the durable monitoring runner."""

from datetime import datetime, timezone, timedelta

from autoresearch.runner import parse_args, refresh_claim
from autoresearch.coordination.claims import Claim, save_claims, load_claims


def test_parse_args_required():
    args = parse_args(["--run-id", "abc", "--claim-id", "hyp_001"])
    assert args.run_id == "abc"
    assert args.claim_id == "hyp_001"
    assert args.poll_interval == 60
    assert args.max_rpm == 31470.0
    assert args.budget == 5_000_000
    assert args.baseline_fitness is None


def test_parse_args_custom():
    args = parse_args([
        "--run-id", "abc", "--claim-id", "hyp_001",
        "--budget", "10000000", "--poll-interval", "30",
        "--baseline-fitness", "5.2",
    ])
    assert args.budget == 10_000_000
    assert args.poll_interval == 30
    assert args.baseline_fitness == 5.2


def test_refresh_claim_updates_timestamp(tmp_path):
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    claim = Claim(
        hypothesis_id="hyp_001", researcher="test", branch="ar/exp-hyp001",
        target_cells=[], scope="hyperparameter",
        timestamp=old_time, wandb_run_id="run_abc", status="running",
    )
    save_claims([claim], tmp_path / "claims.json")

    refresh_claim(tmp_path, "hyp_001")

    loaded = load_claims(tmp_path / "claims.json")
    assert loaded[0].timestamp > old_time


def test_refresh_claim_only_updates_running(tmp_path):
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    running = Claim(
        hypothesis_id="hyp_001", researcher="test", branch="ar/exp-hyp001",
        target_cells=[], scope="hyperparameter",
        timestamp=old_time, wandb_run_id="run_abc", status="running",
    )
    completed = Claim(
        hypothesis_id="hyp_002", researcher="test", branch="ar/exp-hyp002",
        target_cells=[], scope="hyperparameter",
        timestamp=old_time, wandb_run_id="run_def", status="completed",
    )
    save_claims([running, completed], tmp_path / "claims.json")

    refresh_claim(tmp_path, "hyp_002")  # try to refresh completed claim

    loaded = load_claims(tmp_path / "claims.json")
    completed_claim = [c for c in loaded if c.hypothesis_id == "hyp_002"][0]
    assert completed_claim.timestamp == old_time  # should not have changed


def test_config_has_resource_limits():
    from autoresearch.config import AutoResearchConfig
    cfg = AutoResearchConfig()
    assert cfg.resource_limits["max_concurrent_per_machine"] == 1
    assert cfg.resource_limits["max_wall_clock_hours"] == 24
    assert cfg.resource_limits["max_worktrees"] == 5
