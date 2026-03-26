"""Tests for the durable monitoring runner."""

import json
from datetime import datetime, timezone, timedelta

from autoresearch.runner import (
    parse_args,
    refresh_claim,
    compute_fitness,
    extract_metrics,
    on_run_complete,
    on_run_failed,
    RunSnapshot,
    _estimate_descriptors,
)
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


# --- compute_fitness ---

def test_compute_fitness_lower_is_better():
    """Better metrics should yield a lower (more negative) fitness."""
    worse = compute_fitness(gates_per_ep=1.0, success_rate=0.2, avg_speed=2.0)
    better = compute_fitness(gates_per_ep=3.0, success_rate=0.8, avg_speed=5.0)
    assert better < worse


def test_compute_fitness_zero_axis():
    """Zero in any axis should yield zero fitness."""
    assert compute_fitness(gates_per_ep=0.0, success_rate=0.5, avg_speed=3.0) == 0.0
    assert compute_fitness(gates_per_ep=2.0, success_rate=0.0, avg_speed=3.0) == 0.0


def test_compute_fitness_baseline_value():
    """Verify baseline fitness matches expected value."""
    f = compute_fitness(gates_per_ep=2.07, success_rate=0.405, avg_speed=3.0)
    assert abs(f - (-2.5151)) < 0.01


# --- extract_metrics ---

def test_extract_metrics_eval_prefix():
    summary = {
        "eval/gates_per_episode": 3.5,
        "eval/success_rate": 0.6,
        "eval/avg_speed": 4.2,
        "eval/mean_reward": 150.0,
    }
    m = extract_metrics(summary)
    assert m["gates_per_ep"] == 3.5
    assert m["success_rate"] == 0.6
    assert m["avg_speed"] == 4.2


def test_extract_metrics_unprefixed():
    summary = {
        "gates_per_episode": 2.0,
        "success_rate": 0.3,
        "avg_speed": 3.0,
    }
    m = extract_metrics(summary)
    assert m["gates_per_ep"] == 2.0


def test_extract_metrics_empty():
    assert extract_metrics({}) == {}


# --- estimate_descriptors ---

def test_estimate_descriptors_range():
    d = _estimate_descriptors({"avg_speed": 5.0})
    assert 0.25 <= d.actuator_utilization <= 1.0
    assert d.aero_regime == 5.0
    assert d.control_smoothness == 0.5


def test_estimate_descriptors_defaults():
    d = _estimate_descriptors({})
    assert d.aero_regime == 3.0  # default avg_speed


# --- on_run_complete ---

def _setup_state(tmp_path):
    """Create minimal state files for completion tests."""
    from autoresearch.archive.serialization import save_archive
    from autoresearch.archive.map_elites import MapElitesArchive
    from autoresearch.tree.serialization import save_tree
    from autoresearch.tree.research_tree import ResearchTree

    archive = MapElitesArchive()
    save_archive(archive, tmp_path / "archive.json")

    tree = ResearchTree.create_with_baseline("test", "abc123")
    tree.add_experiment("hyp_test", parent_id="baseline_v1", scope="hyperparameter",
                        description="test experiment", status="running")
    save_tree(tree, tmp_path / "tree.json")

    claim = Claim(
        hypothesis_id="hyp_test", researcher="test", branch="ar/test",
        target_cells=[], scope="hyperparameter",
        timestamp=datetime.now(timezone.utc), wandb_run_id="run_test",
        status="running",
    )
    save_claims([claim], tmp_path / "claims.json")
    return archive, tree


def test_on_run_complete_updates_state(tmp_path):
    _setup_state(tmp_path)

    args = parse_args([
        "--run-id", "run_test", "--claim-id", "hyp_test",
        "--state-dir", str(tmp_path),
    ])
    snapshot = RunSnapshot(
        state="finished", step=5_000_000,
        summary={
            "eval/gates_per_episode": 3.0,
            "eval/success_rate": 0.5,
            "eval/avg_speed": 4.0,
        },
        config={},
    )

    on_run_complete(args, snapshot)

    # Check archive got an entry
    from autoresearch.archive.serialization import load_archive
    archive = load_archive(tmp_path / "archive.json")
    assert archive.n_occupied == 1

    # Check tree was updated
    from autoresearch.tree.serialization import load_tree
    tree = load_tree(tmp_path / "tree.json")
    node = tree.get_node("hyp_test")
    assert node.status == "completed"
    assert node.fitness < 0  # negative = good

    # Check claim released
    claims = load_claims(tmp_path / "claims.json")
    assert claims[0].status == "completed"


def test_on_run_failed_updates_state(tmp_path):
    _setup_state(tmp_path)

    args = parse_args([
        "--run-id", "run_test", "--claim-id", "hyp_test",
        "--state-dir", str(tmp_path),
    ])
    snapshot = RunSnapshot(
        state="crashed", step=1_000_000,
        summary={}, config={},
    )

    on_run_failed(args, snapshot)

    from autoresearch.tree.serialization import load_tree
    tree = load_tree(tmp_path / "tree.json")
    node = tree.get_node("hyp_test")
    assert node.status == "failed"

    claims = load_claims(tmp_path / "claims.json")
    assert claims[0].status == "failed"


# --- parse_args new flags ---

def test_parse_args_new_flags():
    args = parse_args([
        "--run-id", "abc", "--claim-id", "hyp_001",
        "--wandb-entity", "myteam",
        "--baseline-threshold", "0.8",
        "--trajectory-dir", "/tmp/trajs",
    ])
    assert args.wandb_entity == "myteam"
    assert args.baseline_threshold == 0.8
    assert args.trajectory_dir == "/tmp/trajs"
