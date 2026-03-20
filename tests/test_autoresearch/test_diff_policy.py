"""Tests for edit surface diff policy enforcement."""

import pytest
from autoresearch.constraints.diff_policy import DiffPolicy, DiffValidationResult, FileChange


@pytest.fixture
def policy():
    return DiffPolicy()


class TestDenylist:
    def test_sim_dynamics_denied(self, policy):
        changes = [FileChange("sim/dynamics/numpy_quad.py", added=10, removed=5)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False
        assert "denylist" in result.violations[0].lower()

    def test_rewards_denied(self, policy):
        changes = [FileChange("sim/rewards.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_rewards_mavlab_denied(self, policy):
        changes = [FileChange("sim/rewards_mavlab.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_metrics_denied(self, policy):
        changes = [FileChange("metrics/contract.py", added=1, removed=0)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False

    def test_autoresearch_denied(self, policy):
        changes = [FileChange("autoresearch/archive/map_elites.py", added=5, removed=2)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_training_callbacks_denied(self, policy):
        changes = [FileChange("training/callbacks.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_tests_deletion_denied(self, policy):
        changes = [FileChange("tests/test_sim/test_rewards.py", added=0, removed=5)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False

    def test_tests_addition_allowed(self, policy):
        changes = [FileChange("tests/test_control/test_new.py", added=50, removed=0)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True


class TestScopeAllowlist:
    def test_hyperparameter_rejects_any_file_change(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=1, removed=0)]
        result = policy.validate(changes, scope="hyperparameter")
        assert result.passed is False
        assert "hyperparameter" in result.violations[0].lower()

    def test_algorithm_allows_control(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=10, removed=5)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True

    def test_algorithm_allows_configs(self, policy):
        changes = [FileChange("configs/experiment/new_exp.yaml", added=20, removed=0)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True

    def test_algorithm_rejects_sim_envs(self, policy):
        changes = [FileChange("sim/envs/gate_race_env.py", added=10, removed=5)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False

    def test_architecture_allows_perception_detectors(self, policy):
        changes = [FileChange("perception/detectors/gatenet.py", added=20, removed=10)]
        result = policy.validate(changes, scope="architecture")
        assert result.passed is True

    def test_system_allows_sim_envs(self, policy):
        changes = [FileChange("sim/envs/gate_race_env.py", added=10, removed=5)]
        result = policy.validate(changes, scope="system")
        assert result.passed is True

    def test_system_rejects_sim_dynamics(self, policy):
        changes = [FileChange("sim/dynamics/numpy_quad.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False


class TestMaxDiffSize:
    def test_algorithm_within_limit(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=200, removed=100)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True

    def test_algorithm_exceeds_limit(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=400, removed=200)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False
        assert "lines" in result.violations[0].lower() or "diff" in result.violations[0].lower()

    def test_system_larger_limit(self, policy):
        changes = [FileChange("sim/envs/new_env.py", added=1500, removed=200)]
        result = policy.validate(changes, scope="system")
        assert result.passed is True


class TestFileSummary:
    def test_summary_included(self, policy):
        changes = [
            FileChange("control/algorithms/ppo.py", added=10, removed=5),
            FileChange("configs/experiment/new.yaml", added=20, removed=0),
        ]
        result = policy.validate(changes, scope="algorithm")
        assert len(result.file_summary) == 2
        assert result.file_summary["control/algorithms/ppo.py"] == {"added": 10, "removed": 5}
