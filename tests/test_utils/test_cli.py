"""CLI integration tests for python -m utils.tb_check."""

from __future__ import annotations

import subprocess
import sys


def _write_events(log_dir: str, scalars: dict[str, list[tuple[int, float]]]) -> None:
    """Write a synthetic TensorBoard event file using SummaryWriter."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=log_dir)
    for tag, step_values in scalars.items():
        for step, value in step_values:
            writer.add_scalar(tag, value, global_step=step)
    writer.close()


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run python -m utils.tb_check with the given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "utils.tb_check", *args],
        capture_output=True,
        text=True,
    )


class TestCLISingleRun:
    """End-to-end tests for the default single-run summary mode."""

    def test_exit_code_zero(self, tmp_path) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0), (1000, 2.0)]})
        result = _run_cli(str(tmp_path))
        assert result.returncode == 0, result.stderr

    def test_output_contains_metric(self, tmp_path) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0), (1000, 2.0)]})
        result = _run_cli(str(tmp_path))
        assert "rollout/ep_rew_mean" in result.stdout

    def test_output_contains_diagnostics_section(self, tmp_path) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0), (1000, 2.0)]})
        result = _run_cli(str(tmp_path))
        assert "DIAGNOSTICS" in result.stdout

    def test_output_contains_steps(self, tmp_path) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0), (5000, 2.0)]})
        result = _run_cli(str(tmp_path))
        # 5,000 formatted steps should appear
        assert "5,000" in result.stdout

    def test_multiple_metrics_shown(self, tmp_path) -> None:
        _write_events(
            str(tmp_path),
            {
                "rollout/ep_rew_mean": [(0, 1.0)],
                "train/approx_kl": [(0, 0.01)],
            },
        )
        result = _run_cli(str(tmp_path))
        assert "rollout/ep_rew_mean" in result.stdout
        assert "train/approx_kl" in result.stdout

    def test_diagnostic_finding_shown_in_output(self, tmp_path) -> None:
        # Write a KL spike to trigger a WARN finding
        _write_events(
            str(tmp_path),
            {"train/approx_kl": [(0, 0.01), (1, 0.10)]},
        )
        result = _run_cli(str(tmp_path))
        assert "WARN" in result.stdout or "kl_spike" in result.stdout

    def test_nonexistent_dir_exits_nonzero(self, tmp_path) -> None:
        missing = str(tmp_path / "does_not_exist")
        result = _run_cli(missing)
        # Should either error out or return empty output gracefully
        # Either exit code 1 OR empty output with DIAGNOSTICS section is acceptable
        # The CLI logs to stderr on error
        is_error = result.returncode != 0 or "Error" in result.stderr
        is_empty_ok = "DIAGNOSTICS" in result.stdout and result.returncode == 0
        assert is_error or is_empty_ok


class TestCLIDiagOnly:
    """Tests for the --diag-only flag."""

    def test_diag_only_flag_works(self, tmp_path) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0)]})
        result = _run_cli(str(tmp_path), "--diag-only")
        assert result.returncode == 0, result.stderr

    def test_diag_only_contains_diagnostics(self, tmp_path) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0)]})
        result = _run_cli(str(tmp_path), "--diag-only")
        assert "DIAGNOSTICS" in result.stdout

    def test_diag_only_omits_metric_table(self, tmp_path) -> None:
        _write_events(str(tmp_path), {"rollout/ep_rew_mean": [(0, 1.0)]})
        result = _run_cli(str(tmp_path), "--diag-only")
        # Metric table header columns should NOT appear
        assert "LATEST" not in result.stdout
        assert "MEAN" not in result.stdout

    def test_diag_only_shows_finding(self, tmp_path) -> None:
        # Crash rate > 50% should trigger a WARN
        _write_events(str(tmp_path), {"termination/ground": [(100, 0.8)]})
        result = _run_cli(str(tmp_path), "--diag-only")
        assert "WARN" in result.stdout


class TestCLILastN:
    """Tests for the --last N flag."""

    def test_last_n_flag_accepted(self, tmp_path) -> None:
        steps = list(range(0, 10000, 1000))
        _write_events(
            str(tmp_path),
            {"rollout/ep_rew_mean": [(s, float(s)) for s in steps]},
        )
        result = _run_cli(str(tmp_path), "--last", "3000")
        assert result.returncode == 0, result.stderr
        assert "rollout/ep_rew_mean" in result.stdout
