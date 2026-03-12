"""Tests for utils.tb_check.formatter."""

from __future__ import annotations

import pandas as pd
import pytest

from utils.tb_check.diagnostics import Finding
from utils.tb_check.formatter import (
    format_comparison,
    format_diagnostics_only,
    format_summary,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """Build a long-format DataFrame from (step, tag, value) tuples."""
    return pd.DataFrame(rows, columns=["step", "tag", "value"])


def _warn(rule: str = "kl_spike", metric: str = "train/approx_kl") -> Finding:
    return Finding(severity="WARN", rule=rule, message="test warn", metric=metric, step=100)


def _critical(rule: str = "nan_detected", metric: str = "train/value_loss") -> Finding:
    return Finding(
        severity="CRITICAL", rule=rule, message="test critical", metric=metric, step=50
    )


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    """format_summary produces header, metric table, and diagnostics section."""

    def test_contains_header(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        result = format_summary(df, "/some/path", [])
        assert "TB Check" in result
        assert "/some/path" in result

    def test_contains_steps(self) -> None:
        df = _make_df([(1000, "rollout/ep_rew_mean", 1.0)])
        result = format_summary(df, "/path", [])
        assert "1,000" in result

    def test_contains_metric_name(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 5.0)])
        result = format_summary(df, "/path", [])
        assert "rollout/ep_rew_mean" in result

    def test_contains_diagnostics_section(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        result = format_summary(df, "/path", [])
        assert "DIAGNOSTICS" in result

    def test_finding_appears_in_output(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        finding = _warn()
        result = format_summary(df, "/path", [finding])
        assert "WARN" in result
        assert "test warn" in result

    def test_critical_finding_appears_in_output(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        finding = _critical()
        result = format_summary(df, "/path", [finding])
        assert "CRITICAL" in result

    def test_no_findings_shows_ok_message(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        result = format_summary(df, "/path", [])
        assert "No critical issues" in result

    def test_column_headers_present(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        result = format_summary(df, "/path", [])
        assert "METRIC" in result
        assert "LATEST" in result
        assert "MEAN" in result


# ---------------------------------------------------------------------------
# format_diagnostics_only
# ---------------------------------------------------------------------------

class TestFormatDiagnosticsOnly:
    """format_diagnostics_only formats findings without the metric table."""

    def test_contains_diagnostics_header(self) -> None:
        result = format_diagnostics_only([])
        assert "DIAGNOSTICS" in result

    def test_with_findings_shows_severity(self) -> None:
        result = format_diagnostics_only([_warn(), _critical()])
        assert "WARN" in result
        assert "CRITICAL" in result

    def test_with_findings_shows_messages(self) -> None:
        result = format_diagnostics_only([_warn()])
        assert "test warn" in result

    def test_no_findings_shows_ok_message(self) -> None:
        result = format_diagnostics_only([])
        assert "No critical issues" in result

    def test_does_not_contain_metric_table(self) -> None:
        result = format_diagnostics_only([_warn()])
        # Metric table header would have LATEST/MEAN columns
        assert "LATEST" not in result
        assert "MEAN" not in result


# ---------------------------------------------------------------------------
# format_comparison
# ---------------------------------------------------------------------------

class TestFormatComparison:
    """format_comparison produces RUN1, RUN2, DELTA columns."""

    def test_contains_run1_run2_delta_headers(self) -> None:
        df1 = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        df2 = _make_df([(0, "rollout/ep_rew_mean", 2.0)])
        result = format_comparison(df1, df2, "/run1", "/run2", [], [])
        assert "RUN1" in result
        assert "RUN2" in result
        assert "DELTA" in result

    def test_contains_path_labels(self) -> None:
        df1 = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        df2 = _make_df([(0, "rollout/ep_rew_mean", 2.0)])
        result = format_comparison(df1, df2, "/run1", "/run2", [], [])
        assert "/run1" in result
        assert "/run2" in result

    def test_contains_metric_name(self) -> None:
        df1 = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        df2 = _make_df([(0, "rollout/ep_rew_mean", 2.0)])
        result = format_comparison(df1, df2, "/run1", "/run2", [], [])
        assert "rollout/ep_rew_mean" in result

    def test_diagnostics_sections_present(self) -> None:
        df1 = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        df2 = _make_df([(0, "rollout/ep_rew_mean", 2.0)])
        result = format_comparison(df1, df2, "/run1", "/run2", [_warn()], [])
        assert "DIAGNOSTICS" in result

    def test_findings_for_each_run(self) -> None:
        df1 = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        df2 = _make_df([(0, "rollout/ep_rew_mean", 2.0)])
        w1 = _warn(rule="kl_spike")
        w2 = _critical(rule="nan_detected")
        result = format_comparison(df1, df2, "/run1", "/run2", [w1], [w2])
        assert "kl_spike" in result or "test warn" in result
        assert "nan_detected" in result or "test critical" in result


# ---------------------------------------------------------------------------
# Metric ordering
# ---------------------------------------------------------------------------

class TestMetricOrdering:
    """Metrics appear in group order: racing/ before rollout/ before train/ before termination/."""

    def test_racing_before_rollout(self) -> None:
        df = _make_df([
            (0, "rollout/ep_rew_mean", 1.0),
            (0, "racing/gates_per_ep", 2.0),
        ])
        result = format_summary(df, "/path", [])
        racing_pos = result.index("racing/gates_per_ep")
        rollout_pos = result.index("rollout/ep_rew_mean")
        assert racing_pos < rollout_pos

    def test_rollout_before_train(self) -> None:
        df = _make_df([
            (0, "train/approx_kl", 0.01),
            (0, "rollout/ep_rew_mean", 1.0),
        ])
        result = format_summary(df, "/path", [])
        rollout_pos = result.index("rollout/ep_rew_mean")
        train_pos = result.index("train/approx_kl")
        assert rollout_pos < train_pos

    def test_train_before_termination(self) -> None:
        df = _make_df([
            (0, "termination/ground", 0.1),
            (0, "train/approx_kl", 0.01),
        ])
        result = format_summary(df, "/path", [])
        train_pos = result.index("train/approx_kl")
        term_pos = result.index("termination/ground")
        assert train_pos < term_pos


# ---------------------------------------------------------------------------
# Trend symbols
# ---------------------------------------------------------------------------

class TestTrendSymbols:
    """Trend symbols (↑, ↓, →) are emitted based on tail vs head comparison."""

    def test_trend_up_for_increasing_series(self) -> None:
        # Large upward slope — tail >> head
        rows = [(i, "rollout/ep_rew_mean", float(i * 10)) for i in range(20)]
        df = _make_df(rows)
        result = format_summary(df, "/path", [])
        assert "↑" in result

    def test_trend_down_for_decreasing_series(self) -> None:
        # Large downward slope — tail << head
        rows = [(i, "rollout/ep_rew_mean", float(100 - i * 10)) for i in range(10)]
        df = _make_df(rows)
        result = format_summary(df, "/path", [])
        assert "↓" in result

    def test_trend_flat_for_constant_series(self) -> None:
        rows = [(i, "rollout/ep_rew_mean", 5.0) for i in range(10)]
        df = _make_df(rows)
        result = format_summary(df, "/path", [])
        assert "→" in result
