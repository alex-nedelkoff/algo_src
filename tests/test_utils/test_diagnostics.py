"""Tests for utils.tb_check.diagnostics."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from utils.tb_check.diagnostics import Finding, run_diagnostics


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """Build a long-format DataFrame from (step, tag, value) tuples."""
    return pd.DataFrame(rows, columns=["step", "tag", "value"])


# ---------------------------------------------------------------------------
# reward_plateau
# ---------------------------------------------------------------------------

class TestRewardPlateau:
    """WARN when rollout/ep_rew_mean is flat over the last 20% of training."""

    def test_flat_series_triggers_warn(self) -> None:
        # 20 identical values — no change at all
        rows = [(i, "rollout/ep_rew_mean", 5.0) for i in range(20)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "reward_plateau" in rules

    def test_flat_series_severity_is_warn(self) -> None:
        rows = [(i, "rollout/ep_rew_mean", 5.0) for i in range(20)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        plateau = next(f for f in findings if f.rule == "reward_plateau")
        assert plateau.severity == "WARN"

    def test_improving_series_no_trigger(self) -> None:
        # Strongly increasing — tail >> head
        rows = [(i, "rollout/ep_rew_mean", float(i) * 2) for i in range(20)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "reward_plateau" not in rules

    def test_too_few_points_no_trigger(self) -> None:
        # < 5 points — rule should skip
        rows = [(i, "rollout/ep_rew_mean", 1.0) for i in range(4)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "reward_plateau" not in rules


# ---------------------------------------------------------------------------
# reward_collapse
# ---------------------------------------------------------------------------

class TestRewardCollapse:
    """CRITICAL when rollout/ep_rew_mean drops > 30% from peak."""

    def test_collapse_triggers_critical(self) -> None:
        # Peak = 10, current = 6 → 40% drop
        rows = (
            [(i, "rollout/ep_rew_mean", 10.0) for i in range(5)]
            + [(5, "rollout/ep_rew_mean", 6.0)]
        )
        df = _make_df(rows)
        findings = run_diagnostics(df)
        collapse = next((f for f in findings if f.rule == "reward_collapse"), None)
        assert collapse is not None
        assert collapse.severity == "CRITICAL"

    def test_small_drop_no_trigger(self) -> None:
        # Peak = 10, current = 8 → only 20% drop
        rows = (
            [(i, "rollout/ep_rew_mean", 10.0) for i in range(5)]
            + [(5, "rollout/ep_rew_mean", 8.0)]
        )
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "reward_collapse" not in rules

    def test_negative_peak_skipped(self) -> None:
        # All negative rewards — peak <= 0 so rule should not fire
        rows = [(i, "rollout/ep_rew_mean", -float(i + 1)) for i in range(10)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "reward_collapse" not in rules


# ---------------------------------------------------------------------------
# kl_spike
# ---------------------------------------------------------------------------

class TestKLSpike:
    """WARN when train/approx_kl exceeds 0.05."""

    def test_spike_triggers_warn(self) -> None:
        rows = [
            (0, "train/approx_kl", 0.01),
            (1, "train/approx_kl", 0.02),
            (2, "train/approx_kl", 0.10),  # spike
        ]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        kl = next((f for f in findings if f.rule == "kl_spike"), None)
        assert kl is not None
        assert kl.severity == "WARN"

    def test_within_threshold_no_trigger(self) -> None:
        rows = [
            (0, "train/approx_kl", 0.01),
            (1, "train/approx_kl", 0.02),
            (2, "train/approx_kl", 0.04),
        ]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "kl_spike" not in rules

    def test_step_recorded_correctly(self) -> None:
        rows = [
            (100, "train/approx_kl", 0.01),
            (200, "train/approx_kl", 0.08),
        ]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        kl = next(f for f in findings if f.rule == "kl_spike")
        assert kl.step == 200


# ---------------------------------------------------------------------------
# entropy_collapse
# ---------------------------------------------------------------------------

class TestEntropyCollapse:
    """WARN when train/entropy_loss drops below 10% of its initial value."""

    def test_collapse_triggers_warn(self) -> None:
        # Initial values around -2.0; final value near 0 (< 10% of |-2|)
        rows = [(i, "train/entropy_loss", -2.0) for i in range(10)]
        rows.append((10, "train/entropy_loss", -0.01))  # collapsed
        df = _make_df(rows)
        findings = run_diagnostics(df)
        ec = next((f for f in findings if f.rule == "entropy_collapse"), None)
        assert ec is not None
        assert ec.severity == "WARN"

    def test_healthy_entropy_no_trigger(self) -> None:
        # Values are stable
        rows = [(i, "train/entropy_loss", -2.0 + i * 0.01) for i in range(20)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "entropy_collapse" not in rules

    def test_too_few_points_no_trigger(self) -> None:
        rows = [(i, "train/entropy_loss", -2.0) for i in range(4)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "entropy_collapse" not in rules


# ---------------------------------------------------------------------------
# high_crash_rate
# ---------------------------------------------------------------------------

class TestHighCrashRate:
    """WARN when any non-excluded termination reason exceeds 50%."""

    def test_ground_crash_triggers_warn(self) -> None:
        rows = [(100, "termination/ground", 0.8)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        hcr = next((f for f in findings if f.rule == "high_crash_rate"), None)
        assert hcr is not None
        assert hcr.severity == "WARN"
        assert hcr.metric == "termination/ground"

    def test_timeout_excluded(self) -> None:
        # termination/timeout should not trigger even above 50%
        rows = [(100, "termination/timeout", 0.9)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "high_crash_rate" not in rules

    def test_none_excluded(self) -> None:
        rows = [(100, "termination/none", 0.9)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "high_crash_rate" not in rules

    def test_below_threshold_no_trigger(self) -> None:
        rows = [(100, "termination/ground", 0.4)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "high_crash_rate" not in rules


# ---------------------------------------------------------------------------
# nan_detected
# ---------------------------------------------------------------------------

class TestNaNDetected:
    """CRITICAL if any NaN value appears in any metric."""

    def test_nan_triggers_critical(self) -> None:
        rows = [
            (0, "rollout/ep_rew_mean", 1.0),
            (1, "rollout/ep_rew_mean", float("nan")),
        ]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        nan = next((f for f in findings if f.rule == "nan_detected"), None)
        assert nan is not None
        assert nan.severity == "CRITICAL"

    def test_nan_records_first_step(self) -> None:
        rows = [
            (50, "train/value_loss", float("nan")),
            (100, "train/value_loss", 2.0),
        ]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        nan = next(f for f in findings if f.rule == "nan_detected")
        assert nan.step == 50

    def test_no_nan_no_trigger(self) -> None:
        rows = [(i, "rollout/ep_rew_mean", float(i)) for i in range(10)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "nan_detected" not in rules


# ---------------------------------------------------------------------------
# no_gate_progress
# ---------------------------------------------------------------------------

class TestNoGateProgress:
    """CRITICAL when racing/gates_per_ep stays at 0 after 500K steps."""

    def test_stuck_at_zero_after_500k_triggers_critical(self) -> None:
        rows = [(i * 100_000, "racing/gates_per_ep", 0.0) for i in range(1, 7)]
        df = _make_df(rows)  # max_step = 600_000
        findings = run_diagnostics(df)
        ngp = next((f for f in findings if f.rule == "no_gate_progress"), None)
        assert ngp is not None
        assert ngp.severity == "CRITICAL"

    def test_stuck_at_zero_before_500k_no_trigger(self) -> None:
        rows = [(i * 100_000, "racing/gates_per_ep", 0.0) for i in range(1, 5)]
        df = _make_df(rows)  # max_step = 400_000
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "no_gate_progress" not in rules

    def test_any_gate_progress_no_trigger(self) -> None:
        rows = (
            [(i * 100_000, "racing/gates_per_ep", 0.0) for i in range(1, 6)]
            + [(600_000, "racing/gates_per_ep", 1.0)]
        )
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "no_gate_progress" not in rules

    def test_tag_absent_no_trigger(self) -> None:
        rows = [(100_000, "rollout/ep_rew_mean", 1.0)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "no_gate_progress" not in rules


# ---------------------------------------------------------------------------
# value_loss_explosion
# ---------------------------------------------------------------------------

class TestValueLossExplosion:
    """WARN when train/value_loss exceeds 10x its rolling average."""

    def test_spike_triggers_warn(self) -> None:
        # 15 stable values then a massive spike
        rows = [(i, "train/value_loss", 1.0) for i in range(15)]
        rows.append((15, "train/value_loss", 50.0))  # 50x spike
        df = _make_df(rows)
        findings = run_diagnostics(df)
        vle = next((f for f in findings if f.rule == "value_loss_explosion"), None)
        assert vle is not None
        assert vle.severity == "WARN"

    def test_no_spike_no_trigger(self) -> None:
        rows = [(i, "train/value_loss", 1.0 + i * 0.1) for i in range(20)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "value_loss_explosion" not in rules

    def test_fewer_than_10_points_no_trigger(self) -> None:
        rows = [(i, "train/value_loss", 1.0) for i in range(9)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "value_loss_explosion" not in rules


# ---------------------------------------------------------------------------
# clip_fraction_saturation
# ---------------------------------------------------------------------------

class TestClipFractionSaturation:
    """WARN when train/clip_fraction mean over the last 20% of data > 0.3."""

    def test_high_tail_mean_triggers_warn(self) -> None:
        # 20 points: first 16 at 0.1, last 4 at 0.5 → tail mean = 0.5
        rows = [(i, "train/clip_fraction", 0.1) for i in range(16)]
        rows += [(i + 16, "train/clip_fraction", 0.5) for i in range(4)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        cfs = next((f for f in findings if f.rule == "clip_fraction_saturation"), None)
        assert cfs is not None
        assert cfs.severity == "WARN"

    def test_low_clip_fraction_no_trigger(self) -> None:
        rows = [(i, "train/clip_fraction", 0.1) for i in range(20)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "clip_fraction_saturation" not in rules

    def test_too_few_points_no_trigger(self) -> None:
        rows = [(i, "train/clip_fraction", 0.9) for i in range(4)]
        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = [f.rule for f in findings]
        assert "clip_fraction_saturation" not in rules


# ---------------------------------------------------------------------------
# clean_run
# ---------------------------------------------------------------------------

class TestCleanRun:
    """Healthy metrics produce no findings."""

    def test_clean_metrics_no_findings(self) -> None:
        rows = []
        # Improving reward
        for i in range(20):
            rows.append((i * 1000, "rollout/ep_rew_mean", float(i)))
        # Normal KL
        for i in range(20):
            rows.append((i * 1000, "train/approx_kl", 0.01))
        # Stable entropy
        for i in range(20):
            rows.append((i * 1000, "train/entropy_loss", -2.0))
        # Moderate crash rate
        for i in range(20):
            rows.append((i * 1000, "termination/ground", 0.2))
        # Reasonable clip fraction
        for i in range(20):
            rows.append((i * 1000, "train/clip_fraction", 0.1))
        # Stable value loss
        for i in range(20):
            rows.append((i * 1000, "train/value_loss", 1.0))
        # Gate progress
        for i in range(20):
            rows.append((i * 1000, "racing/gates_per_ep", float(i + 1)))

        df = _make_df(rows)
        findings = run_diagnostics(df)
        assert findings == []


# ---------------------------------------------------------------------------
# run_diagnostics integration
# ---------------------------------------------------------------------------

class TestRunDiagnosticsIntegration:
    """run_diagnostics() collects findings from all rules."""

    def test_multiple_rules_all_reported(self) -> None:
        rows = []
        # Trigger reward_collapse: peak=10, current=3 (70% drop)
        for i in range(5):
            rows.append((i, "rollout/ep_rew_mean", 10.0))
        rows.append((5, "rollout/ep_rew_mean", 3.0))
        # Trigger kl_spike
        rows.append((0, "train/approx_kl", 0.10))
        # Trigger nan_detected
        rows.append((0, "train/value_loss", float("nan")))

        df = _make_df(rows)
        findings = run_diagnostics(df)
        rules = {f.rule for f in findings}
        assert "reward_collapse" in rules
        assert "kl_spike" in rules
        assert "nan_detected" in rules

    def test_returns_list_of_findings(self) -> None:
        df = _make_df([(0, "rollout/ep_rew_mean", 1.0)])
        result = run_diagnostics(df)
        assert isinstance(result, list)
        assert all(isinstance(f, Finding) for f in result)

    def test_empty_df_returns_empty_list(self) -> None:
        df = pd.DataFrame(columns=["step", "tag", "value"])
        findings = run_diagnostics(df)
        assert findings == []
