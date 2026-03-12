"""formatter.py — terminal output formatter for tb_check.

Formats TensorBoard metric DataFrames into human-readable terminal output.
Supports single-run summary, two-run comparison, and diagnostics-only modes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from utils.tb_check.diagnostics import Finding


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Metric group ordering (groups appear in this order; unknown groups go last).
_GROUP_ORDER = ["racing", "rollout", "train", "termination"]

# Trend symbols
_TREND_UP = "↑"
_TREND_DOWN = "↓"
_TREND_FLAT = "→"

# Severity icons
_ICON = {
    "CRITICAL": "🔴",
    "WARN": "⚠",
    "INFO": "ℹ",
}

# Column widths for the metrics table
_COL_METRIC = 32
_COL_VALUE = 9
_COL_DELTA = 12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_key(tag: str) -> tuple[int, str]:
    """Return a sort key that groups metrics by prefix in _GROUP_ORDER order."""
    prefix = tag.split("/")[0] if "/" in tag else ""
    try:
        group_idx = _GROUP_ORDER.index(prefix)
    except ValueError:
        group_idx = len(_GROUP_ORDER)
    return (group_idx, tag)


def _sorted_tags(tags: Sequence[str]) -> list[str]:
    """Return tags sorted by group then alphabetically within each group."""
    return sorted(tags, key=_group_key)


def _compute_trend(series: pd.Series) -> str:
    """Compare mean of last 20% of data points to mean of prior 80%.

    Returns ↑ if > 5% increase, ↓ if > 5% decrease, → otherwise.
    """
    n = len(series)
    if n < 2:
        return _TREND_FLAT

    split = max(1, int(n * 0.8))
    head = series.iloc[:split]
    tail = series.iloc[split:]

    if tail.empty or head.empty:
        return _TREND_FLAT

    head_mean = head.mean()
    tail_mean = tail.mean()

    ref = abs(head_mean) if abs(head_mean) > 1e-9 else 1e-9
    relative_change = (tail_mean - head_mean) / ref

    if relative_change > 0.05:
        return _TREND_UP
    if relative_change < -0.05:
        return _TREND_DOWN
    return _TREND_FLAT


def _metric_stats(df: pd.DataFrame, tag: str) -> dict:
    """Return latest, mean, min, max, trend for a single tag."""
    subset = df[df["tag"] == tag].sort_values("step")
    values = subset["value"].dropna()
    if values.empty:
        return {
            "latest": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "trend": _TREND_FLAT,
        }
    return {
        "latest": float(values.iloc[-1]),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "trend": _compute_trend(values.reset_index(drop=True)),
    }


def _fmt_val(value: float, tag: str) -> str:
    """Format a numeric value for display, with 's' suffix for lap times."""
    if np.isnan(value):
        return "nan"
    suffix = "s" if "lap_time" in tag else ""
    # Use appropriate precision
    if abs(value) >= 1000:
        return f"{value:,.0f}{suffix}"
    if abs(value) >= 100:
        return f"{value:.1f}{suffix}"
    if abs(value) >= 10:
        return f"{value:.2f}{suffix}"
    return f"{value:.3f}{suffix}"


def _estimate_episodes(df: pd.DataFrame) -> int | None:
    """Estimate episode count from length of rollout/ep_rew_mean series."""
    subset = df[df["tag"] == "rollout/ep_rew_mean"]
    if subset.empty:
        return None
    return len(subset)


def _max_step(df: pd.DataFrame) -> int | None:
    """Return the maximum step value in the DataFrame, or None if empty."""
    if df.empty:
        return None
    return int(df["step"].max())


def _header_line(path: str, total_steps: int | None, episodes: int | None) -> str:
    """Build the header line with path, steps, and episodes."""
    step_str = f"{total_steps:,}" if total_steps is not None else "?"
    ep_str = f"~{episodes:,}" if episodes is not None else None

    parts = [f"Steps: {step_str}"]
    if ep_str:
        parts.append(f"Episodes: {ep_str}")
    meta = " | ".join(parts)

    border_char = "═"
    title = f" TB Check: {path} "
    border = border_char * 2
    header = f"{border}{title}{border}"
    return f"{header}\n{meta}"


# ---------------------------------------------------------------------------
# Summary mode formatting
# ---------------------------------------------------------------------------

def _format_metrics_table(df: pd.DataFrame) -> str:
    """Render the metrics table for a single run."""
    tags = _sorted_tags(df["tag"].unique().tolist())
    if not tags:
        return " (no metrics)"

    # Column header
    mc = _COL_METRIC
    vc = _COL_VALUE
    header = (
        f" {'METRIC':<{mc}} {'LATEST':>{vc}} {'MEAN':>{vc}} {'MIN':>{vc}}"
        f" {'MAX':>{vc}}  {'TREND'}"
    )
    separator = " " + "-" * (mc + vc * 4 + 8)

    rows = [header, separator]
    prev_group = None
    for tag in tags:
        current_group = tag.split("/")[0] if "/" in tag else ""
        if prev_group is not None and current_group != prev_group:
            rows.append("")  # blank line between groups
        prev_group = current_group

        stats = _metric_stats(df, tag)
        latest_str = _fmt_val(stats["latest"], tag)
        mean_str = _fmt_val(stats["mean"], tag)
        min_str = _fmt_val(stats["min"], tag)
        max_str = _fmt_val(stats["max"], tag)

        row = (
            f" {tag:<{mc}} {latest_str:>{vc}} {mean_str:>{vc}}"
            f" {min_str:>{vc}} {max_str:>{vc}}   {stats['trend']}"
        )
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Comparison mode formatting
# ---------------------------------------------------------------------------

def _format_comparison_table(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    """Render the two-run comparison metrics table."""
    tags1 = set(df1["tag"].unique())
    tags2 = set(df2["tag"].unique())
    all_tags = _sorted_tags(list(tags1 | tags2))

    if not all_tags:
        return " (no metrics)"

    mc = _COL_METRIC
    vc = _COL_VALUE
    dc = _COL_DELTA
    header = (
        f" {'METRIC':<{mc}} {'RUN1':>{vc}} {'RUN2':>{vc}} {'DELTA':>{dc}}"
    )
    separator = " " + "-" * (mc + vc * 2 + dc + 4)

    rows = [header, separator]
    prev_group = None
    for tag in all_tags:
        current_group = tag.split("/")[0] if "/" in tag else ""
        if prev_group is not None and current_group != prev_group:
            rows.append("")
        prev_group = current_group

        stats1 = _metric_stats(df1, tag) if tag in tags1 else None
        stats2 = _metric_stats(df2, tag) if tag in tags2 else None

        latest1 = stats1["latest"] if stats1 else float("nan")
        latest2 = stats2["latest"] if stats2 else float("nan")

        val1_str = _fmt_val(latest1, tag) if stats1 else "—"
        val2_str = _fmt_val(latest2, tag) if stats2 else "—"

        if not np.isnan(latest1) and not np.isnan(latest2):
            delta = latest2 - latest1
            if abs(latest1) > 1e-9:
                pct = (delta / abs(latest1)) * 100
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{_fmt_val(delta, tag)} ({sign}{pct:.1f}%)"
            else:
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{_fmt_val(delta, tag)}"
        else:
            delta_str = "—"

        row = (
            f" {tag:<{mc}} {val1_str:>{vc}} {val2_str:>{vc}} {delta_str:>{dc}}"
        )
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Diagnostics section formatting
# ---------------------------------------------------------------------------

def _format_diagnostics_section(findings: list[Finding]) -> str:
    """Render the DIAGNOSTICS section from a list of findings."""
    lines = [" DIAGNOSTICS"]

    critical = [f for f in findings if f.severity == "CRITICAL"]
    warn = [f for f in findings if f.severity == "WARN"]
    info = [f for f in findings if f.severity == "INFO"]

    all_shown = critical + warn + info
    if not all_shown:
        lines.append(" ✓ No critical issues detected")
        return "\n".join(lines)

    for finding in all_shown:
        icon = _ICON.get(finding.severity, "·")
        lines.append(f" {icon} {finding.severity:<8} {finding.message}")

    if not critical:
        lines.append(" ✓ No critical issues detected")

    return "\n".join(lines)


def _format_comparison_diagnostics(
    findings1: list[Finding], findings2: list[Finding]
) -> str:
    """Render diagnostics for a two-run comparison (run1 then run2)."""
    sections = []
    if findings1:
        sections.append(" DIAGNOSTICS — run1")
        for f in findings1:
            icon = _ICON.get(f.severity, "·")
            sections.append(f"  {icon} {f.severity:<8} {f.message}")
        if not any(f.severity == "CRITICAL" for f in findings1):
            sections.append("  ✓ No critical issues detected")
    else:
        sections.append(" DIAGNOSTICS — run1")
        sections.append("  ✓ No critical issues detected")

    sections.append("")

    if findings2:
        sections.append(" DIAGNOSTICS — run2")
        for f in findings2:
            icon = _ICON.get(f.severity, "·")
            sections.append(f"  {icon} {f.severity:<8} {f.message}")
        if not any(f.severity == "CRITICAL" for f in findings2):
            sections.append("  ✓ No critical issues detected")
    else:
        sections.append(" DIAGNOSTICS — run2")
        sections.append("  ✓ No critical issues detected")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_summary(
    df: pd.DataFrame, path: str, findings: list[Finding]
) -> str:
    """Format a single-run summary for terminal display.

    Args:
        df: Long-format DataFrame with columns ``step``, ``tag``, ``value``.
        path: Path to the TB run directory (used in the header).
        findings: Diagnostic findings from ``run_diagnostics``.

    Returns:
        Multi-line string ready for ``print()``.
    """
    steps = _max_step(df)
    episodes = _estimate_episodes(df)

    header = _header_line(path, steps, episodes)
    metrics = _format_metrics_table(df)
    diagnostics = _format_diagnostics_section(findings)

    return "\n".join([header, "", metrics, "", diagnostics])


def format_comparison(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    path1: str,
    path2: str,
    findings1: list[Finding],
    findings2: list[Finding],
) -> str:
    """Format a two-run comparison for terminal display.

    Adds a DELTA column showing difference between run2 and run1 latest values.
    No step alignment — runs may have different lengths.

    Args:
        df1: Long-format DataFrame for run1.
        df2: Long-format DataFrame for run2.
        path1: Path to run1 (used in the header).
        path2: Path to run2 (used in the header).
        findings1: Diagnostic findings for run1.
        findings2: Diagnostic findings for run2.

    Returns:
        Multi-line string ready for ``print()``.
    """
    steps1 = _max_step(df1)
    steps2 = _max_step(df2)
    episodes1 = _estimate_episodes(df1)
    episodes2 = _estimate_episodes(df2)

    border = "═" * 2
    lines = [
        f"{border} TB Check (comparison) {border}",
        f" run1: {path1}",
        f" run2: {path2}",
    ]

    # Per-run step/episode info
    def _run_meta(steps: int | None, episodes: int | None) -> str:
        step_str = f"{steps:,}" if steps is not None else "?"
        parts = [f"Steps: {step_str}"]
        if episodes is not None:
            parts.append(f"Episodes: ~{episodes:,}")
        return " | ".join(parts)

    lines.append(f" run1: {_run_meta(steps1, episodes1)}")
    lines.append(f" run2: {_run_meta(steps2, episodes2)}")
    lines.append("")
    lines.append(_format_comparison_table(df1, df2))
    lines.append("")
    lines.append(_format_comparison_diagnostics(findings1, findings2))

    return "\n".join(lines)


def format_diagnostics_only(findings: list[Finding]) -> str:
    """Format diagnostics findings only (for --diag-only mode).

    Args:
        findings: Diagnostic findings from ``run_diagnostics``.

    Returns:
        Multi-line string ready for ``print()``.
    """
    return _format_diagnostics_section(findings)
