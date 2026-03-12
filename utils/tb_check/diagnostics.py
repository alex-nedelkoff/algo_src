"""diagnostics.py — rules engine for TensorBoard log analysis.

Each rule is a function ``(df: pd.DataFrame) -> list[Finding]`` that inspects
metric timeseries in the long-format DataFrame produced by ``loader.load_run``
and returns zero or more findings.

Call ``run_diagnostics(df)`` to execute all rules and collect results.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Finding:
    severity: Literal["INFO", "WARN", "CRITICAL"]
    rule: str        # rule name
    message: str     # human-readable description
    metric: str | None  # tag that triggered it
    step: int | None    # step where detected


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_series(df: pd.DataFrame, tag: str) -> pd.Series | None:
    """Return the ``value`` series for a single tag, sorted by step.

    Returns ``None`` if the tag is absent or the filtered slice is empty.
    """
    subset = df[df["tag"] == tag]
    if subset.empty:
        return None
    return subset.sort_values("step")["value"].reset_index(drop=True)


def _split_tail(series: pd.Series, tail_fraction: float = 0.2):
    """Split *series* into (head, tail) by fraction of data points."""
    n = len(series)
    split = max(1, int(n * (1 - tail_fraction)))
    return series.iloc[:split], series.iloc[split:]


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def _rule_reward_plateau(df: pd.DataFrame) -> list[Finding]:
    """WARN when rollout/ep_rew_mean is flat over the last 20% of training.

    "Flat" means the mean of the tail differs from the mean of the head by
    less than 1% of the absolute head mean (or less than 1e-6 if head ≈ 0).
    """
    tag = "rollout/ep_rew_mean"
    series = _get_series(df, tag)
    if series is None or len(series) < 5:
        return []

    head, tail = _split_tail(series, tail_fraction=0.2)
    if tail.empty or head.empty:
        return []

    head_mean = head.mean()
    tail_mean = tail.mean()
    ref = abs(head_mean) if abs(head_mean) > 1e-6 else 1e-6
    relative_change = abs(tail_mean - head_mean) / ref

    if relative_change < 0.01:
        return [Finding(
            severity="WARN",
            rule="reward_plateau",
            message=(
                f"Reward appears flat over the last 20% of training "
                f"(head_mean={head_mean:.4f}, tail_mean={tail_mean:.4f}, "
                f"relative_change={relative_change:.4%})"
            ),
            metric=tag,
            step=None,
        )]
    return []


def _rule_reward_collapse(df: pd.DataFrame) -> list[Finding]:
    """CRITICAL when rollout/ep_rew_mean drops > 30% from its peak."""
    tag = "rollout/ep_rew_mean"
    series = _get_series(df, tag)
    if series is None or len(series) < 2:
        return []

    peak = series.max()
    # Only meaningful if peak is positive; if reward is always ≤ 0, skip.
    if peak <= 0:
        return []

    current = series.iloc[-1]
    drop_fraction = (peak - current) / peak
    if drop_fraction > 0.30:
        # Find the step of the last value for reporting.
        subset = df[df["tag"] == tag].sort_values("step")
        last_step = int(subset["step"].iloc[-1])
        return [Finding(
            severity="CRITICAL",
            rule="reward_collapse",
            message=(
                f"Reward collapsed: peak={peak:.4f}, current={current:.4f}, "
                f"drop={drop_fraction:.1%} (> 30%)"
            ),
            metric=tag,
            step=last_step,
        )]
    return []


def _rule_kl_spike(df: pd.DataFrame) -> list[Finding]:
    """WARN when train/approx_kl exceeds 0.05 at any point."""
    tag = "train/approx_kl"
    series = _get_series(df, tag)
    if series is None:
        return []

    threshold = 0.05
    mask = series > threshold
    if not mask.any():
        return []

    subset = df[df["tag"] == tag].sort_values("step").reset_index(drop=True)
    first_idx = int(mask.idxmax())
    spike_value = float(series.iloc[first_idx])
    step_val = int(subset["step"].iloc[first_idx])

    return [Finding(
        severity="WARN",
        rule="kl_spike",
        message=(
            f"KL divergence exceeded 0.05 (value={spike_value:.4f} at step {step_val})"
        ),
        metric=tag,
        step=step_val,
    )]


def _rule_entropy_collapse(df: pd.DataFrame) -> list[Finding]:
    """WARN when train/entropy_loss drops below 10% of its initial value."""
    tag = "train/entropy_loss"
    series = _get_series(df, tag)
    if series is None or len(series) < 5:
        return []

    n_init = max(1, int(len(series) * 0.1))
    initial_mean = series.iloc[:n_init].mean()
    # Entropy loss is typically negative in SB3 (negative entropy).
    # We compare magnitudes: collapse when |current| < 10% of |initial|.
    if abs(initial_mean) < 1e-8:
        return []

    threshold = 0.10 * abs(initial_mean)
    current = series.iloc[-1]

    # Collapse: the absolute value of entropy loss has shrunk drastically.
    if abs(current) < threshold:
        subset = df[df["tag"] == tag].sort_values("step")
        last_step = int(subset["step"].iloc[-1])
        return [Finding(
            severity="WARN",
            rule="entropy_collapse",
            message=(
                f"Entropy collapsed: current={current:.4f} is below 10% of "
                f"initial mean ({initial_mean:.4f})"
            ),
            metric=tag,
            step=last_step,
        )]
    return []


def _rule_high_crash_rate(df: pd.DataFrame) -> list[Finding]:
    """WARN when any termination reason (excluding timeout/none) exceeds 50%."""
    excluded = {"termination/timeout", "termination/none"}
    term_tags = [
        t for t in df["tag"].unique()
        if t.startswith("termination/") and t not in excluded
    ]

    findings: list[Finding] = []
    for tag in term_tags:
        subset = df[df["tag"] == tag].sort_values("step")
        if subset.empty:
            continue
        latest_value = float(subset["value"].iloc[-1])
        latest_step = int(subset["step"].iloc[-1])
        if latest_value > 0.50:
            findings.append(Finding(
                severity="WARN",
                rule="high_crash_rate",
                message=(
                    f"Termination reason '{tag}' is {latest_value:.1%} "
                    f"(> 50%) at step {latest_step}"
                ),
                metric=tag,
                step=latest_step,
            ))
    return findings


def _rule_nan_detected(df: pd.DataFrame) -> list[Finding]:
    """CRITICAL if any NaN appears in any metric."""
    nan_mask = df["value"].isna()
    if not nan_mask.any():
        return []

    nan_tags = df.loc[nan_mask, "tag"].unique().tolist()
    first_nan_row = df[nan_mask].sort_values("step").iloc[0]
    first_step = int(first_nan_row["step"])
    first_tag = str(first_nan_row["tag"])

    return [Finding(
        severity="CRITICAL",
        rule="nan_detected",
        message=(
            f"NaN values detected in metrics: {nan_tags}. "
            f"First occurrence: tag='{first_tag}' at step {first_step}"
        ),
        metric=first_tag,
        step=first_step,
    )]


def _rule_no_gate_progress(df: pd.DataFrame) -> list[Finding]:
    """CRITICAL when racing/gates_per_ep is stuck at 0 after 500K steps."""
    tag = "racing/gates_per_ep"
    subset = df[df["tag"] == tag]
    if subset.empty:
        return []

    max_step = int(subset["step"].max())
    max_value = float(subset["value"].max())

    if max_value == 0 and max_step > 500_000:
        return [Finding(
            severity="CRITICAL",
            rule="no_gate_progress",
            message=(
                f"No gate progress: {tag} has remained at 0 up to step {max_step:,} "
                f"(threshold: 500K steps)"
            ),
            metric=tag,
            step=max_step,
        )]
    return []


def _rule_value_loss_explosion(df: pd.DataFrame) -> list[Finding]:
    """WARN when train/value_loss exceeds 10x its rolling average."""
    tag = "train/value_loss"
    series = _get_series(df, tag)
    if series is None or len(series) < 10:
        return []

    window = min(100, len(series) - 1)
    rolling_avg = series.rolling(window=window, min_periods=1).mean()

    # Find first point where value > 10x its rolling average (both positive).
    ratio = series / rolling_avg.replace(0, np.nan)
    explosion_mask = ratio > 10.0

    if not explosion_mask.any():
        return []

    first_idx = int(explosion_mask.idxmax())
    explosion_value = float(series.iloc[first_idx])
    avg_value = float(rolling_avg.iloc[first_idx])

    subset = df[df["tag"] == tag].sort_values("step").reset_index(drop=True)
    step_val = int(subset["step"].iloc[first_idx])

    return [Finding(
        severity="WARN",
        rule="value_loss_explosion",
        message=(
            f"Value loss explosion: {explosion_value:.4f} > 10x rolling avg "
            f"({avg_value:.4f}) at step {step_val}"
        ),
        metric=tag,
        step=step_val,
    )]


def _rule_clip_fraction_saturation(df: pd.DataFrame) -> list[Finding]:
    """WARN when train/clip_fraction mean over the last 20% of data > 0.3."""
    tag = "train/clip_fraction"
    series = _get_series(df, tag)
    if series is None or len(series) < 5:
        return []

    _, tail = _split_tail(series, tail_fraction=0.2)
    if tail.empty:
        return []

    tail_mean = float(tail.mean())
    if tail_mean > 0.30:
        subset = df[df["tag"] == tag].sort_values("step")
        last_step = int(subset["step"].iloc[-1])
        return [Finding(
            severity="WARN",
            rule="clip_fraction_saturation",
            message=(
                f"Clip fraction consistently high: mean of last 20% = {tail_mean:.3f} "
                f"(> 0.3) at step {last_step}"
            ),
            metric=tag,
            step=last_step,
        )]
    return []


# ---------------------------------------------------------------------------
# Rule registry and top-level entry point
# ---------------------------------------------------------------------------

_RULES = [
    _rule_reward_plateau,
    _rule_reward_collapse,
    _rule_kl_spike,
    _rule_entropy_collapse,
    _rule_high_crash_rate,
    _rule_nan_detected,
    _rule_no_gate_progress,
    _rule_value_loss_explosion,
    _rule_clip_fraction_saturation,
]


def run_diagnostics(df: pd.DataFrame) -> list[Finding]:
    """Run all diagnostic rules against *df* and return collected findings.

    Args:
        df: Long-format DataFrame with columns ``step`` (int), ``tag`` (str),
            ``value`` (float) as produced by ``loader.load_run``.

    Returns:
        List of :class:`Finding` instances, one per triggered rule condition.
        Rules that find no issues return an empty list; findings from all
        rules are concatenated and returned together.
    """
    findings: list[Finding] = []
    for rule_fn in _RULES:
        findings.extend(rule_fn(df))
    return findings
