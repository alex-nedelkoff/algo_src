"""Benchmark results dataclasses and aggregation functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass
class EnvResult:
    """Result from a single environment (one layout + one starting condition)."""
    layout_name: str
    variant_index: int
    lap_time: float | None
    laps_completed: int
    gates_passed: int
    crash: bool
    avg_speed: float
    termination_reason: str
    rerun_url: str | None = None


@dataclass
class LayoutResult:
    """Aggregated result for one track layout across its 5 starting variants."""
    layout_name: str
    fastest_lap: float | None
    avg_lap: float | None
    laps_completed: float
    gates_passed: float
    crash_rate: float
    avg_speed: float
    variants: list[EnvResult] = field(default_factory=list)


@dataclass
class AggregateResult:
    """Aggregated result across all layouts."""
    fastest_lap: float | None
    avg_lap: float | None
    laps_completed: float
    gates_passed: float
    crash_rate: float
    avg_speed: float


@dataclass
class BenchmarkResults:
    """Complete benchmark output."""
    golden_set_version: str
    checkpoint_path: str
    golden_set_mode: str
    per_environment: list[EnvResult]
    per_layout: list[LayoutResult]
    aggregate: AggregateResult


def aggregate_env_results(layout_name: str, envs: list[EnvResult]) -> LayoutResult:
    """Aggregate EnvResults for a single layout into a LayoutResult."""
    lap_times = [e.lap_time for e in envs if e.lap_time is not None]
    return LayoutResult(
        layout_name=layout_name,
        fastest_lap=min(lap_times) if lap_times else None,
        avg_lap=mean(lap_times) if lap_times else None,
        laps_completed=mean(e.laps_completed for e in envs),
        gates_passed=mean(e.gates_passed for e in envs),
        crash_rate=mean(1.0 if e.crash else 0.0 for e in envs),
        avg_speed=mean(e.avg_speed for e in envs),
        variants=envs,
    )


def aggregate_layout_results(layouts: list[LayoutResult]) -> AggregateResult:
    """Aggregate LayoutResults across all layouts into an AggregateResult."""
    fastest_laps = [lr.fastest_lap for lr in layouts if lr.fastest_lap is not None]
    avg_laps = [lr.avg_lap for lr in layouts if lr.avg_lap is not None]
    return AggregateResult(
        fastest_lap=mean(fastest_laps) if fastest_laps else None,
        avg_lap=mean(avg_laps) if avg_laps else None,
        laps_completed=mean(lr.laps_completed for lr in layouts),
        gates_passed=mean(lr.gates_passed for lr in layouts),
        crash_rate=mean(lr.crash_rate for lr in layouts),
        avg_speed=mean(lr.avg_speed for lr in layouts),
    )
