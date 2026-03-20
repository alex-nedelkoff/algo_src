"""Top-level constraint validator that combines all checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.constraints.physics import check_physics, ConstraintCheck
from autoresearch.constraints.racing import check_gate_passage, check_behavioral_sanity


@dataclass
class ValidationResult:
    passed: bool
    constraint_results: dict[str, ConstraintCheck] = field(default_factory=dict)
    reasoning: str = ""
    soft_signals: dict[str, float] = field(default_factory=dict)


class ConstraintValidator:
    def __init__(
        self,
        max_rpm: float = 31470.0,
        max_accel_g: float = 4.0,
        min_success_rate: float = 0.8,
        min_avg_speed: float = 2.0,
        max_gate_offset_ratio: float = 0.8,
    ) -> None:
        self.max_rpm = max_rpm
        self.max_accel_g = max_accel_g
        self.min_success_rate = min_success_rate
        self.min_avg_speed = min_avg_speed
        self.max_gate_offset_ratio = max_gate_offset_ratio

    def validate(
        self,
        traj: TrajectoryData,
        metrics: dict,
        dr_active: bool = True,
    ) -> ValidationResult:
        """Run all constraint checks.

        Args:
            traj: Parsed trajectory data from .npz file.
            metrics: Pre-aggregated metrics dict with keys:
                - "success_rate" (float): fraction of eval episodes that succeeded
                - "lap_times" (list[float]): per-episode lap times
                Caller aggregates raw EpisodeMetrics into this format.
            dr_active: Whether domain randomization was active during eval.
        """
        results: dict[str, ConstraintCheck] = {}

        success_rate = metrics.get("success_rate", 0.0)
        if success_rate < self.min_success_rate:
            results["success_rate"] = ConstraintCheck(False, f"Success rate {success_rate:.1%} < {self.min_success_rate:.1%}")
        else:
            results["success_rate"] = ConstraintCheck(True, f"Success rate: {success_rate:.1%}")

        results["physics"] = check_physics(traj, self.max_rpm, self.max_accel_g)
        results["gate_passage"] = check_gate_passage(traj, self.max_gate_offset_ratio)
        results["behavioral_sanity"] = check_behavioral_sanity(traj, self.min_avg_speed, dr_active)

        all_passed = all(r.passed for r in results.values())
        failed = [f"{k}: {v.reason}" for k, v in results.items() if not v.passed]
        reasoning = "All constraints passed" if all_passed else "; ".join(failed)

        return ValidationResult(passed=all_passed, constraint_results=results, reasoning=reasoning)
