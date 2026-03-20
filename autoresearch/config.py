"""Configuration for the auto-research system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AutoResearchConfig:
    """Top-level auto-research configuration.

    Mirrors the plugin settings.yaml schema.
    """

    wandb_project: str = "corvidx-drone-racing"
    mode: str = "interactive"  # interactive | autonomous | yolo

    budgets: dict[str, int] = field(default_factory=lambda: {
        "hyperparameter": 5_000_000,
        "algorithm": 15_000_000,
        "architecture": 30_000_000,
        "system": 50_000_000,
    })

    early_stop: dict[str, object] = field(default_factory=lambda: {
        "baseline_threshold": 0.7,
        "archive_redundancy": True,
        "poll_interval_seconds": 60,
    })

    constraints: dict[str, float] = field(default_factory=lambda: {
        "min_success_rate": 0.8,
        "min_avg_speed": 2.0,
        "max_gate_offset_ratio": 0.8,
        "max_acceleration_g": 4.0,
    })

    diff_policy: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "max_diff_lines": {
            "algorithm": 500,
            "architecture": 1000,
            "system": 2000,
        },
    })

    branch_selection: dict[str, float] = field(default_factory=lambda: {
        "exploit_weight": 1.0,
        "explore_weight": 1.0,
        "gap_weight": 0.5,
        "diversity_weight": 0.3,
        "random_restart_pct": 0.2,
        "min_branch_experiments": 3,
        "tie_threshold": 0.05,
    })

    coordination: dict[str, object] = field(default_factory=lambda: {
        "claim_timeout_hours": 4,
        "claim_refresh_minutes": 15,
    })

    resource_limits: dict[str, object] = field(default_factory=lambda: {
        "max_concurrent_per_machine": 1,
        "max_wall_clock_hours": 24,
        "max_worktrees": 5,
    })
