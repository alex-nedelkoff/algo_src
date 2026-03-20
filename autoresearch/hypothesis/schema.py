"""Hypothesis data contract and content hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

SCOPE_BUDGETS = {
    "hyperparameter": 5_000_000,
    "algorithm": 15_000_000,
    "architecture": 30_000_000,
    "system": 50_000_000,
}


@dataclass(frozen=True)
class HydraOverride:
    """A single Hydra config override."""

    key: str
    value: str

    def to_cli_arg(self) -> str:
        return f"{self.key}={self.value}"


@dataclass(frozen=True)
class FileDiff:
    """A source file modification (Phase 2+)."""

    path: str
    description: str


# Union type for changes — HydraOverride for config-only, FileDiff for code changes
Change = HydraOverride | FileDiff


def compute_hypothesis_id(
    scope: str,
    changes: list[Change],
    target_cells: list[tuple[int, int, int]] | None = None,
) -> str:
    """Compute a deterministic content hash for deduplication.

    NOTE: Phase 2 changed hash format to include change type prefix.
    Hypothesis IDs from Phase 1 are not compatible — existing state files
    should be cleared or migrated when upgrading.
    """
    change_tuples = []
    for c in changes:
        if isinstance(c, HydraOverride):
            change_tuples.append(("override", c.key, c.value))
        elif isinstance(c, FileDiff):
            change_tuples.append(("filediff", c.path, c.description))
    content = {
        "scope": scope,
        "changes": sorted(change_tuples),
        "target_cells": sorted([list(c) for c in (target_cells or [])]),
    }
    blob = json.dumps(content, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class Hypothesis:
    """A research hypothesis specifying what to change and why."""

    id: str
    parent_id: str | None
    inspired_by: str | None
    scope: Literal["hyperparameter", "algorithm", "architecture", "system"]
    description: str
    changes: list[Change]
    target_cells: list[tuple[int, int, int]]
    predicted_descriptor_range: dict
    rationale: str
    estimated_budget: int
    risk_level: Literal["low", "medium", "high"]
    novelty_score: float

    @classmethod
    def create(
        cls,
        scope: Literal["hyperparameter", "algorithm", "architecture", "system"],
        description: str,
        changes: list[Change],
        rationale: str,
        parent_id: str | None = None,
        inspired_by: str | None = None,
        target_cells: list[tuple[int, int, int]] | None = None,
        predicted_descriptor_range: dict | None = None,
        estimated_budget: int | None = None,
        risk_level: Literal["low", "medium", "high"] = "low",
        novelty_score: float = 1.0,
    ) -> Hypothesis:
        """Create a hypothesis with auto-computed ID and defaults."""
        return cls(
            id=compute_hypothesis_id(scope, changes, target_cells),
            parent_id=parent_id,
            inspired_by=inspired_by,
            scope=scope,
            description=description,
            changes=changes,
            target_cells=target_cells or [],
            predicted_descriptor_range=predicted_descriptor_range or {},
            rationale=rationale,
            estimated_budget=estimated_budget or SCOPE_BUDGETS[scope],
            risk_level=risk_level,
            novelty_score=novelty_score,
        )

    def to_cli_overrides(self) -> list[str]:
        """Generate Hydra CLI override arguments (ignores FileDiff changes)."""
        return [c.to_cli_arg() for c in self.changes if isinstance(c, HydraOverride)]
