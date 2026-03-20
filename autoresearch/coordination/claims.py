"""Experiment claiming and coordination via git-synced claims.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path


@dataclass
class Claim:
    hypothesis_id: str
    researcher: str
    branch: str
    target_cells: list[tuple[int, ...]]
    scope: str
    timestamp: datetime
    wandb_run_id: str | None
    status: str  # running | completed | failed


def save_claims(claims: list[Claim], path: Path | str) -> None:
    path = Path(path)
    data = {"claims": [_claim_to_dict(c) for c in claims]}
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_claims(path: Path | str) -> list[Claim]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [_dict_to_claim(d) for d in data.get("claims", [])]


def add_claim(claims: list[Claim], claim: Claim) -> list[Claim]:
    return claims + [claim]


def expire_claims(claims: list[Claim], timeout_hours: int = 4) -> list[Claim]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)
    return [c for c in claims if c.status != "running" or c.timestamp > cutoff]


def release_claim(claims: list[Claim], hypothesis_id: str, new_status: str) -> list[Claim]:
    for c in claims:
        if c.hypothesis_id == hypothesis_id:
            c.status = new_status
    return claims


def _claim_to_dict(c: Claim) -> dict:
    return {
        "hypothesis_id": c.hypothesis_id,
        "researcher": c.researcher,
        "branch": c.branch,
        "target_cells": [list(t) for t in c.target_cells],
        "scope": c.scope,
        "timestamp": c.timestamp.isoformat(),
        "wandb_run_id": c.wandb_run_id,
        "status": c.status,
    }


def _dict_to_claim(d: dict) -> Claim:
    return Claim(
        hypothesis_id=d["hypothesis_id"],
        researcher=d["researcher"],
        branch=d["branch"],
        target_cells=[tuple(t) for t in d["target_cells"]],
        scope=d["scope"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        wandb_run_id=d.get("wandb_run_id"),
        status=d["status"],
    )
