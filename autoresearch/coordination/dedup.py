"""Hypothesis deduplication against claims and completed experiments."""

from __future__ import annotations

from autoresearch.hypothesis.schema import Hypothesis


def is_duplicate(
    hypothesis: Hypothesis,
    completed_ids: set[str],
    active_claim_ids: set[str],
) -> bool:
    return hypothesis.id in completed_ids or hypothesis.id in active_claim_ids
