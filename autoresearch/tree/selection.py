"""Branch selection algorithm for research tree exploration."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np

from autoresearch.archive.map_elites import MapElitesArchive
from autoresearch.descriptors.compute import DescriptorVector
from autoresearch.tree.research_tree import ResearchTree


def behavioral_distance(
    a: DescriptorVector,
    b: DescriptorVector,
    axis_ranges: list[tuple[float, float]] | None = None,
) -> float:
    """Compute normalized Euclidean distance between two descriptor vectors."""
    ranges = axis_ranges or [(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)]
    a_vals = a.to_tuple()
    b_vals = b.to_tuple()
    total = 0.0
    for av, bv, (lo, hi) in zip(a_vals, b_vals, ranges):
        span = hi - lo
        if span > 0:
            total += ((av - bv) / span) ** 2
    return sqrt(total)


@dataclass
class BranchStats:
    """Aggregated statistics for a research tree branch."""

    best_fitness: float
    visit_count: int
    centroid: DescriptorVector | None = None
    empty_target_ratio: float = 1.0  # fraction of target cells still empty

    @classmethod
    def from_tree(cls, tree: ResearchTree, branch_root_id: str) -> BranchStats:
        root = tree.get_node(branch_root_id)
        descendants = tree.get_descendants(branch_root_id)
        all_nodes = [root] + descendants

        completed = [n for n in all_nodes if n.status == "completed" and n.fitness is not None]
        visit_count = len([n for n in all_nodes if n.scope != "baseline"])
        best_fitness = min((n.fitness for n in completed), default=float("inf"))

        return cls(
            best_fitness=best_fitness,
            visit_count=visit_count,
            centroid=None,
        )


def score_branch(
    stats: BranchStats,
    archive: MapElitesArchive,
    tree: ResearchTree,
    config: dict,
) -> float:
    """Score a branch for selection. Higher = more likely to be explored next.

    Implements the spec's UCB-style scoring with exploit, explore, gap_bonus,
    and diversity_bonus terms.
    """
    # Minimum exploration guarantee
    if stats.visit_count < config.get("min_branch_experiments", 3):
        return float("inf")

    # Guard against zero/inf fitness
    best_fitness = max(stats.best_fitness, 1e-6)
    if best_fitness == float("inf"):
        best_fitness = 1e6

    total_experiments = max(tree.total_experiments, 1)
    visit_count = max(stats.visit_count, 1)

    # UCB-style score
    exploit = config.get("exploit_weight", 1.0) / best_fitness
    explore = config.get("explore_weight", 1.0) * sqrt(
        log(total_experiments) / visit_count
    )

    # Gap bonus: reward branches targeting unfilled archive cells
    gap_bonus = config.get("gap_weight", 0.5) * stats.empty_target_ratio

    # Diversity bonus (if centroid available and archive has entries)
    diversity_bonus = 0.0
    if stats.centroid is not None:
        approved = archive.approved_cells()
        if approved:
            best_entry = min(approved.values(), key=lambda e: e.fitness)
            diversity_bonus = config.get("diversity_weight", 0.3) * behavioral_distance(
                stats.centroid, best_entry.descriptors, archive.axis_ranges
            )

    return exploit + explore + gap_bonus + diversity_bonus


def select_next_branch(
    tree: ResearchTree,
    archive: MapElitesArchive,
    config: dict,
    seed: int | None = None,
) -> str | None:
    """Select the next branch to explore. Returns branch_root_id, or None for random restart."""
    rng = np.random.default_rng(seed)

    # Random restart check
    if rng.random() < config.get("random_restart_pct", 0.2):
        return None

    branches = tree.get_level1_branches()
    if not branches:
        return None

    scores = {}
    for branch_id in branches:
        stats = BranchStats.from_tree(tree, branch_id)
        scores[branch_id] = score_branch(stats, archive, tree, config)

    # Handle inf scores (minimum exploration guarantee)
    inf_branches = [b for b, s in scores.items() if s == float("inf")]
    if inf_branches:
        return str(rng.choice(inf_branches))

    # Tie-breaking
    max_score = max(scores.values())
    threshold = config.get("tie_threshold", 0.05)
    tied = [b for b, s in scores.items() if s >= max_score * (1 - threshold)]

    return str(rng.choice(tied))
