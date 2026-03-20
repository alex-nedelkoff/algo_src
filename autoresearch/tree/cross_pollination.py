"""Cross-pollination: identify successful patterns and propose grafting across branches."""

from __future__ import annotations
from dataclasses import dataclass
from autoresearch.tree.research_tree import ResearchTree, TreeNode


@dataclass
class SuccessfulPattern:
    branch_root_id: str
    best_fitness: float
    improvement_trajectory: list[float]
    description: str


@dataclass
class CrossPollinationProposal:
    source_branch: str
    target_branch: str
    inspired_by: str
    description: str
    rationale: str


def find_successful_patterns(tree: ResearchTree) -> list[SuccessfulPattern]:
    """Find branches with completed experiments that show improvement."""
    branches = tree.get_level1_branches()
    patterns = []
    for branch_id in branches:
        root = tree.get_node(branch_id)
        descendants = tree.get_descendants(branch_id)
        all_nodes = [root] + descendants
        completed = [n for n in all_nodes if n.status == "completed" and n.fitness is not None]
        if not completed:
            continue
        completed.sort(key=lambda n: n.fitness)
        fitness_trajectory = [n.fitness for n in completed]
        best_fitness = completed[0].fitness
        root_fitness = root.fitness
        if root_fitness is not None and best_fitness >= root_fitness:
            continue
        patterns.append(SuccessfulPattern(
            branch_root_id=branch_id, best_fitness=best_fitness,
            improvement_trajectory=fitness_trajectory, description=root.description,
        ))
    patterns.sort(key=lambda p: p.best_fitness)
    return patterns


def propose_cross_pollinations(tree: ResearchTree) -> list[CrossPollinationProposal]:
    """Propose grafting successful patterns onto other branches."""
    patterns = find_successful_patterns(tree)
    if not patterns:
        return []
    all_branches = tree.get_level1_branches()
    proposals = []
    for pattern in patterns:
        for target_branch in all_branches:
            if target_branch == pattern.branch_root_id:
                continue
            target_descendants = tree.get_descendants(target_branch)
            already_grafted = any(n.inspired_by == pattern.branch_root_id for n in target_descendants)
            if already_grafted:
                continue
            source_descendants = tree.get_descendants(pattern.branch_root_id)
            source_root = tree.get_node(pattern.branch_root_id)
            all_source = [source_root] + source_descendants
            best_source = min(
                (n for n in all_source if n.status == "completed" and n.fitness is not None),
                key=lambda n: n.fitness,
            )
            target_node = tree.get_node(target_branch)
            proposals.append(CrossPollinationProposal(
                source_branch=pattern.branch_root_id,
                target_branch=target_branch,
                inspired_by=best_source.hypothesis_id,
                description=f"Apply '{pattern.description}' pattern (fitness={pattern.best_fitness:.2f}) to branch '{target_node.description}'",
                rationale=f"Branch '{pattern.description}' achieved fitness {pattern.best_fitness:.2f}. Grafting onto '{target_node.description}' may yield similar improvements.",
            ))
    return proposals
