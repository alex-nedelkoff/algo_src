"""Research tree for tracking experiment lineage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TreeNode:
    """A node in the research tree."""

    hypothesis_id: str
    parent_id: str | None
    inspired_by: str | None = None
    scope: str = "baseline"
    description: str = ""
    status: str = "pending"  # pending | running | completed | failed
    fitness: float | None = None
    wandb_run_id: str | None = None
    git_commit: str | None = None
    failure_reason: str | None = None
    archive_cell: tuple[int, int, int] | None = None


class ResearchTree:
    """Tree structure tracking all experiments and their lineage."""

    def __init__(self) -> None:
        self._nodes: dict[str, TreeNode] = {}
        self._root_id: str | None = None

    @classmethod
    def create_with_baseline(cls, baseline_name: str, git_commit: str) -> ResearchTree:
        tree = cls()
        root = TreeNode(
            hypothesis_id="baseline_v1",
            parent_id=None,
            scope="baseline",
            description=baseline_name,
            status="baseline",
            git_commit=git_commit,
        )
        tree._nodes[root.hypothesis_id] = root
        tree._root_id = root.hypothesis_id
        return tree

    @property
    def root(self) -> TreeNode:
        assert self._root_id is not None
        return self._nodes[self._root_id]

    @property
    def total_experiments(self) -> int:
        return sum(1 for n in self._nodes.values() if n.scope != "baseline")

    def get_node(self, hypothesis_id: str) -> TreeNode:
        return self._nodes[hypothesis_id]

    def add_experiment(
        self,
        hypothesis_id: str,
        parent_id: str,
        scope: str,
        description: str,
        status: str = "pending",
        inspired_by: str | None = None,
        git_commit: str | None = None,
    ) -> TreeNode:
        node = TreeNode(
            hypothesis_id=hypothesis_id,
            parent_id=parent_id,
            inspired_by=inspired_by,
            scope=scope,
            description=description,
            status=status,
            git_commit=git_commit,
        )
        self._nodes[hypothesis_id] = node
        return node

    def complete_experiment(
        self,
        hypothesis_id: str,
        fitness: float,
        wandb_run_id: str | None = None,
        archive_cell: tuple[int, int, int] | None = None,
    ) -> None:
        node = self._nodes[hypothesis_id]
        node.status = "completed"
        node.fitness = fitness
        node.wandb_run_id = wandb_run_id
        node.archive_cell = archive_cell

    def fail_experiment(self, hypothesis_id: str, reason: str) -> None:
        node = self._nodes[hypothesis_id]
        node.status = "failed"
        node.failure_reason = reason

    def all_nodes(self) -> list[TreeNode]:
        return list(self._nodes.values())

    def get_children(self, hypothesis_id: str) -> list[TreeNode]:
        """Get all direct children of a node."""
        return [n for n in self._nodes.values() if n.parent_id == hypothesis_id]

    def get_descendants(self, hypothesis_id: str) -> list[TreeNode]:
        """Get all descendants (children, grandchildren, etc.) of a node."""
        descendants = []
        queue = list(self.get_children(hypothesis_id))
        while queue:
            node = queue.pop(0)
            descendants.append(node)
            queue.extend(self.get_children(node.hypothesis_id))
        return descendants

    def get_level1_branches(self) -> list[str]:
        """Get hypothesis IDs of Level 1 branches (direct children of root)."""
        if self._root_id is None:
            return []
        return [n.hypothesis_id for n in self.get_children(self._root_id)]

    def promote_baseline(
        self,
        new_baseline_id: str,
        git_commit: str,
        description: str = "",
        wandb_run_id: str | None = None,
        promoter: str | None = None,
    ) -> TreeNode:
        """Create a new baseline root, linking to the previous baseline."""
        new_root = TreeNode(
            hypothesis_id=new_baseline_id,
            parent_id=self._root_id,
            scope="baseline",
            description=description,
            status="baseline",
            git_commit=git_commit,
            wandb_run_id=wandb_run_id,
        )
        self._nodes[new_baseline_id] = new_root
        self._root_id = new_baseline_id
        return new_root
