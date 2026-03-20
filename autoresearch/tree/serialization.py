"""Serialize/deserialize research tree to/from JSON."""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch.tree.research_tree import ResearchTree, TreeNode


def save_tree(tree: ResearchTree, path: Path | str) -> None:
    path = Path(path)
    data = {
        "root_id": tree._root_id,
        "nodes": {},
    }
    for node in tree.all_nodes():
        data["nodes"][node.hypothesis_id] = {
            "parent_id": node.parent_id,
            "inspired_by": node.inspired_by,
            "scope": node.scope,
            "description": node.description,
            "status": node.status,
            "fitness": node.fitness,
            "wandb_run_id": node.wandb_run_id,
            "git_commit": node.git_commit,
            "failure_reason": node.failure_reason,
            "archive_cell": list(node.archive_cell) if node.archive_cell else None,
        }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_tree(path: Path | str) -> ResearchTree:
    path = Path(path)
    data = json.loads(path.read_text())
    tree = ResearchTree()
    tree._root_id = data["root_id"]
    for hyp_id, node_data in data["nodes"].items():
        cell = node_data.get("archive_cell")
        tree._nodes[hyp_id] = TreeNode(
            hypothesis_id=hyp_id,
            parent_id=node_data["parent_id"],
            inspired_by=node_data.get("inspired_by"),
            scope=node_data["scope"],
            description=node_data["description"],
            status=node_data["status"],
            fitness=node_data.get("fitness"),
            wandb_run_id=node_data.get("wandb_run_id"),
            git_commit=node_data.get("git_commit"),
            failure_reason=node_data.get("failure_reason"),
            archive_cell=tuple(cell) if cell else None,
        )
    return tree
