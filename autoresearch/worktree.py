"""Git worktree lifecycle management for experiment isolation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

BRANCH_PREFIX = "ar/exp-"
MAX_WORKTREES = 5


@dataclass
class WorktreeInfo:
    """Information about an active git worktree."""
    path: str
    branch: str
    hypothesis_id: str
    created_at: datetime | None = None


class WorktreeManager:
    """Manages git worktrees for experiment isolation."""

    def __init__(
        self,
        repo_root: Path | str | None = None,
        worktree_dir: Path | str | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.worktree_dir = Path(worktree_dir) if worktree_dir else self.repo_root / ".worktrees"

    def branch_name(self, hypothesis_id: str) -> str:
        return f"{BRANCH_PREFIX}{hypothesis_id}"

    def worktree_path(self, hypothesis_id: str) -> Path:
        return self.worktree_dir / hypothesis_id

    def create_worktree(self, hypothesis_id: str, base_ref: str = "main") -> str:
        """Create a new worktree. Returns worktree path. Raises RuntimeError if max exceeded."""
        current = self.list_worktrees()
        if len(current) >= MAX_WORKTREES:
            raise RuntimeError(
                f"Max worktrees ({MAX_WORKTREES}) exceeded. Run cleanup first."
            )
        branch = self.branch_name(hypothesis_id)
        wt_path = self.worktree_path(hypothesis_id)
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_git("worktree", "add", "-b", branch, str(wt_path), base_ref)
        return str(wt_path)

    def remove_worktree(self, hypothesis_id: str) -> None:
        wt_path = self.worktree_path(hypothesis_id)
        if wt_path.exists():
            self._run_git("worktree", "remove", str(wt_path), "--force")
        branch = self.branch_name(hypothesis_id)
        try:
            self._run_git("branch", "-D", branch)
        except subprocess.CalledProcessError:
            pass

    def list_worktrees(self) -> list[WorktreeInfo]:
        try:
            output = self._run_git("worktree", "list", "--porcelain")
        except subprocess.CalledProcessError:
            return []
        if not output.strip():
            return []

        worktrees = []
        current_path = None
        current_branch = None
        lines = output.strip().split("\n") + [""]  # sentinel empty line to flush last block
        for line in lines:
            if line.startswith("worktree "):
                current_path = line[len("worktree "):]
            elif line.startswith("branch "):
                ref = line[len("branch "):]
                if ref.startswith("refs/heads/"):
                    current_branch = ref[len("refs/heads/"):]
            elif line == "":
                if current_path and current_branch and current_branch.startswith(BRANCH_PREFIX):
                    hyp_id = current_branch[len(BRANCH_PREFIX):]
                    worktrees.append(WorktreeInfo(
                        path=current_path, branch=current_branch, hypothesis_id=hyp_id,
                    ))
                current_path = None
                current_branch = None
        return worktrees

    def cleanup_stale(self, max_age_days: int = 7) -> list[str]:
        worktrees = self.list_worktrees()
        stale = self._find_stale(worktrees, max_age_days)
        removed = []
        for wt in stale:
            self.remove_worktree(wt.hypothesis_id)
            removed.append(wt.hypothesis_id)
        return removed

    def _find_stale(self, worktrees: list[WorktreeInfo], max_age_days: int = 7) -> list[WorktreeInfo]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return [wt for wt in worktrees if wt.created_at is not None and wt.created_at < cutoff]

    def _run_git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.repo_root,
            capture_output=True, text=True, check=True,
        )
        return result.stdout
