"""Tests for git worktree lifecycle management."""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest
from autoresearch.worktree import WorktreeManager, WorktreeInfo, BRANCH_PREFIX, MAX_WORKTREES


@pytest.fixture
def manager(tmp_path):
    return WorktreeManager(repo_root=tmp_path, worktree_dir=tmp_path / "worktrees")


class TestBranchNaming:
    def test_branch_name_format(self, manager):
        assert manager.branch_name("abc123") == f"{BRANCH_PREFIX}abc123"

    def test_worktree_path(self, manager):
        path = manager.worktree_path("abc123")
        assert "abc123" in str(path)


class TestWorktreeInfo:
    def test_creation(self):
        info = WorktreeInfo(
            path="/tmp/worktrees/abc", branch="ar/exp-abc",
            hypothesis_id="abc", created_at=None,
        )
        assert info.hypothesis_id == "abc"


class TestConstants:
    def test_max_worktrees(self):
        assert MAX_WORKTREES == 5

    def test_branch_prefix(self):
        assert BRANCH_PREFIX == "ar/exp-"


class TestCleanupPolicy:
    def test_find_stale_filters_by_age(self, manager):
        old = WorktreeInfo(
            path="/tmp/old", branch="ar/exp-old", hypothesis_id="old",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        recent = WorktreeInfo(
            path="/tmp/recent", branch="ar/exp-recent", hypothesis_id="recent",
            created_at=datetime.now(timezone.utc),
        )
        stale = manager._find_stale([old, recent], max_age_days=7)
        assert len(stale) == 1
        assert stale[0].hypothesis_id == "old"

    def test_find_stale_ignores_no_timestamp(self, manager):
        no_ts = WorktreeInfo(
            path="/tmp/x", branch="ar/exp-x", hypothesis_id="x", created_at=None,
        )
        stale = manager._find_stale([no_ts], max_age_days=7)
        assert len(stale) == 0


class TestListParsing:
    def test_list_empty(self, manager):
        with patch.object(manager, "_run_git", return_value=""):
            assert manager.list_worktrees() == []

    def test_list_parses_porcelain(self, manager):
        output = (
            "worktree /repo\n"
            "branch refs/heads/main\n"
            "\n"
            "worktree /repo/.worktrees/abc123\n"
            "branch refs/heads/ar/exp-abc123\n"
            "\n"
        )
        with patch.object(manager, "_run_git", return_value=output):
            result = manager.list_worktrees()
            assert len(result) == 1
            assert result[0].hypothesis_id == "abc123"
            assert result[0].branch == "ar/exp-abc123"


class TestMaxWorktreeEnforcement:
    def test_create_raises_when_max_exceeded(self, manager):
        fake_worktrees = [
            WorktreeInfo(f"/tmp/{i}", f"ar/exp-{i}", str(i)) for i in range(MAX_WORKTREES)
        ]
        with patch.object(manager, "list_worktrees", return_value=fake_worktrees):
            with pytest.raises(RuntimeError, match="Max worktrees"):
                manager.create_worktree("new_experiment")
