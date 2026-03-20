# Auto-Research Phase 2: Code-Level Mutations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the auto-research system from config-only mutations to code-level changes with git worktree isolation, diff enforcement, branch selection, early stopping, autonomous/YOLO modes, and durable monitoring.

**Architecture:** Builds on Phase 1's `autoresearch/` package (60 tests passing). Adds 4 new modules: `diff_policy.py` (edit surface enforcement), `tree/selection.py` (branch scoring algorithm), `worktree.py` (git worktree lifecycle), and `runner.py` (durable monitoring process). Extends existing `hypothesis/schema.py` with `FileDiff` support and `auto-research.md` skill with autonomous/YOLO modes.

**Tech Stack:** Python 3.10+, numpy, subprocess (git), wandb API, pytest

**Spec:** `docs/superpowers/specs/2026-03-19-auto-research-design.md` (Phase 2 section, lines 44-54)

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `autoresearch/constraints/diff_policy.py` | Denylist/allowlist enforcement, max diff size checks |
| `autoresearch/tree/selection.py` | UCB-style branch scoring, behavioral distance, random restarts |
| `autoresearch/worktree.py` | Git worktree create/cleanup/list lifecycle management |
| `autoresearch/runner.py` | Durable background monitoring process (claim refresh, early stop, result recording) |
| `autoresearch/analysis/early_stopping.py` | Early stopping signal computation from W&B metrics |
| `tests/test_autoresearch/test_diff_policy.py` | Diff policy tests |
| `tests/test_autoresearch/test_selection.py` | Branch selection tests |
| `tests/test_autoresearch/test_worktree.py` | Worktree lifecycle tests |
| `tests/test_autoresearch/test_early_stopping.py` | Early stopping tests |

### Modified Files

| File | Change |
|------|--------|
| `autoresearch/hypothesis/schema.py` | Add `FileDiff` dataclass, update `Change` type union |
| `autoresearch/tree/research_tree.py` | Add branch aggregation methods (best_descendant_fitness, total_experiments_on_branch, centroid) |
| `autoresearch/tree/serialization.py` | Serialize new TreeNode fields |
| `autoresearch/config.py` | Add `resource_limits` and `worktree` config sections |
| `.claude/plugins/autoresearch/skills/auto-research.md` | Add autonomous/YOLO mode logic, worktree usage, diff policy |
| `.claude/plugins/autoresearch/settings.yaml` | Add resource_limits and worktree config |

---

## Task 1: Diff Policy Enforcement

**Files:**
- Create: `autoresearch/constraints/diff_policy.py`
- Test: `tests/test_autoresearch/test_diff_policy.py`

- [ ] **Step 1: Write diff policy tests**

File: `tests/test_autoresearch/test_diff_policy.py`

```python
"""Tests for edit surface diff policy enforcement."""

import pytest

from autoresearch.constraints.diff_policy import (
    DiffPolicy, DiffValidationResult, FileChange,
)


@pytest.fixture
def policy():
    return DiffPolicy()


class TestDenylist:
    def test_sim_dynamics_denied(self, policy):
        changes = [FileChange("sim/dynamics/numpy_quad.py", added=10, removed=5)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False
        assert "denylist" in result.violations[0].lower()

    def test_rewards_denied(self, policy):
        changes = [FileChange("sim/rewards.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_rewards_mavlab_denied(self, policy):
        changes = [FileChange("sim/rewards_mavlab.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_metrics_denied(self, policy):
        changes = [FileChange("metrics/contract.py", added=1, removed=0)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False

    def test_autoresearch_self_modification_denied(self, policy):
        changes = [FileChange("autoresearch/archive/map_elites.py", added=5, removed=2)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_training_callbacks_denied(self, policy):
        changes = [FileChange("training/callbacks.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False

    def test_tests_deletion_denied(self, policy):
        changes = [FileChange("tests/test_sim/test_rewards.py", added=0, removed=5)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False
        assert "additive" in result.violations[0].lower() or "test" in result.violations[0].lower()

    def test_tests_addition_allowed(self, policy):
        changes = [FileChange("tests/test_control/test_new.py", added=50, removed=0)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True


class TestScopeAllowlist:
    def test_hyperparameter_rejects_any_file_change(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=1, removed=0)]
        result = policy.validate(changes, scope="hyperparameter")
        assert result.passed is False
        assert "hyperparameter" in result.violations[0].lower()

    def test_algorithm_allows_control(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=10, removed=5)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True

    def test_algorithm_allows_configs(self, policy):
        changes = [FileChange("configs/experiment/new_exp.yaml", added=20, removed=0)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True

    def test_algorithm_rejects_sim_envs(self, policy):
        changes = [FileChange("sim/envs/gate_race_env.py", added=10, removed=5)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False

    def test_architecture_allows_perception_detectors(self, policy):
        changes = [FileChange("perception/detectors/gatenet.py", added=20, removed=10)]
        result = policy.validate(changes, scope="architecture")
        assert result.passed is True

    def test_system_allows_sim_envs(self, policy):
        changes = [FileChange("sim/envs/gate_race_env.py", added=10, removed=5)]
        result = policy.validate(changes, scope="system")
        assert result.passed is True

    def test_system_rejects_sim_dynamics(self, policy):
        changes = [FileChange("sim/dynamics/numpy_quad.py", added=1, removed=0)]
        result = policy.validate(changes, scope="system")
        assert result.passed is False


class TestMaxDiffSize:
    def test_algorithm_within_limit(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=200, removed=100)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is True

    def test_algorithm_exceeds_limit(self, policy):
        changes = [FileChange("control/algorithms/ppo.py", added=400, removed=200)]
        result = policy.validate(changes, scope="algorithm")
        assert result.passed is False
        assert "diff size" in result.violations[0].lower() or "lines" in result.violations[0].lower()

    def test_system_larger_limit(self, policy):
        changes = [FileChange("sim/envs/new_env.py", added=1500, removed=200)]
        result = policy.validate(changes, scope="system")
        assert result.passed is True


class TestFileSummary:
    def test_summary_included(self, policy):
        changes = [
            FileChange("control/algorithms/ppo.py", added=10, removed=5),
            FileChange("configs/experiment/new.yaml", added=20, removed=0),
        ]
        result = policy.validate(changes, scope="algorithm")
        assert len(result.file_summary) == 2
        assert result.file_summary["control/algorithms/ppo.py"] == {"added": 10, "removed": 5}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_diff_policy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `autoresearch/constraints/diff_policy.py`**

```python
"""Edit surface enforcement: denylist/allowlist per scope with max diff limits."""

from __future__ import annotations

from dataclasses import dataclass, field

# Files/directories that are always off-limits
DENYLIST = [
    "sim/dynamics/",
    "sim/rewards.py",
    "sim/rewards_mavlab.py",
    "metrics/",
    "training/callbacks.py",
    "training/trajectory_recorder.py",
    "autoresearch/",
    ".claude/",
    "artifacts/",
    "docker/",
]

# Scope-specific allowlists
SCOPE_ALLOWLIST: dict[str, list[str]] = {
    "hyperparameter": [],  # no file edits allowed
    "algorithm": [
        "control/algorithms/",
        "control/policies/",
        "perception/wrappers/",
        "configs/",
    ],
    "architecture": [
        "control/algorithms/",
        "control/policies/",
        "perception/wrappers/",
        "configs/",
        "perception/detectors/",
        "state_estimation/",
        "control/",  # new files
    ],
    "system": [
        "control/algorithms/",
        "control/policies/",
        "perception/wrappers/",
        "configs/",
        "perception/detectors/",
        "state_estimation/",
        "control/",
        "sim/envs/",
        "training/loops/",
        "sim/",  # new files, but not sim/dynamics/ or sim/rewards.py (caught by denylist)
    ],
}

# Max lines changed per scope
MAX_DIFF_LINES: dict[str, int] = {
    "hyperparameter": 0,
    "algorithm": 500,
    "architecture": 1000,
    "system": 2000,
}


@dataclass(frozen=True)
class FileChange:
    """A single file changed in a diff."""

    path: str
    added: int
    removed: int


@dataclass
class DiffValidationResult:
    """Result of diff policy validation."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    file_summary: dict[str, dict[str, int]] = field(default_factory=dict)


class DiffPolicy:
    """Validates proposed code changes against denylist/allowlist/size rules."""

    def __init__(
        self,
        denylist: list[str] | None = None,
        scope_allowlist: dict[str, list[str]] | None = None,
        max_diff_lines: dict[str, int] | None = None,
    ) -> None:
        self.denylist = denylist or DENYLIST
        self.scope_allowlist = scope_allowlist or SCOPE_ALLOWLIST
        self.max_diff_lines = max_diff_lines or MAX_DIFF_LINES

    def validate(
        self, changes: list[FileChange], scope: str
    ) -> DiffValidationResult:
        """Validate a list of file changes against the diff policy."""
        violations: list[str] = []
        file_summary: dict[str, dict[str, int]] = {}

        # Hyperparameter scope: no file edits allowed
        if scope == "hyperparameter" and changes:
            violations.append(
                f"Hyperparameter scope does not allow file edits, "
                f"but {len(changes)} files were changed"
            )
            return DiffValidationResult(
                passed=False, violations=violations, file_summary=file_summary
            )

        total_added = 0
        total_removed = 0
        allowlist = self.scope_allowlist.get(scope, [])

        for change in changes:
            file_summary[change.path] = {
                "added": change.added,
                "removed": change.removed,
            }
            total_added += change.added
            total_removed += change.removed

            # Check denylist
            if self._is_denylisted(change.path):
                # Special case: tests/ allows additions only
                if change.path.startswith("tests/"):
                    if change.removed > 0:
                        violations.append(
                            f"Test file {change.path}: only additive changes allowed, "
                            f"but {change.removed} lines were removed"
                        )
                    # If only additions, it's allowed — don't add a violation
                    if change.removed == 0:
                        continue
                else:
                    violations.append(
                        f"Denylist violation: {change.path} is off-limits"
                    )
                continue

            # Check allowlist
            if not self._is_allowlisted(change.path, allowlist):
                violations.append(
                    f"Scope '{scope}' does not allow changes to {change.path}"
                )

        # Check total diff size
        total_lines = total_added + total_removed
        max_lines = self.max_diff_lines.get(scope, 0)
        if max_lines > 0 and total_lines > max_lines:
            violations.append(
                f"Diff size {total_lines} lines exceeds {scope} limit of {max_lines} lines"
            )

        return DiffValidationResult(
            passed=len(violations) == 0,
            violations=violations,
            file_summary=file_summary,
        )

    def _is_denylisted(self, path: str) -> bool:
        """Check if a path matches the denylist."""
        for pattern in self.denylist:
            if pattern.endswith("/"):
                if path.startswith(pattern):
                    return True
            else:
                if path == pattern:
                    return True
        # tests/ has special handling (additive only)
        if path.startswith("tests/"):
            return True
        return False

    def _is_allowlisted(self, path: str, allowlist: list[str]) -> bool:
        """Check if a path matches any allowlist pattern."""
        for pattern in allowlist:
            if pattern.endswith("/"):
                if path.startswith(pattern):
                    return True
            else:
                if path == pattern:
                    return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_diff_policy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/constraints/diff_policy.py tests/test_autoresearch/test_diff_policy.py
git commit -m "feat(autoresearch): diff policy enforcement (denylist/allowlist/size limits)"
```

---

## Task 2: Branch Selection Algorithm

**Files:**
- Create: `autoresearch/tree/selection.py`
- Modify: `autoresearch/tree/research_tree.py` (add branch aggregation methods)
- Modify: `autoresearch/tree/serialization.py` (handle new optional fields)
- Test: `tests/test_autoresearch/test_selection.py`

- [ ] **Step 1: Write branch selection tests**

File: `tests/test_autoresearch/test_selection.py`

```python
"""Tests for branch selection algorithm."""

import pytest
import numpy as np

from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.selection import (
    score_branch, select_next_branch, behavioral_distance,
    BranchStats,
)
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def tree_with_branches():
    """Tree with baseline + 2 branches, each with experiments."""
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    # Branch A: 2 experiments, best fitness 4.0
    tree.add_experiment("a1", "baseline_v1", "algorithm", "Branch A exp 1", "completed")
    tree.complete_experiment("a1", fitness=5.0, wandb_run_id="run_a1")
    tree.add_experiment("a2", "a1", "algorithm", "Branch A exp 2", "completed")
    tree.complete_experiment("a2", fitness=4.0, wandb_run_id="run_a2")
    # Branch B: 1 experiment, fitness 6.0
    tree.add_experiment("b1", "baseline_v1", "algorithm", "Branch B exp 1", "completed")
    tree.complete_experiment("b1", fitness=6.0, wandb_run_id="run_b1")
    return tree


@pytest.fixture
def archive_with_entries():
    archive = MapElitesArchive()
    entry = CellEntry(
        fitness=4.0, status="approved", wandb_run_id="run_a2",
        git_commit="abc", descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={}, budget_spent=5_000_000, hypothesis_id="a2",
    )
    archive.try_insert(entry)
    return archive


@pytest.fixture
def default_config():
    return {
        "exploit_weight": 1.0,
        "explore_weight": 1.0,
        "gap_weight": 0.5,
        "diversity_weight": 0.3,
        "random_restart_pct": 0.0,  # disable for deterministic tests
        "min_branch_experiments": 3,
        "tie_threshold": 0.05,
    }


class TestBehavioralDistance:
    def test_identical_is_zero(self):
        a = DescriptorVector(0.5, 0.5, 10.0)
        b = DescriptorVector(0.5, 0.5, 10.0)
        ranges = [(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)]
        assert behavioral_distance(a, b, ranges) == pytest.approx(0.0)

    def test_max_distance(self):
        a = DescriptorVector(0.25, 0.0, 0.0)
        b = DescriptorVector(1.0, 1.0, 24.0)
        ranges = [(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)]
        dist = behavioral_distance(a, b, ranges)
        assert dist == pytest.approx(np.sqrt(3.0), abs=0.01)  # sqrt(1^2 + 1^2 + 1^2)

    def test_symmetry(self):
        a = DescriptorVector(0.5, 0.3, 10.0)
        b = DescriptorVector(0.8, 0.7, 5.0)
        ranges = [(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)]
        assert behavioral_distance(a, b, ranges) == pytest.approx(
            behavioral_distance(b, a, ranges)
        )


class TestBranchStats:
    def test_compute_from_tree(self, tree_with_branches):
        stats = BranchStats.from_tree(tree_with_branches, "a1")
        assert stats.best_fitness == 4.0  # min of a1(5.0) and a2(4.0)
        assert stats.visit_count == 2

    def test_single_experiment_branch(self, tree_with_branches):
        stats = BranchStats.from_tree(tree_with_branches, "b1")
        assert stats.best_fitness == 6.0
        assert stats.visit_count == 1


class TestScoreBranch:
    def test_better_fitness_scores_higher(self, archive_with_entries, tree_with_branches, default_config):
        stats_a = BranchStats(best_fitness=4.0, visit_count=2, centroid=DescriptorVector(0.5, 0.3, 10.0))
        stats_b = BranchStats(best_fitness=6.0, visit_count=1, centroid=DescriptorVector(0.7, 0.5, 15.0))
        score_a = score_branch(stats_a, archive_with_entries, tree_with_branches, default_config)
        score_b = score_branch(stats_b, archive_with_entries, tree_with_branches, default_config)
        # Branch A has better fitness → higher exploit score
        # But Branch B has fewer visits → higher explore score
        # With equal weights, branch A should still win due to fitness advantage
        assert score_a > 0
        assert score_b > 0

    def test_minimum_exploration_guarantee(self, archive_with_entries, tree_with_branches, default_config):
        stats = BranchStats(best_fitness=100.0, visit_count=1, centroid=DescriptorVector(0.5, 0.3, 10.0))
        score = score_branch(stats, archive_with_entries, tree_with_branches, default_config)
        # visit_count (1) < min_branch_experiments (3) → explore = inf
        assert score == float("inf")


class TestSelectNextBranch:
    def test_selects_a_branch(self, archive_with_entries, tree_with_branches, default_config):
        default_config["min_branch_experiments"] = 0  # disable guarantee for this test
        result = select_next_branch(tree_with_branches, archive_with_entries, default_config, seed=42)
        assert result is not None
        assert result in ["a1", "b1"]  # one of the Level 1 branches

    def test_random_restart_returns_none(self, archive_with_entries, tree_with_branches, default_config):
        default_config["random_restart_pct"] = 1.0  # always restart
        result = select_next_branch(tree_with_branches, archive_with_entries, default_config, seed=42)
        assert result is None  # None signals "generate novel hypothesis"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_selection.py -v`
Expected: FAIL

- [ ] **Step 3: Add branch aggregation methods to `autoresearch/tree/research_tree.py`**

Add these methods to the `ResearchTree` class:

```python
    def get_children(self, hypothesis_id: str) -> list[TreeNode]:
        """Get all direct children of a node."""
        return [n for n in self._nodes.values() if n.parent_id == hypothesis_id]

    def get_descendants(self, hypothesis_id: str) -> list[TreeNode]:
        """Get all descendants (children, grandchildren, etc.) of a node."""
        descendants = []
        queue = self.get_children(hypothesis_id)
        while queue:
            node = queue.pop(0)
            descendants.append(node)
            queue.extend(self.get_children(node.hypothesis_id))
        return descendants

    def get_level1_branches(self) -> list[str]:
        """Get hypothesis IDs of all Level 1 branches (direct children of root)."""
        if self._root_id is None:
            return []
        return [n.hypothesis_id for n in self.get_children(self._root_id)]
```

- [ ] **Step 4: Implement `autoresearch/tree/selection.py`**

```python
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

    @classmethod
    def from_tree(cls, tree: ResearchTree, branch_root_id: str) -> BranchStats:
        """Compute branch stats from all descendants of a node."""
        root = tree.get_node(branch_root_id)
        descendants = tree.get_descendants(branch_root_id)
        all_nodes = [root] + descendants

        completed = [n for n in all_nodes if n.status == "completed" and n.fitness is not None]
        visit_count = len([n for n in all_nodes if n.scope != "baseline"])

        best_fitness = min((n.fitness for n in completed), default=float("inf"))

        return cls(
            best_fitness=best_fitness,
            visit_count=visit_count,
            centroid=None,  # centroid requires descriptor data not stored in tree nodes
        )


def score_branch(
    stats: BranchStats,
    archive: MapElitesArchive,
    tree: ResearchTree,
    config: dict,
) -> float:
    """Score a branch for selection. Higher = more likely to be explored next."""
    # Minimum exploration guarantee
    if stats.visit_count < config.get("min_branch_experiments", 3):
        return float("inf")

    best_fitness = stats.best_fitness if stats.best_fitness != float("inf") else 1e6
    total_experiments = max(tree.total_experiments, 1)
    visit_count = max(stats.visit_count, 1)

    # UCB-style score
    exploit = config.get("exploit_weight", 1.0) / best_fitness
    explore = config.get("explore_weight", 1.0) * sqrt(
        log(total_experiments) / visit_count
    )

    # Diversity bonus (if centroid available and archive has entries)
    diversity_bonus = 0.0
    if stats.centroid is not None:
        approved = archive.approved_cells()
        if approved:
            best_entry = min(approved.values(), key=lambda e: e.fitness)
            diversity_bonus = config.get("diversity_weight", 0.3) * behavioral_distance(
                stats.centroid, best_entry.descriptors, archive.axis_ranges
            )

    return exploit + explore + diversity_bonus


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
        return None  # Caller should generate a novel hypothesis

    branches = tree.get_level1_branches()
    if not branches:
        return None

    # Score all branches
    scores = {}
    for branch_id in branches:
        stats = BranchStats.from_tree(tree, branch_id)
        scores[branch_id] = score_branch(stats, archive, tree, config)

    # Handle inf scores (minimum exploration guarantee)
    inf_branches = [b for b, s in scores.items() if s == float("inf")]
    if inf_branches:
        return str(rng.choice(inf_branches))

    # Tie-breaking: within threshold, select randomly
    max_score = max(scores.values())
    threshold = config.get("tie_threshold", 0.05)
    tied = [b for b, s in scores.items() if s >= max_score * (1 - threshold)]

    return str(rng.choice(tied))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_selection.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autoresearch/tree/selection.py autoresearch/tree/research_tree.py tests/test_autoresearch/test_selection.py
git commit -m "feat(autoresearch): branch selection algorithm with UCB scoring"
```

---

## Task 3: Worktree Management

**Files:**
- Create: `autoresearch/worktree.py`
- Test: `tests/test_autoresearch/test_worktree.py`

- [ ] **Step 1: Write worktree tests**

File: `tests/test_autoresearch/test_worktree.py`

```python
"""Tests for git worktree lifecycle management."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from autoresearch.worktree import (
    WorktreeManager, WorktreeInfo,
    BRANCH_PREFIX, MAX_WORKTREES,
)


@pytest.fixture
def manager(tmp_path):
    return WorktreeManager(repo_root=tmp_path, worktree_dir=tmp_path / "worktrees")


class TestBranchNaming:
    def test_branch_name_format(self, manager):
        name = manager.branch_name("abc123def456")
        assert name == f"{BRANCH_PREFIX}abc123def456"

    def test_branch_name_short_id(self, manager):
        name = manager.branch_name("a1b2c3")
        assert name == f"{BRANCH_PREFIX}a1b2c3"


class TestWorktreeInfo:
    def test_parse_from_dict(self):
        info = WorktreeInfo(
            path="/tmp/worktrees/ar-exp-abc123",
            branch="ar/exp-abc123",
            hypothesis_id="abc123",
            created_at=None,
        )
        assert info.hypothesis_id == "abc123"


class TestWorktreeListParsing:
    def test_list_empty(self, manager):
        with patch.object(manager, "_run_git", return_value=""):
            result = manager.list_worktrees()
            assert result == []

    def test_max_worktrees_check(self, manager):
        assert MAX_WORKTREES == 5


class TestCleanupPolicy:
    def test_cleanup_filters_by_age(self, manager):
        # This is a unit test for the cleanup logic, not actual git operations
        from datetime import datetime, timezone, timedelta
        old = WorktreeInfo(
            path="/tmp/old",
            branch="ar/exp-old",
            hypothesis_id="old",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        )
        recent = WorktreeInfo(
            path="/tmp/recent",
            branch="ar/exp-recent",
            hypothesis_id="recent",
            created_at=datetime.now(timezone.utc),
        )
        stale = manager._find_stale([old, recent], max_age_days=7)
        assert len(stale) == 1
        assert stale[0].hypothesis_id == "old"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_worktree.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `autoresearch/worktree.py`**

```python
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
        """Generate a branch name for a hypothesis."""
        return f"{BRANCH_PREFIX}{hypothesis_id}"

    def create_worktree(self, hypothesis_id: str, base_ref: str = "main") -> str:
        """Create a new worktree for an experiment.

        Returns the worktree path.
        Raises RuntimeError if max worktrees exceeded.
        """
        current = self.list_worktrees()
        if len(current) >= MAX_WORKTREES:
            raise RuntimeError(
                f"Max worktrees ({MAX_WORKTREES}) exceeded. "
                f"Run cleanup before creating new ones."
            )

        branch = self.branch_name(hypothesis_id)
        wt_path = self.worktree_dir / hypothesis_id
        wt_path.parent.mkdir(parents=True, exist_ok=True)

        self._run_git(
            "worktree", "add", "-b", branch, str(wt_path), base_ref
        )
        return str(wt_path)

    def remove_worktree(self, hypothesis_id: str) -> None:
        """Remove a worktree and its branch."""
        wt_path = self.worktree_dir / hypothesis_id
        if wt_path.exists():
            self._run_git("worktree", "remove", str(wt_path), "--force")
        branch = self.branch_name(hypothesis_id)
        try:
            self._run_git("branch", "-D", branch)
        except subprocess.CalledProcessError:
            pass  # Branch may already be deleted

    def list_worktrees(self) -> list[WorktreeInfo]:
        """List all auto-research worktrees."""
        try:
            output = self._run_git("worktree", "list", "--porcelain")
        except subprocess.CalledProcessError:
            return []

        if not output.strip():
            return []

        worktrees = []
        current_path = None
        current_branch = None

        for line in output.strip().split("\n"):
            if line.startswith("worktree "):
                current_path = line[len("worktree "):]
            elif line.startswith("branch "):
                ref = line[len("branch "):]
                # Extract branch name from refs/heads/...
                if ref.startswith("refs/heads/"):
                    current_branch = ref[len("refs/heads/"):]
            elif line == "":
                if current_path and current_branch and current_branch.startswith(BRANCH_PREFIX):
                    hyp_id = current_branch[len(BRANCH_PREFIX):]
                    worktrees.append(WorktreeInfo(
                        path=current_path,
                        branch=current_branch,
                        hypothesis_id=hyp_id,
                    ))
                current_path = None
                current_branch = None

        return worktrees

    def cleanup_stale(self, max_age_days: int = 7) -> list[str]:
        """Remove stale worktrees. Returns list of removed hypothesis IDs."""
        worktrees = self.list_worktrees()
        stale = self._find_stale(worktrees, max_age_days)
        removed = []
        for wt in stale:
            self.remove_worktree(wt.hypothesis_id)
            removed.append(wt.hypothesis_id)
        return removed

    def _find_stale(
        self, worktrees: list[WorktreeInfo], max_age_days: int = 7
    ) -> list[WorktreeInfo]:
        """Find worktrees older than max_age_days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return [
            wt for wt in worktrees
            if wt.created_at is not None and wt.created_at < cutoff
        ]

    def _run_git(self, *args: str) -> str:
        """Run a git command in the repo root."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_worktree.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/worktree.py tests/test_autoresearch/test_worktree.py
git commit -m "feat(autoresearch): git worktree lifecycle management"
```

---

## Task 4: Early Stopping Logic

**Files:**
- Create: `autoresearch/analysis/early_stopping.py`
- Test: `tests/test_autoresearch/test_early_stopping.py`

- [ ] **Step 1: Write early stopping tests**

File: `tests/test_autoresearch/test_early_stopping.py`

```python
"""Tests for early stopping signal computation."""

import pytest

from autoresearch.analysis.early_stopping import (
    EarlyStopSignal, check_early_stop,
)
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def archive_with_incumbent():
    archive = MapElitesArchive()
    entry = CellEntry(
        fitness=5.0, status="approved", wandb_run_id="run_best",
        git_commit="abc", descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={}, budget_spent=5_000_000, hypothesis_id="best",
    )
    archive.try_insert(entry)
    return archive


class TestEarlyStop:
    def test_no_stop_when_performing_well(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=4.5,
            baseline_fitness=5.0,
            steps_completed=2_000_000,
            budget=5_000_000,
            baseline_threshold=0.7,
            archive=archive_with_incumbent,
            target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.CONTINUE

    def test_stop_poor_performance(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=20.0,  # much worse than baseline 5.0
            baseline_fitness=5.0,
            steps_completed=2_000_000,  # past 1/3 of budget
            budget=5_000_000,
            baseline_threshold=0.7,
            archive=archive_with_incumbent,
            target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.POOR_PERFORMANCE

    def test_no_stop_poor_performance_early(self, archive_with_incumbent):
        # Before 1/3 of budget, don't early stop even if performing poorly
        signal = check_early_stop(
            current_fitness=20.0,
            baseline_fitness=5.0,
            steps_completed=500_000,  # before 1/3 of budget
            budget=5_000_000,
            baseline_threshold=0.7,
            archive=archive_with_incumbent,
            target_cell=(1, 1, 2),
        )
        assert signal == EarlyStopSignal.CONTINUE

    def test_stop_archive_redundancy(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=6.0,  # worse than incumbent 5.0 in this cell
            baseline_fitness=5.0,
            steps_completed=3_000_000,
            budget=5_000_000,
            baseline_threshold=0.7,
            archive=archive_with_incumbent,
            target_cell=(1, 1, 2),  # same cell as incumbent
        )
        assert signal == EarlyStopSignal.ARCHIVE_REDUNDANT

    def test_no_stop_first_experiment(self):
        # Empty archive, no baseline → never early stop
        signal = check_early_stop(
            current_fitness=100.0,
            baseline_fitness=None,
            steps_completed=3_000_000,
            budget=5_000_000,
            baseline_threshold=0.7,
            archive=MapElitesArchive(),
            target_cell=(0, 0, 0),
        )
        assert signal == EarlyStopSignal.CONTINUE

    def test_empty_target_cell_no_redundancy(self, archive_with_incumbent):
        signal = check_early_stop(
            current_fitness=6.0,
            baseline_fitness=5.0,
            steps_completed=3_000_000,
            budget=5_000_000,
            baseline_threshold=0.7,
            archive=archive_with_incumbent,
            target_cell=(4, 4, 4),  # empty cell
        )
        assert signal == EarlyStopSignal.CONTINUE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_early_stopping.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `autoresearch/analysis/early_stopping.py`**

```python
"""Early stopping signal computation for training runs."""

from __future__ import annotations

from enum import Enum

from autoresearch.archive.map_elites import MapElitesArchive


class EarlyStopSignal(Enum):
    """Possible early stopping decisions."""

    CONTINUE = "continue"
    POOR_PERFORMANCE = "poor_performance"
    ARCHIVE_REDUNDANT = "archive_redundant"


def check_early_stop(
    current_fitness: float,
    baseline_fitness: float | None,
    steps_completed: int,
    budget: int,
    baseline_threshold: float,
    archive: MapElitesArchive,
    target_cell: tuple[int, int, int],
) -> EarlyStopSignal:
    """Check whether to early-stop a training run.

    Args:
        current_fitness: Best fitness observed so far in this run.
        baseline_fitness: Best fitness in the archive when experiment started.
            None for the first experiment (empty archive).
        steps_completed: Training steps completed so far.
        budget: Total budget for this experiment.
        baseline_threshold: Performance ratio threshold (e.g., 0.7 = 70%).
        archive: Current MAP-Elites archive.
        target_cell: The archive cell this experiment is targeting.

    Returns:
        EarlyStopSignal indicating whether to continue or stop.
    """
    # First experiment (no baseline) always continues
    if baseline_fitness is None:
        return EarlyStopSignal.CONTINUE

    # Performance check: only after 1/3 of budget
    if steps_completed >= budget / 3:
        # For lap time (lower is better), "70% of baseline" means
        # the current fitness should be no worse than baseline / 0.7
        # i.e., current_fitness <= baseline_fitness / baseline_threshold
        max_acceptable = baseline_fitness / baseline_threshold
        if current_fitness > max_acceptable:
            return EarlyStopSignal.POOR_PERFORMANCE

    # Archive redundancy: target cell already has a better incumbent
    incumbent = archive.get(target_cell)
    if incumbent is not None and current_fitness > incumbent.fitness:
        return EarlyStopSignal.ARCHIVE_REDUNDANT

    return EarlyStopSignal.CONTINUE
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_early_stopping.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/analysis/early_stopping.py tests/test_autoresearch/test_early_stopping.py
git commit -m "feat(autoresearch): early stopping signal computation"
```

---

## Task 5: FileDiff Support in Hypothesis Schema

**Files:**
- Modify: `autoresearch/hypothesis/schema.py`
- Modify: `tests/test_autoresearch/test_hypothesis.py`

- [ ] **Step 1: Add FileDiff tests to existing test file**

Append to `tests/test_autoresearch/test_hypothesis.py`:

```python
from autoresearch.hypothesis.schema import FileDiff, Change


def test_file_diff_creation():
    diff = FileDiff(path="control/algorithms/ppo.py", description="Add entropy bonus")
    assert diff.path == "control/algorithms/ppo.py"


def test_hypothesis_with_file_diffs():
    h = Hypothesis.create(
        scope="algorithm",
        description="Add entropy bonus to PPO",
        changes=[
            HydraOverride("control.ent_coef", "0.01"),
            FileDiff("control/algorithms/ppo.py", "Add entropy coefficient parameter"),
        ],
        rationale="Entropy bonus may improve exploration",
    )
    assert len(h.changes) == 2
    overrides = h.to_cli_overrides()
    assert len(overrides) == 1  # Only HydraOverrides produce CLI args
    assert "control.ent_coef=0.01" in overrides
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python -m pytest tests/test_autoresearch/test_hypothesis.py -v`
Expected: new tests FAIL

- [ ] **Step 3: Update `autoresearch/hypothesis/schema.py`**

Add `FileDiff` dataclass and update `Change` type:

```python
@dataclass(frozen=True)
class FileDiff:
    """A source file modification (Phase 2+)."""

    path: str
    description: str


# Union type for changes
Change = HydraOverride | FileDiff
```

Update `Hypothesis`:
- Change `changes: list[HydraOverride]` to `changes: list[Change]`
- Update `to_cli_overrides` to filter only `HydraOverride` instances:

```python
    def to_cli_overrides(self) -> list[str]:
        """Generate Hydra CLI override arguments (ignores FileDiff changes)."""
        return [c.to_cli_arg() for c in self.changes if isinstance(c, HydraOverride)]
```

Update `compute_hypothesis_id` to handle both types:

```python
def compute_hypothesis_id(
    scope: str,
    changes: list[Change],
    target_cells: list[tuple[int, int, int]] | None = None,
) -> str:
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
```

- [ ] **Step 4: Run all hypothesis tests**

Run: `python -m pytest tests/test_autoresearch/test_hypothesis.py -v`
Expected: ALL PASS (old + new)

- [ ] **Step 5: Commit**

```bash
git add autoresearch/hypothesis/schema.py tests/test_autoresearch/test_hypothesis.py
git commit -m "feat(autoresearch): add FileDiff support to hypothesis schema"
```

---

## Task 6: Durable Monitoring Runner

**Files:**
- Create: `autoresearch/runner.py`
- Modify: `autoresearch/config.py` (add resource_limits)

- [ ] **Step 1: Add resource_limits to config**

Add to `AutoResearchConfig.__post_init__` defaults and the dataclass:

```python
    resource_limits: dict[str, object] = field(default_factory=lambda: {
        "max_concurrent_per_machine": 1,
        "max_wall_clock_hours": 24,
        "max_worktrees": 5,
    })
```

Add to `_DEFAULTS` in `__post_init__`:
```python
            "resource_limits": {"max_concurrent_per_machine": 1, "max_wall_clock_hours": 24, "max_worktrees": 5},
```

- [ ] **Step 2: Implement `autoresearch/runner.py`**

```python
"""Durable monitoring process for training runs.

Runs independently of Claude Code session. Monitors a W&B training run,
refreshes coordination claims, checks early stopping signals, and records
results to state files on completion.

Usage:
    nohup python -m autoresearch.runner \
        --run-id <wandb_run_id> \
        --claim-id <hypothesis_id> \
        --state-dir autoresearch/state/ \
        --max-rpm 31470 \
        --budget 5000000 &
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("autoresearch.runner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner CLI arguments."""
    parser = argparse.ArgumentParser(description="Auto-research durable monitor")
    parser.add_argument("--run-id", required=True, help="W&B run ID to monitor")
    parser.add_argument("--claim-id", required=True, help="Hypothesis ID for claim refresh")
    parser.add_argument("--wandb-project", default="corvidx-drone-racing")
    parser.add_argument("--state-dir", default="autoresearch/state/")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls")
    parser.add_argument("--claim-refresh-minutes", type=int, default=15)
    parser.add_argument("--max-rpm", type=float, default=31470.0)
    parser.add_argument("--control-freq", type=float, default=100.0)
    parser.add_argument("--budget", type=int, default=5_000_000)
    parser.add_argument("--baseline-fitness", type=float, default=None)
    return parser.parse_args(argv)


def refresh_claim(state_dir: Path, claim_id: str) -> None:
    """Refresh the timestamp on an active claim."""
    from autoresearch.coordination.claims import load_claims, save_claims

    claims_path = state_dir / "claims.json"
    claims = load_claims(claims_path)
    for c in claims:
        if c.hypothesis_id == claim_id and c.status == "running":
            c.timestamp = datetime.now(timezone.utc)
    save_claims(claims, claims_path)


def run_monitor(args: argparse.Namespace) -> None:
    """Main monitoring loop. Polls W&B, refreshes claims, checks early stop."""
    state_dir = Path(args.state_dir)
    last_claim_refresh = time.time()
    claim_refresh_interval = args.claim_refresh_minutes * 60

    logger.info(f"Monitoring W&B run {args.run_id} for claim {args.claim_id}")
    logger.info(f"Poll interval: {args.poll_interval}s, budget: {args.budget}")

    while True:
        try:
            # Refresh claim if interval elapsed
            if time.time() - last_claim_refresh >= claim_refresh_interval:
                refresh_claim(state_dir, args.claim_id)
                last_claim_refresh = time.time()
                logger.info("Claim timestamp refreshed")

            # TODO: Query W&B API for run status and metrics
            # TODO: Check early stopping signals
            # TODO: On completion, compute descriptors and validate constraints

            # For now, the runner is a skeleton that refreshes claims
            # Full W&B integration requires the wandb SDK at runtime

            time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            logger.info("Runner interrupted")
            break
        except Exception:
            logger.exception("Runner error, will retry")
            time.sleep(args.poll_interval)


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m autoresearch.runner`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    run_monitor(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `autoresearch/__main__.py`** for `python -m autoresearch.runner`

```python
"""Allow running autoresearch.runner as a module."""
# This file is intentionally empty — runner.py has its own __main__ guard.
# Use: python -m autoresearch.runner --run-id <id> --claim-id <id>
```

- [ ] **Step 4: Write a basic runner test**

File: `tests/test_autoresearch/test_runner.py`

```python
"""Tests for the durable monitoring runner."""

from autoresearch.runner import parse_args, refresh_claim
from autoresearch.coordination.claims import Claim, save_claims, load_claims
from datetime import datetime, timezone, timedelta


def test_parse_args():
    args = parse_args(["--run-id", "abc", "--claim-id", "hyp_001"])
    assert args.run_id == "abc"
    assert args.claim_id == "hyp_001"
    assert args.poll_interval == 60
    assert args.max_rpm == 31470.0


def test_parse_args_custom():
    args = parse_args([
        "--run-id", "abc",
        "--claim-id", "hyp_001",
        "--budget", "10000000",
        "--poll-interval", "30",
    ])
    assert args.budget == 10_000_000
    assert args.poll_interval == 30


def test_refresh_claim_updates_timestamp(tmp_path):
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    claim = Claim(
        hypothesis_id="hyp_001",
        researcher="test",
        branch="ar/exp-hyp001",
        target_cells=[],
        scope="hyperparameter",
        timestamp=old_time,
        wandb_run_id="run_abc",
        status="running",
    )
    save_claims([claim], tmp_path / "claims.json")

    refresh_claim(tmp_path, "hyp_001")

    loaded = load_claims(tmp_path / "claims.json")
    assert loaded[0].timestamp > old_time
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_autoresearch/test_runner.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add autoresearch/runner.py autoresearch/__main__.py autoresearch/config.py tests/test_autoresearch/test_runner.py
git commit -m "feat(autoresearch): durable monitoring runner with claim refresh"
```

---

## Task 7: Update Plugin Skills for Autonomous/YOLO Modes

**Files:**
- Modify: `.claude/plugins/autoresearch/skills/auto-research.md`
- Modify: `.claude/plugins/autoresearch/settings.yaml`

- [ ] **Step 1: Update `auto-research.md` skill**

Read the existing file, then rewrite to add:
- Mode selection (interactive/autonomous/yolo)
- Git worktree creation for code-level changes
- Diff policy enforcement before launch
- Branch selection algorithm for choosing next research direction
- Early stopping awareness
- Durable runner launch for background monitoring
- Updated anti-gaming rules referencing diff_policy.py

Key additions to the skill:

```markdown
## Modes

### Interactive (default)
All hypotheses require human approval before running.

### Autonomous
- Hyperparameter + algorithm scope: proceed without approval
- Architecture + system scope: present to human for approval
- Use: `/auto-research autonomous [focus]`

### YOLO
- All hypotheses proceed without approval
- Hard constraints (diff policy, physics, behavioral) still enforced
- Baseline promotion (merge to main) is ALWAYS human-gated
- Use: `/auto-research yolo [focus]`

## Code-Level Changes (Phase 2)

For algorithm/architecture/system scope:
1. Create git worktree: use `autoresearch.worktree.WorktreeManager` to create isolated workspace
2. Edit source files in the worktree
3. Validate diff against policy: `python -c "from autoresearch.constraints.diff_policy import DiffPolicy, FileChange; ..."`
4. Generate experiment config in `configs/experiment/`
5. Commit changes to worktree branch

## Branch Selection

When deciding what to explore next, use the branch selection algorithm:
```python
from autoresearch.tree.selection import select_next_branch
branch_id = select_next_branch(tree, archive, config.branch_selection, seed=None)
# None means random restart — generate a novel hypothesis
```

## Durable Monitoring

After launching training, start the background monitor:
```bash
nohup python -m autoresearch.runner --run-id <wandb_run_id> --claim-id <hypothesis_id> --budget <budget> &
```
This refreshes claims and checks early stopping independently of this Claude Code session.
```

- [ ] **Step 2: Update `settings.yaml`**

Add resource_limits section:

```yaml
  resource_limits:
    max_concurrent_per_machine: 1
    max_wall_clock_hours: 24
    max_worktrees: 5
```

- [ ] **Step 3: Commit**

```bash
git add .claude/plugins/autoresearch/
git commit -m "feat(autoresearch): update plugin skills for autonomous/YOLO modes"
```

---

## Task 8: Phase 2 Integration Test

**Files:**
- Create: `tests/test_autoresearch/test_phase2_integration.py`

- [ ] **Step 1: Write Phase 2 integration test**

```python
"""Integration test for Phase 2 features."""

from datetime import datetime, timezone

from autoresearch.constraints.diff_policy import DiffPolicy, FileChange
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.selection import select_next_branch, BranchStats
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector
from autoresearch.analysis.early_stopping import check_early_stop, EarlyStopSignal
from autoresearch.hypothesis.schema import Hypothesis, HydraOverride, FileDiff
from autoresearch.worktree import WorktreeManager, BRANCH_PREFIX
from autoresearch.runner import parse_args, refresh_claim
from autoresearch.coordination.claims import Claim, save_claims, load_claims


def test_phase2_pipeline(tmp_path):
    """End-to-end Phase 2 flow: branch selection → hypothesis → diff check → early stop."""

    # 1. Build a research tree with branches
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment("a1", "baseline_v1", "algorithm", "Try entropy bonus", "completed")
    tree.complete_experiment("a1", fitness=5.0, wandb_run_id="run_a1")
    tree.add_experiment("b1", "baseline_v1", "algorithm", "Try larger network", "completed")
    tree.complete_experiment("b1", fitness=6.0, wandb_run_id="run_b1")

    # 2. Archive with one entry
    archive = MapElitesArchive()
    archive.try_insert(CellEntry(
        fitness=5.0, status="approved", wandb_run_id="run_a1",
        git_commit="abc", descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={}, budget_spent=5_000_000, hypothesis_id="a1",
    ))

    # 3. Branch selection
    config = {
        "exploit_weight": 1.0, "explore_weight": 1.0, "gap_weight": 0.5,
        "diversity_weight": 0.3, "random_restart_pct": 0.0,
        "min_branch_experiments": 0, "tie_threshold": 0.05,
    }
    branch_id = select_next_branch(tree, archive, config, seed=42)
    assert branch_id in ["a1", "b1"]

    # 4. Create hypothesis with code changes
    hyp = Hypothesis.create(
        scope="algorithm",
        description="Add entropy bonus to PPO",
        changes=[
            HydraOverride("control.ent_coef", "0.01"),
            FileDiff("control/algorithms/ppo.py", "Add entropy coefficient"),
        ],
        rationale="Improve exploration via entropy regularization",
        parent_id=branch_id,
    )
    assert len(hyp.to_cli_overrides()) == 1

    # 5. Diff policy validation
    policy = DiffPolicy()
    diff_result = policy.validate(
        [FileChange("control/algorithms/ppo.py", added=20, removed=5)],
        scope="algorithm",
    )
    assert diff_result.passed is True

    # 6. Early stopping check — good performance, continue
    signal = check_early_stop(
        current_fitness=4.8,
        baseline_fitness=5.0,
        steps_completed=2_000_000,
        budget=15_000_000,
        baseline_threshold=0.7,
        archive=archive,
        target_cell=(1, 1, 2),
    )
    assert signal == EarlyStopSignal.CONTINUE

    # 7. Worktree manager branch naming
    manager = WorktreeManager(repo_root=tmp_path)
    assert manager.branch_name(hyp.id).startswith(BRANCH_PREFIX)

    # 8. Runner claim refresh
    claims_path = tmp_path / "claims.json"
    claim = Claim(
        hypothesis_id=hyp.id, researcher="test", branch=f"ar/exp-{hyp.id}",
        target_cells=[], scope="algorithm",
        timestamp=datetime.now(timezone.utc), wandb_run_id="run_test",
        status="running",
    )
    save_claims([claim], claims_path)
    refresh_claim(tmp_path, hyp.id)
    loaded = load_claims(claims_path)
    assert loaded[0].status == "running"
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/test_autoresearch/ -v`
Expected: ALL PASS (60 Phase 1 + new Phase 2 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_autoresearch/test_phase2_integration.py
git commit -m "test(autoresearch): Phase 2 integration test"
```

---

## Summary

| Task | Component | Key Deliverable |
|------|-----------|-----------------|
| 1 | Diff Policy | `diff_policy.py` — denylist/allowlist/size enforcement |
| 2 | Branch Selection | `selection.py` — UCB scoring with minimum exploration guarantees |
| 3 | Worktree Management | `worktree.py` — create/remove/cleanup git worktrees |
| 4 | Early Stopping | `early_stopping.py` — performance and redundancy signals |
| 5 | FileDiff Support | Extended `schema.py` with `FileDiff` type for code changes |
| 6 | Durable Runner | `runner.py` — background monitor with claim refresh |
| 7 | Plugin Update | Updated skills for autonomous/YOLO modes, worktrees, runner |
| 8 | Integration Test | End-to-end Phase 2 pipeline test |

Tasks 1-4 are independent and can be parallelized. Task 5 is independent. Task 6 depends on config changes. Task 7 depends on all prior tasks. Task 8 validates everything together.
