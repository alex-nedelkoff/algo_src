# Auto-Research Phase 3: Full Autonomy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the auto-research system with advanced trajectory analysis for smarter hypothesis generation, cross-pollination of successful ideas across branches, adaptive MAP-Elites grid refinement, and the baseline promotion workflow.

**Architecture:** Builds on Phase 1+2's `autoresearch/` package (116 tests passing). Adds 4 modules: `analysis/trajectory_insights.py` (failure pattern extraction), `tree/cross_pollination.py` (idea grafting), `archive/subdivision.py` (adaptive grid), and `promotion.py` (baseline promotion). Adds the `/ar-promote` plugin skill.

**Tech Stack:** Python 3.10+, numpy, subprocess (git/gh), pytest

**Spec:** `docs/superpowers/specs/2026-03-19-auto-research-design.md` (Phase 3 section + Baseline Promotion + Cross-pollination)

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `autoresearch/analysis/trajectory_insights.py` | Extract failure patterns, gate-specific issues, and behavioral signals from trajectory data |
| `autoresearch/tree/cross_pollination.py` | Identify successful patterns and propose cross-branch hypotheses with `inspired_by` |
| `autoresearch/archive/subdivision.py` | Adaptive cell subdivision — split promising cells into 2×2×2 sub-cells |
| `autoresearch/promotion.py` | Baseline promotion: verify, create PR body, update state after merge |
| `.claude/plugins/autoresearch/skills/ar-promote.md` | Promotion skill for Claude Code |
| `tests/test_autoresearch/test_trajectory_insights.py` | Tests for trajectory analysis |
| `tests/test_autoresearch/test_cross_pollination.py` | Tests for cross-pollination |
| `tests/test_autoresearch/test_subdivision.py` | Tests for adaptive grid subdivision |
| `tests/test_autoresearch/test_promotion.py` | Tests for baseline promotion logic |

### Modified Files

| File | Change |
|------|--------|
| `autoresearch/archive/map_elites.py` | Add subdivision support to `MapElitesArchive` |
| `autoresearch/archive/serialization.py` | Handle subdivided cells in JSON |
| `autoresearch/tree/research_tree.py` | Add `promote_baseline()` method for baseline chain |
| `autoresearch/tree/serialization.py` | Serialize new baseline promotion fields |

---

## Task 0: System Scope Test Gate

**Files:**
- Modify: `autoresearch/constraints/validator.py`
- Test: `tests/test_autoresearch/test_constraints.py` (append)

- [ ] **Step 1: Add test for system scope test gate**

Append to `tests/test_autoresearch/test_constraints.py`:

```python
def test_system_scope_requires_test_pass():
    """System scope experiments must pass the test suite."""
    from autoresearch.constraints.validator import check_test_suite
    # Mock a passing test run
    result = check_test_suite(test_dir="tests/test_autoresearch/", timeout=60)
    # This runs real tests — should pass since our suite is healthy
    assert result.passed is True
    assert result.reason != ""
```

- [ ] **Step 2: Implement `check_test_suite` in `autoresearch/constraints/validator.py`**

Read the existing file, then add this function:

```python
import subprocess

def check_test_suite(
    test_dir: str = "tests/",
    timeout: int = 300,
    python: str = "python",
) -> ConstraintCheck:
    """Run pytest on the test suite. Required for system-scope experiments."""
    try:
        result = subprocess.run(
            [python, "-m", "pytest", test_dir, "-x", "-q", "--tb=line"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return ConstraintCheck(True, f"Test suite passed: {result.stdout.strip().splitlines()[-1]}")
        else:
            return ConstraintCheck(False, f"Test suite failed: {result.stdout.strip().splitlines()[-1]}")
    except subprocess.TimeoutExpired:
        return ConstraintCheck(False, f"Test suite timed out after {timeout}s")
    except Exception as e:
        return ConstraintCheck(False, f"Test suite error: {e}")
```

- [ ] **Step 3: Run tests, verify pass**

Run: `python -m pytest tests/test_autoresearch/test_constraints.py -v`

- [ ] **Step 4: Commit**

```bash
git add autoresearch/constraints/validator.py tests/test_autoresearch/test_constraints.py
git commit -m "feat(autoresearch): system scope mandatory test pass gate"
```

---

## Task 1: Trajectory Insights

**Files:**
- Create: `autoresearch/analysis/trajectory_insights.py`
- Test: `tests/test_autoresearch/test_trajectory_insights.py`

- [ ] **Step 1: Write trajectory insights tests**

File: `tests/test_autoresearch/test_trajectory_insights.py`

```python
"""Tests for trajectory insight extraction."""

import numpy as np
import pytest

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.analysis.trajectory_insights import (
    analyze_trajectory,
    TrajectoryInsights,
    GateAnalysis,
    TerminationAnalysis,
)


def _make_trajectory(**overrides) -> TrajectoryData:
    """Helper to create a TrajectoryData with sensible defaults."""
    T = overrides.pop("n_timesteps", 200)
    defaults = dict(
        schema_version=2,
        positions=np.column_stack([
            np.linspace(0, 20, T), np.zeros(T), np.full(T, 2.0),
        ]),
        quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)),
        velocities=np.tile([8.0, 0.0, 0.0], (T, 1)),
        body_rates=np.zeros((T, 3)),
        motor_rpms=np.full((T, 4), 15000.0),
        actions=np.zeros((T, 4)),
        rewards=np.ones(T),
        reward_components=np.zeros((T, 6)),
        reward_component_names=["progress", "body_rate", "action_smooth",
                                "gate_passage", "gate_offset", "crash_penalty"],
        gate_events=np.array([[50, 0], [100, 1], [150, 2]], dtype=np.int64),
        gate_positions=np.array([[5, 0, 2], [10, 0, 2], [15, 0, 2]], dtype=np.float64),
        gate_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        gate_half_extents=np.full((3, 2), 0.5),
        dt=0.01,
    )
    defaults.update(overrides)
    return TrajectoryData(**defaults)


class TestTrajectoryInsights:
    def test_returns_insights_object(self):
        traj = _make_trajectory()
        insights = analyze_trajectory(traj)
        assert isinstance(insights, TrajectoryInsights)

    def test_speed_statistics(self):
        vels = np.tile([10.0, 0.0, 0.0], (200, 1))
        traj = _make_trajectory(velocities=vels)
        insights = analyze_trajectory(traj)
        assert insights.mean_speed == pytest.approx(10.0, abs=0.1)
        assert insights.max_speed == pytest.approx(10.0, abs=0.1)

    def test_gate_analysis_timing(self):
        traj = _make_trajectory()
        insights = analyze_trajectory(traj)
        assert len(insights.gate_analyses) == 3  # 3 gate events
        # First gate at timestep 50, dt=0.01 → 0.5s
        assert insights.gate_analyses[0].passage_time == pytest.approx(0.5, abs=0.01)

    def test_gate_approach_speed(self):
        traj = _make_trajectory()
        insights = analyze_trajectory(traj)
        # Constant velocity of 8 m/s
        for ga in insights.gate_analyses:
            assert ga.approach_speed > 0

    def test_motor_utilization_stats(self):
        rpms = np.full((200, 4), 20000.0)
        traj = _make_trajectory(motor_rpms=rpms)
        insights = analyze_trajectory(traj, max_rpm=31470.0)
        assert 0.5 < insights.mean_motor_utilization < 0.8

    def test_body_rate_stats(self):
        rates = np.zeros((200, 3))
        rates[:, 0] = 2.0  # constant roll rate
        traj = _make_trajectory(body_rates=rates)
        insights = analyze_trajectory(traj)
        assert insights.max_body_rate > 1.5

    def test_reward_component_breakdown(self):
        rc = np.zeros((200, 6))
        rc[:, 0] = 1.0   # progress
        rc[:, 3] = 5.0   # gate_passage
        traj = _make_trajectory(reward_components=rc)
        insights = analyze_trajectory(traj)
        assert "gate_passage" in insights.reward_breakdown
        assert insights.reward_breakdown["gate_passage"] > insights.reward_breakdown["progress"]
        assert insights.dominant_reward_component == "gate_passage"

    def test_no_gate_events(self):
        traj = _make_trajectory(gate_events=np.empty((0, 2), dtype=np.int64))
        insights = analyze_trajectory(traj)
        assert len(insights.gate_analyses) == 0

    def test_summary_string(self):
        traj = _make_trajectory()
        insights = analyze_trajectory(traj)
        summary = insights.to_summary()
        assert isinstance(summary, str)
        assert "speed" in summary.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_trajectory_insights.py -v`

- [ ] **Step 3: Implement `autoresearch/analysis/trajectory_insights.py`**

```python
"""Extract failure patterns and behavioral signals from trajectory data."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autoresearch.analysis.trajectory import TrajectoryData


@dataclass
class GateAnalysis:
    """Analysis of a single gate passage."""

    gate_idx: int
    timestep: int
    passage_time: float  # seconds from trajectory start
    approach_speed: float  # speed at passage timestep
    offset_distance: float  # distance from gate center at passage


@dataclass
class TerminationAnalysis:
    """Analysis of trajectory termination."""

    final_position: tuple[float, float, float]
    final_speed: float
    total_time: float


@dataclass
class TrajectoryInsights:
    """Extracted insights from a trajectory for hypothesis generation."""

    # Speed
    mean_speed: float
    max_speed: float
    min_speed: float

    # Motor utilization
    mean_motor_utilization: float
    max_motor_utilization: float

    # Body rates
    max_body_rate: float
    mean_body_rate: float

    # Gate passage
    gate_analyses: list[GateAnalysis] = field(default_factory=list)
    gates_passed: int = 0

    # Reward breakdown
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    dominant_reward_component: str = ""

    # Termination
    termination: TerminationAnalysis | None = None

    def to_summary(self) -> str:
        """Generate a human-readable summary for hypothesis generation prompts."""
        lines = [
            f"Speed: mean={self.mean_speed:.1f} max={self.max_speed:.1f} m/s",
            f"Motor utilization: mean={self.mean_motor_utilization:.2f} max={self.max_motor_utilization:.2f}",
            f"Body rates: max={self.max_body_rate:.1f} rad/s",
            f"Gates passed: {self.gates_passed}",
        ]
        if self.gate_analyses:
            times = [ga.passage_time for ga in self.gate_analyses]
            speeds = [ga.approach_speed for ga in self.gate_analyses]
            lines.append(f"Gate times: {[f'{t:.2f}s' for t in times]}")
            lines.append(f"Gate approach speeds: {[f'{s:.1f}' for s in speeds]}")
        if self.reward_breakdown:
            top = sorted(self.reward_breakdown.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lines.append(f"Top reward components: {[(k, f'{v:.2f}') for k, v in top]}")
            lines.append(f"Dominant: {self.dominant_reward_component}")
        return "\n".join(lines)


def analyze_trajectory(
    traj: TrajectoryData,
    max_rpm: float = 31470.0,
) -> TrajectoryInsights:
    """Extract insights from a trajectory for informing hypothesis generation."""
    # Speed statistics
    speeds = np.linalg.norm(traj.velocities, axis=1)
    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    min_speed = float(np.min(speeds))

    # Motor utilization
    motor_norms = np.linalg.norm(traj.motor_rpms, axis=1)
    max_motor_norm = np.linalg.norm(np.full(4, max_rpm))
    utilization = motor_norms / max_motor_norm
    mean_motor_util = float(np.mean(utilization))
    max_motor_util = float(np.max(utilization))

    # Body rates
    rate_mags = np.linalg.norm(traj.body_rates, axis=1)
    max_body_rate = float(np.max(rate_mags))
    mean_body_rate = float(np.mean(rate_mags))

    # Gate passage analysis
    gate_analyses = []
    for i in range(traj.gate_events.shape[0]):
        timestep = int(traj.gate_events[i, 0])
        gate_idx = int(traj.gate_events[i, 1])

        if timestep >= traj.n_timesteps:
            continue

        passage_time = timestep * traj.dt
        approach_speed = float(speeds[timestep])

        # Offset from gate center
        drone_pos = traj.positions[timestep]
        gate_pos = traj.gate_positions[gate_idx]
        offset = float(np.linalg.norm(drone_pos - gate_pos))

        gate_analyses.append(GateAnalysis(
            gate_idx=gate_idx,
            timestep=timestep,
            passage_time=passage_time,
            approach_speed=approach_speed,
            offset_distance=offset,
        ))

    # Reward component breakdown
    reward_breakdown = {}
    if traj.reward_components.shape[1] > 0 and len(traj.reward_component_names) > 0:
        for j, name in enumerate(traj.reward_component_names):
            reward_breakdown[name] = float(np.sum(traj.reward_components[:, j]))

    dominant = ""
    if reward_breakdown:
        dominant = max(reward_breakdown, key=lambda k: abs(reward_breakdown[k]))

    # Termination analysis
    final_pos = tuple(float(x) for x in traj.positions[-1])
    termination = TerminationAnalysis(
        final_position=final_pos,
        final_speed=float(speeds[-1]),
        total_time=traj.n_timesteps * traj.dt,
    )

    return TrajectoryInsights(
        mean_speed=mean_speed,
        max_speed=max_speed,
        min_speed=min_speed,
        mean_motor_utilization=mean_motor_util,
        max_motor_utilization=max_motor_util,
        max_body_rate=max_body_rate,
        mean_body_rate=mean_body_rate,
        gate_analyses=gate_analyses,
        gates_passed=len(gate_analyses),
        reward_breakdown=reward_breakdown,
        dominant_reward_component=dominant,
        termination=termination,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_trajectory_insights.py -v`

- [ ] **Step 5: Commit**

```bash
git add autoresearch/analysis/trajectory_insights.py tests/test_autoresearch/test_trajectory_insights.py
git commit -m "feat(autoresearch): trajectory insight extraction for hypothesis generation"
```

---

## Task 2: Cross-Pollination

**Files:**
- Create: `autoresearch/tree/cross_pollination.py`
- Test: `tests/test_autoresearch/test_cross_pollination.py`

- [ ] **Step 1: Write cross-pollination tests**

File: `tests/test_autoresearch/test_cross_pollination.py`

```python
"""Tests for cross-pollination of successful ideas across branches."""

import pytest

from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.cross_pollination import (
    find_successful_patterns,
    propose_cross_pollinations,
    CrossPollinationProposal,
)


@pytest.fixture
def tree_with_successful_branches():
    """Tree with two branches, one has a successful pattern."""
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    # Branch A: tried high gate_passage weight, succeeded
    a1 = tree.add_experiment(
        "a1", "baseline_v1", "hyperparameter",
        "Increase gate_passage weight to 30", "completed",
    )
    tree.complete_experiment("a1", fitness=4.0, wandb_run_id="run_a1")
    # Branch A refinement
    tree.add_experiment(
        "a2", "a1", "hyperparameter",
        "Also reduce crash_penalty to 5", "completed",
    )
    tree.complete_experiment("a2", fitness=3.5, wandb_run_id="run_a2")

    # Branch B: different approach, worse results
    tree.add_experiment(
        "b1", "baseline_v1", "hyperparameter",
        "Try larger network [128,128,128]", "completed",
    )
    tree.complete_experiment("b1", fitness=7.0, wandb_run_id="run_b1")

    return tree


class TestFindSuccessfulPatterns:
    def test_finds_improving_branches(self, tree_with_successful_branches):
        patterns = find_successful_patterns(tree_with_successful_branches)
        assert len(patterns) >= 1
        # Branch A improved (4.0 -> 3.5), Branch B did not
        best = patterns[0]
        assert best.branch_root_id == "a1"

    def test_empty_tree_returns_empty(self):
        tree = ResearchTree.create_with_baseline("baseline", "abc")
        patterns = find_successful_patterns(tree)
        assert patterns == []


class TestProposeCrossPollinations:
    def test_proposes_grafting_to_other_branches(self, tree_with_successful_branches):
        proposals = propose_cross_pollinations(tree_with_successful_branches)
        assert len(proposals) >= 1
        # Should propose applying A's pattern to branch B
        for p in proposals:
            assert isinstance(p, CrossPollinationProposal)
            assert p.source_branch != p.target_branch
            assert p.inspired_by is not None

    def test_no_proposals_when_no_successful_branches(self):
        tree = ResearchTree.create_with_baseline("baseline", "abc")
        tree.add_experiment("a1", "baseline_v1", "hyperparameter", "test", "failed")
        tree.fail_experiment("a1", "diverged")
        proposals = propose_cross_pollinations(tree)
        assert proposals == []

    def test_proposal_has_description(self, tree_with_successful_branches):
        proposals = propose_cross_pollinations(tree_with_successful_branches)
        if proposals:
            assert len(proposals[0].description) > 0
            assert proposals[0].inspired_by is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_cross_pollination.py -v`

- [ ] **Step 3: Implement `autoresearch/tree/cross_pollination.py`**

```python
"""Cross-pollination: identify successful patterns and propose grafting across branches."""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.tree.research_tree import ResearchTree, TreeNode


@dataclass
class SuccessfulPattern:
    """A branch that showed consistent improvement."""

    branch_root_id: str
    best_fitness: float
    improvement_trajectory: list[float]  # fitness values in order
    description: str


@dataclass
class CrossPollinationProposal:
    """A proposal to graft a successful pattern onto another branch."""

    source_branch: str
    target_branch: str
    inspired_by: str  # hypothesis_id of the successful experiment
    description: str
    rationale: str


def find_successful_patterns(tree: ResearchTree) -> list[SuccessfulPattern]:
    """Identify branches that show consistent improvement.

    A branch is "successful" if it has at least one completed experiment
    with a non-None fitness value that improves over the branch root.
    """
    branches = tree.get_level1_branches()
    patterns = []

    for branch_id in branches:
        root = tree.get_node(branch_id)
        descendants = tree.get_descendants(branch_id)
        all_nodes = [root] + descendants

        completed = [
            n for n in all_nodes
            if n.status == "completed" and n.fitness is not None
        ]
        if not completed:
            continue

        # Sort by fitness (best first for lap time = lowest)
        completed.sort(key=lambda n: n.fitness)
        fitness_trajectory = [n.fitness for n in completed]

        # A pattern is successful if best descendant improved over root
        root_fitness = root.fitness
        best_fitness = completed[0].fitness

        if root_fitness is not None and best_fitness >= root_fitness:
            continue  # No improvement
        if root_fitness is None and len(completed) < 2:
            # Single experiment with no baseline to compare — still include it
            pass

        patterns.append(SuccessfulPattern(
            branch_root_id=branch_id,
            best_fitness=best_fitness,
            improvement_trajectory=fitness_trajectory,
            description=root.description,
        ))

    # Sort by best fitness (lowest = best for lap time)
    patterns.sort(key=lambda p: p.best_fitness)
    return patterns


def propose_cross_pollinations(
    tree: ResearchTree,
) -> list[CrossPollinationProposal]:
    """Propose grafting successful patterns onto other branches.

    For each successful branch, propose applying its approach to branches
    that haven't tried it yet. The proposal uses the `inspired_by` field
    to track provenance without creating a structural parent-child edge.
    """
    patterns = find_successful_patterns(tree)
    if not patterns:
        return []

    all_branches = tree.get_level1_branches()
    proposals = []

    for pattern in patterns:
        # Find branches that haven't been cross-pollinated from this pattern
        for target_branch in all_branches:
            if target_branch == pattern.branch_root_id:
                continue

            # Check if target branch already has an experiment inspired by this pattern
            target_descendants = tree.get_descendants(target_branch)
            already_grafted = any(
                n.inspired_by == pattern.branch_root_id
                for n in target_descendants
            )
            if already_grafted:
                continue

            # Find the best experiment in the source branch to use as inspiration
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
                description=(
                    f"Apply '{pattern.description}' pattern "
                    f"(fitness={pattern.best_fitness:.2f}) to branch '{target_node.description}'"
                ),
                rationale=(
                    f"Branch '{pattern.description}' achieved fitness {pattern.best_fitness:.2f}. "
                    f"Grafting this approach onto '{target_node.description}' "
                    f"may yield similar improvements in a different context."
                ),
            ))

    return proposals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_cross_pollination.py -v`

- [ ] **Step 5: Commit**

```bash
git add autoresearch/tree/cross_pollination.py tests/test_autoresearch/test_cross_pollination.py
git commit -m "feat(autoresearch): cross-pollination of successful patterns across branches"
```

---

## Task 3: Adaptive Cell Subdivision

**Files:**
- Create: `autoresearch/archive/subdivision.py`
- Modify: `autoresearch/archive/map_elites.py` (add subdivision tracking)
- Modify: `autoresearch/archive/serialization.py` (serialize subdivision state)
- Test: `tests/test_autoresearch/test_subdivision.py`

- [ ] **Step 1: Write subdivision tests**

File: `tests/test_autoresearch/test_subdivision.py`

```python
"""Tests for adaptive MAP-Elites cell subdivision."""

import pytest

from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.subdivision import (
    should_subdivide,
    subdivide_cell,
    SubdividedArchive,
)
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def archive_with_insertions():
    """Archive with 3+ insertions in one cell (triggers subdivision)."""
    archive = MapElitesArchive()
    # Insert 3 entries that map to the same cell with improving fitness
    descs = [
        DescriptorVector(0.5, 0.3, 10.0),
        DescriptorVector(0.52, 0.31, 10.1),  # same cell as above
        DescriptorVector(0.48, 0.29, 9.9),   # same cell as above
    ]
    # Only the best fitness stays in the archive, but we track insertion count
    for i, desc in enumerate(descs):
        entry = CellEntry(
            fitness=5.0 - i * 0.5,
            status="approved",
            wandb_run_id=f"run_{i}",
            git_commit=f"commit_{i}",
            descriptors=desc,
            constraint_results={},
            budget_spent=5_000_000,
            hypothesis_id=f"hyp_{i}",
        )
        archive.try_insert(entry)
    return archive


class TestShouldSubdivide:
    def test_cell_with_enough_insertions(self, archive_with_insertions):
        cell = archive_with_insertions.descriptor_to_cell(DescriptorVector(0.5, 0.3, 10.0))
        # Need to track insertion count — for now test the function interface
        assert should_subdivide(archive_with_insertions, cell, min_insertions=1) is True

    def test_empty_cell_does_not_subdivide(self):
        archive = MapElitesArchive()
        assert should_subdivide(archive, (2, 2, 2), min_insertions=3) is False


class TestSubdivideCell:
    def test_creates_subcells(self):
        archive = MapElitesArchive()
        entry = CellEntry(
            fitness=4.0, status="approved", wandb_run_id="run_0",
            git_commit="abc", descriptors=DescriptorVector(0.5, 0.3, 10.0),
            constraint_results={}, budget_spent=5_000_000, hypothesis_id="hyp_0",
        )
        archive.try_insert(entry)
        cell = archive.descriptor_to_cell(entry.descriptors)

        sub_archive = subdivide_cell(archive, cell)
        assert isinstance(sub_archive, SubdividedArchive)
        # Original entry should be in one of the subcells
        assert sub_archive.n_occupied >= 1

    def test_subcell_ranges_cover_parent(self):
        archive = MapElitesArchive()
        cell = (2, 1, 2)
        sub_archive = subdivide_cell(archive, cell)
        # Subdivided archive should have 8 subcells (2x2x2)
        assert sub_archive.n_cells == 8


class TestSubdividedArchiveSerialization:
    def test_round_trip_with_subdivision(self, archive_with_insertions, tmp_path):
        cell = archive_with_insertions.descriptor_to_cell(DescriptorVector(0.5, 0.3, 10.0))
        # Track that this cell has been subdivided
        archive_with_insertions.mark_subdivided(cell)
        assert archive_with_insertions.is_subdivided(cell) is True

        path = tmp_path / "archive.json"
        save_archive(archive_with_insertions, path)
        loaded = load_archive(path)
        assert loaded.is_subdivided(cell) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_subdivision.py -v`

- [ ] **Step 3: Add subdivision tracking to `autoresearch/archive/map_elites.py`**

Read the existing file, then add these methods and a `_subdivided` set to `MapElitesArchive`:

Add to `__init__`:
```python
        self._subdivided: set[tuple[int, int, int]] = set()
        self._insertion_counts: dict[tuple[int, int, int], int] = {}
```

Add methods:
```python
    def mark_subdivided(self, cell: tuple[int, int, int]) -> None:
        """Mark a cell as having been subdivided."""
        self._subdivided.add(cell)

    def is_subdivided(self, cell: tuple[int, int, int]) -> bool:
        """Check if a cell has been subdivided."""
        return cell in self._subdivided

    def get_insertion_count(self, cell: tuple[int, int, int]) -> int:
        """Get the number of insertion attempts for a cell."""
        return self._insertion_counts.get(cell, 0)
```

Update `try_insert` to track insertion counts:
```python
    def try_insert(self, entry: CellEntry) -> bool:
        cell = self.descriptor_to_cell(entry.descriptors)
        # Track insertion attempts
        self._insertion_counts[cell] = self._insertion_counts.get(cell, 0) + 1
        existing = self._grid.get(cell)
        if existing is None or entry.fitness < existing.fitness:
            self._grid[cell] = entry
            return True
        return False
```

- [ ] **Step 4: Update `autoresearch/archive/serialization.py`**

Read the existing file, then update `save_archive` to include subdivision state:

In `save_archive`, add after the cells loop:
```python
    data["subdivided"] = [list(c) for c in archive._subdivided]
    data["insertion_counts"] = {
        f"{c[0]},{c[1]},{c[2]}": count
        for c, count in archive._insertion_counts.items()
    }
```

In `load_archive`, add after loading cells:
```python
    for sub_cell in data.get("subdivided", []):
        archive._subdivided.add(tuple(sub_cell))
    for key, count in data.get("insertion_counts", {}).items():
        cell_indices = tuple(int(x) for x in key.split(","))
        archive._insertion_counts[cell_indices] = count
```

- [ ] **Step 5: Implement `autoresearch/archive/subdivision.py`**

```python
"""Adaptive cell subdivision for MAP-Elites archive."""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


@dataclass
class SubdividedArchive:
    """A 2x2x2 sub-archive within a parent cell."""

    parent_cell: tuple[int, int, int]
    sub_grid: dict[tuple[int, int, int], CellEntry]
    axis_ranges: list[tuple[float, float]]  # narrower ranges for this sub-region

    @property
    def n_cells(self) -> int:
        return 8  # 2x2x2

    @property
    def n_occupied(self) -> int:
        return len(self.sub_grid)


def should_subdivide(
    archive: MapElitesArchive,
    cell: tuple[int, int, int],
    min_insertions: int = 3,
) -> bool:
    """Check if a cell should be subdivided based on insertion count."""
    if archive.is_subdivided(cell):
        return False  # Already subdivided
    if archive.get(cell) is None:
        return False  # Empty cell
    return archive.get_insertion_count(cell) >= min_insertions


def _cell_ranges(
    archive: MapElitesArchive,
    cell: tuple[int, int, int],
) -> list[tuple[float, float]]:
    """Compute the descriptor ranges for a specific cell."""
    ranges = []
    for axis_idx, (lo, hi) in enumerate(archive.axis_ranges):
        bin_width = (hi - lo) / archive.bins_per_axis
        cell_lo = lo + cell[axis_idx] * bin_width
        cell_hi = cell_lo + bin_width
        ranges.append((cell_lo, cell_hi))
    return ranges


def subdivide_cell(
    archive: MapElitesArchive,
    cell: tuple[int, int, int],
) -> SubdividedArchive:
    """Create a 2x2x2 sub-archive within a parent cell.

    The parent cell's range is split into 2 bins per axis,
    creating 8 subcells with finer resolution.
    """
    cell_ranges = _cell_ranges(archive, cell)
    sub_grid: dict[tuple[int, int, int], CellEntry] = {}

    # If the parent cell has an entry, place it in the correct subcell
    existing = archive.get(cell)
    if existing is not None:
        desc = existing.descriptors.to_tuple()
        sub_cell = []
        for val, (lo, hi) in zip(desc, cell_ranges):
            mid = (lo + hi) / 2
            sub_cell.append(0 if val < mid else 1)
        sub_grid[tuple(sub_cell)] = existing

    return SubdividedArchive(
        parent_cell=cell,
        sub_grid=sub_grid,
        axis_ranges=cell_ranges,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_subdivision.py -v`

Also verify existing archive tests still pass:
Run: `python -m pytest tests/test_autoresearch/test_archive.py -v`

- [ ] **Step 7: Commit**

```bash
git add autoresearch/archive/ tests/test_autoresearch/test_subdivision.py
git commit -m "feat(autoresearch): adaptive MAP-Elites cell subdivision"
```

---

## Task 4: Baseline Promotion

**Files:**
- Create: `autoresearch/promotion.py`
- Modify: `autoresearch/tree/research_tree.py` (add `promote_baseline` method)
- Create: `.claude/plugins/autoresearch/skills/ar-promote.md`
- Test: `tests/test_autoresearch/test_promotion.py`

- [ ] **Step 1: Write promotion tests**

File: `tests/test_autoresearch/test_promotion.py`

```python
"""Tests for baseline promotion workflow."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autoresearch.promotion import (
    verify_promotion_candidate,
    generate_pr_body,
    update_state_after_promotion,
    PromotionCandidate,
    PromotionResult,
)
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.serialization import save_tree, load_tree
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def promotion_candidate():
    return PromotionCandidate(
        hypothesis_id="hyp_winner",
        wandb_run_id="run_winner",
        fitness=3.5,
        descriptors=DescriptorVector(0.6, 0.2, 12.0),
        branch="ar/exp-hyp_winner",
        git_commit="winner123",
    )


@pytest.fixture
def archive_with_winner():
    archive = MapElitesArchive()
    archive.try_insert(CellEntry(
        fitness=3.5, status="approved", wandb_run_id="run_winner",
        git_commit="winner123", descriptors=DescriptorVector(0.6, 0.2, 12.0),
        constraint_results={"passed": True}, budget_spent=15_000_000,
        hypothesis_id="hyp_winner",
    ))
    return archive


@pytest.fixture
def tree_with_winner():
    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment(
        "hyp_winner", "baseline_v1", "algorithm",
        "Winner experiment", "completed",
    )
    tree.complete_experiment("hyp_winner", fitness=3.5, wandb_run_id="run_winner")
    return tree


class TestVerifyCandidate:
    def test_valid_candidate_passes(self, promotion_candidate, archive_with_winner):
        result = verify_promotion_candidate(promotion_candidate, archive_with_winner)
        assert result.verified is True

    def test_candidate_not_in_archive_fails(self, promotion_candidate):
        empty_archive = MapElitesArchive()
        result = verify_promotion_candidate(promotion_candidate, empty_archive)
        assert result.verified is False
        assert "not found" in result.reason.lower()

    def test_candidate_not_approved_fails(self, promotion_candidate):
        archive = MapElitesArchive()
        archive.try_insert(CellEntry(
            fitness=3.5, status="candidate", wandb_run_id="run_winner",
            git_commit="winner123", descriptors=DescriptorVector(0.6, 0.2, 12.0),
            constraint_results={}, budget_spent=15_000_000,
            hypothesis_id="hyp_winner",
        ))
        result = verify_promotion_candidate(promotion_candidate, archive)
        assert result.verified is False
        assert "approved" in result.reason.lower()


class TestGeneratePrBody:
    def test_contains_required_sections(self, promotion_candidate):
        body = generate_pr_body(
            promotion_candidate,
            old_baseline_fitness=5.0,
            wandb_url="https://wandb.ai/test/run",
        )
        assert "## Summary" in body
        assert "hyp_winner" in body
        assert "3.5" in body  # fitness
        assert "5.0" in body  # old baseline
        assert "wandb.ai" in body

    def test_shows_improvement(self, promotion_candidate):
        body = generate_pr_body(
            promotion_candidate,
            old_baseline_fitness=5.0,
            wandb_url="https://wandb.ai/test/run",
        )
        assert "improvement" in body.lower() or "better" in body.lower()


class TestUpdateStateAfterPromotion:
    def test_archives_old_grid(self, archive_with_winner, tree_with_winner, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        save_archive(archive_with_winner, state_dir / "archive.json")
        save_tree(tree_with_winner, state_dir / "tree.json")

        update_state_after_promotion(
            state_dir=state_dir,
            hypothesis_id="hyp_winner",
            merge_commit="merge_abc",
            new_baseline_name="baseline_v2",
            promoter="shaan",
        )

        # Old archive should be preserved
        assert (state_dir / "archive_v1.json").exists()

        # New archive should have 0 occupied cells (fresh start, entry re-inserted later)
        new_archive = load_archive(state_dir / "archive.json")
        # The implementation should start fresh
        assert new_archive.n_occupied == 0

        # Tree should have new baseline root
        new_tree = load_tree(state_dir / "tree.json")
        root = new_tree.root
        assert root.hypothesis_id == "baseline_v2"
        assert root.git_commit == "merge_abc"

    def test_baseline_chain_preserved(self, archive_with_winner, tree_with_winner, tmp_path):
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        save_archive(archive_with_winner, state_dir / "archive.json")
        save_tree(tree_with_winner, state_dir / "tree.json")

        update_state_after_promotion(
            state_dir=state_dir,
            hypothesis_id="hyp_winner",
            merge_commit="merge_abc",
            new_baseline_name="baseline_v2",
            promoter="shaan",
        )

        tree = load_tree(state_dir / "tree.json")
        # Old baseline should still exist as a node
        old_root = tree.get_node("baseline_v1")
        assert old_root is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_autoresearch/test_promotion.py -v`

- [ ] **Step 3: Add `promote_baseline` method to `autoresearch/tree/research_tree.py`**

Read the existing file, then add this method to `ResearchTree`:

```python
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
```

- [ ] **Step 4: Implement `autoresearch/promotion.py`**

```python
"""Baseline promotion: verify, create PR body, update state after merge."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.descriptors.compute import DescriptorVector
from autoresearch.tree.research_tree import ResearchTree, TreeNode
from autoresearch.tree.serialization import save_tree, load_tree


@dataclass
class PromotionCandidate:
    """An experiment being considered for baseline promotion."""

    hypothesis_id: str
    wandb_run_id: str
    fitness: float
    descriptors: DescriptorVector
    branch: str
    git_commit: str


@dataclass
class PromotionResult:
    """Result of verifying a promotion candidate."""

    verified: bool
    reason: str


def verify_promotion_candidate(
    candidate: PromotionCandidate,
    archive: MapElitesArchive,
) -> PromotionResult:
    """Verify that a candidate is eligible for baseline promotion.

    Checks:
    - Candidate exists in the archive
    - Candidate has 'approved' status
    """
    # Find the candidate in the archive
    found = False
    for cell, entry in archive.occupied_cells().items():
        if entry.hypothesis_id == candidate.hypothesis_id:
            found = True
            if entry.status != "approved":
                return PromotionResult(
                    verified=False,
                    reason=f"Candidate {candidate.hypothesis_id} has status "
                           f"'{entry.status}', must be 'approved' for promotion",
                )
            break

    if not found:
        return PromotionResult(
            verified=False,
            reason=f"Candidate {candidate.hypothesis_id} not found in archive",
        )

    return PromotionResult(verified=True, reason="Candidate verified for promotion")


def generate_pr_body(
    candidate: PromotionCandidate,
    old_baseline_fitness: float | None,
    wandb_url: str,
) -> str:
    """Generate a pull request body for baseline promotion."""
    improvement = ""
    if old_baseline_fitness is not None:
        delta = old_baseline_fitness - candidate.fitness
        pct = (delta / old_baseline_fitness) * 100 if old_baseline_fitness > 0 else 0
        improvement = f"\n**Improvement:** {delta:.2f}s ({pct:.1f}% better lap time)"

    desc = candidate.descriptors
    return f"""## Summary

Promote experiment `{candidate.hypothesis_id}` to new baseline.

**Fitness (lap time):** {candidate.fitness:.3f}s (old baseline: {old_baseline_fitness or 'N/A'}){improvement}

## Behavioral Profile

| Descriptor | Value |
|-----------|-------|
| Actuator Utilization | {desc.actuator_utilization:.3f} |
| Control Smoothness | {desc.control_smoothness:.3f} |
| Aero Regime Index | {desc.aero_regime:.3f} |

## Links

- W&B Run: {wandb_url}
- Branch: `{candidate.branch}`
- Commit: `{candidate.git_commit}`

---
🤖 Generated by auto-research system
"""


def update_state_after_promotion(
    state_dir: Path | str,
    hypothesis_id: str,
    merge_commit: str,
    new_baseline_name: str,
    promoter: str,
) -> None:
    """Update state files after a baseline promotion is merged.

    1. Archive the current archive as archive_v<N>.json
    2. Create a fresh archive
    3. Update the research tree with new baseline root
    """
    state_dir = Path(state_dir)

    # Determine version number
    existing_versions = list(state_dir.glob("archive_v*.json"))
    version = len(existing_versions) + 1

    # Archive current grid
    current_archive_path = state_dir / "archive.json"
    archived_path = state_dir / f"archive_v{version}.json"
    shutil.copy2(current_archive_path, archived_path)

    # Create fresh archive
    fresh_archive = MapElitesArchive()
    save_archive(fresh_archive, current_archive_path)

    # Update research tree
    tree = load_tree(state_dir / "tree.json")

    # Promote to new baseline via public API
    tree.promote_baseline(
        new_baseline_id=new_baseline_name,
        git_commit=merge_commit,
        description=f"Promoted from {hypothesis_id}",
        wandb_run_id=None,  # populated by caller if available
        promoter=promoter,
    )

    save_tree(tree, state_dir / "tree.json")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_autoresearch/test_promotion.py -v`

- [ ] **Step 5: Create `/ar-promote` skill**

File: `.claude/plugins/autoresearch/skills/ar-promote.md`

```markdown
---
name: ar-promote
description: Promote a winning experiment to the new baseline. Always requires human approval regardless of mode.
---

# AR Promote

Promote a winning experiment to become the new baseline for all future research.

**This action is ALWAYS human-gated** — even in YOLO mode.

## Usage

`/ar-promote <hypothesis_id or wandb_run_id>`

## Workflow

### 1. Verify Candidate

```python
from autoresearch.promotion import verify_promotion_candidate, PromotionCandidate
from autoresearch.archive.serialization import load_archive

archive = load_archive("autoresearch/state/archive.json")
candidate = PromotionCandidate(
    hypothesis_id="<id>",
    wandb_run_id="<run_id>",
    fitness=<lap_time>,
    descriptors=<descriptor_vector>,
    branch="ar/exp-<id>",
    git_commit="<commit>",
)
result = verify_promotion_candidate(candidate, archive)
```

Display verification results to the user.

### 2. Show Comparison

Present to the user:
- Old baseline fitness vs candidate fitness
- Behavioral descriptor profile
- Constraint validation summary
- W&B run link and Rerun trajectory URL
- Full diff summary (`git diff main...<branch>`)

### 3. Create PR

If user approves, create a PR:

```bash
gh pr create --title "Promote <hypothesis_id> to baseline" --body "$(python -c "
from autoresearch.promotion import generate_pr_body, PromotionCandidate
from autoresearch.descriptors.compute import DescriptorVector
candidate = PromotionCandidate(...)
print(generate_pr_body(candidate, old_baseline_fitness=<old>, wandb_url='<url>'))
")"
```

### 4. After Merge

Once the PR is merged by a human, run:

```python
from autoresearch.promotion import update_state_after_promotion

update_state_after_promotion(
    state_dir="autoresearch/state/",
    hypothesis_id="<id>",
    merge_commit="<merge_sha>",
    new_baseline_name="baseline_v<N>",
    promoter="<username>",
)
```

Then commit and push the updated state files.
```

- [ ] **Step 6: Commit**

```bash
git add autoresearch/promotion.py tests/test_autoresearch/test_promotion.py .claude/plugins/autoresearch/skills/ar-promote.md
git commit -m "feat(autoresearch): baseline promotion workflow with /ar-promote skill"
```

---

## Task 5: Phase 3 Integration Test

**Files:**
- Create: `tests/test_autoresearch/test_phase3_integration.py`

- [ ] **Step 1: Write Phase 3 integration test**

```python
"""Integration test for Phase 3 features."""

import shutil
from pathlib import Path

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.analysis.trajectory_insights import analyze_trajectory, TrajectoryInsights
from autoresearch.tree.research_tree import ResearchTree
from autoresearch.tree.cross_pollination import find_successful_patterns, propose_cross_pollinations
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.subdivision import should_subdivide, subdivide_cell
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.promotion import (
    verify_promotion_candidate, generate_pr_body,
    update_state_after_promotion, PromotionCandidate,
)
from autoresearch.tree.serialization import save_tree, load_tree
from autoresearch.descriptors.compute import DescriptorVector
import numpy as np


def test_trajectory_to_insights():
    """Trajectory data produces actionable insights."""
    T = 200
    traj = TrajectoryData(
        schema_version=2,
        positions=np.column_stack([np.linspace(0, 20, T), np.zeros(T), np.full(T, 2.0)]),
        quaternions=np.tile([1.0, 0.0, 0.0, 0.0], (T, 1)),
        velocities=np.tile([10.0, 0.0, 0.0], (T, 1)),
        body_rates=np.zeros((T, 3)),
        motor_rpms=np.full((T, 4), 20000.0),
        actions=np.zeros((T, 4)),
        rewards=np.ones(T),
        reward_components=np.column_stack([np.ones(T) * 5, np.zeros(T), np.zeros(T),
                                           np.ones(T) * 10, np.zeros(T), np.zeros(T)]),
        reward_component_names=["progress", "body_rate", "action_smooth",
                                "gate_passage", "gate_offset", "crash_penalty"],
        gate_events=np.array([[50, 0], [100, 1]], dtype=np.int64),
        gate_positions=np.array([[5, 0, 2], [10, 0, 2], [15, 0, 2]], dtype=np.float64),
        gate_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        gate_half_extents=np.full((3, 2), 0.5),
        dt=0.01,
    )
    insights = analyze_trajectory(traj)
    assert insights.mean_speed == 10.0
    assert insights.dominant_reward_component == "gate_passage"
    assert len(insights.gate_analyses) == 2
    summary = insights.to_summary()
    assert "gate_passage" in summary


def test_cross_pollination_pipeline():
    """Successful branch patterns get proposed for other branches."""
    tree = ResearchTree.create_with_baseline("baseline", "abc")
    tree.add_experiment("a1", "baseline_v1", "hyperparameter", "High gate weight", "completed")
    tree.complete_experiment("a1", fitness=4.0)
    tree.add_experiment("a2", "a1", "hyperparameter", "Refine gate weight", "completed")
    tree.complete_experiment("a2", fitness=3.5)
    tree.add_experiment("b1", "baseline_v1", "algorithm", "New network", "completed")
    tree.complete_experiment("b1", fitness=7.0)

    patterns = find_successful_patterns(tree)
    assert len(patterns) >= 1
    proposals = propose_cross_pollinations(tree)
    assert len(proposals) >= 1
    assert proposals[0].source_branch == "a1"
    assert proposals[0].target_branch == "b1"


def test_subdivision_pipeline():
    """Archive cells with enough insertions can be subdivided."""
    archive = MapElitesArchive()
    desc = DescriptorVector(0.5, 0.3, 10.0)
    for i in range(4):
        archive.try_insert(CellEntry(
            fitness=5.0 - i * 0.3, status="approved", wandb_run_id=f"run_{i}",
            git_commit=f"c{i}", descriptors=desc,
            constraint_results={}, budget_spent=5_000_000, hypothesis_id=f"h{i}",
        ))
    cell = archive.descriptor_to_cell(desc)
    assert archive.get_insertion_count(cell) == 4
    assert should_subdivide(archive, cell, min_insertions=3)
    sub = subdivide_cell(archive, cell)
    assert sub.n_occupied >= 1


def test_promotion_pipeline(tmp_path):
    """Full promotion flow: verify → PR body → state update."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    # Setup
    archive = MapElitesArchive()
    desc = DescriptorVector(0.6, 0.2, 12.0)
    archive.try_insert(CellEntry(
        fitness=3.5, status="approved", wandb_run_id="run_win",
        git_commit="win123", descriptors=desc,
        constraint_results={}, budget_spent=15_000_000, hypothesis_id="hyp_win",
    ))
    save_archive(archive, state_dir / "archive.json")

    tree = ResearchTree.create_with_baseline("monorace_baseline", "abc123")
    tree.add_experiment("hyp_win", "baseline_v1", "algorithm", "Winner", "completed")
    tree.complete_experiment("hyp_win", fitness=3.5, wandb_run_id="run_win")
    save_tree(tree, state_dir / "tree.json")

    # Verify
    candidate = PromotionCandidate(
        hypothesis_id="hyp_win", wandb_run_id="run_win", fitness=3.5,
        descriptors=desc, branch="ar/exp-hyp_win", git_commit="win123",
    )
    result = verify_promotion_candidate(candidate, archive)
    assert result.verified

    # PR body
    body = generate_pr_body(candidate, old_baseline_fitness=5.0,
                            wandb_url="https://wandb.ai/test/run")
    assert "3.500" in body
    assert "improvement" in body.lower() or "better" in body.lower()

    # State update
    update_state_after_promotion(
        state_dir=state_dir, hypothesis_id="hyp_win",
        merge_commit="merge_abc", new_baseline_name="baseline_v2",
        promoter="shaan",
    )

    # Verify state
    assert (state_dir / "archive_v1.json").exists()
    new_archive = load_archive(state_dir / "archive.json")
    assert new_archive.n_occupied == 0  # fresh start
    new_tree = load_tree(state_dir / "tree.json")
    assert new_tree.root.hypothesis_id == "baseline_v2"
    assert new_tree.root.git_commit == "merge_abc"
    assert new_tree.get_node("baseline_v1") is not None  # old baseline preserved
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/test_autoresearch/ -v`
Expected: ALL PASS (116 Phase 1+2 + new Phase 3 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_autoresearch/test_phase3_integration.py
git commit -m "test(autoresearch): Phase 3 integration tests"
```

---

## Summary

| Task | Component | Key Deliverable |
|------|-----------|-----------------|
| 1 | Trajectory Insights | `trajectory_insights.py` — extract speed, gates, rewards, failure patterns for smarter hypotheses |
| 2 | Cross-Pollination | `cross_pollination.py` — find successful patterns and propose grafting across branches |
| 3 | Adaptive Subdivision | `subdivision.py` — 2×2×2 cell refinement when insertion count ≥ 3 |
| 4 | Baseline Promotion | `promotion.py` + `/ar-promote` — verify, create PR, update state chain |
| 5 | Integration Test | End-to-end tests for all Phase 3 features |

Tasks 1-4 are independent and can be parallelized. Task 5 validates everything together.
