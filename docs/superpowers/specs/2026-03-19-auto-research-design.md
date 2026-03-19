# Auto-Research System Design

**Date**: 2026-03-19
**Status**: Draft
**Scope**: Claude Code plugin for autonomous research exploration across the algo_src drone racing stack

## Overview

An automated research system that uses Claude Code to systematically explore the solution space for autonomous drone racing. Inspired by DeepMind's AlphaEvolve and Karpathy's autoresearch, the system generates hypotheses, implements changes, runs experiments, and evaluates results — ranging from hyperparameter tweaks to entirely new architectures.

The system uses constrained MAP-Elites to maintain a diverse archive of high-performing solutions across behavioral niches, coordinates across team members via git-synced state files to prevent duplicate work, and enforces hard safety constraints to prevent reward gaming.

## Goals

- Increase the area of the solution space explored by leveraging Claude Code's ability to read, understand, and modify the codebase
- Explore changes at all scope levels: hyperparameters, algorithms, architectures, and system-level redesigns
- Maintain safety and control through git isolation, behavioral validation, and anti-gaming constraints
- Enable team-wide parallel research without duplication or collision
- Build on existing W&B, Rerun, and Hydra infrastructure

## Non-Goals

- Replacing human judgment for high-level research direction (though YOLO mode allows full autonomy)
- Building a standalone ML pipeline independent of Claude Code
- Hyperparameter sweep / grid search (this is hypothesis-driven research, not brute force)

---

## Delivery Phases (MVP → Full)

### Phase 1 — MVP: Config-Only Mutations

Scope: hyperparameter and reward weight exploration only. No autonomous code edits.

- Hydra override strings only (no source file changes)
- Fixed 5×5×5 MAP-Elites grid (no adaptive subdivision)
- Human review before any archive promotion (interactive mode only)
- Git branch per experiment (standard branches, not worktrees)
- Coordination via git-synced state file
- Basic constraint validation (success rate floor, physics plausibility)

**Delivers**: working research loop, archive, coordination, descriptors, constraints. Validates the behavioral descriptor axes empirically before relying on them.

### Phase 2 — Code-Level Mutations

Adds: algorithm and architecture scope changes with code edits.

- Git worktrees for experiment isolation
- Diff allowlist/denylist enforcement (see Edit Surface Constraints)
- Autonomous and YOLO modes enabled
- Full constraint validation suite
- Research tree with branch selection
- Early stopping

### Phase 3 — Full Autonomy

Adds: system-level changes, cross-pollination, adaptive grid refinement.

- System scope changes with mandatory test pass
- Cross-pollination across branches
- Adaptive cell subdivision in promising archive regions
- Advanced trajectory/Rerun analysis for hypothesis generation

---

## Architecture

### Two Layers

**1. `autoresearch/` Python package** — infrastructure utilities Claude Code calls:
- `archive/` — MAP-Elites archive backed by git-synced JSON state
- `descriptors/` — Behavioral descriptor computation from trajectory data
- `constraints/` — Hard constraint validators (anti-gaming, physics plausibility, diff surface)
- `coordination/` — Experiment locking/claiming via git-synced state file
- `analysis/` — W&B data pulling, learning curve analysis, early stopping, Rerun/trajectory analysis
- `tree/` — Research tree structure, branch selection with exploration guarantees

**2. `autoresearch-plugin/` Claude Code plugin** — installed into the project's `.claude/` directory following Claude Code plugin conventions:

```
.claude/
├── plugins/
│   └── autoresearch/
│       ├── plugin.json         # Plugin manifest (name, version, description)
│       ├── skills/
│       │   ├── auto-research.md    # Main research loop skill
│       │   ├── ar-status.md        # Archive status display
│       │   ├── ar-review.md        # Experiment review skill
│       │   └── ar-branch.md        # Manual branch creation
│       ├── hooks/
│       │   ├── post-training.sh    # Descriptor computation + archive update
│       │   └── session-start.sh    # Pull latest archive state
│       └── settings.yaml           # Default plugin configuration
```

### Package Structure

```
autoresearch/
├── archive/
│   ├── map_elites.py      # Grid storage, cell insertion, state sync
│   └── serialization.py   # JSON state read/write
├── descriptors/
│   ├── actuator.py        # Axis 1: actuator utilization
│   ├── smoothness.py      # Axis 2: control smoothness
│   ├── aero_regime.py     # Axis 3: speed x angle-of-attack
│   └── compute.py         # Orchestrates all 3 from trajectory data
├── constraints/
│   ├── validator.py       # Hard constraint checks
│   ├── physics.py         # Physics plausibility
│   ├── racing.py          # Gate passage quality, behavioral sanity
│   └── diff_policy.py     # Edit surface allowlist/denylist enforcement
├── coordination/
│   ├── claims.py          # Experiment claiming/locking
│   └── dedup.py           # Hypothesis deduplication
├── analysis/
│   ├── wandb_pull.py      # Fetch run history, metrics, configs
│   ├── trajectory.py      # Load/analyze .npz trajectory data
│   ├── learning_curve.py  # Early stopping signals
│   └── rerun_analysis.py  # Extract insights from .rrd/.npz recordings
├── tree/
│   ├── research_tree.py   # Tree structure, branch management
│   └── selection.py       # UCB scoring, diversity bonus, random restarts
├── hypothesis/
│   ├── schema.py          # Hypothesis data contract
│   └── prompt.py          # Prompt construction for hypothesis generation
└── config.py              # Plugin configuration schema
```

### State Management

**W&B** handles what it's good at: run tracking, metrics logging, artifact storage (.npz, .rrd), Rerun URL linking.

**Git-synced state files** handle coordination and research state:

```
autoresearch/
├── state/
│   ├── archive.json       # MAP-Elites grid (cell contents, fitness, descriptors)
│   ├── tree.json           # Research tree (nodes, edges, hypotheses, results)
│   ├── claims.json         # Active experiment claims (who is doing what)
│   └── config.json         # Shared config overrides
```

These files live in the repo and are synced via git pull/push. Each auto-research session pulls before starting and pushes state updates after experiments complete. Merge conflicts in JSON state files are resolved by taking the entry with better fitness (archive) or newer timestamp (claims/tree).

---

## Edit Surface Constraints

The auto-research system is restricted in what files it can modify. This prevents gaming by altering the simulation, evaluation, or metrics infrastructure to produce artificially better scores.

### Denylist (never modify)

These files/directories are off-limits in all modes and scopes:

| Path | Reason |
|------|--------|
| `sim/dynamics/` | Physics simulation — changing dynamics to make the drone faster is gaming |
| `sim/rewards.py` | Reward function — modifying rewards to inflate scores is gaming |
| `metrics/` | Metrics contract — changing how success is measured is gaming |
| `training/callbacks.py` | Logging/eval callbacks — altering what gets logged corrupts data |
| `training/trajectory_recorder.py` | Trajectory recording — corrupting training data |
| `autoresearch/` | The research system itself — no self-modification |
| `autoresearch/state/` | State files (modified only through the autoresearch API, not raw edits) |
| `.claude/` | Plugin infrastructure |
| `artifacts/` | Artifact pipeline — altering upload/logging corrupts provenance |
| `docker/` | Docker infrastructure |
| `tests/` | Test suite (should only be extended, not weakened) |

### Allowlist by scope

| Scope | Allowed edits |
|-------|--------------|
| **Hyperparameter** | No file edits — Hydra override strings only |
| **Algorithm** | `control/algorithms/`, `control/policies/`, `perception/wrappers/`, `configs/` |
| **Architecture** | Algorithm scope + `perception/detectors/`, `state_estimation/`, new files in `control/` |
| **System** | Architecture scope + `sim/envs/`, `training/loops/`, new files in `sim/` (not `sim/dynamics/` or `sim/rewards.py`) |

### Enforcement

Before any experiment launches, `diff_policy.py` validates the proposed changes:
1. Compute the diff between the worktree/branch and main
2. Check every modified file against the denylist — reject if any match
3. Check every modified file against the scope-appropriate allowlist — reject if any file is outside scope
4. Log the diff summary to the research tree for auditability
5. Reject diffs exceeding a configurable max size (default: 500 lines added/removed for algorithm scope, 1000 for architecture, 2000 for system) to prevent uncontrolled churn

---

## MAP-Elites Archive

### Evaluation Strategy

**Primary fitness**: Lap time (minimized). The only metric being optimized.

**Hard constraints** (pass/fail, gate entry to archive):
- Success rate > configurable threshold (default 80%)
- Physics plausibility (no sim exploitation)
- Gate passage quality (sequential, through center, no clipping)
- Behavioral sanity (forward progress, minimum speed, no degenerate patterns)

**Behavioral descriptors** (3 axes, define the map):
1. Actuator utilization
2. Control smoothness
3. Aerodynamic regime index

This separates concerns cleanly: constraints gate entry, fitness determines ranking within a cell, and descriptors maintain diversity. No weighted composite score, no hyperparameter explosion.

### Behavioral Descriptor Axes

**Axis 1 — Actuator Utilization (Aggressiveness)**
- Computation: `mean(||motor_RPMs(t)|| / ||max_RPMs||)` over the trajectory
- `max_RPM` source: `VehicleParams.max_rpm` (defined in `sim/dynamics/params.py`, currently 31470). Loaded from the Hydra sim config and stored in trajectory `.npz` metadata. The descriptor module reads it from either the trajectory metadata or the sim config as fallback.
- Range: ~0.25 (hover) to ~1.0 (full saturation)
- Physics: measures controllability margin. Near saturation, thrust curve flattens, motor model errors are amplified, and no control authority remains for disturbance rejection. Strongest predictor of sim-to-real transfer failure (Kaufmann et al., Nature 2023).

**Axis 2 — Control Smoothness (Command Rate of Change)**
- Computation: `mean(||d(motor_RPMs)/dt(t)||)` normalized by max_RPM × control_frequency (max_RPM sourced identically to Axis 1)
- Range: [0, 1] dimensionless
- Physics: motors are low-pass filters. Smooth commands track well despite model error; high-frequency commands expose inaccuracies in motor transient response (time constants, voltage sag, temperature). The Swift paper's successful sim-to-real policy was notably smoother than the sim-only optimum.
- Independence: genuinely orthogonal to Axis 1 — fast-but-smooth (well-planned) vs slow-but-jerky (reactive) represent meaningfully different transfer characteristics.

**Axis 3 — Aerodynamic Regime (Speed × Angle-of-Attack)**
- Computation: `mean(||v(t)|| * sin(alpha(t)))` where alpha = angle between velocity vector and body z-axis (from quaternion)
- Physics: simplified drag models (constant `drag_coeff`) diverge from reality when both speed and angle-of-attack are high. NeuroBEM (Bauersfeld et al., RSS 2021) shows errors grow dramatically above ~10 m/s at alpha > 30°. Captures diving, aggressive cornering, sideways flight — regimes where prop wash, advance ratio effects, and lateral forces are unmodeled.
- Moderate correlation with Axis 1, but the alpha component adds independent information about aerodynamic model fidelity.

**Validation prerequisite** (Phase 1): before relying on the archive for branch selection, run a descriptor stability analysis across 10+ training runs with varied configs to verify: (1) descriptors are stable across tracks and DR settings, (2) axes are sufficiently uncorrelated, (3) bin boundaries produce meaningful behavioral distinctions.

### Grid Resolution

Each axis discretized into 5 bins → 125 total cells. Fixed grid (no adaptive subdivision until Phase 3).
- Actuator utilization: 5 bins across [0.25, 1.0]
- Control smoothness: 5 bins across [0, 1]
- Aero regime index: 5 bins across [0, v_max] where v_max is the physics-derived upper bound = `sqrt(4 * k_thrust * max_omega^2 / max(drag_coeff))` (theoretical terminal velocity from quadratic thrust model). This is a fixed constant for a given vehicle configuration (~25-30 m/s for our quad), ensuring bin boundaries never shift as new experiments are added.

### Cell Contents

Each occupied cell stores:
- Best fitness (lap time) for that behavioral niche
- W&B run ID (links to all metrics, Rerun recordings, config)
- Git commit hash (exact code that produced it)
- Behavioral descriptor values
- Constraint validation results (pass/fail + details)
- Total compute budget spent (timesteps trained) — helps UCB-style selection distinguish "well-explored" from "barely tried" cells

### Storage & Sync

Stored in `autoresearch/state/archive.json`, synced via git.

**Conflict resolution protocol** (optimistic concurrency):
1. `git pull` to get latest state
2. Compute new cell insertion (fitness comparison against current incumbent)
3. Before writing, `git pull` again — if archive has changed:
   a. Re-check if the insertion is still valid against the updated archive
   b. If the target cell now has a better incumbent (another writer beat us), skip insertion but still log the result in the research tree
   c. If still valid, write the update
4. `git commit && git push` — if push fails (another writer pushed first), pull, re-evaluate, retry (max 3 attempts)

---

## Hypothesis Generation

### Hypothesis Schema

Every hypothesis must conform to this contract before it can proceed:

```python
@dataclass
class Hypothesis:
    # Identity
    id: str                          # Content hash of (scope + changes + target)
    parent_id: str | None            # Research tree parent
    inspired_by: str | None          # Cross-pollination source (metadata only)

    # What
    scope: Literal["hyperparameter", "algorithm", "architecture", "system"]
    description: str                 # Human-readable: what and why
    changes: list[Change]            # Concrete list of modifications
    # Change = HydraOverride(key, value) | FileDiff(path, description)

    # Predictions
    target_cells: list[tuple[int,int,int]]  # Archive cells this aims to fill/improve
    predicted_descriptor_range: dict         # Optional — expected behavioral profile
    rationale: str                           # Why this hypothesis should improve fitness

    # Metadata
    estimated_budget: int            # Timesteps (from scope default, can be overridden)
    risk_level: Literal["low", "medium", "high"]  # Based on scope + change magnitude
    novelty_score: float             # 0-1, how different from existing tree nodes
```

### Prompt Construction

The hypothesis generation prompt is assembled from concrete inputs, not open-ended creativity:

**Required inputs** (pulled automatically):
1. Archive state summary — occupied cells, fitness values, empty regions
2. Top-5 best and worst performing runs — W&B config + key metrics
3. Research tree summary — what's been tried, what worked, what failed and why
4. Active claims — what's currently being explored by other team members
5. Reward component breakdown from recent runs — which components dominate
6. Trajectory analysis summary — common failure modes (crash types, termination reasons)

**Optional inputs** (when available):
7. Rerun trajectory analysis — qualitative patterns (e.g., "overshoots gate 3")
8. Learning curve shapes — where training plateaued, diverged, or showed instability
9. Human focus directive — if the user specified a research direction

**Output requirements**:
- Generate 3-5 candidate hypotheses
- Each must satisfy the `Hypothesis` schema
- Each must pass the diff policy check (no denylist violations, within scope allowlist)
- Rank by: `score = novelty_score * 0.4 + expected_impact * 0.4 + archive_gap_fill * 0.2`
- Select the top-ranked hypothesis that is not duplicate (checked against claims + completed experiments)

### Novelty Scoring

A hypothesis is novel if it differs meaningfully from what's been tried:
- `novelty = 1 - max_similarity(hypothesis, completed_experiments)`
- Similarity is computed from the Hydra override keys changed (for config-only) or the files touched (for code changes)
- Hypotheses identical to a completed experiment score 0 and are filtered out
- Hypotheses similar to a failed experiment get a penalty unless the failure reason suggests a different approach could work

---

## Research Tree

### Structure

Tree nodes = experiments. Stored in `autoresearch/state/tree.json`.

Each node contains:
- Parent node (what it refined/branched from)
- `inspired_by` (optional, for cross-pollinated ideas — references source branch without creating a structural edge)
- Scope level: `hyperparameter | algorithm | architecture | system`
- Hypothesis (full `Hypothesis` object)
- Result summary (fitness, archive cell, pass/fail, W&B run ID)
- Children (refinements that branched from this)

### Hierarchy

- **Root** = current baseline (e.g., `monorace_baseline` PPO + GCNet)
- **Level 1 branches** = major divergences (new architecture, new reward structure, new observation space)
- **Level 2+** = refinements within each major branch

### Branch Selection Algorithm

A single concrete scoring function replaces the menu of strategies:

```python
def score_branch(branch, archive, config) -> float:
    """Score a branch for selection. Higher = more likely to be explored next."""

    # Best fitness achieved by any descendant of this branch
    best_fitness = branch.best_descendant_fitness or float('inf')

    # How many experiments have been run on this branch
    visit_count = branch.total_experiments

    # Archive coverage: what fraction of this branch's target cells are still empty
    empty_target_ratio = branch.empty_target_cells / branch.total_target_cells

    # Behavioral distance from the current global best
    diversity = behavioral_distance(branch.centroid, archive.global_best.descriptors)

    # UCB-style score: exploit good branches, explore under-visited ones
    exploit = 1.0 / best_fitness  # lower lap time = higher exploit score
    explore = config.exploration_weight * sqrt(log(total_experiments) / max(visit_count, 1))

    # Bonuses
    gap_bonus = config.gap_weight * empty_target_ratio  # reward filling archive gaps
    diversity_bonus = config.diversity_weight * diversity  # reward behavioral diversity

    return exploit + explore + gap_bonus + diversity_bonus
```

**Tie-breaking**: if scores are within 5%, select randomly among tied branches.

**Random restarts**: before scoring, roll a random number. If < `random_restart_pct` (default 0.2), skip branch selection entirely and generate a novel hypothesis unrelated to any existing branch.

**Minimum exploration**: branches with `visit_count < min_branch_experiments` (default 3) get `explore = float('inf')`, guaranteeing they are selected before any branch can be deprioritized.

### Pruning

Branches where the best descendant is significantly worse than the archive incumbent (after sufficient experiments) are deprioritized — not deleted. They remain in the tree for reference but the system stops actively exploring them.

---

## Research Loop

### Phase 1 — Situational Awareness

1. `git pull` to sync latest state files
2. Load MAP-Elites archive from `state/archive.json`
3. Load research tree from `state/tree.json`
4. Load active claims from `state/claims.json`
5. Pull recent W&B run history (metrics, configs, termination reasons)
6. Identify gaps: empty cells, cells with poor fitness, under-explored branches
7. Analyze trajectory data (`.npz` files) and Rerun recordings for qualitative patterns

### Phase 2 — Hypothesis Generation

1. Assemble the hypothesis prompt from required inputs (see Hypothesis Generation section)
2. Generate 3-5 candidate hypotheses conforming to the `Hypothesis` schema
3. Validate each candidate:
   - Passes diff policy check (denylist/allowlist)
   - Not a duplicate of active claims or completed experiments
   - Within max diff size for its scope
4. Rank candidates and select the top-scoring non-duplicate
5. Scope-based approval:
   - **Interactive mode**: present top 3 candidates with rationale, let human choose or redirect
   - **Autonomous mode**: hyperparameter + algorithm scope proceed freely; architecture+ escalates to human
   - **YOLO mode**: everything proceeds without approval

### Phase 3 — Implementation

1. Create a git worktree for the experiment (Phase 2+) or a branch (Phase 1 MVP)
2. Implement changes — two strategies depending on scope:
   - **Config-only changes** (hyperparameters, reward weights, DR ranges): generate Hydra override strings passed directly on the command line (e.g., `python -m training control.learning_rate=1e-4 reward.gate_passage=2.0`)
   - **Code changes** (algorithm, architecture, system): edit source files in the worktree, generate a new experiment YAML in `configs/experiment/`, commit to the worktree branch
3. Run diff policy validation — reject and regenerate hypothesis if violations found
4. Write claim to `state/claims.json`, commit and push
5. Record reproducibility snapshot: Hydra config dump, git commit hash, Python/package versions, random seed

Launch training: `python -m training [+experiment=<name>] [hydra overrides...]`

**Failure handling:**
- **Training crash (OOM, NaN, sim instability)**: mark experiment as `failed` in the research tree with crash reason, release coordination claim, log the failure to W&B. Do not retry automatically — Claude Code analyzes the failure to inform the next hypothesis (e.g., "NaN at step 50k suggests learning rate too high").
- **Git worktree creation failure**: fall back to a fresh branch from main. If that also fails (dirty state), abort and report to user.
- **W&B upload failure**: retry once after 30s. If still failing, save results locally — the next successful sync will apply pending updates.
- **Coordination claim expiry during long training**: the monitoring loop (Phase 4) refreshes the claim timestamp every `claim_timeout / 2` hours. If the session dies without cleanup, the claim expires naturally and other instances can pick up the work.
- **State file merge conflict**: for `archive.json`, take the entry with better fitness. For `claims.json`, take the entry with newer timestamp. For `tree.json`, merge additively (new nodes are appended, existing nodes keep the version with more data).

### Phase 4 — Monitoring & Early Stopping

Claude Code monitors training via a polling loop within the active session. The `/auto-research` skill runs a check cycle:
1. Query W&B API for latest metrics of the active run
2. Compute provisional behavioral descriptors from available trajectory `.npz` files
3. Refresh the coordination claim timestamp in `state/claims.json`
4. Early stop if:
   - Learning curve significantly below baseline after budget/3 steps
   - Converging to a well-populated archive cell with a better incumbent
5. Adaptive budget extension if learning curve is still improving at budget limit
6. Sleep for a configurable poll interval (default 60s) and repeat

In autonomous/YOLO modes, this loop is self-sustaining within the Claude Code session. The session must remain active for the loop to run — if the session ends, the experiment continues training but monitoring stops. The next `/auto-research` invocation picks up where it left off by checking W&B for completed/in-progress runs. Training runs are self-contained and log to W&B regardless of whether Claude Code is watching.

### Phase 5 — Evaluation & Archive Update

1. Compute final behavioral descriptors from full trajectory data
2. Run constraint validators (see Constraint Validation section for exact thresholds)
3. Attempt archive insertion (better fitness in target cell?)
4. Update research tree in `state/tree.json`
5. Release coordination claim
6. `git commit && git push` state files
7. Decide next action using branch selection algorithm: refine branch, try adjacent cell, backtrack, or start new branch

### Phase 6 — Reporting

1. Log findings: what worked, what didn't, why
2. If autonomous/YOLO: loop back to Phase 1
3. If interactive: present results and ask for direction

---

## Experiment Budgets

Adaptive and scope-dependent:

| Scope | Initial Budget | Extension Criteria |
|-------|---------------|-------------------|
| Hyperparameter | 5M steps | Still improving at budget |
| Algorithm | 15M steps | Still improving at budget |
| Architecture | 30M steps | Still improving at budget |
| System | 50M steps | Still improving at budget |

Early stop triggers:
- Below 70% of baseline performance after 1/3 of budget
- Converging to an archive cell already occupied by a better solution

**Baseline definition**: the baseline is the best fitness (lap time) in the archive at the time the experiment starts. For the very first experiment (empty archive), no early stopping on baseline threshold is applied — the first run always completes its full budget to establish the baseline. Subsequent experiments compare against the archive's best incumbent in their target cell (or global best if targeting an empty cell).

---

## Constraint Validation & Anti-Gaming

### Hard Constraints (must pass for archive entry)

Each constraint has explicit thresholds and aggregation rules:

1. **Success rate floor**
   - Threshold: >80% of eval episodes complete a lap without crashing
   - Aggregation: over the final 200 eval episodes (or all episodes if fewer)
   - Partial laps: count as failures for this metric

2. **Physics plausibility**
   - Ground/ceiling: reject if >1% of timesteps have z ≤ 0.0m or z > 10.0m
   - Motor limits: reject if any motor command exceeds `max_rpm * 1.05` (5% tolerance for numerical noise)
   - Acceleration: reject if any instantaneous acceleration exceeds `4.0 * g` (max thrust-to-weight ratio for our quad is ~3.8, allowing 5% margin)
   - Quaternion: reject if quaternion norm deviates from 1.0 by more than 0.01 at any timestep

3. **Gate passage quality**
   - Must pass gates in sequential order — reject if any gate is skipped
   - Gate offset: compute distance from drone center to gate center plane at passage timestep (using `gate_events`, `positions`, `gate_positions`, `gate_orientations` from `.npz`). Reject if mean offset > 80% of gate half-extent (clipping edges)
   - Minimum gate clearance: reject if any passage has offset > 95% of gate half-extent

4. **Behavioral sanity**
   - Forward progress: reject if the drone's cumulative distance traveled toward the next gate is negative over any 2-second sliding window (circling/oscillating)
   - Minimum speed: reject if mean speed over completed laps < 2.0 m/s
   - Trajectory diversity: reject if the standard deviation of per-episode lap times < 0.01s across 10+ episodes (suspiciously identical trajectories suggesting degenerate memorization)

5. **Edit surface compliance**
   - The diff must pass the allowlist/denylist check (see Edit Surface Constraints)
   - This is checked before launch (Phase 3) and re-verified before archive insertion

### Soft Signals (logged, don't block)

- Domain randomization robustness: performance degradation across DR range
- Sensitivity to initial conditions: high variance = brittleness
- Reward component breakdown: flag if one component dominates suspiciously (>90% of total reward)

### Implementation

A `ConstraintValidator` class that takes trajectory data + metrics and returns:

```python
@dataclass
class ValidationResult:
    passed: bool
    constraint_results: dict[str, ConstraintCheck]  # per-constraint pass/fail + values
    reasoning: str                                    # Human-readable summary
    soft_signals: dict[str, float]                    # Logged but non-blocking
```

Claude Code reads the reasoning to inform future hypotheses.

---

## Coordination & Anti-Duplication

### State File: `autoresearch/state/claims.json`

```json
{
  "claims": [
    {
      "hypothesis_id": "abc123",
      "researcher": "shaan@laptop",
      "branch": "ar/exp-abc123",
      "target_cells": [[2,1,3], [2,2,3]],
      "scope": "hyperparameter",
      "timestamp": "2026-03-19T14:30:00Z",
      "wandb_run_id": "run_xyz",
      "status": "running"
    }
  ]
}
```

### Claiming Work

- Before starting an experiment, append a claim to `claims.json`, commit, and push
- Other instances pull and check claims before generating hypotheses
- Claims expire after configurable timeout (default 4 hours) for dead session cleanup
- On experiment completion or failure, update claim status to `completed` or `failed`

### Hypothesis Deduplication

- Content hash per hypothesis based on: scope + sorted list of changes + target cells
- Check against active claims and completed experiments in `tree.json`
- "Similar but not identical" is allowed (e.g., lr=1e-3 vs lr=8e-4 are different experiments)
- Identical hypotheses (same hash) are rejected

### Branch Allocation

- In YOLO/autonomous mode, different instances can be assigned different Level 1 branches
- `claims.json` tracks which branches each instance is exploring
- Prefer unclaimed branches over claimed ones when scores are similar

---

## Reproducibility

Every experiment records a reproducibility snapshot:

| Field | Source |
|-------|--------|
| Hydra config (resolved) | `OmegaConf.to_yaml(cfg)` dumped at training start |
| Git commit hash | The worktree/branch HEAD at launch |
| Git diff summary | Lines added/removed per file |
| Random seed | From Hydra config `seed` field |
| Python version | `sys.version` |
| Key package versions | PyTorch, SB3, numpy, wandb |
| W&B run ID | From the training run |
| Hypothesis ID | Links back to the research tree |

Stored as W&B run metadata. Enables any experiment to be reproduced exactly.

---

## Plugin Architecture

### Skills

| Skill | Description |
|-------|-------------|
| `/auto-research` | Main entry point. Args: mode (`interactive`/`autonomous`/`yolo`), optional focus area |
| `/ar-status` | Archive state, active experiments, research tree summary |
| `/ar-review` | Review results from recent experiments with trajectory analysis and W&B links |
| `/ar-branch` | Manually create a new Level 1 branch with human-defined research direction |

### Hooks

| Hook | Trigger | Action |
|------|---------|--------|
| Post-training | After `python -m training` completes | Compute descriptors, validate constraints, attempt archive insertion |
| Pre-commit | Committing experiment results | Validate diff policy compliance |
| Session-start | `/auto-research` launch | `git pull` state files, display archive/coordination summary |

### Configuration

```yaml
autoresearch:
  wandb_project: corvidx-drone-racing
  mode: interactive
  budgets:
    hyperparameter: 5_000_000
    algorithm: 15_000_000
    architecture: 30_000_000
    system: 50_000_000
  early_stop:
    baseline_threshold: 0.7
    archive_redundancy: true
    poll_interval_seconds: 60
  constraints:
    min_success_rate: 0.8
    min_avg_speed: 2.0
    max_gate_offset_ratio: 0.8
    max_acceleration_g: 4.0
  diff_policy:
    max_diff_lines:
      algorithm: 500
      architecture: 1000
      system: 2000
  exploration:
    random_restart_pct: 0.2
    min_branch_experiments: 3
    exploration_weight: 1.0
    gap_weight: 0.5
    diversity_weight: 0.3
  coordination:
    claim_timeout_hours: 4
  branch_selection:
    exploit_weight: 1.0
    explore_weight: 1.0
    tie_threshold: 0.05
```

---

## Operating Modes

| Mode | Approval Gate | Use Case | Available From |
|------|--------------|----------|----------------|
| **Interactive** | All hypotheses require approval | Guided exploration, learning the system | Phase 1 (MVP) |
| **Autonomous** | Hyperparameter + algorithm run freely; architecture+ escalates | Day-to-day research with safety net | Phase 2 |
| **YOLO** | No approval gates, everything runs | Maximum exploration throughput | Phase 2 |

All modes enforce hard constraints (anti-gaming, git isolation, behavioral validation, diff policy). YOLO removes only the human approval gate.

---

## Integration Points

### Existing Infrastructure

- **Hydra configs** — experiments are Hydra config overrides, composable with existing experiment configs
- **W&B** — run tracking, metrics logging, artifact storage (.npz, .rrd), Rerun URL linking
- **Rerun** — trajectory visualization and qualitative analysis via .npz → .rrd pipeline
- **Metrics contract** — `EpisodeMetrics` TypedDict provides the structured data for descriptor computation and constraint validation
- **Git worktrees** — experiment isolation, clean rollback (Phase 2+)
- **Training callbacks** — `GateMetricsCallback`, `TrajectoryRecorderCallback` provide the raw data

### Data Flow

```
Claude Code (hypothesis) → git branch/worktree (isolation) → python -m training (run)
    → W&B (metrics) + .npz (trajectories) + .rrd (Rerun recordings)
    → descriptors + constraints (evaluation)
    → state/archive.json (archive update)
    → state/tree.json (research tree update)
    → git push (sync state)
    → branch selection → loop
```
