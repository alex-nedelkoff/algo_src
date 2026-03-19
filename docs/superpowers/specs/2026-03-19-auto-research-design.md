# Auto-Research System Design

**Date**: 2026-03-19
**Status**: Draft
**Scope**: Claude Code plugin for autonomous research exploration across the algo_src drone racing stack

## Overview

An automated research system that uses Claude Code to systematically explore the solution space for autonomous drone racing. Inspired by DeepMind's AlphaEvolve and Karpathy's autoresearch, the system generates hypotheses, implements changes, runs experiments, and evaluates results — ranging from hyperparameter tweaks to entirely new architectures.

The system uses constrained MAP-Elites to maintain a diverse archive of high-performing solutions across behavioral niches, coordinates across team members via W&B to prevent duplicate work, and enforces hard safety constraints to prevent reward gaming.

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

## Architecture

### Two Layers

**1. `autoresearch/` Python package** — infrastructure utilities Claude Code calls:
- `archive/` — MAP-Elites archive backed by W&B artifacts
- `descriptors/` — Behavioral descriptor computation from trajectory data
- `constraints/` — Hard constraint validators (anti-gaming, physics plausibility)
- `coordination/` — Experiment locking/claiming via W&B to prevent team overlap
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
│   ├── map_elites.py      # Grid storage, cell insertion, artifact sync
│   └── serialization.py   # W&B artifact read/write
├── descriptors/
│   ├── actuator.py        # Axis 1: actuator utilization
│   ├── smoothness.py      # Axis 2: control smoothness
│   ├── aero_regime.py     # Axis 3: speed x angle-of-attack
│   └── compute.py         # Orchestrates all 3 from trajectory data
├── constraints/
│   ├── validator.py       # Hard constraint checks
│   ├── physics.py         # Physics plausibility
│   └── racing.py          # Gate passage quality, behavioral sanity
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
│   └── selection.py       # UCB, diversity bonus, random restarts
└── config.py              # Plugin configuration schema
```

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

### Grid Resolution

Each axis discretized into 5 bins → 125 total cells.
- Actuator utilization: 5 bins across [0.25, 1.0]
- Control smoothness: 5 bins across [0, 1]
- Aero regime index: 5 bins across [0, v_max] where v_max is the physics-derived upper bound = `sqrt(4 * k_thrust * max_omega^2 / max(drag_coeff))` (theoretical terminal velocity from quadratic thrust model). This is a fixed constant for a given vehicle configuration (~25-30 m/s for our quad), ensuring bin boundaries never shift as new experiments are added.

Sparse initially. **Adaptive refinement**: when a cell accumulates 3+ archive insertions (indicating an active region), it can be subdivided into 2×2×2 sub-cells along each axis. Subdivision is recorded in the archive metadata so all instances use consistent binning.

### Cell Contents

Each occupied cell stores:
- Best fitness (lap time) for that behavioral niche
- W&B run ID (links to all metrics, Rerun recordings, config)
- Git commit hash (exact code that produced it)
- Behavioral descriptor values
- Constraint validation results (pass/fail + details)
- Total compute budget spent (timesteps trained) — helps UCB-style selection distinguish "well-explored" from "barely tried" cells

### Storage & Sync

Stored as a versioned W&B artifact (`autoresearch-archive`). Full history preserved.

**Conflict resolution protocol** (optimistic concurrency):
1. Read latest archive version, note its version number
2. Compute new cell insertion (fitness comparison against current incumbent)
3. Before writing, re-read the archive — if version has changed since step 1:
   a. Re-check if the insertion is still valid against the updated archive
   b. If the target cell now has a better incumbent (another writer beat us), skip insertion but still log the result in the research tree
   c. If still valid, write the update
4. W&B artifact versioning provides the version tracking — no external locking needed

---

## Research Tree

### Structure

Tree nodes = experiments. Stored as W&B metadata.

Each node contains:
- Parent node (what it refined/branched from)
- `inspired_by` (optional, for cross-pollinated ideas — references source branch without creating a structural edge)
- Scope level: `hyperparameter | algorithm | architecture | system`
- Hypothesis description (what was tried and why)
- Result summary (fitness, archive cell, pass/fail)
- Children (refinements that branched from this)

### Hierarchy

- **Root** = current baseline (e.g., `monorace_baseline` PPO + GCNet)
- **Level 1 branches** = major divergences (new architecture, new reward structure, new observation space)
- **Level 2+** = refinements within each major branch

### Branch Selection (Anti-Local-Optima)

The system must actively avoid getting stuck in local optima. Strategies:

1. **UCB-style selection** — branches with high best-fitness and/or low visit count get priority (exploration-exploitation balance)
2. **Minimum exploration guarantee** — every Level 1 branch gets at least N experiments (configurable, default 3) before deprioritization
3. **Archive-driven exploration** — empty or poorly-populated MAP-Elites cells represent unexplored behavioral niches; periodically target these gaps
4. **Random restarts** — allocate a percentage of experiments (default 20%) to completely novel hypotheses unrelated to any existing branch
5. **Stale branch revival** — when the dominant branch plateaus, revisit deprioritized branches with fresh ideas
6. **Diversity bonus** — branches behaviorally dissimilar to the current best get a selection bonus
7. **Cross-pollination** — successful refinements on one branch are proposed on other branches. This creates a new node with a single parent (the target branch) and an `inspired_by` reference to the source branch — the tree remains a tree in its parent-child structure, with cross-references as metadata rather than structural edges.

### Pruning

Branches where the best descendant is significantly worse than the archive incumbent (after sufficient experiments) are deprioritized — not deleted. They remain in W&B for reference but the system stops actively exploring them.

---

## Research Loop

### Phase 1 — Situational Awareness

1. Pull latest MAP-Elites archive from W&B
2. Pull recent experiment history (tried, running, failed)
3. Identify gaps: empty cells, cells with poor fitness, under-explored regions
4. Analyze trajectory data (`.npz` files) and Rerun recordings for qualitative patterns
5. Check coordination registry for what other team members are researching

### Phase 2 — Hypothesis Generation

1. Generate candidate hypotheses ranked by expected impact, informed by:
   - Archive gaps and performance patterns
   - Trajectory/Rerun analysis (e.g., "drone overshoots gate 3 consistently")
   - Research tree history (what worked, what didn't, and why)
   - W&B learning curves and reward component breakdowns
2. Each hypothesis specifies: what to change, scope level, target archive cells, predicted behavioral descriptor range
3. Scope-based approval:
   - **Interactive mode**: all hypotheses presented for approval
   - **Autonomous mode**: hyperparameter + algorithm scope proceed freely; architecture+ escalates
   - **YOLO mode**: everything proceeds without approval

### Phase 3 — Implementation

1. Create a git worktree for the experiment
2. Implement changes — two strategies depending on scope:
   - **Config-only changes** (hyperparameters, reward weights, DR ranges): generate Hydra override strings passed directly on the command line (e.g., `python -m training control.learning_rate=1e-4 reward.gate_passage=2.0`)
   - **Code changes** (algorithm, architecture, system): edit source files in the worktree, generate a new experiment YAML in `configs/experiment/`, commit to the worktree branch
3. Claim the experiment in W&B coordination registry
4. Launch training: `python -m training [+experiment=<name>] [hydra overrides...]`

**Failure handling:**
- **Training crash (OOM, NaN, sim instability)**: mark experiment as `failed` in the research tree with crash reason, release coordination claim, log the failure to W&B. Do not retry automatically — Claude Code analyzes the failure to inform the next hypothesis (e.g., "NaN at step 50k suggests learning rate too high").
- **Git worktree creation failure**: fall back to a fresh branch from main. If that also fails (dirty state), abort and report to user.
- **W&B artifact upload failure**: retry once after 30s. If still failing, save archive update locally as JSON and log a warning — the next successful sync will apply pending updates.
- **Coordination claim expiry during long training**: the monitoring loop (Phase 4) refreshes the claim timestamp every `claim_timeout / 2` hours. If the session dies without cleanup, the claim expires naturally and other instances can pick up the work.

### Phase 4 — Monitoring & Early Stopping

Claude Code monitors training via a polling loop within the active session. The `/auto-research` skill runs a check cycle:
1. Query W&B API for latest metrics of the active run
2. Compute provisional behavioral descriptors from available trajectory `.npz` files
3. Refresh the coordination claim timestamp
4. Early stop if:
   - Learning curve significantly below baseline after budget/3 steps
   - Converging to a well-populated archive cell with a better incumbent
5. Adaptive budget extension if learning curve is still improving at budget limit
6. Sleep for a configurable poll interval (default 60s) and repeat

In autonomous/YOLO modes, this loop is self-sustaining within the Claude Code session. The session must remain active for the loop to run — if the session ends, the experiment continues training but monitoring stops. The next `/auto-research` invocation picks up where it left off by checking W&B for completed/in-progress runs.

### Phase 5 — Evaluation & Archive Update

1. Compute final behavioral descriptors from full trajectory data
2. Run constraint validators
3. Attempt archive insertion (better fitness in target cell?)
4. Update research tree in W&B (link to parent hypothesis)
5. Decide next action: refine branch, try adjacent cell, backtrack, or start new branch

### Phase 6 — Reporting

1. Log findings: what worked, what didn't, why
2. Update archive artifact
3. If autonomous/YOLO: loop back to Phase 1
4. If interactive: present results and ask for direction

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

1. **Success rate floor** — minimum % of episodes completing a lap without crashing (default >80%)
2. **Physics plausibility**:
   - No sustained flight below ground or above ceiling
   - Motor commands within physical actuator limits
   - No impossible accelerations (exceeding max thrust-to-weight)
   - Quaternion stability (no attitude divergence)
3. **Gate passage quality** — must fly through gates in sequence, not clip edges or skip. Gate offset distances are computed from drone position and gate geometry at gate passage timesteps (using `gate_events`, `positions`, `gate_positions`, `gate_orientations` from the `.npz` trajectory data). The existing `GateMetricsCallback` tracks passage counts; the constraint validator additionally computes per-gate clearance distances.
4. **Behavioral sanity**:
   - Forward progress required (no circling/oscillating near gates)
   - Average speed above minimum threshold (no hovering exploits)
   - No repeated identical trajectories suggesting degenerate behavior

### Soft Signals (logged, don't block)

- Domain randomization robustness: performance degradation across DR range
- Sensitivity to initial conditions: high variance = brittleness
- Reward component breakdown: flag if one component dominates suspiciously

### Implementation

A `ConstraintValidator` class that takes trajectory data + metrics and returns pass/fail with detailed reasoning. Claude Code reads the reasoning to inform future hypotheses.

---

## Coordination & Anti-Duplication

### Claiming Work

- Before starting an experiment, write a "claim" to the coordination artifact: hypothesis hash, target archive cells, researcher identity, timestamp
- Other instances check claims before generating hypotheses
- Claims expire after configurable timeout (default 4 hours) for dead session cleanup

### Hypothesis Deduplication

- Content hash per hypothesis based on: what's being changed, scope, target behavioral region
- Check against active claims and completed experiments
- "Similar but not identical" is allowed (e.g., lr=1e-3 vs lr=8e-4 are different experiments)

### Branch Allocation

- In YOLO/autonomous mode, different instances can be assigned different Level 1 branches
- Coordination artifact tracks which branches each instance is exploring

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
| Pre-commit | Committing experiment results | Validate coordination claim exists and constraints passed |
| Session-start | `/auto-research` launch | Pull latest archive and coordination state, display summary |

### Configuration

```yaml
autoresearch:
  wandb_project: corvidx-drone-racing
  archive_artifact: autoresearch-archive
  mode: interactive
  budgets:
    hyperparameter: 5_000_000
    algorithm: 15_000_000
    architecture: 30_000_000
    system: 50_000_000
  early_stop:
    baseline_threshold: 0.7
    archive_redundancy: true
  constraints:
    min_success_rate: 0.8
    min_avg_speed: 2.0
  exploration:
    random_restart_pct: 0.2
    min_branch_experiments: 3
  coordination:
    claim_timeout_hours: 4
```

---

## Operating Modes

| Mode | Approval Gate | Use Case |
|------|--------------|----------|
| **Interactive** | All hypotheses require approval | Guided exploration, learning the system |
| **Autonomous** | Hyperparameter + algorithm run freely; architecture+ escalates | Day-to-day research with safety net |
| **YOLO** | No approval gates, everything runs | Maximum exploration throughput |

All modes enforce hard constraints (anti-gaming, git isolation, behavioral validation). YOLO removes only the human approval gate.

---

## Integration Points

### Existing Infrastructure

- **Hydra configs** — experiments are Hydra config overrides, composable with existing experiment configs
- **W&B** — archive storage, coordination, experiment tracking, Rerun URL linking
- **Rerun** — trajectory visualization and qualitative analysis via .npz → .rrd pipeline
- **Metrics contract** — `EpisodeMetrics` TypedDict provides the structured data for descriptor computation and constraint validation
- **Git worktrees** — experiment isolation, clean rollback
- **Training callbacks** — `GateMetricsCallback`, `TrajectoryRecorderCallback` provide the raw data

### Data Flow

```
Claude Code (hypothesis) → git worktree (code changes) → python -m training (run)
    → W&B (metrics) + .npz (trajectories) + .rrd (Rerun recordings)
    → descriptors + constraints (evaluation)
    → MAP-Elites archive (archive update)
    → research tree (decision on next hypothesis)
    → loop
```
