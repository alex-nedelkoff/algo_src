---
name: auto-research
description: Launch the auto-research loop. Analyzes W&B data, proposes hypotheses, implements changes, runs experiments with configurable approval gates.
---

# Auto-Research

Start an automated research session exploring the drone racing algorithm stack.

## Usage

`/auto-research [mode] [focus]`

- mode: `interactive` (default), `autonomous`, or `yolo`
- focus: optional area (e.g., "reward shaping", "learning rate", "new policy architecture")

## Modes

### Interactive (default)
All hypotheses require human approval before running. Archive insertions require human review.

### Autonomous
- Hyperparameter + algorithm scope: proceed without approval
- Architecture + system scope: escalate to human for approval
- Use: `/auto-research autonomous [focus]`

### YOLO
- All hypotheses proceed without approval gates
- Hard constraints (diff policy, physics, behavioral) still enforced
- **Baseline promotion (merge to main) is ALWAYS human-gated regardless of mode**
- Use: `/auto-research yolo [focus]`

## Research Loop

### 1. Situational Awareness
- `git pull` to sync state files
- Load archive: `python -c "from autoresearch.archive.serialization import load_archive; a = load_archive('autoresearch/state/archive.json'); print(f'{a.n_occupied}/{a.n_cells} cells occupied')"`
- Load tree: `python -c "from autoresearch.tree.serialization import load_tree; t = load_tree('autoresearch/state/tree.json'); print(f'{t.total_experiments} experiments')"`
- Check claims: `python -c "from autoresearch.coordination.claims import load_claims; c = load_claims('autoresearch/state/claims.json'); print(f'{len([x for x in c if x.status==\"running\"])} active claims')"`
- Check W&B for recent run metrics and completed/orphaned runs

### 2. Branch Selection (Phase 2)
Use the branch selection algorithm to decide where to explore:
```python
from autoresearch.tree.selection import select_next_branch
from autoresearch.tree.serialization import load_tree
from autoresearch.archive.serialization import load_archive

tree = load_tree("autoresearch/state/tree.json")
archive = load_archive("autoresearch/state/archive.json")
config = {
    "exploit_weight": 1.0, "explore_weight": 1.0, "gap_weight": 0.5,
    "diversity_weight": 0.3, "random_restart_pct": 0.2,
    "min_branch_experiments": 3, "tie_threshold": 0.05,
}
branch_id = select_next_branch(tree, archive, config)
# None means random restart — generate a novel hypothesis
```

### 3. Hypothesis Generation
Generate 3-5 candidate hypotheses. For Phase 2, hypotheses can include code changes:

```python
from autoresearch.hypothesis.schema import Hypothesis, HydraOverride, FileDiff

# Config-only (hyperparameter scope)
h = Hypothesis.create(
    scope="hyperparameter",
    description="Increase learning rate",
    changes=[HydraOverride("control.learning_rate", "3e-4")],
    rationale="Current LR may be too conservative",
    parent_id=branch_id,
)

# With code changes (algorithm/architecture/system scope)
h = Hypothesis.create(
    scope="algorithm",
    description="Add entropy bonus to PPO",
    changes=[
        HydraOverride("control.ent_coef", "0.01"),
        FileDiff("control/algorithms/ppo.py", "Add entropy coefficient"),
    ],
    rationale="Improve exploration",
    parent_id=branch_id,
)
```

### 4. Approval Gate
- **Interactive**: present top 3 candidates with rationale, let human choose
- **Autonomous**: auto-approve hyperparameter + algorithm; escalate architecture + system
- **YOLO**: auto-approve all

### 5. Implementation
- **Config-only**: generate Hydra override CLI args via `h.to_cli_overrides()`
- **Code changes**: create git worktree, edit files, validate diff:

```python
from autoresearch.worktree import WorktreeManager
from autoresearch.constraints.diff_policy import DiffPolicy, FileChange

# Create worktree
manager = WorktreeManager()
wt_path = manager.create_worktree(h.id)

# After making code changes, validate the diff
policy = DiffPolicy()
changes = [FileChange("control/algorithms/ppo.py", added=20, removed=5)]
result = policy.validate(changes, scope=h.scope)
if not result.passed:
    print(f"Diff policy violations: {result.violations}")
    # Reject and regenerate hypothesis
```

### 6. Launch Training
Write claim, then launch:
```bash
python -m training [+experiment=<name>] [hydra overrides...]
```

Start durable monitor in background:
```bash
nohup python -m autoresearch.runner --run-id <wandb_run_id> --claim-id <hypothesis_id> --budget <budget> &
```

### 7. Evaluation
On completion, compute descriptors and validate constraints:
```python
from autoresearch.analysis.trajectory import load_trajectory
from autoresearch.descriptors.compute import compute_descriptors
from autoresearch.constraints.validator import ConstraintValidator

traj = load_trajectory("path/to/episode.npz")
desc = compute_descriptors(traj, max_rpm=31470.0, control_freq=100.0)
validator = ConstraintValidator(max_rpm=31470.0)
result = validator.validate(traj, {"success_rate": 0.9, "lap_times": [...]}, dr_active=True)
```

### 8. Archive Update
Insert as candidate, present for review (interactive) or auto-approve (autonomous/yolo).

## Anti-Gaming Rules
- NEVER modify files in: `sim/dynamics/`, `sim/rewards.py`, `sim/rewards_mavlab.py`, `metrics/`, `training/callbacks.py`, `training/trajectory_recorder.py`, `autoresearch/`, `artifacts/`, `docker/`, `.claude/`
- `tests/` files: additions only, no deletions
- All diff changes validated by `DiffPolicy` before launch
- Baseline promotion (merge to main) is ALWAYS human-gated

## Key Commands
- `python -m pytest tests/test_autoresearch/ -v` — verify package works
- `/ar-status` — check archive and coordination state
- `/ar-review` — review pending candidates
