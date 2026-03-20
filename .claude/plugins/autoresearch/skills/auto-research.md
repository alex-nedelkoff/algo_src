---
name: auto-research
description: Launch the auto-research loop. Analyzes W&B data, proposes hypotheses for hyperparameter changes, runs experiments with human approval.
---

# Auto-Research

Start an automated research session exploring the drone racing algorithm stack.

## Usage
`/auto-research [mode] [focus]`
- mode: `interactive` (default) — all hypotheses require human approval
- focus: optional area to focus on (e.g., "reward shaping", "learning rate", "domain randomization")

## Research Loop

1. **Situational Awareness**: Pull latest state files (`git pull`), load archive from `autoresearch/state/archive.json`, load tree from `autoresearch/state/tree.json`, check active claims in `autoresearch/state/claims.json`. Query W&B for recent run metrics.

2. **Hypothesis Generation**: Analyze archive gaps, recent experiment results, and trajectory data. Generate 3-5 candidate hypotheses for hyperparameter changes (MVP scope). Each hypothesis must specify Hydra override strings, rationale, and target archive cells.

3. **Human Approval**: Present top 3 candidates with rationale, expected impact, and novelty score. Wait for human to choose one or provide direction.

4. **Implementation**: Create git branch `ar/exp-<hypothesis-id>`. Build the training command with Hydra overrides. Write claim to `state/claims.json`, commit and push.

5. **Launch Training**: Run `python -m training [hydra overrides...]`. Monitor via polling loop (check W&B every 60s for metrics).

6. **Evaluation**: On completion, load trajectory .npz files. Run:
   - `python -c "from autoresearch.descriptors.compute import compute_descriptors; from autoresearch.analysis.trajectory import load_trajectory; ..."` to compute behavioral descriptors
   - `python -c "from autoresearch.constraints.validator import ConstraintValidator; ..."` to validate constraints

7. **Archive Update**: If constraints pass, insert result as `candidate` in archive. Present to human for approval/rejection.

8. **Report**: Show results, suggest next hypothesis direction, ask if human wants to continue.

## Key Commands
- Use `python -m pytest tests/test_autoresearch/ -v` to verify the autoresearch package works
- Use `/ar-status` to check current archive and coordination state
- Use `/ar-review` to review pending candidates

## Anti-Gaming Rules
- NEVER modify files in: `sim/dynamics/`, `sim/rewards.py`, `sim/rewards_mavlab.py`, `metrics/`, `training/callbacks.py`, `training/trajectory_recorder.py`, `autoresearch/`, `artifacts/`, `docker/`, `tests/`, `.claude/`
- MVP scope is hyperparameter only — use Hydra override strings, NO source file edits
- All archive insertions require human approval in interactive mode
