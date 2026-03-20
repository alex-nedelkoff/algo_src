---
name: ar-review
description: Review pending auto-research experiment results and approve/reject candidates for archive promotion.
---

# AR Review

Review experiment results pending human approval.

## What to Display

For each candidate entry in the archive:
1. **Hypothesis**: What was changed and why
2. **Fitness**: Lap time achieved vs current cell incumbent
3. **Descriptors**: Actuator utilization, control smoothness, aero regime values
4. **Constraints**: Pass/fail for each constraint with details
5. **Links**: W&B run URL, Rerun trajectory URL
6. **Diff**: The Hydra overrides used

## Actions
- **Approve**: Promote candidate to `approved` status in archive
- **Reject**: Set status to `rejected`
- **Skip**: Leave as candidate for later review

## How to Update
Use `python -c` to load the archive, update the status, and save:
```python
from autoresearch.archive.serialization import load_archive, save_archive
archive = load_archive("autoresearch/state/archive.json")
archive.set_status((cell_i, cell_j, cell_k), "approved")  # or "rejected"
save_archive(archive, "autoresearch/state/archive.json")
```
Then commit and push the state file.
