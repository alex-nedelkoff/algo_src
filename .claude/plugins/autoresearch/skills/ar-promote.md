---
name: ar-promote
description: Promote a winning experiment to the new baseline. Always requires human approval regardless of mode.
---

# AR Promote

Promote a winning experiment to become the new baseline.

**This action is ALWAYS human-gated** — even in YOLO mode.

## Usage

`/ar-promote <hypothesis_id>`

## Workflow

1. **Verify**: Check candidate is in archive with approved status
2. **Compare**: Show old vs new fitness, behavioral descriptors, W&B link
3. **PR**: Create pull request from experiment branch to main
4. **Merge**: Human reviews and merges the PR
5. **Update**: Run `update_state_after_promotion()` to archive old grid and create new baseline
