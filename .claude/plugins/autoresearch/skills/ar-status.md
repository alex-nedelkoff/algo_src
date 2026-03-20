---
name: ar-status
description: Display current auto-research archive state, active experiments, and research tree summary.
---

# AR Status

Show the current state of the auto-research system.

## What to Display

1. **Archive Summary**: Run `python -c` to load `autoresearch/state/archive.json` and show:
   - Total cells: 125 (5x5x5 grid)
   - Occupied cells count
   - Best fitness (lowest lap time) across all cells
   - Pending candidates awaiting review

2. **Active Claims**: Load `autoresearch/state/claims.json` and show:
   - Who is running what
   - Target cells and scope
   - Time since claim was made

3. **Research Tree**: Load `autoresearch/state/tree.json` and show:
   - Current baseline
   - Total experiments (completed/failed/running)
   - Recent experiment results

4. **Cleanup** (with `--cleanup` arg): Remove expired claims from `state/claims.json`.

5. **History** (with `--history` arg): Show baseline promotion history chain.
