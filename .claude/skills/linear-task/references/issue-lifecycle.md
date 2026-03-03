# Issue Lifecycle

Creating, structuring, and managing Linear issues.

## Creating Issues

Use `save_issue` with at minimum `title` and `team`:

```
save_issue(
  title: "Implement gate detection preprocessing",
  team: "Corvidx-drone-grand-prix",
  description: <see template below>,
  labels: ["Feature"],
  priority: 3,        # 1=Urgent, 2=High, 3=Normal, 4=Low
  project: "competition requirements and context"
)
```

### Issue Description Template

```markdown
## Context
<!-- Why this work exists. Link parent issue if applicable. -->

## Objective
<!-- Clear goal. What does "done" look like? -->

## Technical Plan
1. Step one
2. Step two
3. Step three

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Relevant Files
- `path/to/file.py` — description

## Session Notes
<!-- Updated by each session. Current state summary. -->
Not started.
```

## Task Decomposition

Break large issues into sub-issues when a task has 3+ distinct implementation steps:

```
# Parent issue
save_issue(title: "Implement EKF pipeline", team: "Corvidx-drone-grand-prix")

# Sub-issues
save_issue(title: "EKF: prediction step", team: "Corvidx-drone-grand-prix", parentId: "<parent-uuid>")
save_issue(title: "EKF: measurement update", team: "Corvidx-drone-grand-prix", parentId: "<parent-uuid>")
save_issue(title: "EKF: IMU integration tests", team: "Corvidx-drone-grand-prix", parentId: "<parent-uuid>")
```

## Dependencies

Use blocking relations when one issue must complete before another can start:

```
save_issue(id: "<blocked-uuid>", blockedBy: ["<blocker-uuid>"])
```

Before picking up an issue, check if it has unresolved blockers via `get_issue` with `includeRelations: true`.

## Status Management

Transition issues through the workflow:

| Trigger | Action |
|---|---|
| Agent starts working | → **In Progress** |
| Implementation complete, needs review | → **In Review** |
| Review approved / merged | → **Done** |
| Work cannot proceed | Stay **In Progress**, comment with blocker details |
| Issue no longer needed | → **Canceled** |
| Duplicate of another issue | → **Duplicate**, set `duplicateOf` |

## Labels

Apply labels at creation. Current labels:
- **Bug** — defect or regression
- **Improvement** — enhancement to existing functionality
- **Feature** — new capability

Create new labels as needed via `create_issue_label` with descriptive names and a color.

## Linking External Resources

Attach PRs, docs, or references:
```
save_issue(id: "<uuid>", links: [
  {"url": "https://github.com/org/repo/pull/42", "title": "PR #42: Implement X"}
])
```

Links are append-only — existing links are never removed.
