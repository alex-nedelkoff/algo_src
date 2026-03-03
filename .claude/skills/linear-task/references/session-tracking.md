# Session Tracking

Patterns for using Linear issues as cross-session context.

## Starting a Session

1. Fetch the issue and its comments:
   ```
   get_issue(id: "COR-XX")
   list_comments(issueId: "<issue-uuid>")
   ```
2. Read the description for the current plan, acceptance criteria, and relevant files.
3. Read the most recent comment for the last session's state (branch, commit, blockers, next steps).
4. Set status to **In Progress**: `save_issue(id: "<uuid>", status: "In Progress")`
5. Set delegate if not already set: `save_issue(id: "<uuid>", delegate: "claude")`
6. Check out the issue's git branch: use `gitBranchName` from the issue.

## During a Session

Post comments for significant milestones — not every minor step, but:
- Completing a major sub-task
- Discovering a bug or unexpected behavior
- Making an architectural decision
- Hitting a blocker

Create sub-issues when discovering new work:
```
save_issue(title: "Fix edge case in X", team: "Corvidx-drone-grand-prix", parentId: "<parent-uuid>", labels: ["Bug"])
```

Link PRs and external resources:
```
save_issue(id: "<uuid>", links: [{"url": "https://github.com/...", "title": "PR #42"}])
```

## Ending a Session

Post a session comment using this structure:

```markdown
## Session Update

### What Was Done
- Implemented X in `src/foo.py`
- Added tests in `tests/test_foo.py` (8/8 passing)
- Discovered issue with Y (created COR-XX)

### Key Decisions
- Chose approach A over B because [reason]

### Current State
- Branch: `janahanr/cor-XX-feature-name`
- Last commit: `abc1234` — "commit message"
- Tests: passing / failing (details)
- Build: clean / errors (details)

### Reproduction
```bash
git checkout janahanr/cor-XX-feature-name
# build/test commands
```

### Next Steps
1. First thing to do next session
2. Second thing
3. Third thing

### Blockers
- None / [describe blocker and link blocking issue]
```

Then update the issue description's **Session Notes** section with a one-line current state summary. Read the full description first to preserve existing content:
```
current_desc = get_issue(id).description
# Append or update the Session Notes section
save_issue(id: "<uuid>", description: updated_desc)
```

## Picking Up Where You Left Off

When starting a new session on an issue with prior work:
1. The last session comment contains the branch, commit, and next steps
2. Check out the branch and verify the commit matches
3. Run the reproduction commands from the comment
4. Start from the "Next Steps" list
