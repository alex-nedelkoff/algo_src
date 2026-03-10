---
name: linear-task
description: Manage Linear issues as persistent context for the Corvidx Drone Grand Prix project. Use when starting or ending a coding session, picking up or creating tasks, tracking implementation progress, updating issue status, or working with Linear in any way. Triggers on mentions of Linear, issues, tasks, COR-* identifiers, session start/end, "what should I work on", progress tracking, or `/linear-task`.
---

# Linear Task Management

MCP server: `linear-grandprix` (if not configured, see [references/setup.md](references/setup.md))

## Workspace

- Team: **Corvidx-drone-grand-prix** (key: `COR`)
- Statuses: Backlog → Todo → In Progress → In Review → Done (also: Canceled, Duplicate)
- Labels: Bug, Improvement, Feature
- `delegate` field: use for agent assignment; `assignee` stays as the human owner

## What Are You Doing?

**Starting a session on a task:**
1. Read the issue via `get_issue` to load context (description, status, relations)
2. Read comments via `list_comments` — comments often contain scope changes, design decisions, and session history that override the original description. This is not optional; `get_issue` does not return comments.
3. Move status to **In Progress** via `save_issue`
4. Follow patterns in [references/session-tracking.md](references/session-tracking.md)

**Reading any issue (even outside a session):**
Always call both `get_issue` AND `list_comments`. The issue description is the original plan; comments are where the plan evolves. Reading only the description means you'll miss critical updates.

**Creating or managing issues:**
- Follow templates and conventions in [references/issue-lifecycle.md](references/issue-lifecycle.md)

**Ending a session:**
1. Post a structured session comment via `create_comment` (template in [references/session-tracking.md](references/session-tracking.md))
2. Update the issue description's Session Notes section with current state
3. Transition status if appropriate (In Progress → In Review, or leave as In Progress)

**Setting up Linear MCP (first time):**
- See [references/setup.md](references/setup.md)

## Core Principles

Issues are the **cross-session memory** for each task. Every issue should contain enough context for a fresh Claude session to pick up exactly where the last one left off.

- **Issue description** = living plan document (update in-place via read-then-write)
- **Comments** = append-only session log (one structured comment per session)
- **Sub-issues** = task decomposition (break large work into child issues)
- **Labels** = routing and categorization
- **Blocking relations** = dependency tracking

## Conventions

### Status Transitions
```
Backlog → Todo → In Progress → In Review → Done
                      ↓
                  (if blocked, comment why and note blocker)
```

### Issue Descriptions
Always preserve existing content when updating — read via `get_issue` first, then write back the full updated description via `save_issue`.

### Comments
Use markdown with code blocks for commands and file paths. Each session comment should be self-contained — a reader should understand the current state without reading prior comments.

### Branch Naming
Linear auto-generates branch names via `gitBranchName` on each issue. Use these.
