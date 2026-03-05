@AGENTS.md

## Research Vault

Shared research knowledge base at `../research-vault/` (sibling repo). Hooks auto-sync on every write.

**Triage workflow** (progressive disclosure — read only as deep as you need):
1. Read `../research-vault/index/topics.md` — scan one-line summaries to find relevant docs
2. Read frontmatter `one_line` + `relevance` fields to triage without opening full docs
3. Read TL;DR + Key Findings for 80% of the value
4. Only read Technical Details when implementing or deeply evaluating

Use `/literature-review` to add papers or concept notes to the vault.

## Linear MCP

MCP integration `linear-grandprix` provides read/write access to the team's Linear workspace from Claude Code.

- New teammates: run `/linear-task` to get setup instructions
- Issues use the `COR` prefix (COR-1, COR-2, etc.)
- Linear issues serve as cross-session context — see the `linear-task` skill for workflows
