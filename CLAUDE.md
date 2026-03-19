@AGENTS.md

## Research Vault

Shared research knowledge base at `../research-vault/` (sibling repo). Hooks auto-sync on every write.

**Triage workflow** (progressive disclosure — read only as deep as you need):
1. Read `../research-vault/index/topics.md` — scan one-line summaries to find relevant docs
2. Read frontmatter `one_line` + `relevance` fields to triage without opening full docs
3. Read TL;DR + Key Findings for 80% of the value
4. Only read Technical Details when implementing or deeply evaluating

Use `/literature-review` to add papers or concept notes to the vault.

## Docker

Docker DNS is broken with the default bridge network on this machine (`"iptables": false` in `/etc/docker/daemon.json`). Always use `--network=host` for builds and runs:

```bash
docker build --network=host -t <image> -f <dockerfile> .
docker run --network=host --gpus all <image> <cmd>
```

Docker images are layered: `base.Dockerfile` → `control.Dockerfile` / `sim.Dockerfile` / `perception.Dockerfile`. Build base first:

```bash
docker build --network=host -t algo-src-base -f docker/base.Dockerfile .
docker build --network=host -t algo-src-control -f docker/control.Dockerfile .
```

## Linear MCP

MCP integration `linear-grandprix` provides read/write access to the team's Linear workspace from Claude Code.

- New teammates: run `/linear-task` to get setup instructions
- Issues use the `COR` prefix (COR-1, COR-2, etc.)
- Linear issues serve as cross-session context — see the `linear-task` skill for workflows
- **Rate limits**: Cloudflare WAF blocks writes after rapid bursts (~8+ calls). Reads still work. Batch Linear operations and avoid retrying failed writes — wait for the block to expire (~1 hour).
