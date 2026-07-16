---
name: workspace-layout
description: "Which algo_src checkout is the live one, and the 2026-07-13 disk cleanup (what was deleted, what backups exist)"
metadata: 
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

**The live repo is `C:\Users\alexj\algo_src`** — branch `vq2-estimation`, remote `algo_src-alex.git` (Alex's fork), the DPVO/VQ2 work. This is the SOURCE per [[../../CLAUDE.md]]. Everything else that looked like algo_src was a stale duplicate.

**2026-07-13 disk cleanup** (laptop had crashed at 3 GB free / 100% full; recovered to ~205 GB free):
- Deleted 3 stale clones in `C:\Users\alexj\Documents\` (~16 GB): `algo_src` (branch vq-course1, Jun 4), `algo_src_main` (detached, Jun 29), `drone-ai-grand-prix/algo_src` (cor-136 teacher branch). Each held UNIQUE unpushed commits + uncommitted edits (a `look_at_gate`/camera-aim line and a classic-teacher speed line) — all captured as verified git bundles in **`C:\Users\alexj\algo_src_backups\*.bundle`** before deletion. Restore via `git clone <bundle>`; the WIP commits live on `backup/pre-cleanup-2026-07-13` branches inside each bundle.
- Deleted `C:\Program Files\Epic Games` (Fortnite + UE 5.5, ~128 GB) — user-approved, games unrelated to VQ2.
- Cleared regenerable caches: uv (5.3 GB), CrashDumps (1.7 GB). conda pkgs cache is all in-use (nothing to clear).

**NOT touched (all live data, no reclaimable slack — compact returned 0):** Docker `docker_data.vhdx` (98 GB live images/volumes), WSL `Ubuntu` distro (30 GB used, 17 GB in /home — active dev env), WSL `Ubuntu-22.04` (6 GB, 3.5 GB home). Reclaiming any of these means destroying real work; only do it with explicit per-item sign-off. Docker prune needs the daemon started first.
