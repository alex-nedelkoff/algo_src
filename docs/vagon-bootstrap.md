# Vagon session bootstrap (2026-07-16 pivot)

You are Claude Code running INSIDE a Vagon cloud Windows desktop (streamed).
Everything runs in this VM — SendKeys/screenshots to the sim work exactly like
the laptop-local sessions. Vagon bills per-minute while the machine is on; the
disk persists across sessions. Shut down when idle, never delete.

## What differs from the laptop (read CLAUDE.md for everything else)
- **GPU**: bigger than the laptop's 4 GB RTX 3050. Run `nvidia-smi` first and
  record model/VRAM. If VRAM >= 16 GB, the GateNet+DPVO+sim co-residency wall
  (GN_UNLOAD gymnastics) may be obsolete — verify empirically before removing
  any of it.
- **WSL2 is probably UNAVAILABLE** (cloud VMs usually lack nested
  virtualization). The DPVO live path on the laptop runs through a WSL2 TCP
  bridge. Decision tree:
  1. Try `wsl --install` / `wsl -l -v` — if WSL2 works, replicate the laptop
     recipe (DPVO_wsl checkout + monorace env + bridge_dpvo.py).
  2. Try NATIVE Windows DPVO on this GPU/driver (the lietorch host-AV was
     laptop-specific; port recipe in CLAUDE.md). 10-minute test:
     import torch first, import dpvo from the repo checkout.
  3. Fallback: run `vq2/live/bridge_dpvo.py` on a Linux cloud GPU (RunPod
     A4000 recipe in HANDOFF docs) and point the client at it — the bridge is
     already TCP (`DPVO_BRIDGE_PORT`, host arg in dpvo_odom_bridge/replay).
- **Paths**: keep the repo at `C:\Users\<user>\algo_src` and deploy flight
  files to `C:\Users\<user>\` (same shape as the laptop). If the username is
  not `alexj`, fix the hardcoded `C:/Users/alexj/...` paths in the `vq2/live/
  *.bat` launchers and `bridge_dpvo.py` env defaults when deploying.
- **No Mac tailnet assumption**: check reachability of 100.101.13.126 before
  Obsidian logging; otherwise append to `obsidian_outbox.md` per the standing
  rule.

## First-session checklist (in order)
1. `nvidia-smi` — record GPU model, VRAM, driver.
2. `wsl -l -v` — settles the WSL question (see decision tree above).
3. Install Git for Windows, clone the repo (`origin` =
   corvidx-drone-grand-prix/algo_src-alex), checkout `vq2-estimation`.
4. Install Miniconda; recreate env `monorace`: python 3.13-compatible,
   torch 2.5.1+cu121 (or the CUDA build matching the Vagon driver),
   `transformers==5.10.2` EXACTLY (gatenet ckpt layout — do not upgrade or
   downgrade), rerun-sdk 0.33, numpy, opencv, scipy, pytest.
5. Download + install the AI-GP simulator (v1.0.3385 or the current
   qualifier build). GUI login to the qualifier — without login the judge is
   inert. Verify: hard reset probe, RACE_STATUS decodes, green pad scene.
6. Copy the flight kit from the repo to `C:\Users\<user>\`:
   vq2/live/vq2wp.py, dpvo_odom.py (laptop deploy copies also carry eskf.py,
   gate_traj.py, gidx2.py, crash.py — pull from repo/laptop zip), plus
   `gatenet_b2_cov.pt` and `C:\Users\<user>\DPVO\dpvo.pth` (weights are NOT
   in the repo — bring via Vagon file upload).
7. GateNet smoke test: load the model, run one frame, confirm inference time.
8. DPVO decision-tree test (above). If a bridge runs, replay
   `vq2/dpvo_bridge_replay.py` against a small corpus block and record dt_ms.
9. Run `python -m pytest vq2/tests -q` — all non-skipped tests must pass.
10. Restore agent memory: copy `docs/handoff/agent-memory/*.md` into
    `~/.claude/projects/<this-project-slug>/memory/` so past war lessons load.

## Map JSON (Janahan)
- Contract: `docs/vq2-map-json-contract.md` — spawn frame, meters, pos =
  aperture CENTER, route_order 1..6.
- Validate on receipt: `python -m vq2.map_ingest path/to/map.json` — it
  cross-checks judge-verified G1/G2 anchors and warns on >1 m disagreement
  (judge evidence wins; resolve before flying).
- Integration step (open): wire `MAP_JSON=<path>` into vq2wp.py to replace the
  hardcoded G1_AP/G2_W/GATES_W and feed build_traj. One variable per flight:
  first flight after wiring should change NOTHING else.

## Standing discipline (unchanged from the laptop)
ONE variable per experiment. gidx2 scoring only, fresh sim only. Frames beat
derived telemetry. Check COLLISION + judge clock before trusting probe data.
No unverified success prints. Per-run recording dirs. Restart the sim after
any crash (wrecked drones survive hard resets and poison the next run).
