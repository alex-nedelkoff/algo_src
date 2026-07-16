---
name: vq2-vagon-pivot
description: "2026-07-16 pivot — survey plan stopped at 0% impl, moving to Vagon cloud, Janahan supplying 6-gate map JSON"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26d47b9b-e6e4-4754-9d51-4847f5d5a58f
---

**2026-07-16 pivot (Alex's call):** the autonomous-survey/mapping plan
(4-subproject spec + 3 implementation plans, `docs/superpowers/{specs,plans}/
2026-07-16-vq2-*survey*`) was stopped at **100% planning / 0% implementation**
(no `vq2/survey/`, no `vq2/mapping/`, docs-only commits). Alex stopped the
implementing agents.

**New direction:**
- Compute moves to **Vagon** (vagon.io cloud Windows desktop, streamed; per-minute
  billing; bigger GPU than the 4 GB RTX 3050 laptop).
- **Janahan** (teammate, mainrepo `janahanr/*` branches) provides a **map JSON of
  the first 6 gates** — replaces the survey-based map bootstrap.
- Map ingest contract + validated loader built: `docs/vq2-map-json-contract.md`,
  `vq2/map_ingest.py`, `vq2/tests/test_map_ingest.py` (8 tests green).
  Key fact: `vq2wp.py` does NOT read a map JSON at runtime — G1_AP/G2_W/GATES_W
  are hardcoded (vq2wp.py:103-112); course_map_v5.json is reference only. Wiring
  map_ingest into vq2wp is the remaining integration step.

**Why it matters for old constraints:** a bigger Vagon GPU may dissolve the
GateNet+DPVO+sim VRAM co-residency problem (GN_UNLOAD gymnastics). WSL2 is
likely UNAVAILABLE on Vagon (nested virtualization) — the DPVO WSL bridge may
not run there; options: (a) test native Windows DPVO on the new GPU/driver
(the lietorch host-AV was laptop-specific, may not reproduce), (b) run
bridge_dpvo.py on a Linux cloud GPU (RunPod A4000 recipe in handoff) — the
bridge is already TCP, so remote DPVO is a config change (DPVO_BRIDGE host),
not a rewrite. Related: [[dpvo-lietorch-launch-crash]], [[vq2-dpvo-live-scale]],
[[workspace-layout]].
