# Vision-Augmented Racing Policy — Design

**Date:** 2026-05-05
**Status:** Spec — pending plan + Linear issue
**Owner:** alex
**Related:** COR-91 (Janahan's exploration / TSDF), COR-96 (sysID), MoE deployment next-steps doc

## Why

The AI Grand Prix sim splits into two phases:

1. **Exploration** (Janahan, COR-91): drone maps the warehouse, builds a TSDF, localizes gates. Output: a frozen world map + gate poses.
2. **Racing**: drone runs laps as fast as possible against the same map.

The exploration output gives us **privileged gate poses** during racing — they don't change between phases. Our existing G&CNet policy already takes gate-relative observations and races at 88 % golden-set completion, so finding gates is **not** the perception problem.

What the existing policy *doesn't* see is anything between the gates: shelving, columns, dynamic clutter, or other drones if it ends up multi-agent. The current racing policy will fly straight through walls. This spec captures the architecture for adding obstacle awareness without throwing away what works.

## Decision: hierarchical pipeline, depth-only vision

```
STATIC (frozen at race start, from exploration):
    • Gate poses (privileged)
    • TSDF / occupancy grid

LIVE (per control step):
    • VIO pose
    • RGB frame
    • IMU

DERIVED:
    • 33-dim gate-relative state  (unchanged from current G&CNet)
    • Depth map from Video Depth Anything (VDA), 10–20 Hz, cached
    • Local obstacle embedding from depth via small CNN

VISION-AWARE PLANNER (5–10 Hz):
    • Inputs: VIO pose, depth, TSDF, next gate index
    • Output: next short-horizon waypoint (next gate by default; detour if blocked)

POLICY (G&CNet, 100 Hz):
    • Inputs: 33-dim state, with waypoint substituted for "current gate"
    • Output: TRPY
```

### Rationale

- **Hierarchical, not end-to-end.** Existing G&CNet at 88 % golden-set works; tossing it for a transformer policy is months of training risk for unclear gain. Hierarchy reuses the policy and isolates vision as a planning concern.
- **Depth-only, no DINO.** For a static map with known gate poses, vision's job is geometric (avoid obstacles), not semantic. Depth gives geometry directly — DINO's semantic embeddings are redundant. Saves Jetson compute and training complexity. If dynamic obstacles or semantic distinctions become important (multi-agent races, etc.), DINO can be added in v2.
- **Video Depth Anything over per-frame DA3.** VDA's temporal consistency matters in a 100 Hz control loop — per-frame depth flickers between consecutive RGB frames, which is the wrong signal for a control loop. VDA smooths this. Inference cost is higher than DA3; needs Jetson benchmarking.
- **G&CNet unchanged at the input layer.** The 33-dim gate-relative obs stays exactly as it is. The "current gate" is just substituted with whatever waypoint the planner emits. This means we can reuse the trained checkpoint with zero retraining; only the planner is new code.

### What we're explicitly *not* doing

- **Not** retraining the G&CNet from scratch with a different obs space. Item from MoE deployment next-steps doc (COR-100 follow-up): we still owe an obs-builder upgrade from 28-dim → 33-dim for AirSim/MAVLink shims. That's separate.
- **Not** adding DINO. Re-evaluate only if obstacle reactions in v1 fail in ways depth alone can't explain.
- **Not** doing closed-loop visual servoing. The waypoint output is the vision-policy interface; tightly-coupled "scrape the wall to make the gate" maneuvers aren't supported by this hierarchy. Acceptable v1 limitation.
- **Not** training a vision-aware policy from scratch with PPO. The planner is classical (cost map + planning) — no RL inside the vision branch.

## Open questions that gate the design

### Q1: VIO + map-matching localization fidelity

The whole hierarchy assumes the drone's pose in the map is accurate enough that the 33-dim gate-relative obs is trustworthy. If VIO drifts >10 cm regularly, the obs frame drifts with it and vision has to take over navigation, not just augment it.

**To resolve:** milestone M1 below.

### Q2: VDA on Jetson Orin NX runtime

Depth quality is the upstream signal for the entire vision branch. If VDA can't run at ≥10 Hz on Jetson with acceptable latency, we either compress it (TensorRT, INT8), drop to per-frame DA3 with a temporal smoothing post-process, or accept lower depth rates.

**To resolve:** milestone M2 below.

## Components (in/out)

| Component | Inputs | Outputs | Rate | Notes |
|---|---|---|---|---|
| **VIO + map-matcher** | RGB, IMU, TSDF | drone pose in world frame | 100 Hz | Janahan's stack or off-the-shelf (ORB-SLAM3 + map alignment) |
| **VDA depth net** | RGB | depth map (H × W) | 10–20 Hz | Cached between frames for control rate |
| **Depth → obstacle embedding** | depth map | feature vector (~64-dim) | 10–20 Hz | Small CNN, trained from scratch or with self-supervised pretext |
| **Vision-aware planner** | drone pose, depth, TSDF, gate index | waypoint (x, y, z, yaw) | 5–10 Hz | Cost-map A* / MPC over the local TSDF + depth-derived occupancy |
| **G&CNet policy** | 33-dim state (waypoint substituted for current gate) | TRPY | 100 Hz | Existing trained checkpoint, unchanged |

## Fallback behavior

If vision is offline / failing for any reason:
- Planner returns "next gate as waypoint" (no detour logic) → policy behaves as it does today (88 % golden-set).
- The pipeline is graceful: vision is additive, not load-bearing for the basic racing case.

If localization confidence drops:
- TBD based on M1 results. Could include: re-localize via gate detection (since gates are visually distinctive) before resuming, or fall back to dead-reckoning + last-known-good gate frame for a short window.

## Milestones (and what each unblocks)

| # | Milestone | Effort | Output | Unblocks |
|---|---|---|---|---|
| **M1** | Localization fidelity benchmark — run exploration → race in sim, measure VIO + map drift over 30 s race | 1–2 days | Drift number; "augmentation" vs "primary navigation" call | Architecture firming for M3-M5 |
| **M2** | VDA on Jetson benchmark — latency, FPS, depth quality on warehouse imagery vs DA3 | 1 day | Effective vision rate, model selection (VDA vs DA3 + smoothing) | M4 (planner needs depth at known rate) |
| **M3** | Waypoint-tracking conversion of G&CNet — item #2 from MoE deployment next-steps | 3 days | Existing policy driven by waypoints, not gates directly | M4 |
| **M4** | Vision-aware planner v0 — cost-map planner taking (pose, depth, TSDF, next gate) → waypoint | 1 week | Hierarchical pipeline end-to-end | M5 |
| **M5** | Closed-loop test in sim — full pipeline on a course with synthetic obstacles | 2–3 days | First obstacle-aware lap; ablation data | Decide v2 (late-fusion ablation, or stop) |

Total ~3 weeks calendar time. Strong dependency chain: M1 + M2 in parallel can run first week, M3 in parallel as well. M4 starts once M3 is done. M5 closes the loop.

## Success criteria

- **MVP**: full pipeline runs without crashes in sim. Lap time within 20 % of the obstacle-free racing time.
- **Target**: lap time within 10 %. Detours actually trigger (not "always go straight to gate"). Visible obstacle avoidance behavior on courses with mid-gate clutter.
- **Stretch**: per-component ablation showing vision actually helps (vs. M3-only baseline).

## Out of scope

- DINO embeddings (re-evaluated in v2 only if needed)
- End-to-end vision policy / transformer architecture
- Multi-agent / dynamic obstacles
- Real-flight testing (sim only for the qualifier)
- AirSim 33-dim shim upgrade (separate work, MoE deployment next-steps item #1)

## Cross-references

- **MoE deployment next-steps:** `docs/superpowers/specs/2026-05-04-moe-deployment-next-steps.md` — items #1 (33-dim shim) and #4 (obstacle awareness) are the two related arcs
- **SysID multi-drone:** `docs/superpowers/artifacts/2026-05-05-sysid-multi-drone-report.html` — capability demo, validates the dynamics layer the policy assumes
- **Janahan's COR-91:** TSDF / frontier exploration — produces the static map this design assumes
- **COR-92:** Janahan's setup notes (camera/depth gotchas — VDA depth correction, etc.)

## Next step

Awaiting go-ahead to produce the implementation plan in `docs/superpowers/plans/`, prioritising M1 + M2 + M3 (week-1 parallel work).
