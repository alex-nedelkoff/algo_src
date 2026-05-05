# Vision-Augmented Racing — Week 1 Implementation Plan

**Date:** 2026-05-05
**Spec:** `docs/superpowers/specs/2026-05-05-vision-augmented-racing-design.md`
**Scope:** First week of execution. Covers milestones M1 + M2 + M3 from the spec. M4 (planner) and M5 (closed-loop test) are deferred until M1 lands, because M1's "augmentation vs primary" call legitimately changes M4's scope.

## Why week-1-only

Three milestones can run **in parallel** because they have no shared dependencies:

- **M1** needs to **stand up a racing-time localization stack** (VIO + map matcher) — Janahan's exploration only produces offline ground-truth poses, so this is on us. Then benchmark its drift in sim.
- **M2** needs VDA running locally + warehouse RGB samples → independent.
- **M3** is pure refactor of existing G&CNet inputs → independent.

The decision point at end of week 1: M1 result determines whether M4 builds a "vision augments map-based navigation" planner (sub-dm localization → vision is detour finder) or a "vision drives navigation" planner (loose localization → vision must close the gap). Different planners, different effort estimates, different code surface. Writing M4 tasks now would be guesswork.

## Goals at end of week 1

1. **Racing-time localization stack + fidelity number**: drift in cm over a 30 s race run. Decides architecture's primary/augmentation positioning. (M1 grew because we now own the localization stack outright.)
2. **VDA inference cost number**: FPS + per-frame latency on host hardware (Jetson benchmarking deferred — sim qualifier runs on dev box, not Jetson). Decides whether VDA is the depth source or DA3 + temporal smoothing.
3. **Waypoint-tracking G&CNet**: existing policy demonstrably driven by an arbitrary waypoint list, identity-tested against gate-list inputs, off-axis-detour test passes.

## M1: Localization stack + fidelity benchmark (2–3 days)

**Confirmed:** Janahan's exploration only produces offline ground-truth poses; he does not have a racing-time localization stack. We're standing one up.

### Tasks

| # | Task | Effort | DoD |
|---|---|---|---|
| 1.1 | **Pick the VIO stack** — short eval of OpenVINS, ORB-SLAM3, and rtabmap-style options against monocular + IMU inputs we can produce in AirSim. Document the choice with rationale | 2 hours | Decision logged in `docs/superpowers/specs/` |
| 1.2 | **Wrap the picked VIO into a Python interface** — input: RGB frame + IMU stream (rate-matched to AirSim's output), output: pose estimate + confidence. Likely a thin wrapper around an existing implementation | 1 day | `perception/localization/<vio>_wrapper.py` running on captured AirSim sequences |
| 1.3 | **Map-matcher** — fuse VIO pose with the static TSDF / occupancy grid from Janahan's exploration to anchor against the world frame (drift correction). Even a simple ICP between current depth scan and TSDF works for v1 | half day | Pose stream is in world frame (matches gate poses' frame), not VIO-local |
| 1.4 | **Instrument the sim** to log ground-truth pose alongside the localized pose at every step | 2 hours | New logging: `ground_truth_pose`, `localized_pose`, `timestamp_ns`, `localization_confidence` |
| 1.5 | **Run a 30 s race** with the current G&CNet using ground-truth pose for gate-relative obs (control of variables — we want to measure *localization* drift, not policy failure) | 2 hours | `.npz` log of both pose streams over 30 s |
| 1.6 | **Drift analysis script** — `scripts/perception/_localization_drift.py`. Outputs: drift over time plot, mean/p50/p99/max position drift, mean/max yaw drift | 2 hours | One PNG + one JSON summary |
| 1.7 | **Decision write-up** — short markdown in `docs/superpowers/artifacts/` documenting the drift number and the architecture call (augmentation vs primary) | 30 min | Architecture decision recorded |

### Success criteria

- Localization stack runs end-to-end in AirSim sim at the policy's control rate.
- Drift number with known confidence (10+ runs across different seeds / starting poses).
- Clear recommendation logged: "VIO + map matching is good enough at <X cm drift over 30 s, vision can be augmentation" OR "drift is too loose at >X cm, vision must drive navigation primary".

### Risks

- **AirSim's monocular + IMU streams may not match what real VIO libraries expect** (timing, calibration). Mitigation: budget half a day for IMU rate matching and camera intrinsics extraction.
- **Map matcher fidelity** — depth-based ICP against the TSDF requires the depth source from M2. If M2 reveals VDA/DA3 isn't ready in time, fall back to feature-based map matching against gate detections (gates are visually distinctive landmarks).

## M2: VDA inference benchmark (1 day)

### Tasks

| # | Task | Effort | DoD |
|---|---|---|---|
| 2.1 | **Capture warehouse RGB samples** — 100 frames from the AirSim warehouse PIE at varied poses (different rooms, different yaws, edge cases like up-close-to-wall) | 1 hour | `outputs/perception/warehouse_rgb_samples/*.png` |
| 2.2 | **Install VDA + DA3** in the monorace env. Document any install gotchas (CUDA versions, model weights download, etc.) | 1 hour | Both models load and run on the captured samples |
| 2.3 | **Latency benchmark** — wall-clock per-frame inference on host GPU (RTX-class), with and without TensorRT compilation if applicable | 2 hours | Latency table: model × resolution × batch (FP16/INT8 if available) |
| 2.4 | **Quality benchmark** — visual inspection on warehouse samples + temporal-consistency metric across 10 consecutive frames (RMSE between adjacent depth maps for static-scene captures) | 2 hours | One side-by-side comparison image per model + stability number |
| 2.5 | **Recommendation write-up** — VDA vs DA3 + smoothing, with what rate, at what resolution | 30 min | Decision logged |

### Success criteria

- Latency + temporal-consistency numbers for both models.
- Confident pick: which one we use, at what input resolution, expected rate (Hz).
- If both are too slow at any usable resolution: documented fallback (lower-res depth, less frequent updates, etc.).

### Risks

- **VDA's PyTorch checkpoint is GPU-specific.** Compilation to TensorRT is a known headache. Mitigation: skip TRT for the benchmark, accept FP32/FP16 PyTorch as baseline; TRT is a deployment-time optimization.
- **Warehouse imagery is out-of-distribution for both models.** Both were trained on natural-image datasets. Depth quality might be poor on synthetic warehouse textures. If so: document the gap and consider fine-tuning on warehouse imagery (out of scope for week 1, flagged as a follow-up).

## M3: Waypoint-tracking G&CNet (3 days)

### Tasks

| # | Task | Effort | DoD |
|---|---|---|---|
| 3.1 | **`WaypointTrack` adapter class** — wraps a list of `np.array` (3,) waypoints into something that mimics the existing `Track` interface (`gates`, `num_gates`, indexing). Each waypoint becomes a synthetic `GateState(position, orientation)`. | 3 hours | New `sim/tracks_waypoint.py` (or similar). Unit test: round-trip a real Track through WaypointTrack identity-mapping → identical observations |
| 3.2 | **Synthetic yaw heuristic** — at each waypoint k, compute `yaw_k = atan2(wp_{k+2}.y - wp_k.y, wp_{k+2}.x - wp_k.x)` (anticipatory: looking 2 waypoints ahead). EMA-smoothed with `alpha=0.5` | 2 hours | Function tested against a sample waypoint list; visualizes correctly in rerun |
| 3.3 | **Density resampler** — given an arbitrary waypoint plan, resample to ~4 m spacing (matching mean training gate spacing). Cubic spline along arc length, then uniform sampling | 3 hours | Test: input dense waypoint plan (0.5 m spacing) → output ~4 m spacing, smooth, no overshoot |
| 3.4 | **Integration with `AirSimPolicyClient` + `MavlinkPolicyClient`** — accept a `WaypointTrack` in place of the gate `Track`, advance index on 0.75 m radius crossing | 4 hours | Existing eval scripts run unchanged when given a `WaypointTrack` constructed from the original gate positions |
| 3.5 | **Identity test** — run the existing 50-episode eval with `WaypointTrack(gate_positions)` as the track, compare to the original gate-based eval. Should be identical (within numerical noise) | 2 hours | Eval metrics within 1 % of the gate-based baseline |
| 3.6 | **Off-axis detour test** — generate a synthetic waypoint sequence with a 5 m lateral excursion between gates 3 and 4, verify the policy actually follows it (not just cuts straight through) | 2 hours | Trajectory log + visual confirmation that the drone executes the detour |
| 3.7 | **Smoke test in golden-set sim** — run 1 episode of golden-set with waypoints derived from gate centers, verify no regression | 1 hour | Episode completes; gates passed in expected order |

### Success criteria

- `WaypointTrack` is a drop-in replacement for `Track` in the existing eval pipeline.
- Identity test passes — no policy regression when waypoints are gate centers.
- Off-axis detour test passes — policy actually responds to non-trivial waypoint placement.
- Both policy clients (`AirSim`, `MAVLink`) accept the new track type.

### Risks

- **Gate-relative obs assumes gate has an orientation; waypoints don't (intrinsically).** The yaw heuristic is the answer, but it can produce sharp transitions if waypoints are dense. The EMA smoothing helps, but we may need to tune `alpha` or insert intermediate yaw-only waypoints for very sharp turns. Out of scope for v1; flagged as a follow-up.
- **`gate_passage_radius` is hardcoded at 0.75 m in eval scripts.** Need to verify this carries through cleanly to waypoint advance logic; may need a config knob.

## Decision point at end of week 1

After M1, M2, M3 land, we know:
1. Whether vision is augmentation or primary navigation (from M1 drift number).
2. What depth source + rate we have (from M2).
3. That the G&CNet can be driven by waypoints (from M3).

**Then we plan M4 (vision-aware planner) and M5 (closed-loop test):**
- M4 architecture branches on M1: a "detour planner" if vision is augmentation, or a "primary navigator" if vision is the navigation signal.
- M4 uses whatever depth source M2 picks.
- M4 outputs waypoints into the M3 pipeline.

We'll write the M4 + M5 plan once those three numbers exist. Estimated effort for that plan: 1–2 days for M4, 2–3 days for M5, totaling weeks 2–3 of the 3-week budget in the spec.

## Linear

Recommend filing as a new `COR-XXX` issue (parent or sibling to COR-91, COR-100). The spec + this plan together form the issue description. M1, M2, M3 become sub-issues or numbered tasks; M4, M5 added later.

## Cross-references

- **Spec:** `docs/superpowers/specs/2026-05-05-vision-augmented-racing-design.md`
- **MoE deployment next-steps:** `docs/superpowers/specs/2026-05-04-moe-deployment-next-steps.md` — M3 here is item #2 in that doc
- **COR-91:** Janahan's exploration / TSDF (M1 dependency)
- **COR-100:** AirSim perception bring-up (provides the RGB capture pipeline for M2)
- **COR-92:** Janahan's setup notes (camera/depth gotchas)
