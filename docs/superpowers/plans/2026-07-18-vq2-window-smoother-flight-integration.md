# VQ2 Window-Smoother Flight Integration (T6c-lite completion)

**Date:** 2026-07-18 · **Branch:** `vq2-estimation` · **Status:** Phase A pending
**Supersedes:** the sequential-EKF aid-consumption thread (measured to its
limit 07-17/18; four circular-gate failures logged, tuning oscillated).
**Reference:** COR-144/T6c `state_estimation_architecture.html` — adopted
~⅔ of its estimation layer, 0% of its Linux infra, plus a pillar-landmark
subsystem it doesn't have. This plan carries the remaining adoptions.

## Measured state (what this plan builds on)

| Metric (tick-truth harness) | Best EKF config | `window_smoother.py` |
|---|---|---|
| Approach: est-to-G1 at judge tick | 1.97–6.83 m | **1.98–4.61 m** |
| Transit: line-dist median post-tick | 9.9–23 m | **0.97–1.76 m** |

- Smoother: batch LS over `[x,y,z,yaw]` @10 Hz, roll/pitch trusted input;
  factors = DR odometry between + gate-anchor unaries (capture-time) +
  pillar bearing/elevation-range (position+yaw coupled, twins as
  min-mixture, **no est-referenced gates**) + G1→G2 corridor unaries.
  5 unit tests incl. retroactive folding + twin disambiguation.
- Pillar map v1 (5 identity-certain entries incl. duplicate '22' twins);
  detector 9–10 ms/frame; OCR reads ~15% of obs (identity-certain).
- Judge-tick decoding: `ENCAPSULATED_DATA` hex, struct `<BQqqIq`
  (gidx2.py errors on these corpora). Ticks: t4 16.6 / t5 20.77 / t6 18.14.
- Anchor availability: 100%/0-gap on test7; ≥84%/≤3 s on well-flown routes.

## Phase A — online sliding-window wrapper (T6c emit pattern) [~1 day]

The last structural T6c adoption: emit the **head pose immediately at
odometry rate**, refine the window behind it, and carry the previous
solution as the next window's prior (naive marginalization: fixed-state
prior on the oldest kept state).

- `vq2/window_smoother.py`: add `SlidingSmoother` class — `push_odom()`,
  `push_anchor()`, `push_pillar()`, `push_corridor()`, `pose()`;
  window 3 s @10 Hz, re-solve on new absolute factor or every 0.3 s.
- Late measurements insert at capture time anywhere in the window
  (already supported by construction).
- Information-driven sigmas (T6c regime switch, done properly): odometry
  `sigma_p` scaled by that interval's flow inlier count; corridor active
  on controller intent (LINE_TRANSIT engaged), not clock.
- **Tests:** sliding vs batch within 0.3 m on synthetic; late-anchor
  insertion mid-window; solve-time budget < 15 ms/solve (target hardware:
  Vagon 4 vCPU — measure, don't assume).
- **Pass:** replay test4/5/6 through the sliding version → within 0.5 m of
  the batch result on both harness metrics.

## Phase B — map completion + duplicate registry [~1 day]

- Tick-anchored surveying (origin = G1 at tick−0..2.5 s, kinematic
  de-rotation via crossing azimuth ≈ 0) extended to every number with ≥5
  reads/corpus; test5 excluded as survey source (est velocity reversed at
  its tick). test7 (100% coverage, longest) surveyed via smoother-pose
  chaining once Phase A lands.
- Grid hypothesis: rows y ≈ {−9.5, +5.3, +12.8}, x-pitch ≈ 9.8 m — fit
  once ≥8 pillars, then snap + extrapolate; keep per-entry provenance.
- Duplicate registry: every number maps to ALL its twins (min-mixture
  consumes them directly).
- **Pass:** every number readable on the racing legs has all visible twins
  mapped; cross-corpus spread < 0.5 m per pillar; rerun harness — transit
  p90 (crash-tail excluded) < 3 m.

## Phase C — 4-DoF gate-corner unaries (last accuracy adoption) [~0.5 day]

T6c's `solve_gravity_constrained` pattern: corners + VP-fixed attitude →
yaw+position unary with χ² gating, replacing the center-vector implieds
(corner data already in detections.jsonl: `corner_xy`, `sigma_diag`,
`visibility`).
- **Pass:** approach G1@tick ≤ 2.0 m on all three ticked corpora.

## Phase D — vq2wp integration, observe-only [~1 day + flights]

- Port detector+OCR to `vq2/pillars.py` (prototypes packaged as .npz);
  smoother runs in-process beside the KF; **no control changes**.
- jlog per solve: smoother pose, KF pose, divergence, health triad
  (step-jump p95, anchor residual, factor counts), pillar read rate.
- **Pass:** 2–3 flights with health green and smoother-vs-tick-truth ≤ 2 m
  live; per-track OCR read rate on the transit measured (the one number
  still unknown).

## Phase E — control handover + spline attempt

- Route follower consumes smoother pose (KF stays as fallback on health
  red — the triad is the switch).
- Fresh-sim, TICKS=2, gidx2-decode scoring per campaign discipline.

## Deferred / conditional

- **DPVO between-factors** (T6c P2): only if the WSL bridge is revalidated
  (tracking is run-dependent; observe-only verdict stands).
- **ESKF attitude-error states:** superseded — the smoother owns the
  entangled-error problem now.
- **Far-panel OCR:** closed (deterministic renders defeat stacking; pad
  pillars are eye-labeled constants; runtime identity = close reads + map).

## Risks

1. Online head-pose degradation vs batch hindsight → measured in Phase A
   pass criterion, T6c accepts the same tradeoff.
2. Flight CPU (4 vCPU shared with sim/GateNet) → solve budget test,
   decimate to 5 Hz states if needed.
3. In-flight OCR under motion → per-track voting; Phase D measures it.
4. Map error → health triad flags, corridor prior bounds the damage,
   KF fallback on red.

## Race-track parallel (unchanged)

Line-follow + FASTGATE gate-2 push continues independently — the qualifier
does not gate on this plan; this plan is the spline-through-gates and
future-course asset.
