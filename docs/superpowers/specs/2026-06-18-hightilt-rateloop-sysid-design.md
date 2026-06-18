# High-tilt rate-loop sysID — design spec

**Date:** 2026-06-18
**Status:** approved (brainstorming) → ready for implementation plan
**Linear:** prerequisite for COR-136 (push the classic teacher / min-time CPC). Create a child issue under COR-136 at implementation start.
**Sub-project of:** the min-time CPC racing-trajectory effort (Phase 0 of 3).

## Context

The classic analytic teacher races VQ1 in ~21.35s (judge 6/6, 0 collisions). The wall is the
**>45° rate-loop "ring"**: the VQ's *onboard* rate loop is underdamped (EXP-20a: ζ≈0.14, ~5–9 Hz),
and aggressive body-rate setpoints at high tilt ring it up into a runaway (true_tilt → 103° =
tumble). We command the loop only via mavlink `ATTITUDE_TARGET` (body-rate + collective setpoints);
the rate PID itself is firmware we cannot retune.

**Key insight (this is why Phase 0 exists):** 45° is **not** a plant limit. The airframe is a 5"
racing quad (acro at any attitude); the ring is the closed-loop behavior of {underdamped firmware
loop + our command shaping}. Empirically this session, with kdatt + ZVD the *actual* tilt reached
**73° and recovered** (pin-recovery) — the system is not hard-unstable past 45°, it rings and
recovers. So a controller that **models the underdamped loop** (rather than capping tilt statically)
can find command trajectories stable well past 45°.

A min-time CPC trajectory optimizer (Phase 1) wants to exploit exactly that. But the existing
matched rate-loop model was **fit on low-tilt/low-speed data (v0–4)** — it is *least trustworthy
exactly where the CPC would exploit it*. The user chose the aggressive path ("Option B"): push the
CPC to the modeled stability edge (60°+), **gated on first refitting the rate loop in that regime.**
This spec is that refit.

## Goal

Refit the VQ rate-loop model so it is **valid in the >45° tilt / high-body-rate ring regime**, with
a quantified validity envelope and a clear **go/no-go**: either deliver a high-tilt-valid model the
CPC can trust, or determine the regime is not cleanly fittable and the CPC reverts to the
bounded-to-validated "Option A" envelope. Either outcome is useful.

## Scope

**In scope**
- Mine existing live recordings into a labeled high-tilt rate-loop dataset.
- Fit the rate loop in the high regime (linear 2nd-order baseline + a nonlinear/amplitude-dependent
  extension), with command-replay holdout validation.
- An optional minimal live excitation mode (`--ringid`) to fill regime gaps the recordings miss.
- A refit `rate_loop` model + a **validated-envelope descriptor** + a go/no-go report.

**Out of scope (downstream specs)**
- The CPC solver itself (Phase 1) and matched-sim validation (Phase 2).
- Live deploy of any CPC trajectory.
- Retuning/bypassing the firmware rate PID (we only have the setpoint interface).

## Components

Three units, each understandable + testable in isolation.

### 1. `scripts/sysid/ring_id.py` — mine + fit + validate (offline, numpy/scipy; no sim deps)

**1a. Mine.** Parse `vq_data/*.npz` (and the flightlog rows) from prior live runs — the recorder logs
the commanded `ω_cmd` + collective per step; the flightlog logs the measured `ω`, quaternion, and
true tilt. Build a per-axis time-series dataset of `(ω_cmd, ω, tilt, |ω|)` and **bin by tilt and
rate-amplitude**. Report regime coverage and the gaps (which tilt/rate cells are unobserved). Seed
data already exists: this session's tilt-66 detonation (true_tilt 103°) and the tilt-73 recovery
runs (`tch_t44k6`, `tch_t48k7`, `tch_l60`, `p_*`), all in `C:\Users\alexj\Documents\vq_data\`.

**1b. Fit.** Per axis (roll/pitch/yaw), the command→measured rate loop:
- *Baseline:* the current linear 2nd-order (`gain_G`, ωₙ, ζ) — the form in `vq_model.json:rate_loop`.
- *Extension:* **amplitude/tilt-dependent** damping (the documented "nonlinear ring" — ζ and/or gain
  as a function of rate amplitude or tilt; a Hammerstein / quasi-LPV form). Fit on a train split.
- Discrete-time exact-ZOH propagation to match the deploy rate-loop integration.

**1c. Validate.** **Command-replay holdout:** fit on a subset, then *replay* the held-out maneuvers'
`ω_cmd` through the fitted loop and compare predicted vs measured `ω` (and the resulting tilt). This
is the metric that took the low-tilt fit from 53%→95% (`project_vq_matched_imu_holdout`). Validate
against **measured/IMU rate**, not differentiated odometry (the latter inflated holdout 5–8×). Emit a
per-cell holdout error → the validity envelope.

### 2. `--ringid` excitation mode (deploy; only if mining leaves gaps)

Open-loop body-rate **chirp + step sweep** on a chosen axis at progressively higher tilt setpoints,
recorded for ID. Safety: **ramped amplitude + auto-abort** (reuse the deploy's `ABORT_TILT` +
auto-level) so we capture the *onset* of ringing without committing to the runaway; run in a safe
area (hover-and-excite), not on the race course. Modeled directly on the existing `--rollprobe`
open-loop probe. The teacher deploy already has `fresh_start` + the send/abort plumbing; add the
excitation generator behind a flag. Start from mined data so the live campaign is minimal.

### 3. Outputs

- Refit `vq_model.json:rate_loop` (or a sibling `vq_model_hightilt.json`) including any nonlinear
  terms — **byte-identical defaults preserved** so existing low-tilt consumers are unaffected unless
  they opt in.
- **Envelope descriptor** (JSON): the tilt/rate box where holdout error < threshold — the constraint
  the CPC (Phase 1) will be bounded to.
- A report: coverage, fit quality per regime, and the **go/no-go** verdict (Option B viable vs revert
  to Option A).

## Data flow

```
prior vq_data/*.npz  ──┐
                       ├─► ring_id.py: bin by (tilt, |ω|) ─► fit (2nd-order + nonlinear)
[--ringid chirp campaign] ─┘                                   │
                                                               ▼
                                       command-replay holdout (vs measured ω)
                                                               │
                                          ┌────────────────────┴───────────────────┐
                                          ▼                                          ▼
                          refit rate_loop + envelope descriptor            go/no-go report
                                          │
                                          ▼
                              (Phase 1 CPC consumes the refit model + envelope)
```

## Success criteria

- A rate-loop model whose **command-replay holdout** (predicted-vs-measured body rate, per axis)
  clears **R² ≥ 0.9** — parity with the validated low-tilt fit — across a stated tilt/rate envelope
  that extends meaningfully past 45° (**target: usable to ~60° tilt**). — OR
- A clear, evidence-backed **no-go**: the >45° regime cannot clear **R² ≈ 0.85** (too variable /
  nonlinear / plant-variant), reverting the CPC to the bounded-to-validated Option A. The decision
  is per-axis and per-tilt-bin, so a partial result (e.g. valid to 52° not 60°) is a legitimate
  envelope, not a binary fail.
- Either way: the validity envelope is quantified, so Phase 1 is constrained to a regime we trust.

## Risks & mitigations

- **Detonation during live excitation.** → Ramped amplitude + auto-abort + hover/safe-area; minimize
  live probing by mining existing recordings first.
- **The ring may be too variable to fit** (per-run plant variance; the absolute origin/state jumps).
  → This *is* the go/no-go; reverting to Option A is an accepted, useful outcome. Quantify variance
  across runs as part of the report.
- **Holdout pitfall** — differentiated-odometry rate inflates apparent accuracy 5–8×. → Validate
  against measured/IMU body rate only (`project_vq_matched_imu_holdout`).
- **Excitation confounds the firmware loop with our shaping** if closed-loop. → Prefer open-loop
  `--ringid` (raw `ω_cmd → ω`) for the bare-loop ID; use mined closed-loop data for coverage, not for
  the core fit.
- **Sim↔live transfer of the refit.** → Validate against held-out *live* maneuvers, not the matched
  sim (the matched sim is downstream of this model).

## Dependencies / follow-on

- **Follow-on Phase 1:** the CPC min-time solver (casadi/ipopt, full-quad, complementary progress
  constraints) — consumes this refit model + envelope. Separate spec, already brainstormed (see
  session notes / the CPC design outline).
- **Phase 2:** matched-sim validation of the CPC trajectory.
- Reuses: the deploy harness (`vq_deploy_teacher_real.py` plumbing), `vq_data` recordings,
  `vq_model.json` schema.
