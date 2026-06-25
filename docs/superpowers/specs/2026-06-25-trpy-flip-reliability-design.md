# TRPY gate flier — reliability fixes (COR-139)

**Date:** 2026-06-25
**Branch:** `cor139-live` (laptop) / `alexnedelkoff/cor-139-...` (Mac)
**Scope:** `scripts/trpy_gate_fly.py` only. Aligned RAW path stays intact; the navigator
control law is unchanged (already 221 tests). Validation is **live sim**, not unit tests.

## Problem

`trpy_gate_fly.py` flies the scored VQ1 course on the direct-TRPY engine. Two open
reliability gaps remain (canon STATE 06-24):

1. **Flipped resets (~15%)** score only gate 1.
2. **Aligned resets** are "reliable 6/6" but occasionally miss.

## Diagnostic (TRPY-DIAG, 06-25, N=6)

Ran the unchanged flier 3× aligned + 3× `--forceflip 160` (a *perfect* translation by
construction — isolates anchor precision from any layout distortion).

| Config | Result | Detail |
|---|---|---|
| ALIGNED (RAW) | 6/6 · 6/6 · **5/6** | the 5/6 grinds the **last gate (gi=5)**, ~73 s |
| FORCED-FLIP (anchored) | **1/6 · 1/6 · 0/6** | flip-detect 3/3 correct; anchor residual **3.3 / 3.7 / 5.6 m short** |

**Root cause, locked.** Forced-flip is a perfect translation yet still fails → the *only*
bottleneck is the **gate0 range estimate**. The single far spawn shot reads **3–6 m short
and ±4 m noisy** (aligned vision-front swung −16 → −20.8 on the identical measurement). The
whole layout inherits gate0's range error. Because range bias is a scale effect
(`Z = FX·1.5/mask`), it **shrinks proportionally as the drone closes on the gate** → one
accurate, close-range anchor fixes *all* gates.

The canon's "gate-3/4 descent wall" is **stale** — the current aligned miss is the *last*
gate, a tick-zone margin issue, not a descent wall.

## Fix 1 — flipped anchor precision

Keep TRACK layout + perpendicular-crossing + constant speed. Only the anchor changes:
single-shot → refined + continuous.

### 1a. Gate0 close-approach refine (the core fix)
On a FLIPPED reset (aligned still flies RAW, untouched):
1. Compute the rough spawn anchor as today (±5 m).
2. **Fly a standoff waypoint ~5 m before the rough-anchored gate0**, camera-forward, gentle.
   Standoff > THRU (2.5 m) + margin so the refine never grinds the gate.
3. At that closer range, **accumulate detections ~1.5 s, per-axis median** → refined `vis0`
   (bias ~0.7 m at 5 m vs ~5 m at spawn; median kills the ±4 m noise).
4. Lock refined `anchor = vis0 − gp_of(gates[0])`, recompute the full course, fly it.

### 1b. Per-gate re-anchor insurance
A background watcher localizes the **current target gate**; when a close-range, high-confidence
estimate shifts the anchor by more than a threshold, **abort + re-issue `follow()`** for the
*remaining* (not-yet-passed) gates from the drone's current position (the `Drone` facade already
aborts+joins the prior mission on a new `follow()`).
- For a perfect translation (forced-flip): fires ~once on gate0, then stays quiet.
- For a real flip with rotation/scale: self-heals per gate.
- Sparingly-fired (significant-shift gate only) to avoid the known mission-handoff velocity spike.

## Fix 2 — aligned last gate (gi=5), config-first

The aligned miss is gate 5 grinding ~1/3 of runs; canon notes "arrives ~1.3 m high." Likely the
uniform `+0.8` zr pushes gate5 above the tick zone (dz −0.43..−1.2). **No code first:** bracket
`--zr` per-gate (e.g. `0.8,0.8,0.8,0.8,0.8,0.4` vs `...,1.2`) against the Rerun dashboard to read
whether it grinds high / low / lateral, then tune. Escalate to a gate5-specific approach speed only
if config can't close it.

## Non-goals
- No `vq_gate_wp` full per-gate vision port (arc-calib, frame-sign probes) — translation is confirmed.
- No change to the navigator control law, the aligned RAW path, or the constant-speed recipe.
- Speed (beat ~23 s) is the *next* objective, not this one.

## Validation
- Re-run the TRPY-DIAG batch (3 aligned + 3 forced-flip). Target: **forced-flip → 6/6**,
  aligned stays ≥ today (ideally 3/3 with the gate5 tune).
- Every run streams telemetry to the Mac Rerun (hard rule).
- Append a TRPY-FLIP2 row to the experiment log.

## Outcome (2026-06-25)

The implementation diverged substantially from the plan above — the original "continuous
re-anchor / refinement" hypothesis was wrong about the root cause. What actually shipped
(commit `e5ec6a7`):

**Flipped 6/6 (forced-flip 5/5) via a DIRECT-fly 3-axis anchor**, not a refine detour. The
diagnostic (TRPY-DIAG) and ~13 live forced-flip iterations (collision + per-gate crossing
telemetry) showed flipped failure was three stacked gate0-localization errors on the tight
~1.5 m hole:
- **X (range):** the assumed 1.5 m aperture under-ranged ~18% → range off the KNOWN 2.7 m
  outer red-square width (`localize_front_gate(gate_w=)`, `Z=FX*gate_w/det.w_px`).
- **Z (height):** vision gives the hole CENTER but `zr` is tuned to the gate BASE → pin gate0
  to a base reference (`--gz0 0`) and let gates 1-5 follow the reliable track-RELATIVE z.
- **Y (lateral):** the H_DEFAULT chart's known ~+1 m cross-track bias → `--ybias 0.7`.
- **Fly DIRECT** (no detour): the close-approach "refine" (`--refine`, kept opt-in) dropped the
  drone to low altitude right before gate0 → crossed settled not mid-descent → wrong hole
  height. The natural descent from takeoff = the same dynamics the aligned RAW path uses.

The TRACK frame can be garbage and it still flies 6/6 — the vision anchor + GZ0 + relative
layout fully absorb the jump.

**Not done (deferred):**
- **Real-flip confirmation** — only `--forceflip 160` (perfect, clean-spawn translation) is
  proven. A real natural flip (`ga1`) exposed the open blocker below.
- **Spawn-localize attitude anomaly** — ~13% of resets the spawn vision is anomalous
  (z ~+3 vs −0.9, range off ~20 m, persistent over the whole spawn window) → flipped placement
  lands off → gate0 pin. DEAD END tried: a post-takeoff hover localize (camera needs the 18°
  spawn tilt to frame the gate; sees nothing at level hover). Real fix = in-flight re-anchor
  during the descent (stable attitude + the camera frames the approaching gate) — keep spawn
  localize for flip-DETECT, re-localize + abort/re-issue the course on a confident anchor shift.
- **Aligned gate5 margin** — investigated, found mostly fine (3/3 6/6 this session); the
  baseline's "1/3 grind" was partly misattributed anomalous-spawn flips. No change.
