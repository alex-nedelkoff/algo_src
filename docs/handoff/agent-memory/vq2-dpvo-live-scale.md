---
name: dpvo-live-scale
description: Live DPVO monocular scale is unreliable and corrupted the gate-1 estimate
metadata: 
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

The live DPVO monocular scale calibration (dpvo_odom_bridge.py, climb-cal
`scale = d_kf/d_dp`) is **unreliable run-to-run and can make gate 1 WORSE than pure
DR** — measured 2026-07-13 (first fresh scored gate-1 attempt, corpus vq2_scored3).

- Scale locked **20.96 m/unit** off a *single* under-constrained cal sample
  (`d_dp=0.068 units, d_kf=1.618 m`). The clean archtest locked **10.87** — a 2×
  swing between runs.
- Under NOFIX=1 (DPVO owns pre-tick position) a wrong scale amplifies DPVO drift
  linearly. Route trace: estimate flew clean to `(3.02,−0.53,−1.33)` on track for
  the pad-locked aperture `[6.22,0.03,−1.3]`, then **jumped +5.6 m in y** in one
  step (bad DPVO fix; GateNet already unloaded at GO so DPVO was the only position
  source) → controller chased it → `LAND: collision (route): imp 3.7`, 0 ticks.
- So the pad lock + spline had the drone within ~0.5 m of track and **DPVO threw
  it 5.6 m off**. For gate 1, DPVO currently hurts.

**Robust scale lock DONE (dpvo_odom_bridge.py, env knobs DPVO_DPMIN/KFMIN/etc.):
removed the timeout lock that froze garbage scale; now requires d_dp>0.12u &
d_kf>1.0m + two-half consistency, else flies DR.** Verified vq2_scored4: locked a
sane 8.64 m/unit (well-constrained cal d_dp=0.229 d_kf=2.4), 0 giveups.

BUT the crash persists — deeper cause: **DPVO's live TRAJECTORY diverges during the
fast gate-1 dive even with a sane scale.** vq2_scored4 route trace: clean DR+spline
to (5.00,−0.68,−1.36) at s=5.18 (only ~0.7 m off the aperture [6.21,0.03,−1.3]),
then over 80 fusions the estimate ran to (14.85,−28.83) → wall (imp 7.5). Both
07-13 scored runs: pad-lock+spline+DR flies gate 1 clean to ~0.7 m, then live DPVO
blows it up. **DPVO currently HURTS gate 1** (short fast dive; DPVO_PATCHES=32 for
VRAM; motion blur / bright gate emissives likely break patch tracking).

**Innovation gate DONE (dpvo_odom_bridge.py, DPVO_MAXJUMP=1.5m): reject a fix if
|p_spawn − kf.p| > MAXJUMP.** Verified vq2_scored5: 42/52 fixes rejected (innov up
to 32) — NO runaway divergence, estimate stayed sane. Good.

BUT still 0 ticks, and it exposed the real wall: with a wrong scale (19.98 that run)
DPVO's positions disagree with the KF → big innovations → the gate rejects nearly
everything → DPVO effectively OFF → drone flies pure blind-chute DR → lands ~1.3 m
off the aperture (−y) and clips the gate post (imp 3.2). Classic blind gate-1 wall.

**Three scored runs, all 0 ticks, one root cause = the live scale is unreliable AND
~2x off (8.64 / 19.98 / 20.96; archtest 10.87).** Climb-cal d_kf/d_dp off a tiny
0.05–0.23-unit DPVO baseline can't pin it. Innovation gate stops the crash-from-
divergence but can't manufacture a good scale.

**GateNet-anchored scale IMPLEMENTED (GNSCALE=1, fly_scored6.bat), untested:**
(1) det_loop keeps GateNet loaded until scale locks or GNSCALE_TMAX (12 s) after GO
instead of unloading immediately; (2) det_loop publishes state['gatenet_p'] = G1_W -
g_w (drift-free metric drone position from the fixed pad-locked gate; does NOT touch
the control KF); (3) bridge calibrates DPVO scale against gatenet_p instead of the
DR/KF position; (4) on lock the bridge sets state['dpvo_scale_locked'] -> GateNet
unloads -> DPVO owns the GPU for the crossing. Robust lock + innovation gate stay on.
VRAM RESULT (vq2_scored6, 07-13): **parallel GateNet + WSL2-DPVO FITS — no OOM.**
GateNet held loaded 12 s past GO alongside DPVO, process survived to normal landing.
So the consecutive buffer-replay is NOT needed for VRAM.

BUT GNSCALE hit a different wall — a GateNet FOV limit, not VRAM: in-flight sightings
were almost all long-range (ranges 14-19 m = far structures/gate-2, rejected by the
<8 m gate-1 filter); only ONE close gate-1 sighting (6.4 m). So gatenet_p never built
a scale baseline -> no_gatenet_p giveup -> DR -> crash, 0 ticks. ROOT: once the drone
LEVELS to fly, the 20°-up nose cam loses the below/ahead gate (same geometry as the
pad wall) -> GateNet can't sustain a close gate-1 lock in flight. Neither parallel nor
consecutive fixes this.

FORK chosen (a): relax gatenet_p range + range-ratio identity. IMPLEMENTED & WORKS
(vq2_scored7, 07-14): det_loop assigns each detection to nearest known gate by
range-ratio (>RANGE_RATIO_MIN), publishes state['gatenet_p']=gate_W-g_w + gate id;
bridge keeps per-gate cal (gcal), calibrates on the gate with the largest DPVO
baseline. Verified: 5 clean gate-1 anchors at 5.6->4.2 m (ratios 0.88-0.99, tagged
gate 0) + 2 gate-2 anchors. Env knobs on the lock: DPVO_NCAL(12), DPVO_DPMIN(0.12).

NEW WALL = TIMING: the drone crashes on the gate-1 approach (blind-chute est-truth
gap, ~4 m out, horiz 0.53 m) BEFORE the scale locks -- got ~5 cal samples, needs
NCAL=12. The gate-1 leg is too short (~6 m / few s) for live DPVO to calibrate AND
correct in time. Chicken-and-egg: DPVO needs baseline flight to calibrate, but the
drone crashes during that flight because DPVO isn't helping yet. 5 scored runs this
session (scored3-7) all 0 ticks, all blind-chute crashes on the gate-1 approach.
Time-align + climb-hold (A+B) DONE & TESTED (vq2_scored8, 07-14): bridge pairs each
lagged anchor with the DPVO pose from its OWN frame (pose_hist + _pose_at, deduped
per frame_ns); RECENTER holds near the pad (safe) creeping to GNSCALE_HOLD_X, waiting
for scale lock. Hold WORKED (no dive-crash, held 18 s at x=0.87). BUT: 13/16 anchors
were GATE 2 (coarsely mapped, 8-17 m) so scale calibrated off the bad gate -> locked
21.04 -> all 193 in-flight DPVO fixes rejected (innov 21-72 m!) -> velocity runaway.
Scale is ~20x off what live DPVO needs in flight.

**ARCHITECTURE VERDICT (systematic-debugging: 3+ fixes each revealing a new failure =
wrong architecture, not a failed hypothesis):** ~8 scored runs, 0 ticks. Fixes tried
= robust lock, innovation gate, gate-anchor, gate-2 anchor, time-align, climb-hold.
Each fixed its target and exposed a NEW failure elsewhere (scale garbage -> divergence
-> gated-out -> blind crash -> latency -> timing-mismatch -> gate-2 corruption ->
runaway). Live DPVO scale is not just noisy but INCONSISTENT between cal and flight
(scale 21 from cal, but flight DPVO displacement implies ~1) -> the live DPVO
trajectory is likely NOT cleanly metric-scalable by one factor (scale drift within a
session). STOP fixing live-scale symptoms. Reconsider: nose-down GateNet servo (fork
b, every historical tick, NO DPVO scale needed for gate 1) with DPVO reserved for
gates 2+/SLAM where there's real baseline. Discussed with Alex 07-14. Related:
[[vq2-perception-latency]], [[vq2-pad-acquisition]], [[dpvo-lietorch-launch-crash]].
RRD recording sink added to vq2wp (RRD=path -> rr.save local .rrd); vq2_scored5.rrd
is a full FPV+3D recording of an UNSUCCESSFUL crossing (no ticked run exists yet).

**ROOT CAUSE FOUND (2026-07-16, offline replay of fg62 via the WSL bridge):** the
scale non-reproducibility is DPVO non-determinism. `DPVO_wsl/config/default.yaml`
sets `CENTROID_SEL_STRAT: 'RANDOM'` -> patch centroids are drawn randomly each run,
so the monocular trajectory (and its arbitrary scale) differs run-to-run even at
IDENTICAL config/corpus. Measured: two back-to-back offline `dpvo_tick_calibrate`
runs on the same fg62 corpus gave scale **8.74 vs 9.38** (and an earlier build 14.22).
This is the same effect as the historical live 10.87 / 20.96 / 21.04 swings — not
noise, not scale drift, but unseeded RNG. **Candidate fix (low-code, untested):** seed
the bridge in `bridge_dpvo._load_runtime` (`torch.manual_seed`, `cudnn.deterministic`)
and add `seed` to `DpvoSessionConfig` (identity already keys resolution/patches/window)
so cal and flight share one deterministic trajectory -> a single fixed scale becomes
valid. Until then DPVO stays OBSERVE-ONLY (scale maps telemetry, not control).

**Latency decomposition (2026-07-16, same replay harness):** WSL transport is NOT the
bottleneck (~20 ms of a ~448 ms/frame live budget). Per-frame cost is DPVO GPU compute
that GROWS with keyframe count (72 ms @ 9 kf -> ~300 ms @ 60 kf, sim-closed) — lietorch
BA, so `cudnn.benchmark` is a no-op. Sim GPU contention adds ~1.5x (300->427 ms live).
Fast config (patches=24 + half-res 320x180 + REMOVAL_WINDOW=12/OPT_WINDOW=6) offline
plateau **300->171 ms sim-closed (~5.8 Hz), ~4 Hz live** — deployed as
`fly_servo_dpvo_fast.bat` + `dpvo_fg62_fast.json` (observe-only). Window params are now
part of `DpvoSessionConfig`/calibration identity (change them -> recalibrate).
Related: [[vq2-perception-latency]], [[dpvo-tracking]], [[dpvo-lietorch-launch-crash]].
