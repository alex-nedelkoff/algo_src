# DPVO scale review - consolidated findings

**Context:** review of the GateNet-to-DPVO monocular scale experiment, grounded
in the `vq2_servo_dpvo_gnscale_obs2` livelog and the flight code
(`bridge_dpvo.py`, `dpvo_odom_bridge.py`, `dpvo_gate_scale.py`, and
`vq2wp.py`). Date: 2026-07-16.

---

## Bottom line

The intended mechanism is sound: GateNet supplies metric gate-relative drone
positions, timestamp-aligned DPVO translations supply nonmetric motion, and
same-gate relative displacements fit a scalar in metres per DPVO unit.

Obs2 did not set or apply that scale. Four issues now need to be separated:

1. DPVO emitted its first live pose 13.9 seconds after the service was ready.
2. The bridge returned only the newest pose after initialization, although
   DPVO retained earlier accepted keyframe poses internally.
3. The calibrator's forward-only pending-anchor logic discards early anchors if
   historical poses arrive too late or in the wrong order.
4. Obs2 produced only seven gate-0 anchors plus one gate-1 anchor, while
   readiness requires eight supported samples from the same gate.

Issue 4 makes obs2 impossible to unlock retroactively. Offline replay is still
valuable as a pose-yield and timing measurement, not as a way to make that run
pass.

## Verified obs2 facts

- Eight `gnscale_anchor` events were recorded: seven for gate 0 and one for
  gate 1.
- One `dpvo_gate_scale_pair` event was recorded.
- The final `dpvo_gate_scale_fit` reported `pair_count: 0`, `sample_count: 1`,
  `scale: null`, and `ready: false`.
- `pair_count` means usable pairwise scale ratios; it does not mean timestamp
  pair events.
- No GateNet-derived scale entered control. The flight remained on metric dead
  reckoning.
- DPVO survived the GateNet overlap without a GPU/process death.
- The separate 7.3-second model initialization completed before the service
  declared `ready`; it must not be added again to the 13.9-second
  ready-to-first-pose interval.

## Pose supply and recoverable history

The log reports:

> `dpvo_first_pose: ready_to_pose_ms = 13877.2`

Most GateNet anchors therefore occurred before the first live DPVO pose reply.
However, the conclusion that backfill cannot help because those poses never
existed is too strong.

At the first successful reply, DPVO has accumulated the keyframes required for
initialization. The service currently returns only
`slam.pg.poses_[slam.n - 1]`. DPVO retains earlier graph poses and also has
`terminate()`/`get_pose()` machinery for reconstructing accepted and skipped
frame timestamps.

A history reply can therefore recover optimized poses for accepted
pre-emission keyframes. Frames rejected by `motion_probe` may inherit an
identity delta, so not every old image will provide useful motion. The actual
recoverable yield must be measured on the saved corpus.

## Backfill ordering must be coordinated with the calibrator

`GateScaleCalibrator._drain_pending()` is forward-only. When the first newer
pose arrives, a pending anchor that is older than `newest_pose_ns` and lacks a
valid earlier bracket is discarded. In obs2, this drops the pre-first-pose
anchors before a naive late history reply can pair them.

Backfill therefore needs one of two designs:

1. **Atomic oldest-first history delivery:** on initialization, the service
   returns the accepted historical keyframes and the client feeds them to the
   calibrator oldest-first before feeding the newest pose. This requires a
   protocol/service/client change, but not necessarily a calibrator change.
2. **Out-of-order retention:** the calibrator retains pending anchors and
   accepts late historical poses. This requires calibrator semantics and tests
   for bounded retention, ordering, and duplicate handling.

Saying that backfill strictly requires a calibrator change is therefore too
strong. It requires coordinated ordering or calibrator retention.

## The observation windows overlap, but narrowly

Three GateNet anchors occurred after the first emitted DPVO pose timestamp:

- One had an exact DPVO timestamp and paired.
- One had a 62.1 ms preceding bracket and a 298.2 ms following bracket.
- One had a 69.6 ms preceding bracket and a 299.5 ms following bracket.

The latter two exceeded the 250 ms maximum bracket. Raising the tolerance to
300 ms would have produced at most three timestamp pairs, still far short of
readiness. Tolerance alone is not the fix.

## Same-gate count is a hard blocker

`GateScaleCalibrator.estimate()` filters by gate ID. Readiness requires at
least eight supported observations for one gate, where each supported
observation has degree at least two in the inlier graph.

Obs2 supplied:

- Gate 0: seven anchors
- Gate 1: one anchor

Even perfect pose backfill could not satisfy the current threshold. The gate-1
sample may represent the farther known gate or a data-association error; in
either case it cannot contribute to a gate-0 displacement fit.

Future pass/fail reports must include per-gate anchor counts, not only a total.

## Metric baseline is required but not yet measured for obs2

Readiness also requires a supported metric baseline of at least 1.0 m. This is
a real acceptance condition, but the claim that obs2 definitely failed it is
not supported by the recorded data.

The hold intentionally commands a slow forward creep toward `x=1.5`. The run
finished RECENTER at estimated `x=7.67`, and gate-0 range observations spanned
5.3-6.6 m. Those figures do not prove a valid one-metre 3D GateNet baseline:
noise and changing view geometry may contribute to the spread. They do show
that describing the interval as having negligible translation is unjustified.

Only one paired metric GateNet position was logged, so the supported 3D anchor
baseline cannot be reconstructed directly from the current livelog. Treat
baseline as a measurement required in replay/future telemetry, not as an
established obs2 failure.

## Backfilled poses may be revised after initialization

DPVO bundle adjustment can revise earlier keyframe poses as new frames and
factors enter the graph. A history pose emitted immediately at initialization
may therefore be provisional.

The current calibrator deduplicates DPVO timestamps and does not replace a pose
when the same timestamp is later re-emitted with an optimized value. A robust
design must choose and test one of these policies:

- Emit historical poses only after a defined stabilization point.
- Permit pose replacement and recompute affected GateNet/DPVO pairs and the
  scale fit.
- Demonstrate empirically that post-initialization revision is negligible for
  the keyframes used by scale calibration.

This must be resolved before historical poses are allowed to lock control
scale.

## Faster DPVO is a plausible lever, not a demonstrated fix

Obs2 used patches=32, 640x360 input, removal window 22, optimization window 10,
and stride 2. The existing fast configuration uses patches=24, 320x180 input,
and a bounded 12/6 graph. Its lower measured latency should increase the number
of processed frames in the overlap window.

It does not guarantee enough accepted-motion keyframes, correctly identified
same-gate anchors, metric baseline, scale stability, or route accuracy.
Resolution, patch count, and graph size should be evaluated offline and treated
as distinct causal variables where practical.

## A single scale still needs route-level validation

Even a successful near-pad fit establishes only local consistency. Before DPVO
becomes control-ready, estimate scale again in a separated motion window later
in the same session:

- Agreement within a predefined tolerance supports a session-wide scalar.
- Material divergence means one locked scalar is unsafe and requires periodic
  metric anchoring or a different fusion architecture.

A gradual scale change may evade the existing discontinuity and speed gates.
This test remains mandatory after the near-pad estimator first reports ready.

Live GateNet scale has an advantage over fixed offline scale:
`CENTROID_SEL_STRAT` is `RANDOM`, and recorded calibrations have produced
different gauges, including 9.3809 and 14.2203. Per-session metric anchoring
does not assume that an arbitrary monocular gauge transfers across runs.

## Evidence-ordered next tests

1. **Confirm obs2 cannot pass:** complete; seven same-gate anchors are below the
   eight-sample threshold.
2. **Offline keyframe audit:** replay obs2 and expose every accepted
   pre-initialization keyframe timestamp plus its source frame timestamp.
3. **Simulate ordered backfill:** feed recovered history oldest-first before
   the newest pose and report how many of the seven gate-0 anchors become
   timestamp pairs. This is a yield metric, not an obs2 rescue.
4. **Measure pose revision:** compare each backfilled pose at initialization
   with the same pose after later bundle-adjustment updates.
5. **Measure metric baseline:** log or reconstruct all metric GateNet positions
   and compute the supported same-gate 3D baseline.
6. **Offline latency comparison:** compare default and fast DPVO on the same
   frame block, measuring initialization time, pose rate, pair yield, pose
   continuity, and pose revision.
7. **Future same-gate supply test:** demonstrate at least eight correctly
   associated gate-0 anchors with sufficient supported baseline. Do not weaken
   readiness merely to make the test pass.
8. **Two-window scale test:** compare near-pad and later-route scale before any
   GateNet-derived DPVO scale is applied to control.

No further flight should merely rerun the unchanged obs2 pipeline. The saved
corpus should resolve history yield, ordering, pose revision, and fast-config
latency first.

---

## Reply - two concessions, one resolution, one refinement

The consolidation is accurate and improves on the prior round. Two of my earlier
points were overstated and are withdrawn; one open item can now be closed; the
new pose-revision point can be made concrete.

### Concession 1 - backfill does not strictly require a calibrator change
Correct. Atomic oldest-first history delivery (service/client ordering) is a
valid alternative to calibrator retention. "Requires a calibrator change" should
read "requires coordinated ordering OR calibrator retention." Withdrawn as
written.

### Concession 2 - I wrongly asserted near-pad low motion
Correct to challenge it. The gate-0 anchors are not stationary.

### Resolution - the 1.0 m baseline was satisfiable (measured from ranges)
Test 5 can be partly closed now without reconstructing 3D positions. The seven
gate-0 anchor ranges, time-ordered, are:

`6.20, 5.30, 5.40, 5.80, 6.20, 6.10, 6.60 m`  (ratios 0.87-0.98)

Because gate 0 is fixed, `|metric_p_i - metric_p_j| = drone displacement`, and by
the reverse triangle inequality that displacement is `>= |range_i - range_j|`.
The range span is `6.60 - 5.30 = 1.30 m`, so the calibrator's own supported
baseline among these anchors is **>= 1.30 m >= the 1.0 m floor**. Baseline was
therefore NOT a blocker in obs2 (noise-inclusive, since this is the calibrator's
own quantity). The ratios 0.87-0.98 also indicate an internally consistent
gate-0 set that would likely form a clean inlier graph.

Net: obs2 has exactly **one hard structural blocker - the same-gate count
(7 < 8)**. Pose supply, ordering, and baseline are all solvable or already
satisfied; only the count made the run impossible.

### Refinement - pose revision has a concrete stabilization rule
The pose-revision risk is real and is the sharpest new point. It also has a
code-level bound: DPVO only revises a keyframe while it sits inside the active
window. Edges outside `REMOVAL_WINDOW` (22) are marginalized and the windowed BA
optimizes only `t0 = n - OPTIMIZATION_WINDOW` (10) forward. So a keyframe that is
more than ~`REMOVAL_WINDOW` keyframes old is effectively frozen.

That turns policy option 1 ("emit after a stabilization point") into a fixed,
testable rule: **backfill only keyframes older than the removal window**; emit
recent in-window poses live as usual. Test 4 should measure revision magnitude
as a function of keyframe age to confirm the freeze point, which also sizes the
safe backfill lag.

### On the plan
Agreed and well-ordered. Suggest folding this Resolution into test 5 (baseline
lower-bounded from ranges; full 3D reconstruction only if a run gets closer to
the eight-sample threshold) and this Refinement into test 4 (revision-vs-age,
which directly yields the safe backfill lag).

---

## Setting DPVO scale without a GateNet co-observation window

Everything above fits scale from GateNet-vs-DPVO **displacement ratios**, which
requires the two to be live and moving **together** over a window (>= 8 same-gate
anchors, >= 1 m baseline). That co-observation window is the real bottleneck: in
obs2 it never existed because DPVO emitted no poses until 13.9 s in, while GateNet
was only visible near the pad.

The following options set DPVO scale **without** that overlap. None is a config
toggle; they differ sharply in cost and robustness. The root reason any of them
is needed: DPVO initializes every patch inverse-depth to a constant
(`net.py`: `if disps is None: disps = torch.ones(...)`), so the whole
reconstruction lives in an arbitrary gauge until something metric pins it.

### 1. Known-object-size (single-frame gate anchor)
DPVO already reconstructs patch 3D points in its own units
(`pops.point_cloud(...)` in `update()`). On any single frame where the gate is in
view, the patches that fall inside the gate's image region reconstruct the gate's
3D extent **in DPVO units**; scaling so that extent equals the known ~1.5 m
aperture sets scale from **one frame** - no co-movement, no displacement pairing.
GateNet (or any detector) is used only to segment which patches lie on the gate.

- Needs: patch-to-gate association (GateNet bbox/mask), enough patches on the gate
  plane, and those patch depths to be well-constrained. A flat, frontal, or
  distant gate gives weak depth on exactly those patches - the random-init
  quantity - so this is the most GateNet-aligned option but also the most fragile.
  Research effort, not a wiring change.

### 2. Metric depth prior (seed the disparities, GateNet-free)
Replace the `disps = 1.0` initialization with a **metric** monocular depth model
(e.g. Metric3D / Depth-Anything-metric) so DPVO is metric from frame 1 and needs
no GateNet at all.

- Needs: another model resident on the 4 GB GPU - the same budget that forced
  patches=24 - though it can be run only at initialization rather than per frame.
  Metric depth nets carry their own ~5-20% scale error. Upside: this removes
  **both** the co-observation requirement **and** the `CENTROID_SEL_STRAT: RANDOM`
  gauge non-determinism, because scale no longer comes from an arbitrary init.
  Strongest candidate if VRAM can be found.

### 3. Inertial scale (IMU)
Metric scale is observable from accelerometer integration under excitation; a
tight VIO (the existing ESKF fused with DPVO) makes scale observable with no
GateNet.

- Needs: enough acceleration for observability, which is weak on the slow
  `VMAX=0.6` creep, plus the known sim-IMU quirks. Best as a complement to another
  anchor, not a sole source on this short leg.

### 4. Known start motion (free ruler)
Takeoff imposes a repeatable, physically known displacement (~1.0 m y / 0.55 m z
per the campaign notes). DPVO's reading of that known motion at t0 sets scale
against a fixed ruler.

- Needs: the takeoff displacement to be metric-known and repeatable enough to
  trust. Speculative, but nearly free to try and needs no GateNet.

### Ordering and a persistence note
Most robust first: **(2) metric-depth prior** - it also kills the random-gauge
problem - VRAM permitting; then **(4) known-start-motion** as the cheapest probe;
**(1) known-gate-size** is the most GateNet-native but the most fragile; **(3)
IMU** is a complement, weak alone here.

Crucially, options **(2) and (4) also unlock the persisted-config lane**: a metric
or known-motion anchor gives DPVO the **same units every session**, so a scale
written once to a config transfers across flights **without** seeding. The
displacement-ratio and known-gate-size options do not - their gauge is still
per-session unless DPVO is seeded.

---

## Rebuttal - claims that remain unverified or incorrect

The new response closes the ordering dispute, but several later conclusions go
beyond what obs2 and the current DPVO code establish.

### 1. Range span does not prove the supported calibrator baseline

For exact observations of the same fixed gate, the reverse triangle inequality
does show that a 5.3-6.6 m range span permits at least 1.3 m of separation
between the measured metric positions. That establishes a **raw metric-baseline
opportunity**.

It does not establish `GateScaleCalibrator.baseline_m >= 1.3`. The calibrator
reports baseline only on edges whose GateNet/DPVO displacement ratios agree
with the fitted scale and whose endpoints are supported by at least two inlier
edges. Obs2 lacks the required DPVO pairs, so that inlier graph does not exist.

The logged values 0.87-0.98 are gate-identity range ratios. They are not
GateNet/DPVO scale ratios and cannot predict a clean scale inlier graph.
Therefore, obs2 demonstrates enough **potential measured range span**, but not
a supported scale baseline.

### 2. Same-gate count was the impossible condition, not the only live blocker

Seven gate-0 anchors make obs2 mathematically incapable of satisfying the
eight-sample rule. That is the only condition currently proved impossible to
repair by replay.

It does not make pose supply and ingestion ordering non-blockers in the live
pipeline. The unchanged implementation still emits history too late and drains
old anchors before pairing them. Those issues are solvable, but they remain
blocking until the ordered-history design is implemented and verified.

### 3. `REMOVAL_WINDOW` is not yet a proven pose-freeze boundary

The ordinary windowed BA path starts at `n - OPTIMIZATION_WINDOW`, which tends
to leave older poses unchanged. However, DPVO also has a global-BA path that
calls `PatchGraph.normalize()` and can optimize from the minimum active frame
index. Keyframe removal also shifts graph entries and preserves removed-frame
transforms through `delta` links.

Consequently, "older than `REMOVAL_WINDOW` means frozen" is a useful hypothesis
for test 4, not a code-level guarantee. Revision magnitude must be measured by
keyframe age before 22 keyframes is adopted as a safety boundary. A lag that
long would also require anchor retention; atomic initialization-time history
delivery cannot wait for a 22-keyframe freeze and still use forward-only
pending anchors.

### 4. Metric depth is not a drop-in disparity initialization

Changing the `disps = ones(...)` fallback in `net.py` would not make the current
DPVO service metric:

- `dpvo.py` overwrites each new patch's disparity with random values before
  initialization, then uses the recent median after initialization.
- `PatchGraph.normalize()` divides all disparities by their mean and scales
  pose translations by the same factor, explicitly choosing a new monocular
  gauge during global BA.
- A monocular reprojection objective remains scale-invariant unless a metric
  prior is retained as a constraint or every similarity transform is tracked
  and correctly undone.

A metric-depth architecture is still researchable, but it requires changes to
patch initialization, gauge normalization, and/or the optimization objective.
Its feasibility and VRAM cost should not be ranked first until those mechanics
are designed and tested.

### 5. Known start motion does not make a persisted scale transferable

Comparing DPVO motion with a known takeoff displacement can estimate a
per-session multiplier. It does not make the next session's random-centroid
gauge equal to the current session's gauge.

A scale written once becomes transferable only if the known-motion information
deterministically seeds or continuously constrains DPVO's gauge every session.
If it is merely observed and divided into raw DPVO displacement, it is another
per-run calibration source, like GateNet displacement scale.

The campaign notes also describe takeoff displacement as variable and partly
misrepresented by the estimator, so repeatability must be measured before it is
treated as a ruler.

### 6. Known gate extent is not truly a single-frame DPVO solution

The DPVO depths used to reconstruct gate-frame extent arise from multiview
tracking and bundle adjustment, even if GateNet supplies a mask from only one
image. Sparse random patches inside the gate bounding box may land on the
background visible through the aperture rather than on the physical gate
frame. A usable implementation would require ring/corner association, enough
well-constrained gate-frame patches, and rejection of background points.

This option can remove **simultaneous live GateNet/DPVO execution** if the old
mask and keyframe are retained until DPVO initializes. It does not remove the
need for multiview DPVO motion or historical keyframe recovery.

## Resulting decision

The no-overlap alternatives remain research candidates, not replacements for
the evidence-ordered plan. The next decisive work remains:

1. Replay obs2 and expose accepted pre-initialization keyframe timestamps.
2. Simulate oldest-first history ingestion and measure recoverable gate-0 pair
   yield.
3. Measure pose revision versus keyframe age instead of assuming a freeze
   boundary.
4. Compute the actual supported scale graph when enough same-gate pairs exist;
   use range span only as a raw-baseline precondition.
5. Compare default and fast DPVO on the same recorded frames.

No alternative should be promoted to control or persisted-config status until
its metric gauge survives these code-level checks.

---

## Reply - rebuttal accepted, with two code confirmations and one synthesis

All six points stand; my previous round overstated them. Conceding, and adding
the code that makes two of them exact plus the principle they collectively imply.

### Concessions
1. **Baseline.** Correct - the 1.30 m range span is a raw baseline *opportunity*,
   not a satisfied `baseline_m`, which only exists over supported inlier edges
   that obs2 never formed. And the 0.87-0.98 are gate-identity range ratios, not
   scale ratios, so they do not predict a clean inlier graph. Restatement:
   baseline is *not expected to bind once pairs exist*; it is a precondition, not
   a met condition.
2. **Blocker count.** Correct - the 7 < 8 count is the only condition proved
   *unrepairable by replay*. Pose supply and ingestion ordering remain live
   blockers until the ordered-history design ships; "solvable" was the wrong word
   for "still blocking."
3-6. Freeze boundary, metric-depth drop-in, known-motion persistence, and
   single-frame gate extent: all conceded as written below.

### Confirmation A - the freeze boundary is broken by `normalize()`
`patchgraph.py:84` `normalize()` does `s = mean(disps); disps /= s;
poses_[:n,:3] *= s` on **every** global BA - it rescales *all* pose translations,
including keyframes far outside `REMOVAL_WINDOW`. So "older than the removal
window is frozen" is false as a scale guarantee; only the *relative* structure of
marginalized frames is fixed, not their absolute translation. Revision-vs-age
(test 3/4) must be measured, exactly as the rebuttal says.

### Confirmation B - metric-depth needs three changes, not one
Beyond the `disps=ones` fallback, `dpvo.py:427` overwrites each new patch's
disparity with `torch.rand_like(...)` and rescales by the recent median at 429;
`normalize()` then re-gauges globally; and the reprojection objective is
scale-invariant. A metric prior must therefore be injected at patch
initialization **and** retained as a constraint through normalization/BA, or it
is washed out. Downgrading it from "most robust, VRAM permitting" to "most
principled but a real research change."

### Also worth noting - a second non-determinism source
`dpvo.py:427`'s `rand_like` disparity init is a *second* RNG draw on top of
`CENTROID_SEL_STRAT: RANDOM`. Any determinism fix (seeding) must cover both;
`torch.manual_seed` does, but a fix that only addressed centroid selection would
not make the gauge reproducible.

### Synthesis - why every one-shot anchor is per-session
The rebuttal's points 4 and 5 generalize to one rule:

> A persisted, cross-session scale requires **either** a deterministic gauge
> (seed both RNG draws) **or** a metric constraint retained through
> `normalize()`/BA. Any *one-shot* anchor - GateNet displacement, known gate
> size, known takeoff motion - is inherently **per-session**, because random
> patch init (`dpvo.py:427`) and `normalize()` re-select the gauge every run.

So none of the no-overlap options "unlocks the persisted-config lane" on its own;
that claim is withdrawn. They are per-session calibration *sources*, exactly like
GateNet displacement - differing only in whether they need co-observation, not in
whether they transfer.

### Bearing on the central question
`normalize()` also re-normalizes mean inverse-depth to 1 on every global BA. As
the scene depth distribution changes along the leg (near gate -> open transit),
the normalization constant `s` shifts, which is a concrete code-level mechanism
for **within-session scale drift** - the thread's core unknown. This does not
prove drift occurs, but it means the **two-window scale test is the decisive
measurement**, not an optional one. Fully agree with the resulting decision and
the evidence-ordered plan; the corpus can answer all of it offline.

---

## Full-course traversal architecture and plan

This scale and pose-history work is the odometry foundation for a full-course
architecture. It becomes reusable beyond gates 1 and 2 only after DPVO's gauge
is stabilized and its metric scale is shown to hold across separated route
windows.

### Intended estimator and navigation stack

```text
IMU -> high-rate ESKF propagation
             ^
DPVO -> continuous visual position corrections
             ^
GateNet -> sparse metric anchors and gate verification
             ^
Offline map -> expected gate locations and reference trajectory
```

The sources have distinct responsibilities:

- The **offline map** supplies course topology, expected gate poses, and the
  reference line.
- **DPVO** supplies continuous relative visual motion between gates.
- **GateNet** supplies sparse metric anchors, confirms gate identity, and
  supports recovery; it should not be the continuous control signal.
- The **ESKF** propagates attitude, velocity, and position at control rate and
  bridges short visual gaps.
- **Judge ticks** are the authoritative course-progress signal and advance the
  route state.

No single source should be trusted alone for a full traversal.

### How the current work contributes

- **Timestamped backfill** can make early GateNet observations usable after
  DPVO's delayed initialization.
- **Normalization telemetry or compensation** can express DPVO output in one
  canonical gauge instead of allowing global BA to silently rescale the route.
- **Per-session GateNet anchoring** can convert the session's arbitrary DPVO
  translation units into metres.
- **Two-window scale validation** determines whether a scale fitted near the
  pad remains valid during later transit.
- **GPU overlap testing** establishes whether GateNet can be invoked
  selectively without killing or starving DPVO.
- **Pose-age, pose-revision, and tracking-health measurements** provide the
  gates required before DPVO measurements enter control.

### Full-course operating cycle

1. At the pad, GateNet establishes the metric relationship to gate 1.
2. DPVO initializes and returns accepted historical keyframes in safe timestamp
   order.
3. Same-gate GateNet/DPVO pairs fit and validate the current session's scale.
4. The canonical, scaled DPVO trajectory is aligned to the offline course map.
5. GateNet becomes sparse: run it near predicted gate windows or when recovery
   is needed, rather than using it continuously for control.
6. During transit, the ESKF propagates at control rate and DPVO supplies visual
   corrections.
7. Before each crossing, the map predicts where the gate should appear;
   GateNet verifies its identity and refines the local gate-relative state.
8. The controller follows a gate-centred reference corridor through the
   aperture.
9. A judge tick confirms the crossing and advances the route to the next gate.
10. If map, DPVO, GateNet, and judge state disagree, the system re-observes,
    relocalizes, or commands a go-around instead of continuing blindly.

### Work remaining before full traversal

1. Produce canonical DPVO poses despite `PatchGraph.normalize()` gauge changes.
2. Prove metric scale consistency across at least two separated route windows.
3. Build a metric gate map and reference line for the complete course.
4. Align each new DPVO session to that map with an explicit session transform.
5. Add tracking-loss detection, restart/reinitialization, and map
   relocalization.
6. Add periodic gate re-anchoring so map error and odometry drift cannot grow
   unchecked.
7. Validate GateNet/DPVO GPU and compute scheduling over a full-length flight.
8. Define control acceptance thresholds for pose age, tracking health, scale
   confidence, pose discontinuity, drift, and gate-relative error.
9. Validate the controller progressively: offline replay, observe-only flight,
   gates 1-2 control, then extend one verified segment at a time.

### Evidence gates for calling DPVO control-ready

- Historical pose yield and ordering are deterministic and timestamp-correct.
- Pose revision is bounded or incorporated through replacement/recomputation.
- Global-BA normalization cannot silently invalidate the active metric scale.
- The near-pad and later-route scale estimates agree within a tolerance derived
  from the gate-crossing error budget.
- DPVO updates remain fresh and continuous at the estimator's consumption rate.
- Short outages degrade safely to ESKF propagation; longer outages trigger
  recovery instead of blind flight.
- Repeated recorded-route trials stay within the planned gate corridors before
  any live full-course attempt.

### Resulting outlook

Once canonical gauge and two-window scale stability pass, extension beyond
gates 1 and 2 becomes primarily a course mapping, trajectory generation, and
recovery problem. It should not require inventing a new odometry solution for
each gate. The intended long-term architecture is continuous DPVO/ESKF motion
against a metric offline map, with GateNet and judge ticks providing sparse
semantic and authoritative corrections throughout the course.

---

## Reply - architecture endorsed; three dependency-order points

The stack is sound and matches the documented target (offline reference line +
continuous DPVO/ESKF, GateNet sparse, judge ticks authoritative). The plan's
content is really its *ordering*, and three things there should be made explicit.

### 1. Work-items 1 and 2 are blocking foundations, not parallel tasks
"Produce canonical DPVO poses despite `normalize()`" and "two-window scale
stability" are not line items beside mapping and control - the rest stands on
them, and both are currently **unsolved with no known in-place fix**:

- `normalize()` (`patchgraph.py:84`) has no free cure. Disabling it risks DPVO's
  numerical conditioning; compensating means tracking every per-BA `s` and
  undoing it on the published route. Until one is built and tested, cycle step 4
  (map alignment) and evidence gate "normalization cannot invalidate scale"
  cannot even be attempted.
- If the two-window test shows drift - and `normalize()` is a concrete mechanism
  for it - a single session scalar is dead, and "scaled DPVO trajectory aligned
  to map" must be replaced by continuous map-relative re-anchoring (work-item 6).
  That is an estimator change, not a threshold, so the plan should pre-commit to
  the branch rather than treat drift as a pass/fail leaf.

Sequence 1-2 as gates; everything downstream is contingent on them.

### 2. Map provenance is circular - name the bootstrap
The metric gate map and reference line (cycle step 4, work-item 3) can only be
built from a run that already has metric ground truth: judge-ticked crossings
plus seeded DPVO over that run's frames. But ticking the gates is the very thing
this DPVO effort exists to enable. So the map cannot be the first deliverable.

The campaign already implies the resolution: the first gate-1/2 ticks come from a
**DPVO-free path** (nose-down GateNet/fastgate servo, no scale needed), and the
metric map + DPVO route are a **second phase bootstrapped from those ticked
frames**. State this explicitly - DPVO full-course is downstream of first getting
ticks by other means, not the means to the first ticks.

### 3. The cold-start is a one-time startup gap, and it is exactly what obs2 hit
"ESKF bridges short visual gaps" is right mid-flight but understates startup.
DPVO's 13.9 s ready-to-first-pose (obs2) is a **one-time** cost - the session
persists across ticks, so it is not per-leg - but it must be fully closed by
prewarm **before the gate-1 metric-anchor window**, or obs2 reproduces exactly
(anchors land during the cold start with no DPVO to pair). Make "prewarm
completes cold-start before the first anchor window" an explicit, measured
precondition, not an assumption.

Also record the measured rate ceiling: ~2-5 Hz (fast config, sim-closed). At
`VMAX=0.6` that is a fix every ~12-30 cm - fine for the slow qualifier transit,
but it caps safe transit speed, so the trajectory generator must respect that
ceiling rather than assume "continuous."

### Agreement
The evidence gates and the progressive validation ladder (offline -> observe ->
gates 1-2 control -> extend one verified segment) are the right discipline and
match the campaign's rules. With work-items 1-2 treated as blocking foundations,
the map bootstrap made explicit, and cold-start a measured precondition, this is
the right long-term shape.

---

## Map anchoring - the concrete "how" behind work-item 4

Work-item 4 says "align each new DPVO session to the map with an explicit session
transform" but does not say how. It is two separate anchorings in two places, and
keeping them separate is what avoids repeating the obs2 struggle.

### Anchoring 1 (offline): map -> world
Build the map from a ticked reference run: run DPVO **seeded** (deterministic) ->
trajectory + point cloud in DPVO units, then metric-align to world/spawn frame
using all available data with no real-time pressure:

- judge ticks pin along-course position at each gate,
- gate PnP + known gate geometry pin gate poses,
- the deterministic spawn pins the start.

This is where ticks and gates do their anchoring - offline, data-abundant, robust.
It **inverts obs2**: the hard metric alignment moves to where data is unlimited,
instead of fighting for eight live GateNet/DPVO pairs in a tiny window.

### Anchoring 2 (online): flight -> map
The live DPVO session lives in its own arbitrary frame (origin, orientation, and
**scale**). Anchoring the flight into the map = estimating the **similarity
transform T** (3 rot + 3 trans + 1 scale = 7 DOF) from live-DPVO frame to map
frame. Two parts:

- **Rotation + translation - nearly free.** The sim spawns identically every run,
  so the t0 world pose is known and bootstraps rot+trans immediately.
- **Scale - the hard DOF.** Spawn says nothing about DPVO's gauge, and it drifts
  (`normalize()`). This is the unsolved thread problem.

The mechanism that supplies scale **and** corrects drift is **visual
relocalization against the map**: match live keyframes to map keyframes (place
recognition) -> 3D/2D correspondences -> solve the **sim3** transform (standard
prior-map relocalization / loop-closure sim3). Because DPVO's scaled points are
aligned to the map's *metric* points, **the scale ratio falls out of the
alignment** - every relocalization re-pins the gauge.

Punchline: relocalizing against a metric map **is** the "metric constraint
retained through `normalize()`" that the scale synthesis called for. Scale stops
being a fit from a sparse gate window and becomes a **byproduct of localizing**,
computed continuously from the whole environment - which also supplies absolute
position between gates (the long-interval problem).

### Online loop
1. Spawn -> initialize T (rot+trans) from the known pad pose.
2. Fly the map-relative reference line on DPVO/ESKF.
3. Continuously relocalize live frames against the map -> refine T including
   scale -> correct drift and re-pin the gauge.
4. Reinforce at gates with PnP; zero the progress axis at each tick.

### Obstacles (so this is not hand-waving)
- **Compute.** Online relocalization needs the environment map + a matcher live on
  the 4 GB GPU - the budget that forced patches=24 and made loop closure be
  disabled (`LOOP_CLOSURE=False`). The capability exists but is switched off;
  project pieces (MASt3R-SLAM worktree, BoQ-VPR pipeline) are infrastructure to
  wire, not invent.
- **Relocalization frequency vs drift.** Scale drifts *between* relocalizations,
  so they must be frequent enough to re-pin before drift blows the
  gate-acquisition budget.
- **Bootstrap.** Building the first map needs one good ticked run, which again
  points to first ticks via the DPVO-free fastgate/servo path.

### Why this closes the loop
Gate-only anchoring (PnP + tick) is sparse and FOV-limited, so between gates the
system dead-reckons on the drifting gauge and long intervals blow the drift
budget. Environment relocalization against a metric map converts sparse gate
anchors into continuous absolute fixes **and** pins scale as a byproduct - the
same mechanism resolves the long-interval problem and the scale problem at once.
This is a stronger claim than work-item 6's "periodic gate re-anchoring": the
re-anchoring must be against the mapped *environment*, not only gates.

---

## Review - map relocalization is promising, with six corrections

The prior-map relocalization direction is likely the strongest full-course
global layer: DPVO/ESKF supplies local motion, while image-to-map localization
supplies periodic absolute metric fixes. The proposal should not be adopted
unchanged, however.

### 1. Map bootstrap is not necessarily circular

The campaign already has the banked fg75 gate-2 tick traversal. That recorded
run can seed an offline map/reference experiment now; obtaining another
DPVO-free tick is not a prerequisite.

More generally, judge ticks validate course progress but do not by themselves
provide every metric map coordinate. Known gate geometry, gate-relative pose,
spawn pose, and offline multi-view optimization can metrically constrain a map.
A judge-blessed run is valuable evidence and a reference-line source, not the
only possible source of metric geometry.

The proposed fallback to a DPVO-free nose-down GateNet/fastgate first-tick path
also conflicts with campaign evidence: blind/GateNet-centre approaches were
unreliable because detection centre did not consistently match the judge
aperture. Use the already banked ticked data before reopening that path.

### 2. Prewarm cannot eliminate motion initialization

Obs2 separated two delays:

- Model/service initialization: approximately 7.3 seconds, completed before
  the service declared `ready`.
- Ready-to-first-pose: approximately 13.9 seconds, dominated by DPVO's need for
  accepted moving keyframes.

Model prewarm can remove the first delay from the race window. It cannot make a
stationary DPVO graph initialize. The live remedy is to feed motion as early as
possible, retain the source timestamps, and recover accepted initialization
history in safe order. Calling the entire 13.9-second interval a prewarmable
cold-start gap would reproduce the obs2 diagnosis error.

The delay is one-time per DPVO session, but the required motion and historical
pose delivery remain explicit startup acceptance conditions.

### 3. PnP and Sim3 are different online localization pipelines

The proposal conflates two geometries:

1. **Metric-map 3D points + live-image 2D features -> PnP -> metric SE3 camera
   pose.** These absolute camera fixes can update the ESKF and the external
   DPVO-to-map transform.
2. **Metric-map 3D points + live-DPVO 3D submap points -> 3D/3D Sim3.** This
   directly estimates rotation, translation, and scale between two maps.

Standard 3D/2D correspondences do not directly solve Sim3. The implementation
must choose which pipeline is being benchmarked. PnP is the lighter first test
because it does not require a stable live DPVO point cloud; Sim3 becomes useful
when a reliable live submap and 3D/3D correspondences exist.

### 4. External relocalization corrects published state, not DPVO BA itself

Repeated map localization can re-estimate an **external** flight-to-map
transform and keep the controller in metric map coordinates despite changes in
DPVO's raw gauge. That is a valid architecture and may avoid modifying DPVO's
optimizer.

It is not automatically a metric constraint retained inside DPVO's bundle
adjustment. Injecting map factors into BA would be a separate estimator change.
The plan should state whether it will:

- leave DPVO untouched and continuously update an external Sim3/SE3 wrapper,
  or
- add metric map constraints to DPVO's optimization.

The external-wrapper path is the lower-risk first implementation.

### 5. Known spawn pose does not completely determine online alignment

The spawn pose supplies a strong world position and orientation prior, but
DPVO's effective origin is its first accepted moving keyframe, not necessarily
the stationary pad frame. Camera/IMU extrinsics, motion before initialization,
and recovered keyframe timestamps still enter the transform.

With correct backfill and attitude alignment, spawn can make rotation and
translation inexpensive to estimate. It is not literally free, and it does not
resolve scale.

### 6. Prior-map localization is not merely disabled DPVO loop closure

`LOOP_CLOSURE=False` disables DPVO's own loop machinery. It does not imply that
a metric prior-map localization service already exists and only needs to be
switched on. BoQ/MASt3R or a lighter feature/PnP stack may provide useful
components, but the complete pipeline still needs measured:

- map construction and storage cost,
- query retrieval recall,
- correspondence and geometric-verification quality,
- pose accuracy and outlier rate,
- latency and fix frequency,
- GPU/CPU memory alongside the simulator, GateNet, and DPVO.

Given the 4 GB GPU and prior disk concerns, these are acceptance measurements,
not implementation details.

## Revised full-course architecture

The preferred division is:

- **DPVO/ESKF:** local, high-rate motion and short-gap propagation.
- **Offline metric environment map:** global course reference.
- **Image-to-map PnP:** lightweight absolute metric camera fixes where
  possible.
- **External DPVO-to-map Sim3/SE3 wrapper:** continuously updated from absolute
  fixes to absorb raw DPVO gauge changes.
- **GateNet:** gate identity, terminal refinement, metric landmark support, and
  recovery.
- **Judge ticks:** authoritative route progression and reference validation.

Environment relocalization can reduce dependence on the fragile live
GateNet/DPVO displacement-overlap window, but only after its compute and
accuracy budget is demonstrated.

## Next decisive offline experiment

1. Use the banked fg75 ticked corpus to build or select a small metric map for
   the gate-1/gate-2 region.
2. Hold out recorded frames as queries.
3. Run place retrieval, 2D/3D correspondence, geometric verification, and PnP.
4. Measure metric pose error, outlier rate, latency, fix frequency, VRAM, RAM,
   and map size.
5. Feed accepted PnP poses into an offline external flight-to-map transform and
   test whether they bound DPVO gauge/position error across separated windows.
6. Only if the lightweight PnP route is insufficient, benchmark live-submap to
   metric-map 3D/3D Sim3.

This experiment can establish whether environment relocalization is a viable
global layer without changing flight control or spending another fresh judge
attempt.

---

## Reply - six corrections accepted, one corpus fix, one unification

All six corrections stand; three of them catch real errors in my map-anchoring
section. Conceding, with one verified correction to the supporting evidence and
one scoping note on the experiment.

### Concessions
- **2 (prewarm != motion init).** Correct, and this was my error. Prewarm removes
  only the ~7.3 s model load; the ~13.9 s ready-to-first-pose is DPVO needing
  accepted *moving* keyframes and cannot be prewarmed away. "Prewarm completes
  cold-start" is withdrawn - the remedy is early motion + safe-order history
  recovery, not model prewarm.
- **4 (external wrapper vs BA constraint).** Correct and important. Relocalization
  should first be an **external** DPVO-to-map Sim3/SE3 wrapper that corrects
  published state; it does not retain a constraint *inside* DPVO's BA. My "metric
  constraint retained through `normalize()`" conflated the two. External wrapper
  is the lower-risk path; map-factors-in-BA is a separate, later estimator change.
- **6 (prior-map reloc != re-enabling loop closure).** Correct. `LOOP_CLOSURE=False`
  is DPVO's *intra-session* loop machinery, not a prior-map localization service.
  I conflated them. The full retrieval/correspondence/PnP pipeline is a build with
  its own acceptance measurements (recall, pose accuracy, latency, VRAM), not a
  toggle.
- **3 (PnP vs Sim3).** Conceded: 2D-3D correspondences -> PnP -> SE3; Sim3 comes
  from 3D-3D submap alignment. The PnP route still recovers scale, but *indirectly*
  - by fitting the external transform (correction 4) from a sequence of SE3 fixes
  against the DPVO trajectory - not from 2D-3D in one shot. PnP-first is the right
  call.
- **5 (spawn not literally free).** Conceded: DPVO's origin is its first accepted
  moving keyframe, not the pad; spawn is a strong rot/trans *prior*, not a free
  anchor, and resolves no scale.

### Corpus fix - use fg62, not fg75
Correction 1's substance is right (banked ticked data exists, so the map bootstrap
is not circular, and my DPVO-free-first-tick suggestion was over-specified). But
the named corpus is wrong, verified via `decode_race_rows`/`find_fresh_tick_pair`:

- **fg62**: `gate_idx 0 -> 1 -> 2`, both gates ticked, fresh pair at
  17545 ms -> 22056 ms. This is the two-tick reference run.
- **fg75**: `gate_idx 0 -> 1` only - ticked gate 1, **never gate 2**. Not a
  gate-2 traversal.

So the experiment's map should be built from **fg62**. Its two ticks 4.5 s apart
also give two naturally separated anchor windows in one session.

### Unification - the experiment subsumes the two-window scale test
Experiment step 5 (feed accepted PnP poses into an external flight-to-map
transform and test whether they bound DPVO gauge/position error across separated
windows) **is** the earlier two-window scale-drift test. fg62's gate-1 and gate-2
windows are the two separated windows. So this single offline experiment answers
both open questions at once: (a) can environment relocalization bound drift, and
(b) does DPVO's scale drift across the leg. Scope it as answering both.

### Agreement
The revised architecture (external Sim3/SE3 wrapper, PnP-first, relocalization
budget as acceptance measurements) and the offline-first experiment are the right
shape and discipline. With the corpus set to fg62 and the two-window test folded
into step 5, this is ready to run against banked data with no fresh flight.

---

## Refinements - acceptance requirements for the offline map experiment

The fg62 correction is independently confirmed with `gidx2`:

- `vq2_servo_fg62`: gate index `0 -> 1 -> 2`, two real ticks at 17.55 s and
  22.06 s.
- `vq2_servo_fg75`: gate index `0 -> 1`, one real tick at 20.87 s.

Use fg62 as the two-gate map/reference source. Add the following requirements so
the offline experiment measures cross-run localization rather than memorizing a
single traversal.

### 1. Cross-run validation is mandatory

Held-out frames from fg62 are useful for plumbing, but adjacent frames from the
same traversal share nearly identical appearance, motion, and lighting. They
can make retrieval and PnP look substantially easier than a new flight.

Use two validation tiers:

1. **Within-run sanity:** build the map from a sparse subset of fg62 and query
   temporally separated fg62 frames, excluding nearby map keyframes.
2. **Cross-run acceptance:** build only from fg62, then query independent
   corpora such as fg75 and obs2 wherever their field of view overlaps the map.
   Add other banked runs for the gate-2 region if available.

Report results separately. Control readiness must depend on cross-run recall,
pose error, and outlier rate, not within-run performance.

The split must prevent leakage of identical or near-identical frames into both
map and query sets. Record map-frame timestamps, query-frame timestamps, and
minimum temporal separation for every within-run result.

### 2. Judge ticks are aperture constraints, not exact camera poses

A judge tick proves that the drone crossed somewhere inside the gate aperture
at the recorded time. It does not prove that the camera passed through the gate
centre or with the nominal gate orientation.

Treat each tick as:

- an authoritative gate identity and crossing-time constraint,
- a plane/aperture membership constraint with lateral and vertical uncertainty,
  not an exact 3D point at the gate centre.

Use gate-corner or gate-frame PnP, known gate geometry, spawn pose, and offline
multi-view optimization to refine metric camera and gate poses. If PnP is not
available for a tick frame, propagate an aperture-sized covariance rather than
silently pinning the camera to the centre.

This uncertainty must flow into map scale and pose-error reporting. Otherwise,
the reference map can appear more accurate than the judge evidence supports.

### 3. Record normalization telemetry alongside PnP fixes

External PnP fixes can show whether the DPVO-to-map wrapper bounds metric pose
error. They do not explain why raw DPVO gauge changed. During replay, record
every `PatchGraph.normalize()` event with:

- source/keyframe timestamp or graph index,
- normalization factor `s`,
- cumulative similarity-scale factor,
- raw poses immediately before and after normalization where practical,
- nearest accepted PnP/map fix and its residual.

Evaluate three trajectories over the same timestamps:

1. raw DPVO output,
2. normalization-compensated DPVO output,
3. externally map-corrected output from accepted PnP fixes.

This separates internal gauge rescaling from ordinary VO drift and determines
whether normalization compensation alone is sufficient or periodic map fixes
remain necessary.

## Updated decisive experiment

1. Build a sparse metric map from fg62 using aperture-aware tick constraints,
   gate geometry/PnP where available, spawn pose, and offline optimization.
2. Run a leakage-controlled within-fg62 localization sanity test.
3. Run cross-run localization on fg75, obs2, and other overlapping banked
   corpora without adding their frames to the map.
4. Measure retrieval recall, PnP inliers, reprojection error, metric pose error,
   outlier rate, latency, fix frequency, VRAM, RAM, and map size.
5. Log normalization factors and compare raw, normalization-compensated, and
   externally map-corrected DPVO trajectories.
6. Use the fg62 gate-1 and gate-2 windows as the separated scale test, with
   aperture uncertainty included.
7. Accept the external-wrapper architecture only if independent-run PnP fixes
   bound position and scale error within the gate-acquisition budget.

This keeps the experiment offline and judge-free while preventing same-run
leakage and false precision from exact-centre tick assumptions.

---

## Reply - refinements endorsed, with a verified coverage constraint

All three refinements are correct and additive; nothing to rebut.

- **Refinement 1 (cross-run, no leakage)** is the critical methodological point.
  Within-run held-out frames share appearance/lighting and would give false
  confidence; control readiness must rest on cross-run recall and pose error.
  Agreed without reservation.
- **Refinement 2 (tick = aperture, not centre)** is exactly the plane-not-point
  result: a tick is an authoritative identity + crossing-time constraint plus a
  +/-~0.75 m in-plane membership, not a centre pin. Propagating aperture-sized
  covariance instead of pinning to `gate*_world` is the right fix, and it also
  bounds how accurate the reference map can honestly claim to be.
- **Refinement 3 (normalization telemetry + three trajectories)** is the right
  instrumentation of `patchgraph.py:84`; the raw / normalization-compensated /
  map-corrected split is what separates internal re-gauging from ordinary VO
  drift.

### Verified constraint on Refinement 1: cross-run coverage is asymmetric
Scanning all 75 banked `fg` runs by max gate index reached
(`decode_race_rows`):

- 46 runs reached gate 0 only (crashed on approach),
- 28 runs ticked gate 1 then failed before gate 2,
- **1 run reached gate 2 - only fg62.**

Implications for the experiment:

- **Approach + gate-1 region: 29 independent traversals** (28 gate-1 ticks + fg62)
  plus 46 partial-approach runs. Cross-run acceptance is strongly supported here,
  with rich appearance/motion diversity.
- **Gate-2 crossing region: fg62 is the only traversal.** No independent
  tick-verified run covers it (the 28 gate-1 runs give only partial coverage of
  the transit *toward* gate 2, not the crossing). So step 7's "independent-run PnP
  fixes bound error" **cannot be evaluated for gate 2 from banked data** - gate-2
  localization conclusions would be within-run only, which Refinement 1 itself
  flags as weak evidence.

### Scoping note that follows
- The **two-window scale test** (step 6) is still valid within fg62: its gate-1
  and gate-2 windows are one session, and within-session drift is exactly what
  that test targets. Keep it.
- But distinguish it from **cross-run localization validation**: the latter is
  achievable for gate 1, not gate 2, without a new recording. So the experiment
  can fully validate the global layer through gate 1 offline; extending
  acceptance to the gate-2 crossing needs at least one more gate-2 traversal
  (which, per the fg62 evidence, does not require DPVO - it can come from any run
  that ticks gate 2). Scope the offline experiment's claims to gate 1
  accordingly, and mark gate-2 cross-run validation as data-limited.

---

## Dedicated survey passes as the map source

The gate-2 coverage limit above exists only because every banked run was a
*threading attempt* - so coverage stops wherever the drone crashed. The fix is to
**decouple map data collection from gate threading**: fly deliberate survey
passes over the whole course (over/around gates, high and slow, no aperture
precision), record frames + IMU, and run DPVO and GateNet on the recording
offline. This is standard SLAM practice - a mapping pass, then localize task runs
against it - and it removes the gate-2 data limitation outright.

### What it gains
- **Full-course coverage, safely.** No threading means no crash truncation; a
  survey reaches the gate-2/3 regions no crash-run covered. Multiple passes add
  viewpoint diversity for cross-run robustness (Refinement 1).
- **Offline escapes every live constraint.** Run GateNet on every frame at full
  resolution (no 4 GB squeeze, no unload/reload, no 0.45-1 s latency), aggregate
  PnP over many views so the 0.3-1.4 m detection noise averages down by
  triangulation, seed DPVO for a deterministic gauge, and run a **full-batch
  bundle adjustment** over the whole pass. The result is a globally consistent
  metric map, far better than any online windowed `normalize()`-drifting
  trajectory.

### Two catches that constrain the survey design
1. **No threading -> no judge ticks -> no authoritative anchor.** A survey
   produces zero ticks, so map scale must come entirely from offline GateNet PnP
   aggregation + known gate geometry + spawn pose. Viable, but only if the survey
   keeps gates inside GateNet's competent viewing envelope (GateNet is trained on
   approach views; too high or too oblique detects poorly and the only metric
   anchor is lost).
2. **Viewpoint overlap decides usability.** The map exists to relocalize real
   threading flights; relocalization degrades when query viewpoint (angle,
   distance, motion blur) differs from the map. A high, slow, straight survey that
   looks nothing like a low, fast racing line builds a map that is hard to
   relocalize against. Fly the survey near the racing line and/or across multiple
   viewpoints.

### What it does not do
- It does not score (no ticks) and does not advance the qualifier.
- It does not validate threading/control - flying past gates says nothing about
  flying through them; that validation stays separate.
- It still needs enough control to cruise the course without crashing, but a slow,
  high, straight-ish pass is far lower risk than threading.

### Consequence for the plan
The map source in Anchoring 1 (map -> world) should be **dedicated survey passes**,
metrically anchored by offline aggregated GateNet PnP + gate geometry + spawn (not
ticks), optimized with full-batch BA. This supersedes relying on scarce ticked
crash-runs and resolves the gate-2 coverage constraint - provided the survey keeps
gates in the detection envelope and spans viewpoints close enough to the racing
line to be relocalizable. Ticked runs (fg62) remain valuable as reference lines
and as the aperture-constrained validation of the survey-built gate poses.
