# VQ2 Autonomous Survey, Metric Map, and DPVO Map Localization

## Goal

Build a full-course navigation stack in four independently promoted stages:

1. A fully autonomous, low-GPU survey flight that follows the course without
   requiring precision gate threading and records lossless mapping corpora.
2. An offline metric map builder that combines final-batch DPVO motion,
   GateNet gate geometry, inertial attitude, and background visual landmarks.
3. A lightweight image-to-map PnP localizer and external DPVO-to-map
   similarity wrapper.
4. Observe-only and then control-enabled race integration after replay and
   live acceptance gates pass.

The resulting race architecture uses DPVO/ESKF for local motion, the metric
map for global position, CPU visual localization for periodic absolute fixes,
FastGate for terminal aperture steering, and judge ticks for authoritative
route progression. GateNet is a survey mapper and a stopped-hover recovery
tool, not a normally resident race-time dependency.

## Why this design

The existing survey and mapping prototypes are useful evidence but are not a
safe foundation:

- `vq2/live/vq2survey.py` is a monolithic blind straight-line flight with old
  camera transforms, old IMU handling, reused output paths, and no fail-closed
  response to detector failure.
- `vq2/build_map.py`, `vq2/local_map.py`, and `vq2/live/map_build4.py` use
  stale or compressed-scale assumptions, do not provide persistent gate data
  association, and do not jointly optimize camera, gate, and background
  landmark geometry.
- Canonical camera geometry is the nose camera in `vq2/camera.py` with
  `fx=fy=226`, principal point `(319.5, 179.5)`, and 20-degree upward pitch.
  Legacy 320-pixel focal-length constants are not accepted as map metadata.
- GateNet's per-frame `gate_id` is connected-component order, not a persistent
  course identity.
- The RTX 3050 has 4 GB VRAM. GateNet and full DPVO have already competed for
  memory during live flight, while offline sequential execution is practical.

The design therefore creates new bounded packages and reuses proven mechanics
from `vq2/live/vq2wp.py` without adding another experimental mode to that
file. Existing concurrent edits to `vq2wp.py` and `dpvo_odom.py` remain
untouched until the final integration stage.

## Alternatives considered

### 1. Modular survey, map, localizer, and wrapper

This is the selected approach. Each component has a replayable interface and
an independent promotion gate. It adds more small modules but minimizes GPU
risk, isolates flight safety from map research, and makes failures auditable.

### 2. Extend the existing survey and race monoliths

This would produce an early flight faster, but it would preserve stale camera
and dead-reckoning assumptions, mix actuation with asynchronous perception,
and collide with current work in `vq2wp.py`. It is rejected.

### 3. Adopt a heavyweight learned localization stack

COLMAP/HLoc-style mapping and learned retrieval/matching may improve weak
texture or viewpoint robustness. They also add substantial dependencies,
derived data, and GPU residency. They are reserved as an offline benchmark or
an escalation if the OpenCV/SciPy baseline fails its frozen validation pass;
they are not the first live dependency.

## System architecture

The three execution profiles never require GateNet and DPVO to be active on
the GPU together:

- **Survey:** GateNet active, FastGate and ribbon extraction on CPU, DPVO off.
- **Offline:** simulator closed; GateNet extraction completes and unloads,
  then final-batch DPVO and map optimization run.
- **Race:** DPVO active, map localization and FastGate on CPU, GateNet off.
  GateNet recovery is allowed only after entering a stopped hover and
  explicitly releasing DPVO GPU ownership.

Data flows in one direction through versioned artifacts:

```text
survey pass A/B -> immutable corpora -> offline observations/trajectories
                -> metric map A -> held-out validation B
                -> approved map -> race PnP fixes -> DPVO-to-map wrapper
                -> observe-only state -> control-approved state
```

No online component may silently rebuild, modify, or tune the deployed map.

## Subproject 1: Autonomous survey acquisition

### Mission semantics

The survey is an autonomous mapping pass, not a scoring attempt. It does not
aim through apertures and does not depend on judge gate-index progression.
Instead, it follows the rendered cyan course ribbon, observes an eligible gate
from racing-relevant stand-off views, performs a gate-relative lateral bypass,
and reacquires the ribbon beyond the structure.

Pass A uses one configured bypass side. Pass B uses the opposite side and a
small image-relative observation fan. This gives independent viewpoints and
parallax while remaining near the views expected during a race. A survey does
not fly high above the visual domain in which GateNet was validated.

An unexpected judge tick means the bypass margin failed. It is recorded and
causes a controlled abort; the pass is not promoted as a clean survey.

### Perception roles

- **GateNet:** full gate quads, metric gate-relative pose, orientation, and
  coarse multi-candidate detection.
- **FastGate:** low-latency aperture continuity and alias rejection; it does
  not establish persistent identity or route order.
- **Cyan ribbon:** course membership and forward tangent. A gate candidate is
  eligible only when the ribbon enters, exits, or consistently extrapolates
  through its aperture.
- **Temporal tracker:** persistent in-pass track IDs formed from geometry,
  motion, visibility, and ribbon agreement. Detector IDs are never reused as
  map IDs.

If candidate costs are ambiguous, the vehicle holds and performs a bounded
scan. It never resolves ambiguity by choosing the nearest or largest hole.

### Controller state machine

1. `PREFLIGHT`: require a unique empty run directory, disk reserve, heartbeat,
   camera, deduplicated IMU, command arbiter, recorder, detector warm-up,
   canonical calibration, and confirmation that DPVO is disabled.
2. `RESET_SETTLE`: require the documented fresh-simulator procedure, verify a
   new race-clock epoch, and discard all pre-reset tracks.
3. `PAD_LOCK`: require multi-frame GateNet consensus and ribbon agreement.
4. `WAIT_GO`: command no motion until race GO is verified. Timeout aborts
   before arming.
5. `ARM_LEVEL_CLIMB`: use the measured low-thrust leveling sequence and climb
   only to the gate-height survey envelope.
6. `FOLLOW_RIBBON`: follow a filtered image-space tangent at conservative
   speed while searching for an eligible forward gate.
7. `ACQUIRE_GATE`: hold or scan until a candidate has stable geometry,
   temporal continuity, and ribbon support.
8. `OBSERVE_FAN`: stop at a safe stand-off and collect center plus bounded
   lateral views. Continue only with a healthy gate-relative solution.
9. `BYPASS`: use GateNet gate-relative geometry and short-horizon inertial
   propagation to pass outside the outer gate frame on the pass-specific side.
   The commanded offset includes the observed outer half-width plus a fixed
   safety margin; the exact initial margin is frozen by replay before flight.
10. `REACQUIRE_RIBBON`: clear the old target only after it is behind the
    vehicle, brake, and perform a bounded forward scan. No ribbon or eligible
    target within the timeout causes landing, not blind continuation.
11. `COMPLETE` or `ABORT_LAND`: finalize the corpus even after failure.

Mission completion is explicit: a configured segment limit or a validated
end-of-ribbon condition. A mission timeout produces a usable partial corpus
but is not reported as full-course success.

### Command and failure ownership

One command-arbiter thread exclusively owns 50 Hz MAVLink actuation. Policy
commands have a short time-to-live. A stale controller, perception worker,
recorder, or command source yields brake/hold and then controlled descent.
Top-level cleanup lands, disarms, stops workers, and atomically finalizes the
manifest.

The survey fails closed on:

- manual abort, `Ctrl+C`, or a local abort sentinel;
- stale heartbeat, camera, IMU, or command loop;
- GateNet, FastGate, ribbon, recorder, or arbiter worker death;
- queue overrun, write failure, per-run byte cap, or disk reserve violation;
- fresh serious collision, excessive attitude, or velocity runaway;
- target ambiguity, detector disagreement, or target loss before bypass;
- unexpected race-clock reset, gate-index regression, or judge tick;
- phase or mission timeout.

Ground contact before takeoff is distinguished from a fresh in-flight
collision. Vision staleness never permits forward motion except a separately
bounded, gate-relative bypass carry whose duration and maximum distance are
fixed by replay tests.

### Survey package boundaries

- `vq2/survey/types.py`: immutable sensor, detection, track, command, health,
  and state-transition records.
- `vq2/survey/recorder.py`: versioned corpus writer, bounded queues, disk
  watchdog, and atomic manifest.
- `vq2/survey/perception.py`: GateNet adapter, FastGate adapter, and ribbon
  extractor.
- `vq2/survey/tracker.py`: temporal association, target eligibility, and
  ambiguity handling.
- `vq2/survey/controller.py`: pure state transition and command policy.
- `vq2/survey/replay.py`: corpus-driven controller simulation with no
  actuation.
- `vq2/live/vq2survey.py`: thin MAVLink/runtime wiring only.

## Corpus and disk contract

The recorder preserves the existing corpus primitives:

- `mavlink.jsonl`
- `frames_dedup.jsonl`
- `frames/<sim_ns>.jpg`
- `cmds.jsonl`

It adds:

- `detections.jsonl`: accepted and rejected GateNet/FastGate/ribbon results,
  including rejection reasons;
- `tracks.jsonl`: temporal associations, eligibility scores, and target
  changes;
- `events.jsonl`: state transitions, reset/GO/tick events, aborts, and worker
  health;
- `manifest.json`: schema, pass ID, configuration, sim build, git SHA and
  dirty-file hashes, camera calibration, model/config hashes, explicit frame
  block, disk counters, sample counts, judge summary, and terminal status.

Every visual observation carries source `sim_ns`, receive wall time, detector
identity, calibration identity, raw image measurements, uncertainty, clipping
and visibility flags, source track ID, and acceptance decision. Live
dead-reckoned positions are telemetry, never ground truth.

Frame metadata and JPEGs remain one-to-one. The live recorder keeps the
original camera-rate JPEG stream because early keyframe deletion would damage
offline DPVO and cross-run localization. Derived products reference source
frames by hash and never create a second JPEG tree.

The initial policy reserves 20 GB free on C:, caps a survey pass at 1 GB and
five minutes, and reports both start and end free space. These limits are
configuration, but a run manifest freezes their effective values. With the
measured 75-87 MB/minute corpus rate and approximately 138 GB currently free,
this retains a wide safety margin.

## Subproject 2: Offline metric map

### Frozen inputs

Survey A is the only map-building survey. Survey B is sealed before map
construction and cannot enter feature selection, association, optimization,
or threshold tuning. Existing `fg62` and `fg75` corpora are external replay
and judge-aperture validation data, not hidden additions to map A.

Corpus loading always names an explicit manifest frame block. The current
implicit "last timestamp gap" behavior is not accepted for map production.

### Sequential extraction

1. Validate corpus hashes, timestamp monotonicity, frame/meta completeness,
   calibration identity, and reset boundaries.
2. Run full-resolution GateNet and preserve all decoded corners, full
   covariance, visibility/suspect masks, gate-relative pose, and quality.
3. Unload GateNet and run DPVO over the same selected stream.
4. Finalize DPVO and export the final post-termination position and
   orientation trajectory, accepted/skipped frame mapping, session identity,
   and normalization provenance. Streamed latest poses are not map truth.
5. Select map keyframes offline. Gate-observation frames, adequate-parallax
   frames, and geometrically verified loop candidates are retained; source
   JPEGs are not copied.
6. Extract OpenCV SIFT background features and descriptors outside gate masks.
   SIFT is the initial CPU baseline because it adds no dependency and is more
   tolerant of cross-run viewpoint change than binary matching. Learned
   features require a separate benchmark and promotion decision.

### Association

Within a run, gate observations are associated by projected bearing, corner
geometry, face/normal evidence, temporal continuity, and DPVO motion—not
detector IDs. Across runs, candidate gates are matched by a coarse spawn/gate
alignment, position, normal, route topology, and shared background tracks.

Repeated gate appearance alone cannot establish a loop closure. Background
matches must pass descriptor matching, spatial coverage, epipolar/pose
verification, and a minimum non-gate inlier count. Equal-cost identical-gate
matches remain separate and tentative. An auditable override file may resolve
an association, but the optimizer never silently merges ambiguous gates.

Stable map gate IDs and `route_order` are distinct fields. Route order is not
inferred by sorting a world coordinate.

### Joint optimization

The initial optimizer uses sparse `scipy.optimize.least_squares` with robust
losses and explicit Jacobian sparsity. Variables are:

- keyframe SE3 poses;
- gate SE3 poses;
- background 3D landmarks;
- one similarity alignment per survey run;
- smooth per-segment DPVO log-scale corrections only if a constant per-run
  scale fails the frozen residual test.

Residuals are:

- adjacent DPVO relative rotation and translation-direction priors;
- SIFT landmark reprojection and verified nonlocal loop tracks;
- full-covariance GateNet corner reprojection against one canonical rendered
  gate geometry;
- IMU roll/pitch priors with timestamp-bridge uncertainty;
- spawn frame and camera-extrinsic priors;
- optional fg62 judge ticks as aperture-plane membership constraints, never
  exact gate-center camera positions;
- scale smoothness when segmented scale variables are enabled.

Gate dimensions make the solution metric. DPVO supplies local sequential
geometry and is never treated as the source of metric scale. Gate in-plane
orientation is handled modulo its symmetry; gravity establishes vertical and
face evidence resolves the gate-normal sign where observable.

Optimization uses robust Huber or Cauchy loss, cheirality and parallax checks,
iterative outlier pruning, and leave-one-anchor-out stability reporting.

### Immutable map artifact

The `vq2/mapping/` package owns versioned schemas and artifacts:

- `schema.py`: run manifests, gate observations, DPVO trajectories, course
  maps, and absolute pose fixes;
- `extract.py`: explicit block validation and GateNet export;
- `dpvo_batch.py`: final-trajectory DPVO export;
- `features.py`: keyframes, SIFT tracks, and loop verification;
- `association.py`: gate tracklets, cross-run matches, and overrides;
- `optimizer.py`: joint metric optimization;
- `artifact.py`: immutable map read/write and digest;
- `validate.py`: holdout evaluation and approval state.

The deployed artifact contains:

- calibration, model, corpus, configuration, and solver digests;
- optimized keyframe poses and uncertainty;
- stable gate IDs, route order, full gate pose, dimensions, derived corners,
  uncertainty, contributing observations, and confirmation state;
- background 3D landmarks, SIFT descriptors, observation graph, and spatial
  keyframe index;
- optimization residuals, rejected associations, and validation report.

JSON/JSONL stores manifests and audit trails. Compressed NPZ stores numeric
arrays. A map is immutable; any change creates a new digest and requires
validation again.

## Subproject 3: Prior-map localization and DPVO wrapper

### Image-to-map PnP

Normal localization starts with the wrapper's predicted map pose and
uncertainty. It selects nearby map keyframes, matches query SIFT features to
their 3D landmarks, runs `solvePnPRansac`, and refines accepted solutions with
iterative PnP. GateNet corners may be added only when GateNet is deliberately
active in stopped-hover recovery.

An absolute fix includes source `frame_ns`, metric SE3 pose, covariance,
inlier count, reprojection RMS, image-grid coverage, innovation against the
prior, association margin, and provenance. Acceptance requires positive-depth
geometry, distributed inliers, bounded residual, and consistency with route
topology. Thresholds are fit using map A only, frozen in the map artifact, and
then evaluated unchanged on B.

Recovery mode widens the keyframe search but raises the geometric evidence
requirement. A single visually identical gate cannot establish a global fix
without a decisive route/pose association margin; background landmarks,
multiple gates, or another independent constraint are required.

### External DPVO-to-map similarity

DPVO remains unmodified. `vq2/mapping/map_alignment.py` maintains a robust
sliding-window similarity transform `S_map_dpvo` from timestamp-paired DPVO
poses and accepted absolute PnP fixes.

- Rotation and translation may initialize from the spawn prior, but scale is
  not published until paired fixes provide an adequately conditioned metric
  baseline.
- Fixes are applied at their source timestamp; the later raw DPVO delta is
  then propagated to the current frame.
- Scale, rotation, and translation updates are rate-limited and carry
  uncertainty. One fix cannot cause a discontinuous control jump.
- Stale, nonfinite, high-residual, weak-baseline, or route-inconsistent fixes
  are rejected and counted.
- A DPVO identity change, timestamp regression, or session discontinuity
  invalidates the transform and requires reinitialization.
- The wrapper publishes health, fix age, residuals, baseline, scale, scale
  uncertainty, and rebase reason with every corrected pose.

The deployment identity hashes the map artifact, camera/extrinsic model,
feature configuration, GateNet configuration used to build the map, and DPVO
identity. A mismatch is fail-closed.

## Subproject 4: Race integration

Normal race-time ownership is:

- DPVO/ESKF: high-rate local propagation;
- CPU PnP: periodic absolute metric fixes;
- external similarity wrapper: corrected map-frame state;
- route map: next gate and approach geometry;
- FastGate: terminal aperture bearing;
- judge ticks: authoritative route advancement;
- GateNet: stopped-hover recovery only.

The first integration is replay-only. The second is live observe-only and
publishes both legacy control state and corrected map state without allowing
the new state into source selection. Control approval is a separate immutable
configuration latch and is impossible unless the map digest and validation
report match.

If normal localization becomes unhealthy, the controller uses a bounded
hold/brake policy rather than forcing a stale map pose. GateNet recovery may
start only after the vehicle is stopped, DPVO frame submission has stopped,
and GPU ownership has been explicitly released. Normal racing does not incur
per-gate DPVO/GateNet reloads.

## Verification and promotion

### Survey unit and replay gates

- State-machine tests prove arming is impossible before all preflight and GO
  conditions pass.
- Stale, ambiguous, or disagreeing perception produces hold or abort, never a
  forward command.
- Command rate, thrust, tilt, phase timeout, and watchdog TTL are bounded.
- Calibration tests prove GateNet, FastGate, ribbon projection, and map code
  share the canonical camera identity.
- Recorder tests cover unique directories, frame/meta one-to-one, queue
  overflow, byte and disk limits, interrupted finalization, and manifests.
- Replays of fg62 and fg75 exercise target continuity, ribbon association,
  blind-motion bounds, and known judge transitions without actuation.

### Live survey ladder

Only one new variable is promoted at each step:

1. no-arm perception and recorder soak;
2. arm-hover-manual-abort;
3. takeoff, stare, land;
4. autonomous ribbon following without gate bypass;
5. one-gate observe and bypass;
6. two-gate observe and bypass;
7. full pass A;
8. independent opposite-side pass B.

A promoted survey has a finalized manifest, clean worker shutdown, near-camera
frame coverage with reported gaps, no unexpected tick or collision, no stale
forward command, and sufficient gate/background observations for offline map
quality checks. A partial or aborted corpus may be analyzed but is never
silently relabeled as a successful full-course pass.

### Map and localization gates

1. Synthetic tests establish camera/extrinsic convention, exact gate geometry,
   symmetry handling, metric scale recovery, outlier rejection, and covariance
   propagation.
2. Map A must satisfy bounded reprojection residuals, adequate per-gate
   baseline/parallax, association uniqueness, and stable leave-one-anchor-out
   scale. Exact limits are declared before B is opened.
3. B is evaluated without map mutation or threshold tuning. Reports include
   localization availability and gap distribution, inlier geometry,
   cross-pass pose repeatability, false-accept count, latency, CPU/RAM, map
   size, and gate-pose residuals.
4. Wrong-gate, shuffled-association, and repeated-appearance tests must produce
   zero accepted global fixes.
5. Wrapper replay compares raw DPVO, normalization-compensated DPVO, and
   map-corrected output. It covers delayed fixes, changing gauge, outliers,
   timestamp discontinuity, and reinitialization.
6. fg62 judge ticks are scored only with `gidx2` and treated as aperture
   membership. fg75 remains a one-tick independent run. Neither is allowed to
   create false exact-center ground truth.

If B fails, the map remains unapproved. The next action is selected from the
failure class: more survey viewpoint coverage, association correction, SIFT
budget tuning, or an offline learned-feature benchmark. B is never used as
training data for the same approval attempt.

### Control promotion

1. Offline wrapper replay on B and banked race corpora.
2. Live map-localizer soak with no arm.
3. Live observe-only flight with legacy control.
4. One-gate corrected-state shadow comparison.
5. One-gate control only after the control-approval latch is generated from
   the exact validated map digest.
6. Two-gate control, then full-course traversal.

Any service death, map identity mismatch, stale fix, unbounded correction,
false global association, unexpected control-source selection, collision, or
missing required telemetry is a failure. A run without the expected fresh
judge evidence is invalid rather than a pass.

## Observability

Every stage emits machine-readable decision records. At minimum the final
system exposes:

- survey state transitions, candidate scores, ambiguity, bypass geometry,
  command age, and abort reason;
- corpus gaps, queue depth, bytes written, and disk reserve;
- GateNet and DPVO model/config identities;
- DPVO final-trajectory provenance and normalization events;
- map associations, overrides, residuals, uncertainty, and digest;
- every PnP query, rejection reason, inlier geometry, latency, and accepted
  absolute fix;
- wrapper pair count, conditioned baseline, scale, uncertainty, residual,
  health, and source-time correction;
- legacy versus corrected state and the exact control source.

Success messages are emitted only after the corresponding artifact or state
transition has been verified.

## Implementation sequence

This design is intentionally too large for one undifferentiated coding pass.
It will be implemented as four detailed plans matching the subprojects above.
Each plan begins with tests and ends at its own promotion gate; later plans may
depend only on versioned artifacts and interfaces approved by earlier plans.

The first plan builds the replayable survey controller and recorder. The map
builder starts only after the survey corpus schema is frozen. Race integration
starts only after an immutable map, held-out validation report, localizer, and
wrapper all pass offline gates.

