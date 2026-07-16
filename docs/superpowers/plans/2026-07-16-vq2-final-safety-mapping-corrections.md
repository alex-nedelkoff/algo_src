# VQ2 Final Safety and Mapping Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the final safety, geometry, mapping, covariance, and live-recovery gaps found by independent review of the end-to-end VQ2 plans.

**Architecture:** This plan replaces unsafe or contradictory examples with fail-closed survey control, the exact GateNet gate frame, a connected multi-run mapping graph, unit-correct absolute-pose fixes, and identity-checked race control. Small contract and synthetic tests are inserted before each affected live or holdout gate.

**Tech Stack:** Python 3, NumPy, SciPy sparse optimization and Rotation, OpenCV SIFT/PnP, pymavlink, existing DPVO bridge, pytest; no new dependency.

## Global Constraints and Ordering

- This document is normative over conflicting examples in every other dated
  `2026-07-16-vq2-*` plan.
- Task 1 runs before offline-map Task 3. Tasks 2 and 3 run during survey Tasks
  1-8 and before any armed survey. Task 4 replaces the incomplete handoff
  among offline-map Tasks 4-7. Task 5 runs before B1. Task 6 runs before any
  live map observation or control. Task 7 is the final offline gate.
- Never change GateNet, DPVO, and control behavior in one live experiment.
- B1 remains sealed until Tasks 1-5 and their frozen policies pass on A-only
  and synthetic data.
- A missing, stale, nonfinite, identity-mismatched, or unit-ambiguous value is
  unhealthy. It must never be replaced by a safe-looking numeric default.

---

### Task 1: Correct the gate frame and type DPVO gauge events

**Files:**
- Create or replace: `vq2/mapping/gate_geometry.py`
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/live/bridge_dpvo.py`
- Modify: `vq2/mapping/dpvo_batch.py`
- Modify: `vq2/tests/test_mapping_gate_geometry.py`
- Modify: `vq2/tests/test_mapping_dpvo_batch.py`

**Interfaces:**
- Gate-local axes are `x=right`, `y=up`, `z=normal`; front is
  `z=-0.064 m`, back is `z=+0.064 m`.
- Slot order is `front_tl, front_tr, front_br, front_bl, back_tl,
  back_tr, back_br, back_bl`.
- Produces `DpvoNormalizationEvent(keyframe_n, factor,
  cumulative_factor)`; no unindexed `tuple[float]` remains.

- [ ] **Step 1: Replace the incorrect depth-axis test with slot parity**

```python
EXPECTED = np.array([
    [-1.3705, +1.3705, -0.064],
    [+1.3705, +1.3705, -0.064],
    [+1.3705, -1.3705, -0.064],
    [-1.3705, -1.3705, -0.064],
    [-1.3705, +1.3705, +0.064],
    [+1.3705, +1.3705, +0.064],
    [+1.3705, -1.3705, +0.064],
    [-1.3705, -1.3705, +0.064],
])


def test_gate_geometry_matches_gatenet_weighted_pnp_slots():
    assert np.array_equal(GATE_CORNERS_8, EXPECTED)
    assert np.ptp(GATE_CORNERS_8[:, 0]) == pytest.approx(2.741)
    assert np.ptp(GATE_CORNERS_8[:, 1]) == pytest.approx(2.741)
    assert np.ptp(GATE_CORNERS_8[:, 2]) == pytest.approx(0.128)
```

Add a projection-parity test using the same synthetic camera pose and
intrinsics through `vq2.mapping.gate_geometry.project()` and the copied
weighted-PnP slot model. Require sub-`1e-9 px` agreement. The earlier
`x/z=width, y=depth` model is explicitly superseded.

- [ ] **Step 2: Implement the sole gate model and identity**

Store the exact array above, `OUTER_WIDTH_M=2.741`,
`APERTURE_WIDTH_M=1.457`, and `DEPTH_M=0.128`. Expose named slot keys and a
SHA-256 over little-endian float64 bytes, dimensions, slot order, and axis
labels. GateNet extraction, PnP, mapping residuals, and map artifacts all
record and compare that geometry digest.

- [ ] **Step 3: Type normalization telemetry**

```python
@dataclass(frozen=True)
class DpvoNormalizationEvent:
    keyframe_n: int
    factor: float
    cumulative_factor: float


@dataclass(frozen=True)
class DpvoTrajectory:
    poses: tuple[PoseSE3, ...]
    normalization_events: tuple[DpvoNormalizationEvent, ...]
    final: bool
```

The bridge wrapper reads the inverse-disparity mean immediately before each
original `slam.pg.normalize()` call, appends an event after the call returns,
and updates `cumulative_factor *= factor`. Validate that keyframe indices are
strictly increasing and factors are positive finite. Do not change DPVO's
factor, call count, or optimizer.

- [ ] **Step 4: Test normalization compensation and final export**

Use two fake events, factors `2.0` then `0.5`, and poses before/after each
normalization. Assert the stable-gauge position is continuous and the final
trajectory retains `(keyframe_n, factor, cumulative_factor)` in order. Assert
the response is rejected if it is streamed before `terminate()` or if any
source frame index cannot map to `frame_ns`.

Run: `python -m pytest vq2/tests/test_mapping_gate_geometry.py vq2/tests/test_mapping_dpvo_batch.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit geometry and gauge provenance**

```bash
git add vq2/mapping/gate_geometry.py vq2/mapping/schema.py vq2/live/bridge_dpvo.py vq2/mapping/dpvo_batch.py vq2/tests/test_mapping_gate_geometry.py vq2/tests/test_mapping_dpvo_batch.py
git commit -m "fix(vq2): bind metric gate and DPVO gauge"
```

### Task 2: Make survey progress measured and gate bypass closed-loop

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/tracker.py`
- Modify: `vq2/survey/controller.py`
- Test: `vq2/tests/test_survey_controller_complete.py`
- Test: `vq2/tests/test_survey_bypass_closed_loop.py`

**Interfaces:**
- `GateRelativeState` carries source `frame_ns`, receive age, gate center in a
  documented body frame, normal, range, covariance, track age, and coherence.
- `TransitionEvidence` records the measured value, threshold, streak, source
  timestamp, and state transition it authorized.
- The runtime does not inject `climb_complete`, `fan_complete`,
  `bypass_complete`, or `gate_behind` booleans. The controller derives them
  from timestamped observations and its own bounded maneuver state.

- [ ] **Step 1: Write a complete one-gate replay test**

Feed time-ordered snapshots through:

```text
PREFLIGHT -> RESET_SETTLE -> PAD_LOCK -> WAIT_GO -> LEVEL_CLIMB
-> FOLLOW_RIBBON -> ACQUIRE_GATE -> OBSERVE_FAN -> SHIFT_TO_BYPASS
-> PASS_GATE_SIDE -> REACQUIRE_RIBBON -> FOLLOW_RIBBON
```

Every transition must include `TransitionEvidence`; no test-only completion
boolean is permitted. Replay must show the lateral offset converging, lateral
command returning to zero, minimum clearance maintained, the gate becoming
side/behind only after achieved clearance, and forward motion stopping on a
stale track.

- [ ] **Step 2: Define measurable progress producers**

- `climb_complete`: gate-relative vertical error under `0.20 m`, vertical
  speed under `0.10 m/s`, and fresh coherent GateNet state for 10 frames.
- `fan_complete`: IMU yaw visits the configured left, center, and right angles
  within `3 deg`, returns to center, and every view has a recorded frame.
- `offset_achieved`: signed gate lateral displacement is at least the target
  clearance and lateral speed is under `0.10 m/s` for 5 fresh frames.
- `side_passed`: offset was achieved first; then gate forward coordinate is
  under `0.75 m` or the bearing exceeds `70 deg`, followed by at most `0.5 s`
  bounded carry while measured speed stays under `0.30 m/s`.
- `ribbon_reacquired`: fresh ribbon confidence and tangent agreement for 5
  frames after `side_passed`.

If the detector clips before `offset_achieved`, command hold/reacquire. If it
clips after achieved offset, only the separately bounded side-pass carry may
continue. A timeout lands; it never asserts progress.

- [ ] **Step 3: Replace absolute-offset-as-velocity with feedback**

```python
error_m = target_signed_offset_m - gate_state.lateral_m
lateral_mps = np.clip(KP_LATERAL * error_m, -0.25, 0.25)
if abs(error_m) <= 0.10:
    lateral_mps = 0.0
```

During `SHIFT_TO_BYPASS`, forward speed is zero. During `PASS_GATE_SIDE`, use
the same lateral feedback while forward speed is at most `0.25 m/s`. Compute
the target as `max(2.5, outer_half_width + 1.0)` metres from gate center.
Covariance must prove the lower 3-sigma clearance bound is outside the outer
gate edge; otherwise increase stand-off or abort.

- [ ] **Step 4: Add clearance and target-loss adversarial replays**

Cover noisy lateral pose, delayed GateNet, clipped corners, wrong-side motion,
overshoot, sudden covariance growth, ribbon/gate disagreement, and a collision
event. Every case must either converge within the envelope or command zero
forward velocity and abort. Require zero collision-mask intersection in saved
frames for synthetic gate projections.

- [ ] **Step 5: Run survey state and guidance suites**

Run: `python -m pytest vq2/tests/test_survey_controller_lifecycle.py vq2/tests/test_survey_controller_guidance.py vq2/tests/test_survey_controller_complete.py vq2/tests/test_survey_bypass_closed_loop.py -v`

Expected: all tests PASS with a complete measured transition chain.

- [ ] **Step 6: Commit measured survey guidance**

```bash
git add vq2/survey/types.py vq2/survey/tracker.py vq2/survey/controller.py vq2/tests/test_survey_controller_complete.py vq2/tests/test_survey_bypass_closed_loop.py
git commit -m "fix(vq2): close the survey bypass loop"
```

### Task 3: Fail closed on survey telemetry and finalize corpora transactionally

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/controller.py`
- Modify: `vq2/survey/recorder.py`
- Modify: `vq2/survey/replay.py`
- Modify: `vq2/live/vq2survey.py`
- Modify: `vq2/tests/test_survey_recorder.py`
- Test: `vq2/tests/test_survey_health_fail_closed.py`
- Test: `vq2/tests/test_survey_coverage.py`

**Interfaces:**
- Required finite snapshot fields include collision state and age, tilt,
  measured speed, IMU age, camera age, GateNet result age/coherence, recorder
  health, command-loop age, race clock, and gate index.
- `CoverageReport` contains immutable A1/A2 digests and typed coverage gaps;
  A3 consumes that exact report digest.
- A completed manifest is published only after JPEG/metadata one-to-one
  verification and frame-block endpoint verification.

- [ ] **Step 1: Write missing/nonfinite telemetry tests**

Parametrize every required field over `None`, `nan`, and `inf`. Each case must
produce `ABORT_LAND`, zero forward/lateral velocity, a precise reason, and an
event row. A fresh camera with a stale GateNet result must fail before any
gate-relative command. Missing collision data must not become `False`;
missing tilt/speed must not become zero.

- [ ] **Step 2: Implement a single health validator**

`validate_snapshot(snapshot, now_ns) -> tuple[HealthViolation, ...]` owns all
freshness, finiteness, monotonicity, and coherence checks. Controller states
may only consume a validated snapshot. GateNet observation `frame_ns` must
match a recorded camera frame and be no older than the frozen threshold;
tracker and controller log both source age and receive age.

- [ ] **Step 3: Budget and publish frame pairs safely**

Before writing, calculate JPEG bytes plus the exact serialized metadata-row
bytes and reserve both against the per-run cap and 20 GB disk floor. Write and
fsync a unique temporary JPEG, atomically rename it, then append and flush its
metadata row. On append failure, remove the just-created JPEG, mark the
recorder failed, and forbid `status="complete"`. Crash recovery and finalizer
delete only `.tmp` files and reject any completed manifest with orphan JPEGs
or metadata.

Update `first_frame_ns`/`last_frame_ns` only after the pair succeeds. Serialize
`PassProfile` enums through `.value`; never send `asdict()` output containing
Enum objects directly to JSON. `finalize(status, reason)` is idempotent only
for the identical status/reason and returns the same manifest path.

- [ ] **Step 4: Define and consume real A3 coverage gaps**

```python
@dataclass(frozen=True)
class CoverageGap:
    segment_id: str
    stable_gate_id: str | None
    missing_view: str       # left, right, high, low, approach, departure
    priority: int
    target_yaw_deg: float
    target_standoff_m: float


@dataclass(frozen=True)
class CoverageReport:
    a_manifest_sha256: tuple[str, str]
    gaps: tuple[CoverageGap, ...]
    digest: str
```

Build gaps from A1/A2 view-bin counts, baselines, parallax, gate covariance,
and ribbon segment coverage. A3 must receive `SURVEY_COVERAGE=<path>`, verify
its digest, and use each gap to select observation-fan direction and side. If
the report is empty or invalid, A3 does not arm. The placeholder
`coverage_gaps: []` and executable angle-bracket commands are superseded.

- [ ] **Step 5: Use the real corpus API and concrete commands**

Replay imports `vq2.corpus.load`, not nonexistent `load_corpus`. Use the
concrete attempt-001 paths from the self-review companion. Every retry changes
the attempt suffix and uses a new empty directory.

- [ ] **Step 6: Run recorder, health, coverage, and runtime tests**

Run: `python -m pytest vq2/tests/test_survey_recorder.py vq2/tests/test_cross_plan_survey_manifest.py vq2/tests/test_survey_health_fail_closed.py vq2/tests/test_survey_coverage.py vq2/tests/test_survey_runtime_wiring.py -v`

Expected: all tests PASS, including injected write failure, crash residue,
double-finalize, enum serialization, nonempty A3 gap consumption, and stale
GateNet rejection.

- [ ] **Step 7: Commit survey health and corpus contracts**

```bash
git add vq2/survey/types.py vq2/survey/controller.py vq2/survey/recorder.py vq2/survey/replay.py vq2/live/vq2survey.py vq2/tests/test_survey_recorder.py vq2/tests/test_survey_health_fail_closed.py vq2/tests/test_survey_coverage.py
git commit -m "fix(vq2): fail closed and seal survey data"
```

### Task 4: Connect feature tracks, metric initialization, optimization, and artifacts

**Files:**
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/mapping/features.py`
- Modify: `vq2/mapping/association.py`
- Modify: `vq2/mapping/optimizer.py`
- Modify: `vq2/mapping/artifact.py`
- Modify: `vq2/tests/test_mapping_features.py`
- Modify: `vq2/tests/test_mapping_association.py`
- Modify: `vq2/tests/test_mapping_optimizer.py`
- Modify: `vq2/tests/test_mapping_artifact.py`
- Test: `vq2/tests/test_mapping_end_to_end_synthetic.py`

**Interfaces:**
- Artifact adds `MapKeyframe`, `LandmarkObservation`, and explicit landmark
  semantic/group provenance.
- `build_tracks(keyframes, descriptors, gate_masks) -> TrackGraph` produces
  geometrically verified multi-frame observations, not pairwise matches only.
- `initialize_problem(inputs, tracks, gate_tracklets) -> MapProblem` supplies
  every variable and residual consumed by `optimize_map()`.
- `derive_route_order()` uses consistent encounter/ribbon topology, never x
  sorting.

- [ ] **Step 1: Freeze the missing artifact tables**

```python
@dataclass(frozen=True)
class MapKeyframe:
    keyframe_id: int
    run_id: str
    frame_ns: int
    pose: PoseSE3
    observation_start: int
    observation_count: int


@dataclass(frozen=True)
class LandmarkObservation:
    keyframe_id: int
    landmark_id: int
    uv: tuple[float, float]
    octave: int
    response: float


@dataclass(frozen=True)
class MapLandmark:
    landmark_id: int
    xyz_m: tuple[float, float, float]
    descriptor_row: int
    semantic: str             # background or gate
    stable_gate_id: str | None
    spatial_group_id: int
    observation_count: int
```

Store these as keyed NPZ arrays with schema, dtype, shape, and per-file hashes.
The localizer must be able to retrieve the exact landmark subset observed by
each candidate keyframe. Recovery's `stable_gate_id` and
`spatial_group_id` come from the artifact, not test-only fixtures.

- [ ] **Step 2: Build multi-frame tracks and triangulation inputs**

For adjacent and retrieval candidate pairs: ratio-test SIFT matches, exclude
gate-mask-only loops, run essential/fundamental RANSAC, enforce bidirectional
consistency, then union observations into tracks. Reject duplicate keyframe
membership, negative depth, insufficient parallax, and tracks with fewer than
two keyframes. Triangulate only after coarse run alignment; retain observation
rows and rejected-reason audit.

- [ ] **Step 3: Define the complete metric initializer**

1. Use final DPVO poses as each run's local camera graph.
2. Estimate coarse cross-run Sim3 from shared verified background tracks plus
   robust GateNet PnP tracklets; ambiguous identical gates remain separate.
3. Transform per-frame metric gate PnP centers/normals into the coarse common
   frame and associate by one-to-one cost, topology, normal, and temporal
   encounter order.
4. Triangulate background landmarks in the coarse common frame.
5. Initialize each gate pose by robust averaging of its associated metric PnP
   poses modulo the Z4 in-plane gauge.
6. Fix the first A1 keyframe SE3 as the world gauge. Gate dimensions provide
   metric scale; spawn/judge anchors are soft priors, not exact tick poses.

The initializer returns camera rotvec/translation blocks, gate
rotvec/translation blocks, landmark XYZ, per-run log-scale knots, residual
index tables, and a sparse Jacobian pattern. Association must not consume
globally metric gate centers before step 2 creates the coarse common frame.

- [ ] **Step 4: Implement one connected sparse optimization**

Residual blocks are DPVO relative rotation/translation, SIFT observation
reprojection, full-covariance eight-corner GateNet reprojection, gravity,
scale-knot smoothness, and soft judge aperture-plane priors. Use rotvec plus
translation for SE3 increments, log scale for positivity, robust losses, and
iterative outlier pruning. Keep route order out of numeric optimization; derive
it afterward from majority encounter order and ribbon adjacency, and fail on a
cycle or unresolved tie.

Compute gate/pose uncertainty with sparse block solves or Schur marginals.
Never materialize or invert the full dense normal matrix.

- [ ] **Step 5: Add a three-run synthetic end-to-end map**

Generate at least four identical gates, background landmarks, A1/A2/A3 camera
paths, different monocular scales, smooth scale drift, one loop, shuffled gate
candidate IDs, and 10% outliers. Require:

- all stable gate associations correct and ambiguous decoys unmerged;
- metric gate centers within `0.15 m`, normals within `5 deg`;
- background reprojection RMS under the frozen synthetic threshold;
- route order matches encounter topology despite nonmonotonic x;
- keyframe lookup returns only its stored observation subset;
- removing any one A run still leaves bounded scale;
- B1 input raises before extraction or optimization.

- [ ] **Step 6: Run the connected mapping suite**

Run: `python -m pytest vq2/tests/test_mapping_features.py vq2/tests/test_mapping_association.py vq2/tests/test_mapping_optimizer.py vq2/tests/test_mapping_artifact.py vq2/tests/test_mapping_end_to_end_synthetic.py -v`

Expected: all tests PASS and every optimizer input traces to an artifact or
initializer output.

- [ ] **Step 7: Commit the connected map graph**

```bash
git add vq2/mapping/schema.py vq2/mapping/features.py vq2/mapping/association.py vq2/mapping/optimizer.py vq2/mapping/artifact.py vq2/tests/test_mapping_features.py vq2/tests/test_mapping_association.py vq2/tests/test_mapping_optimizer.py vq2/tests/test_mapping_artifact.py vq2/tests/test_mapping_end_to_end_synthetic.py
git commit -m "fix(vq2): connect the metric map graph"
```

### Task 5: Make PnP covariance and delayed Sim3 propagation unit-correct

**Files:**
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/mapping/localizer.py`
- Modify: `vq2/mapping/map_alignment.py`
- Modify: `vq2/tests/test_map_localizer.py`
- Modify: `vq2/tests/test_map_alignment.py`
- Modify: `vq2/tests/test_cross_plan_pose_bootstrap.py`

**Interfaces:**
- `AbsolutePoseFix.covariance_6x6` is tangent-space covariance ordered
  `[translation_m, rotation_rad]`; pixel residual variance is a separate field.
- `propagate_source_fix(fix, source_dpvo, current_dpvo, scale)` applies a DPVO
  camera-relative delta in the metric map frame.
- Offline `DpvoPose` is removed or aliased to the canonical `PoseSE3`; no
  second pose layout exists.

- [ ] **Step 1: Write covariance-unit and conditioning tests**

Use synthetic projections at two depths with known pixel noise. Require finite
positive-semidefinite 6x6 pose covariance, metric translation variance that
increases at weaker geometry, radian rotation variance, and explicit rejection
of a rank-deficient/ill-conditioned Jacobian. Assert pixel RMS never appears
directly on a metric covariance diagonal.

- [ ] **Step 2: Estimate pose covariance in the correct tangent space**

After PnP refinement, obtain the projection Jacobian for the inlier set,
weight it with corner/feature pixel covariance, and compute the damped 6x6
inverse information only when its condition number passes the frozen policy.
Transform camera-from-world covariance to camera-to-world with the SE3
adjoint. Store pixel residual variance, covariance method, condition number,
and correspondence count separately. If the approximation is not trustworthy,
reject the fix; do not invent metric confidence from RMS.

- [ ] **Step 3: Test delayed propagation with nontrivial scale and rotation**

Use alignment rotation `Rz(90 deg)`, scale `2.0`, a source absolute fix with
nonidentity orientation, and a later raw DPVO pose. Verify both formulations
agree:

```python
R_rel = R_dpvo_source.T @ R_dpvo_current
t_rel = R_dpvo_source.T @ (p_dpvo_current - p_dpvo_source)
R_map_current = R_map_source @ R_rel
p_map_current = p_map_source + R_map_source @ (scale * t_rel)
```

and applying the current global Sim3 to the current DPVO pose. Test delayed
fix arrival, normalization events, timestamp regression, identity reset, and
quaternion sign equivalence. The earlier unit-scale identity-rotation-only
test is insufficient and is superseded.

- [ ] **Step 4: Use canonical pose and alignment APIs everywhere**

Base-plan calls to `add_pair()`, `correct_current()`, `history.add()`,
`predicted_map_pose()`, and `correct(frame_ns, pose)` are superseded by the
pose-bootstrap addendum. Offline artifacts, localizer, alignment, worker, and
test fakes use `PoseSE3(frame_ns, p, q_xyzw)` and the public methods
`add_dpvo_pose`, `predicted_pose`, `add_fix`, `correct`, and `reset` only.

- [ ] **Step 5: Run localization and alignment suites**

Run: `python -m pytest vq2/tests/test_map_localizer.py vq2/tests/test_map_alignment.py vq2/tests/test_cross_plan_pose_api.py vq2/tests/test_cross_plan_pose_bootstrap.py -v`

Expected: all tests PASS with metre/radian covariance and the nontrivial
delayed-fix case.

- [ ] **Step 6: Commit unit-correct localization**

```bash
git add vq2/mapping/schema.py vq2/mapping/localizer.py vq2/mapping/map_alignment.py vq2/tests/test_map_localizer.py vq2/tests/test_map_alignment.py vq2/tests/test_cross_plan_pose_bootstrap.py
git commit -m "fix(vq2): propagate metric pose fixes"
```

### Task 6: Fail closed in race control and make GPU recovery rebootstrapable

**Files:**
- Modify: `vq2/live/map_control.py`
- Modify: `vq2/live/map_pose_worker.py`
- Modify: `vq2/live/gpu_ownership.py`
- Modify: `vq2/live/vq2wp.py`
- Modify: `vq2/tests/test_map_control_live.py`
- Modify: `vq2/tests/test_map_gpu_recovery_live.py`
- Test: `vq2/tests/test_map_failsafe_live.py`
- Test: `vq2/tests/test_map_rebootstrap_live.py`

**Interfaces:**
- Runtime identity separates stable `dpvo_config_id` from changing
  `dpvo_session_id`.
- `select_control_position()` compares every `MapPoseState` with the verified
  map/config identity on every call.
- `CorrectionLimiter.reset(anchor_position, anchor_q, now_s)` anchors recovery
  to the currently commanded/legacy pose; it never clears into an unbounded
  first jump.
- Any approved-map failure enters an expiring hold/land arbiter path before
  the loop can continue or exit.

- [ ] **Step 1: Write per-cycle identity and stale-command tests**

Mutate map digest, map ID, policy digest, camera ID, feature ID, DPVO config ID,
and DPVO session ID one at a time. Every mismatch must return a fail-closed
decision before the map position is read. In observe-only mode legacy control
continues and the mismatch is logged. In approved-control mode the arbiter
must publish a zero-velocity hold command in the same cycle and keep refreshing
it until controlled land/disarm.

- [ ] **Step 2: Verify identity inside source selection**

```python
def select_control_position(legacy_p, map_state, *, observe_only,
                            approval, runtime_identity):
    mismatch = runtime_identity.compare_state(map_state)
    if mismatch:
        return SourceDecision(legacy_p if observe_only else None,
                              "legacy" if observe_only else "abort",
                              f"identity:{mismatch}")
    # Continue with approval, health, age, finiteness, and uncertainty checks.
```

`MapPoseState` carries the verified map digest/ID, policy, camera, feature,
stable DPVO config ID, and current session ID. A session change immediately
resets alignment and disables map control until the new session is healthy;
it does not invalidate an approval bound to the same stable config/model.

- [ ] **Step 3: Replace `aborted; continue` with an owned failsafe**

The control loop calls `failsafe.enter(reason, now_ns)` and the command owner
sends hold at its normal rate with a short TTL. If health is not restored
inside the frozen recovery window, it commands controlled descent, verifies
ground contact, then disarms. No `continue`, exception, or worker exit may
leave the previous velocity command active.

- [ ] **Step 4: Correct the limiter's first-sample and time behavior**

Initialize from current control position/orientation. For `dt <= 0`, retain
the previous output and return unhealthy; timestamp regression enters the
failsafe. On reset, require a finite anchor. Limit position and orientation
increments separately. Test a 10 m recovered target at `dt=0`, a reset after
unhealthy alignment, and a later `0.1 s` update: the first two produce no
jump, the third moves at most `0.05 m` under the frozen `0.5 m/s` limit.

- [ ] **Step 5: Define a feasible post-GateNet DPVO rebootstrap**

Recovery sequence:

1. Hold and prove speed below `0.15 m/s`.
2. Stop DPVO, confirm process exit and CUDA release, then load GateNet.
3. Use route prior plus GateNet PnP only to confirm gate association and a
   collision-free map corridor; ambiguity lands.
4. Unload GateNet, start DPVO with the approved config and a new session ID,
   and reset alignment.
5. While map control remains disabled, execute a preapproved low-speed
   rebootstrap dither of at least `1.0 m` baseline inside the verified corridor
   at no more than `0.15 m/s`. CPU localizer uses strict recovery and must
   accept at least three source-time fixes.
6. Resume only after Sim3 conditioning, residual, covariance, and identity
   gates pass. Timeout, target loss, corridor violation, or failed PnP lands.

Do not claim fresh Sim3 initialization while stationary. Test an unavailable
corridor, fewer than three fixes, weak baseline, session rollover, successful
dither, and cancellation by manual abort.

- [ ] **Step 6: Run live control, failsafe, and recovery tests**

Run: `python -m pytest vq2/tests/test_map_control_live.py vq2/tests/test_map_failsafe_live.py vq2/tests/test_map_gpu_recovery_live.py vq2/tests/test_map_rebootstrap_live.py vq2/tests/test_map_control_wiring.py -v`

Expected: all tests PASS with no stale command path and no stationary
rebootstrap claim.

- [ ] **Step 7: Commit fail-closed map control**

```bash
git add vq2/live/map_control.py vq2/live/map_pose_worker.py vq2/live/gpu_ownership.py vq2/live/vq2wp.py vq2/tests/test_map_control_live.py vq2/tests/test_map_failsafe_live.py vq2/tests/test_map_gpu_recovery_live.py vq2/tests/test_map_rebootstrap_live.py
git commit -m "fix(vq2): fail closed and rebootstrap DPVO"
```

### Task 7: Prove the corrected contracts end to end before B1

**Files:**
- Test: `vq2/tests/test_vq2_map_stack_end_to_end.py`
- Create outside git: `C:\Users\alexj\vq2_maps\pre_b1_report_001\`

**Interfaces:**
- One deterministic CPU synthetic fixture flows through survey manifest,
  map input firewall, track graph, metric optimizer, immutable artifact,
  keyframe localizer, Sim3 bootstrap, live worker, identity gate, and route
  manager.
- Produces a hashed pre-B1 report without reading B1.

- [ ] **Step 1: Build the end-to-end synthetic fixture**

Generate A1/A2/A3 manifests and image/feature observations with exact frame
blocks, four gates, route topology, background groups, scale drift, delayed
fixes, and one DPVO normalization event. Generate a B1 manifest path but make
any attempted open raise in this test.

- [ ] **Step 2: Assert every cross-plan invariant**

Require canonical geometry digest, no B1 input, keyframe-observation lookup,
correct gate route order, metric covariance units, recovery-to-normal
bootstrap, position/orientation Sim3 correctness, external map digest match,
per-cycle runtime identity match, bounded correction output, and judge-only
route advancement.

- [ ] **Step 3: Run the complete non-GPU test set**

Run: `python -m pytest vq2/tests/test_survey_*.py vq2/tests/test_mapping_*.py vq2/tests/test_map_*.py vq2/tests/test_cross_plan_*.py vq2/tests/test_vq2_map_stack_end_to_end.py -v`

Expected: all tests PASS. Record exact command, Git SHA, elapsed time, peak
RSS, map size, and test count.

- [ ] **Step 4: Produce and verify the pre-B1 report**

Run: `python -m vq2.mapping.validate pre-b1 --map C:\Users\alexj\vq2_maps\map_A_candidate --output C:\Users\alexj\vq2_maps\pre_b1_report_001`

Expected: map and policy digests frozen; input roles exactly A1/A2/A3;
geometry/camera/feature/DPVO-config identities present; zero unresolved
associations; synthetic false accepts zero; B1 receipt absent.

- [ ] **Step 5: Commit the final integration test**

```bash
git add vq2/tests/test_vq2_map_stack_end_to_end.py
git commit -m "test(vq2): prove map stack contracts"
```

---

## Final correction gate

No survey may arm until Tasks 2-3 pass. No map may be frozen until Tasks 1 and
4 pass. B1 may open only after Tasks 1-5 and the Task 7 pre-B1 report pass.
Live map observation may begin only after Task 6 also passes. Control remains
observe-only until the separate localization and shadow-control approvals are
digest-bound and the staged fresh-sim ladder succeeds.
