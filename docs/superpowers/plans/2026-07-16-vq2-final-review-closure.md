# VQ2 Final Review Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the last survey-runtime ambiguities found after the independent closure review: arm-state authorization, recorder backpressure, coverage-view setpoints, runtime identity, and detached manual abort.

**Architecture:** Explicit runtime modes separate a prohibited no-arm soak from an authorized pre-arm state; a bounded writer worker isolates source reception from disk latency; every requested coverage view has gate-relative measured setpoints; and one hashed runtime identity plus a run-local abort sentinel makes collection auditable and operable.

**Tech Stack:** Python enums/dataclasses, threading and bounded queues, SHA-256, argparse/PowerShell launcher integration, pytest; no new dependency.

## Global Constraints

- This is the final highest-precedence erratum for the dated VQ2 plan set.
- Apply Tasks 1-5 while implementing survey-runtime addendum Tasks 2-4 and
  before any armed survey.
- A queue overflow, missing identity field, stale abort watcher, unsupported
  coverage target, or ambiguous arm state makes the corpus/run invalid and
  fails closed.

---

### Task 1: Separate no-arm mode from the pre-arm telemetry state

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/arbiter.py`
- Modify: `vq2/live/vq2survey.py`
- Modify: `vq2/tests/test_survey_arbiter_runner.py`
- Modify: `vq2/tests/test_survey_runtime_integration.py`

**Interfaces:**
- `RuntimeMode.NO_ARM` prohibits every outgoing arm, rate/thrust, land, and
  disarm message.
- `RuntimeMode.FLIGHT` starts with `ArmState.DISARMED`; after all preconditions
  and GO, `request_arm(preflight_token)` permits exactly one arm request.
- Rate/thrust commands are prohibited until fresh telemetry confirms
  `ArmState.ARMED`.

- [ ] **Step 1: Write the arm authorization matrix test**

Cover `(NO_ARM, DISARMED)`, `(FLIGHT, DISARMED)`,
`(FLIGHT, ARM_REQUESTED)`, and `(FLIGHT, ARMED)`. Assert no-arm sends nothing;
flight/disarmed can send only one validated arm request; arm-requested sends no
duplicate and no thrust; armed can send bounded commands. Timeout or rejection
enters abort without sending land/disarm for a vehicle that never armed.

- [ ] **Step 2: Implement a single-use preflight token**

The runtime creates the token only after fresh reset, pad lock, GO, stream and
recorder health, model warmup, disk reserve, and DPVO-off checks pass. The
arbiter consumes it atomically in `request_arm()`. Reuse, wrong run ID, expired
token, or missing precondition is rejected and logged.

- [ ] **Step 3: Run arm and runtime tests**

Run: `python -m pytest vq2/tests/test_survey_arbiter_runner.py vq2/tests/test_survey_runtime_integration.py -k 'arm or no_arm' -v`

Expected: all tests PASS.

- [ ] **Step 4: Commit arm-state closure**

```bash
git add vq2/survey/types.py vq2/survey/arbiter.py vq2/live/vq2survey.py vq2/tests/test_survey_arbiter_runner.py vq2/tests/test_survey_runtime_integration.py
git commit -m "fix(vq2): authorize survey arming explicitly"
```

### Task 2: Add a bounded transactional recorder worker

**Files:**
- Modify: `vq2/survey/recorder.py`
- Modify: `vq2/live/vq2survey.py`
- Modify: `vq2/tests/test_survey_recorder.py`
- Test: `vq2/tests/test_survey_recorder_backpressure.py`

**Interfaces:**
- `RecorderWorker(max_pending_frames=64)` owns all disk writes.
- Receiver calls nonblocking `submit_frame()`; it never waits for disk.
- Queue saturation latches unhealthy state, records an overflow event through
  the emergency event channel, aborts flight, and forbids a complete manifest.
  It is never counted as a successful/deduplicated frame.

- [ ] **Step 1: Write slow-disk and saturation tests**

Block the writer after its first frame and submit through the queue limit.
Assert receiver calls remain nonblocking, queue depth is reported, the next
submission returns an explicit overflow failure, controller sees recorder
unhealthy, arbiter force-lands, and final manifest is `aborted` with overflow
reason. Assert no silent frame gap can appear in a `complete` corpus.

- [ ] **Step 2: Implement writer lifecycle and emergency event path**

Use a bounded `queue.Queue`, one writer thread, monotonic heartbeat, queue
depth/high-water telemetry, error latch, `stop_accepting`, `flush`, `stop`, and
`join`. Disk writes use the transactional pair contract. The emergency event
path is pre-opened and bounded so a recorder data-queue failure can still
record the failure reason; if it also fails, retain the in-memory reason in the
aborted manifest.

- [ ] **Step 3: Verify flush/finalize ordering**

Successful shutdown stops submissions, drains the queue, joins the writer,
verifies content, and only then finalizes. Aborted shutdown attempts a bounded
drain but never blocks landing; it finalizes after the writer stops or records
a join-timeout failure. Duplicate/nonmonotonic timestamps are rejected before
queue insertion.

- [ ] **Step 4: Run recorder suites**

Run: `python -m pytest vq2/tests/test_survey_recorder.py vq2/tests/test_survey_recorder_backpressure.py vq2/tests/test_survey_runtime_integration.py -k 'recorder or queue or overflow' -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit bounded recording**

```bash
git add vq2/survey/recorder.py vq2/live/vq2survey.py vq2/tests/test_survey_recorder.py vq2/tests/test_survey_recorder_backpressure.py
git commit -m "fix(vq2): bound survey recording pressure"
```

### Task 3: Give every A3 coverage gap a measurable gate-relative target

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/controller.py`
- Modify: `vq2/survey/replay.py`
- Modify: `vq2/tests/test_survey_coverage.py`
- Test: `vq2/tests/test_survey_coverage_guidance.py`

**Interfaces:**
- `CoverageGap` includes target lateral, vertical, standoff, and yaw values in
  the documented gate-local/body convention plus tolerances and required
  consecutive fresh frames.
- A3 records measured start/end pose, achieved error, source frames, and
  completion evidence for every attempted gap.

- [ ] **Step 1: Expand the coverage schema**

```python
@dataclass(frozen=True)
class CoverageGap:
    segment_id: str
    stable_gate_id: str | None
    view_id: str
    target_lateral_m: float
    target_vertical_m: float
    target_standoff_m: float
    target_yaw_deg: float
    tolerance_m: float
    tolerance_deg: float
    required_frames: int
    priority: int
```

Remove string-only `high`/`low` behavior. Every view, including high/low, is a
full numeric target that passes the same clearance and covariance checks as a
bypass target.

- [ ] **Step 2: Implement measured coverage guidance**

Use bounded feedback on lateral, vertical, standoff, and yaw errors. A view is
complete only after all errors and measured speeds remain inside tolerance for
the required fresh coherent frames and each frame is committed to the
recorder. Target loss holds; it does not advance the gap index.

- [ ] **Step 3: Test left/right/high/low/approach/departure gaps**

For each view, replay convergence and verify achieved translation, clearance,
frame evidence, and zero unsupported completion. Inject stale/noisy PnP and
require hold/abort.

- [ ] **Step 4: Run coverage suites**

Run: `python -m pytest vq2/tests/test_survey_coverage.py vq2/tests/test_survey_coverage_guidance.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit executable A3 targets**

```bash
git add vq2/survey/types.py vq2/survey/controller.py vq2/survey/replay.py vq2/tests/test_survey_coverage.py vq2/tests/test_survey_coverage_guidance.py
git commit -m "fix(vq2): execute measured A3 coverage"
```

### Task 4: Freeze the survey runtime identity schema

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/live/vq2survey.py`
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/tests/test_survey_runtime_wiring.py`
- Test: `vq2/tests/test_survey_runtime_identity.py`

**Interfaces:**
- `SurveyRuntimeIdentity` has a common stack identity plus role-specific pass
  profile/coverage/collection-context digests.
- Mapping rejects a missing common field or an incompatible common identity;
  A1/A2/A3 are allowed distinct expected profile digests.

- [ ] **Step 1: Define mandatory identity fields**

```text
schema, git_sha, dirty_file_hashes, sim_build,
camera_id, camera_extrinsic_id, gate_geometry_id,
gatenet_checkpoint_sha256, gatenet_config_sha256, gatenet_decode_id,
fastgate_code_sha256, ribbon_config_sha256,
controller_code_sha256, controller_policy_sha256,
pass_profile_sha256, coverage_sha256, collection_context_sha256,
dpvo_disabled
```

Hash dirty files by relative path and content; never record only a boolean
dirty flag. `coverage_sha256` is required for A3 and null for A1/A2.
`collection_context_sha256` is required for B1. `dpvo_disabled` must be true.

- [ ] **Step 2: Add identity completeness and compatibility tests**

Delete/mutate every field one at a time. Recorder creation rejects missing
fields; mapping rejects mismatched common stack fields; it accepts the frozen
role-specific A1/A2/A3 profile differences only when their profile digests
match the declared pass definitions.

- [ ] **Step 3: Run identity tests**

Run: `python -m pytest vq2/tests/test_survey_runtime_identity.py vq2/tests/test_survey_runtime_wiring.py vq2/tests/test_mapping_schema.py -v`

Expected: all tests PASS.

- [ ] **Step 4: Commit runtime identity**

```bash
git add vq2/survey/types.py vq2/live/vq2survey.py vq2/mapping/schema.py vq2/tests/test_survey_runtime_identity.py vq2/tests/test_survey_runtime_wiring.py
git commit -m "fix(vq2): freeze survey runtime identity"
```

### Task 5: Add a concrete detached manual-abort interface

**Files:**
- Modify: `vq2/live/vq2survey.py`
- Modify: `vq2/live/run_survey.bat`
- Modify: `vq2/tests/test_survey_runtime_integration.py`
- Test: `vq2/tests/test_survey_manual_abort.py`

**Interfaces:**
- Ctrl+C and a unique run-local `abort.request` file both call the same
  idempotent `request_abort(reason)` path, which immediately calls
  `arbiter.force_land(reason)` when armed.
- `SURVEY_ABORT_FILE` defaults to `<unique-run-dir>\abort.request`; runtime
  refuses to start if it already exists.

- [ ] **Step 1: Write Ctrl+C and sentinel tests**

Create the sentinel while a fake detached run is armed. Require detection
within `100 ms`, an abort event with source `sentinel`, immediate force-land,
and retained sentinel evidence. Repeat with a simulated `KeyboardInterrupt`.
Repeated requests must not restart or bypass the landing state.

- [ ] **Step 2: Implement the watcher and signal path**

Poll the run-local sentinel at 20 Hz from a small watcher thread using
monotonic deadlines. `KeyboardInterrupt` calls the same method in the main
thread. Watcher health is a required runtime field; watcher death aborts. Stop
and join it during cleanup.

- [ ] **Step 3: Print and log the operator command before arm**

For attempt 001, the launcher/runtime displays and records:

```powershell
New-Item -ItemType File -Force 'C:\Users\alexj\vq2_survey_A1_001\abort.request'
```

The actual path always comes from the unique run directory; do not reuse the
example for another pass/attempt.

- [ ] **Step 4: Run abort and runtime tests**

Run: `python -m pytest vq2/tests/test_survey_manual_abort.py vq2/tests/test_survey_runtime_integration.py -k 'abort or sentinel or keyboard' -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit manual abort**

```bash
git add vq2/live/vq2survey.py vq2/live/run_survey.bat vq2/tests/test_survey_manual_abort.py vq2/tests/test_survey_runtime_integration.py
git commit -m "fix(vq2): add detached survey abort"
```

---

## Closure gate

The survey implementation is ready for its no-arm and armed ladders only when
all five tasks pass: arm authorization is unambiguous, disk latency cannot
block or silently drop frames, A3 targets are numeric and measured, runtime
identity is complete, and both interactive and detached aborts reach the
owned landing path.
