# VQ2 Survey Runtime and Corpus Addendum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the autonomous survey runtime executable, asynchronous, actuator-owned, and cryptographically bound to the complete corpus used by mapping and B1 validation.

**Architecture:** Camera-rate CPU perception is isolated from slower GateNet inference; a fused tracker publishes timestamped decisions; a monotonic 50 Hz arbiter is the only MAVLink command owner; and a concrete runtime wires reset verification, sensor snapshots, warmup, control, recording, and armed-aware shutdown. Completed corpora hash every source stream and frame, while aborted preflight runs may finalize with no frame block.

**Tech Stack:** Python 3, threading and queues, pymavlink, NumPy, OpenCV, GateNet/PyTorch, SHA-256, argparse, pytest; no new dependency.

## Global Constraints and Ordering

- This addendum is normative over conflicting survey-runtime, recorder,
  perception, replay, and B1 examples in every other dated VQ2 plan.
- Tasks 1-3 must pass before an armed survey. Task 4 runs after A1/A2 and
  before A3. B1 is not collected until map A and its policy are frozen.
- Only the arbiter thread sends flight commands. Receive, perception,
  controller, recorder, and main threads never call MAVLink actuation directly.
- `SURVEY_NO_ARM=1` means no arm, rate/thrust, land, or disarm command may be
  emitted under success, failure, or `finally` cleanup.

---

### Task 1: Split perception rates and fuse FastGate instead of discarding it

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/perception.py`
- Modify: `vq2/survey/tracker.py`
- Modify: `vq2/tests/test_survey_perception.py`
- Modify: `vq2/tests/test_survey_tracker.py`
- Test: `vq2/tests/test_survey_perception_scheduling.py`

**Interfaces:**
- `CameraPerception.process(frame) -> CameraPerceptionResult` runs ribbon and
  FastGate at camera rate on CPU.
- `GateNetWorker.submit_latest(frame)` uses a size-one queue and publishes
  `GateNetResult`; it never blocks the camera receiver.
- Every result carries `frame_ns`, receive monotonic time, latency, health,
  calibration ID, and accepted/rejected candidates.
- `SurveyTracker.update(camera_result, gatenet_result) -> TrackDecision`
  consumes both detector families and logs disagreement.

- [ ] **Step 1: Write nonblocking scheduling and consumption tests**

Use a GateNet fake blocked on an event while submitting ten frames. Assert all
ten FastGate/ribbon results complete, GateNet retains only the newest pending
frame, the camera receiver never waits on GateNet, and the tracker receives a
FastGate result rather than losing it. Inject worker death and require an
unhealthy snapshot before the controller's next forward command.

- [ ] **Step 2: Implement separate workers and timestamped mailboxes**

`CameraPerception` owns no GPU state. `GateNetWorker` loads, warms, and owns
GateNet in its thread, uses a latest-frame mailbox, and exposes explicit
`start`, `warm`, `health`, `stop`, and `join`. Publish immutable records through
an atomic latest-result slot. Never run GateNet synchronously inside
`PerceptionEngine.detect()`.

- [ ] **Step 3: Fuse by continuity and source-time coherence**

GateNet supplies coarse quad/pose candidates. FastGate supplies terminal
aperture continuity only when its timestamp is within the frozen coherence
window and its aperture overlaps the projected GateNet track. Ribbon supplies
course membership. Require:

- GateNet+ribbon to acquire a stable gate track;
- FastGate agreement before any close/side maneuver that uses its aperture;
- a hold on GateNet/FastGate disagreement or stale source timestamps;
- FastGate through-hole width-drop and clipping rejection from the proven
  implementation;
- every accepted and rejected candidate recorded with reason.

FastGate never selects a course gate alone and GateNet per-frame IDs never
become persistent identities.

- [ ] **Step 4: Run perception and tracker tests**

Run: `python -m pytest vq2/tests/test_survey_perception.py vq2/tests/test_survey_tracker.py vq2/tests/test_survey_perception_scheduling.py -v`

Expected: all tests PASS with camera-rate CPU results during a blocked GateNet
fake and explicit disagreement behavior.

- [ ] **Step 5: Commit asynchronous fused perception**

```bash
git add vq2/survey/types.py vq2/survey/perception.py vq2/survey/tracker.py vq2/tests/test_survey_perception.py vq2/tests/test_survey_tracker.py vq2/tests/test_survey_perception_scheduling.py
git commit -m "fix(vq2): fuse asynchronous survey perception"
```

### Task 2: Implement an active, armed-aware 50 Hz command owner

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/arbiter.py`
- Test: `vq2/tests/test_survey_arbiter.py`
- Test: `vq2/tests/test_survey_arbiter_runner.py`

**Interfaces:**
- `CommandArbiter.start(sender, state_provider)`, `submit(command)`,
  `force_land(reason)`, `stop()`, and `join(timeout)`.
- `VelocityRateAdapter.convert(command, measured_state) -> RateThrustCommand`
  is the sole velocity-to-rate/thrust conversion and enforces the measured
  `RATE_MAX=0.6` and thrust envelope.
- Sender exceptions, policy TTL expiry, invalid/nonfinite commands, loop
  stalls, or missing measured state enter the same brake/descent failsafe.

- [ ] **Step 1: Write clocked runner and no-arm tests**

With a fake monotonic clock and sender, require approximately 50 sends per
second, bounded jitter, command validation, and automatic zero-velocity output
after TTL expiry. Inject a sender error and verify `force_land` latches. With
`armed=False` or `no_arm=True`, assert the sender receives no arm, flight,
land, or disarm messages—even during cleanup.

- [ ] **Step 2: Implement the exclusive actuation thread**

The thread wakes from monotonic deadlines, reads the latest validated command,
converts it with the latest measured attitude/rates, and sends one bounded
rate/thrust command. It publishes heartbeat, last-send time, state, failure,
and sequence number. `submit()` only updates an atomic mailbox. `stop()` is
idempotent; `join()` proves exit.

- [ ] **Step 3: Implement armed-aware brake, descent, and disarm**

`force_land` immediately replaces policy output with bounded level brake, then
controlled descent after speed/tilt settle, then zero thrust/disarm only after
fresh ground contact. Ground contact before takeoff is normal; contact after
takeoff outside the landing state is a collision. A hard tilt over `55 deg`
remains emergency; a conservative `35 deg` soft threshold brakes and lands if
not recovered. Survey speed over `1.0 m/s` is a soft emergency even though the
legacy hard sanity ceiling is higher.

- [ ] **Step 4: Verify axis/sign conversion against proven flight code**

Create table-driven tests for forward, right, up/down, and yaw commands using
the current sim/body conventions and the measured vq2wp signs. Assert
`RATE_MAX=0.6`, bounded thrust, zero command on nonfinite input, and no
environment-dependent sign defaults.

- [ ] **Step 5: Run arbiter suites**

Run: `python -m pytest vq2/tests/test_survey_arbiter.py vq2/tests/test_survey_arbiter_runner.py -v`

Expected: all tests PASS, including TTL, sender failure, shutdown, landing,
axis signs, and no-arm cleanup.

- [ ] **Step 6: Commit the command owner**

```bash
git add vq2/survey/types.py vq2/survey/arbiter.py vq2/tests/test_survey_arbiter.py vq2/tests/test_survey_arbiter_runner.py
git commit -m "fix(vq2): own survey actuation at 50 hz"
```

### Task 3: Implement the concrete live runtime and launcher

**Files:**
- Modify: `vq2/live/vq2survey.py`
- Modify: `vq2/live/run_survey.bat`
- Modify: `vq2/tests/test_survey_runtime_wiring.py`
- Test: `vq2/tests/test_survey_runtime_integration.py`

**Interfaces:**
- Defines concrete `SurveyRuntime`, `runtime_identity`,
  `build_sensor_snapshot`, `send_rate_thrust`, and `land_and_disarm` symbols;
  no placeholder runtime call remains.
- `SurveyRuntime` owns receiver, camera perception, GateNet worker, controller,
  recorder, and arbiter lifecycle, but delegates all actuation to the arbiter.
- Preflight proves DPVO disabled, unique output, disk reserve, stream health,
  GateNet load/warmup, FastGate/ribbon self-test, recorder health, and fresh
  race state before arm.

- [ ] **Step 1: Write a fake-MAVLink lifecycle integration test**

Exercise fresh reset verification, pad lock, GO, arm, one controller command,
manual abort, controlled landing, recorder finalization, and thread joins.
Assert the initial gate index and race-clock epoch are captured after reset;
clock regression, unexpected reset, gate-index regression/advance during a
bypass survey, collision staleness, and every worker death fail closed.

Run the same fixture with `SURVEY_NO_ARM=1`; require perception/recording but
zero outgoing actuation messages in all paths.

- [ ] **Step 2: Implement typed receive state and monotonic deadlines**

Adapt the proven vq2wp receiver pieces for raw JPEG passthrough, duplicate IMU
suppression, race-status decoding, and collision/ground distinction. Populate
typed snapshot fields; do not hide safety state in `extras`. Controller phase
and mission timeouts use monotonic time independent of recorder writes or race
clock. Verify the pause/restart procedure externally before starting, then
hard-reset the race and confirm a new zero epoch before `PAD_LOCK`.

- [ ] **Step 3: Implement explicit lifecycle and cleanup**

Start recorder and receiver, start camera perception, start/warm GateNet,
start the arbiter last, then enter the controller loop. On exit: stop policy,
if and only if the runtime actually armed then call `arbiter.force_land`, wait
for land/disarm, stop/join workers, and finalize the recorder. A no-arm run
only stops workers and finalizes. Check every exit code and worker join result.

- [ ] **Step 4: Make the launcher package-correct**

```bat
@echo off
setlocal
cd /d C:\Users\alexj\algo_src
set DPVO=0
python -m vq2.live.vq2survey
exit /b %ERRORLEVEL%
```

Keep pass/output/coverage/context variables explicit and validated in Python.
Do not launch `vq2survey.py` by a path whose working directory may hide the
`vq2` package.

- [ ] **Step 5: Run runtime and source-wiring tests**

Run: `python -m pytest vq2/tests/test_survey_runtime_wiring.py vq2/tests/test_survey_runtime_integration.py -v`

Expected: all tests PASS with every named symbol implemented, GateNet warmup
before readiness, and no-arm producing no actuation.

- [ ] **Step 6: Commit executable runtime wiring**

```bash
git add vq2/live/vq2survey.py vq2/live/run_survey.bat vq2/tests/test_survey_runtime_wiring.py vq2/tests/test_survey_runtime_integration.py
git commit -m "feat(vq2): wire the autonomous survey runtime"
```

### Task 4: Seal complete corpus content and enforce A3/B1 chronology

**Files:**
- Modify: `vq2/survey/recorder.py`
- Modify: `vq2/survey/replay.py`
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/mapping/extract.py`
- Modify: `vq2/mapping/benchmark.py`
- Modify: `vq2/tests/test_survey_recorder.py`
- Modify: `vq2/tests/test_survey_coverage.py`
- Modify: `vq2/tests/test_mapping_extract.py`
- Modify: `vq2/tests/test_map_benchmark.py`
- Test: `vq2/tests/test_survey_corpus_digest.py`

**Interfaces:**
- A completed corpus manifest contains per-stream hashes, a deterministic
  frame-tree hash, and `content_sha256`; the external manifest hash is recorded
  by consumers and seals.
- An aborted run may contain zero frames and `frame_block: null`; only
  `status="complete"` requires a non-null block and one-to-one source data.
- Coverage requires exactly one distinct complete A1 and A2 corpus. A3 binds
  the resulting coverage digest. B1 collection binds the already-frozen map
  and policy digests and never influences either.

- [ ] **Step 1: Test duplicate frames, zero-frame abort, and content mutation**

Reject duplicate/nonmonotonic `frame_ns` before creating a file. Finalize a
preflight abort with no frames and verify `frame_block is None`; verify the
mapper rejects it. Build a completed corpus, mutate each JSONL stream and one
JPEG byte in turn, and require verification/seal failure. A manifest-only hash
must not pass these tests.

- [ ] **Step 2: Compute a deterministic source-content digest**

Hash canonical relative path, length, and bytes for `mavlink.jsonl`,
`frames_dedup.jsonl`, `cmds.jsonl`, `detections.jsonl`, `tracks.jsonl`,
`events.jsonl`, and every sorted `frames/<frame_ns>.jpg`. Exclude temporary,
manifest, seal, and digest files. Write per-stream/frame-tree hashes plus the
combined `content_sha256` into `manifest.json`, which is written last. Mapping
recomputes all hashes before reading a frame.

- [ ] **Step 3: Enforce coverage input identity and A3 binding**

`coverage` accepts exactly one complete A1 and one complete A2 with different
manifest and content digests. It verifies both before deriving gaps. A3
manifest records `coverage_sha256` and the selected gap IDs; its runtime
refuses a missing/mismatched report. Mapping requires the A3 digest to point
back to the A1/A2 inputs it already received.

- [ ] **Step 4: Defer and bind B1 collection**

B1 preflight requires read-only paths to frozen `map_A_candidate/digest.sha256`
and `approval-policy.json`; it records their digests plus camera, geometry,
GateNet, and sim identities in `collection_context`. It does not load map poses
or use map control. After flight, the B1 seal contains both
`manifest_sha256` and `content_sha256` plus the frozen map/policy digests. Any
map or policy change requires a new B1 flight and seal.

- [ ] **Step 5: Implement exact replay CLI entry points**

```text
python -m vq2.survey.replay run <corpus> --profile A1 --output <report>
python -m vq2.survey.replay coverage <A1-manifest> <A2-manifest> --output <coverage>
python -m vq2.survey.replay seal <B1-manifest> --output <seal>
python -m vq2.survey.replay verify <manifest-or-corpus>
```

Use `vq2.corpus.load`. Add argparse tests for all four subcommands and their
exit codes. Add a synthetic gate-local A1/A2 replay that measures minimum
outer-frame clearance; fg62/fg75 can test threading/timestamps but cannot prove
a bypass envelope.

- [ ] **Step 6: Run corpus, coverage, mapping-input, and B1 tests**

Run: `python -m pytest vq2/tests/test_survey_recorder.py vq2/tests/test_survey_corpus_digest.py vq2/tests/test_survey_coverage.py vq2/tests/test_mapping_extract.py vq2/tests/test_map_benchmark.py -v`

Expected: all tests PASS, including source mutation, zero-frame abort,
duplicate timestamp, A3 linkage, B1 chronology, and CLI parsing.

- [ ] **Step 7: Commit content-bound corpora**

```bash
git add vq2/survey/recorder.py vq2/survey/replay.py vq2/mapping/schema.py vq2/mapping/extract.py vq2/mapping/benchmark.py vq2/tests/test_survey_recorder.py vq2/tests/test_survey_corpus_digest.py vq2/tests/test_survey_coverage.py vq2/tests/test_mapping_extract.py vq2/tests/test_map_benchmark.py
git commit -m "fix(vq2): bind survey corpus content"
```

---

## Addendum completion gate

An armed survey requires asynchronous fused perception, a live 50 Hz command
owner, executable runtime wiring, and passing no-arm/failsafe integration
tests. A3 requires verified A1/A2 coverage. B1 is flown only after map A and
policy freeze, and its seal binds every source byte plus that exact frozen
identity.
