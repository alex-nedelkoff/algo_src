# VQ2 Survey State-Machine Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make gate acquisition, the observation fan, route-end verification, and autonomous landing reachable from measured survey inputs without injected completion booleans.

**Architecture:** Stable ribbon-supported GateNet tracks trigger a stopped gate-acquisition state; a controller-owned queue of measured fan setpoints produces its own completion evidence; and a persistent visible ribbon terminus plus a bounded stopped scan distinguishes route end from ordinary ribbon loss. Completion requests landing through the command arbiter.

**Tech Stack:** Existing survey dataclasses/controller/tracker/recorder/arbiter, NumPy, pytest; no new dependency.

## Global Constraints

- This is the highest-precedence controller-state correction in the dated VQ2
  plan set.
- Apply it during final-safety Task 2 before an armed survey.
- Ribbon loss alone is never route completion. A timeout may land safely but
  produces `incomplete`, not `complete`.
- Every fan/route-end condition requires fresh coherent perception, finite
  measured motion, and recorder acknowledgement for its evidence frames.

---

### Task 1: Trigger acquisition from a stable ribbon-supported gate

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/tracker.py`
- Modify: `vq2/survey/controller.py`
- Modify: `vq2/tests/test_survey_controller_complete.py`
- Test: `vq2/tests/test_survey_gate_acquisition.py`

**Interfaces:**
- `GateAcquisitionEvidence` records track ID, source frame, consecutive-frame
  count, ribbon score, ambiguity margin, PnP range/covariance, detector
  agreement, image bearing, and measured speed.
- The controller owns all phase transitions; the runtime cannot set an
  `acquire_gate` boolean.

- [ ] **Step 1: Write the follow-to-acquire transition matrix**

From `FOLLOW_RIBBON`, require the same track for at least 5 fresh frames,
ribbon support at least `0.7`, frozen ambiguity margin at least `0.15`,
GateNet/FastGate agreement when FastGate is eligible, finite PnP range between
`3.5 m` and `8.0 m`, 3-sigma center uncertainty below the frozen bound, image
bearing under `20 deg`, and measured speed under `0.35 m/s`. Vary each field
below threshold and assert the controller remains in ribbon follow or holds;
no case enters acquisition ambiguously.

- [ ] **Step 2: Implement stopped `ACQUIRE_GATE` convergence**

On entry, latch the track and command forward speed toward zero. Use bounded
yaw, vertical, and standoff feedback until image center error, gate-relative
vertical error, standoff error, and speed are inside thresholds for 10 fresh
frames. Track change, stale pose, ribbon disagreement, or timeout holds/aborts.
Only this convergence evidence enters `OBSERVE_FAN`.

- [ ] **Step 3: Run acquisition tests**

Run: `python -m pytest vq2/tests/test_survey_gate_acquisition.py vq2/tests/test_survey_controller_complete.py -k 'follow or acquire' -v`

Expected: all tests PASS.

- [ ] **Step 4: Commit measured acquisition**

```bash
git add vq2/survey/types.py vq2/survey/tracker.py vq2/survey/controller.py vq2/tests/test_survey_gate_acquisition.py vq2/tests/test_survey_controller_complete.py
git commit -m "fix(vq2): trigger measured gate acquisition"
```

### Task 2: Execute and complete the observation fan internally

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/controller.py`
- Modify: `vq2/survey/recorder.py`
- Modify: `vq2/tests/test_survey_controller_complete.py`
- Test: `vq2/tests/test_survey_observation_fan.py`

**Interfaces:**
- `ObservationSetpoint` contains relative yaw, lateral, vertical, standoff,
  tolerances, and required committed frames.
- `ObservationFanProgress` records current setpoint, measured errors, dwell
  count, and committed frame IDs.
- Default safe fan uses centered standoff and relative yaw sequence
  `[-12 deg, 0 deg, +12 deg, 0 deg]`; A3 may replace it with the numeric
  coverage setpoints from the final-review closure.

- [ ] **Step 1: Write setpoint/dwell/acknowledgement tests**

For every default setpoint, require yaw error under `3 deg`, gate-relative
lateral/vertical/standoff errors under frozen bounds, speed under `0.10 m/s`,
fresh coherent track, and 5 distinct frames acknowledged committed by the
recorder. Missing record acknowledgement, stale detection, or a skipped
setpoint must prevent completion.

- [ ] **Step 2: Implement bounded fan commands**

The controller advances one setpoint at a time using bounded yaw plus
gate-relative position feedback. It resets dwell on any threshold violation.
It never integrates a caller-supplied `fan_complete`. After the final centered
setpoint succeeds, emit `TransitionEvidence(kind="fan_complete", ...)` and
enter `SHIFT_TO_BYPASS` with the same latched track.

- [ ] **Step 3: Test abort and recenter behavior**

Inject target loss, yaw overshoot, recorder stall, timeout, and manual abort.
All cases command no forward motion; timeout/manual abort land. Verify the
normal fan ends centered before lateral bypass begins.

- [ ] **Step 4: Run fan suites**

Run: `python -m pytest vq2/tests/test_survey_observation_fan.py vq2/tests/test_survey_controller_complete.py -k 'fan or bypass' -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the autonomous fan**

```bash
git add vq2/survey/types.py vq2/survey/controller.py vq2/survey/recorder.py vq2/tests/test_survey_observation_fan.py vq2/tests/test_survey_controller_complete.py
git commit -m "fix(vq2): execute the survey observation fan"
```

### Task 3: Verify a visible route terminus and land autonomously

**Files:**
- Modify: `vq2/survey/types.py`
- Modify: `vq2/survey/perception.py`
- Modify: `vq2/survey/controller.py`
- Modify: `vq2/live/vq2survey.py`
- Modify: `vq2/tests/test_survey_controller_complete.py`
- Test: `vq2/tests/test_survey_route_end.py`

**Interfaces:**
- `RibbonObservation` adds `terminal_visible`, terminal pixel/location
  confidence, skeleton support, and source timestamp.
- `VERIFY_ROUTE_END` is a stopped bounded scan, distinct from ordinary
  `REACQUIRE_RIBBON`.
- `RouteEndEvidence` requires a persistent visible terminus, completed scan,
  no ribbon-supported gate, low motion, and committed evidence frames.

- [ ] **Step 1: Write route-end versus ribbon-loss tests**

Ordinary ribbon disappearance, clipping, a dark frame, and a transient
skeleton endpoint must enter hold/reacquire, never completion. A synthetic
visible course terminus persisting for 20 fresh frames after at least one
completed bypass enters `VERIFY_ROUTE_END`.

- [ ] **Step 2: Implement the stopped verification scan**

Brake below `0.10 m/s`, retain the terminal track, then execute bounded yaw
setpoints `[-45 deg, 0 deg, +45 deg, 0 deg]`. At each setpoint require fresh
ribbon analysis, no eligible ribbon-supported GateNet track, and committed
frames. If a gate or continuing ribbon branch appears, return to
`FOLLOW_RIBBON`; if perception goes stale, hold/abort. Do not fly forward
during the scan.

- [ ] **Step 3: Produce the route-end and landing transition**

After the centered final scan setpoint, require the terminus still visible and
emit `RouteEndEvidence`. Transition `VERIFY_ROUTE_END -> COMPLETE ->
ABORT_LAND/LAND` by calling the arbiter's normal controlled-land request with
reason `survey_complete`. Mark the corpus complete only after ground contact,
disarm, writer drain, and content verification. A mission timeout uses reason
`route_incomplete` and an aborted/incomplete manifest.

- [ ] **Step 4: Replay reset through gate, terminus, and landing**

Run a deterministic replay through reset, GO, climb, ribbon follow, gate
acquisition, fan, closed-loop bypass, ribbon reacquisition, visible terminus,
verification scan, land, and finalized complete manifest. Assert every phase
transition has measured evidence and no injected completion boolean.

- [ ] **Step 5: Run route and complete-controller suites**

Run: `python -m pytest vq2/tests/test_survey_route_end.py vq2/tests/test_survey_controller_complete.py vq2/tests/test_survey_runtime_integration.py -k 'route or complete or land' -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit route completion**

```bash
git add vq2/survey/types.py vq2/survey/perception.py vq2/survey/controller.py vq2/live/vq2survey.py vq2/tests/test_survey_route_end.py vq2/tests/test_survey_controller_complete.py
git commit -m "fix(vq2): verify and land at survey end"
```

---

## State-machine completion gate

The survey controller is executable end to end only when a replay reaches
controlled landing through measured acquisition, internally commanded fan,
closed-loop bypass, ribbon reacquisition, and verified route-end evidence.
Ribbon loss or timeout alone never produces a complete corpus.
