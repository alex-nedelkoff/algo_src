# VQ2 Race Map Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate approved prior-map localization into the VQ2 race stack through shadow observation first, then digest-bound corrected-state control with FastGate terminal steering and stopped-hover GateNet recovery.

**Architecture:** A CPU map-pose worker consumes camera frames and raw DPVO poses, runs the approved localizer, updates the external Sim3, and publishes a high-rate corrected map-frame state without touching the KF. A route manager selects gates by stable `route_order` and advances only on judge ticks. Source selection remains legacy during observe-only trials and becomes map-corrected only when a separate control-approval artifact matches every runtime identity.

**Tech Stack:** Python 3, existing `vq2wp.py` MAVLink controller, approved mapping package, NumPy, OpenCV CPU localizer, pytest; DPVO owns normal race GPU residency.

## Global Constraints

- Depends on an immutable map, `localization-approval.json`, and passing wrapper replay from the prior-map plan.
- Normal race profile is DPVO GPU plus CPU PnP/FastGate; GateNet is unloaded.
- GateNet recovery is allowed only in stopped hover after explicit DPVO GPU release; it is not toggled at every gate.
- Judge `gate_idx` advances route order; a tick is not an exact camera-position correction.
- Map fixes are applied at source `frame_ns`; corrected state propagates to now using later DPVO motion.
- Observe-only mode is structurally unable to change control position.
- Control requires an exact digest match for map, policy, localization approval, camera, feature configuration, and DPVO identity.
- Map state older than `2.0 s`, unhealthy Sim3, nonfinite output, identity change, or route inconsistency is fail-closed.
- Preserve `RATE_MAX=0.6`, bounded thrust, collision guards, fresh-sim discipline, one-variable experiments, and `gidx2` scoring.
- Do not mix unrelated cleanup or refactoring into `vq2wp.py`; make narrow, source-tested insertions after concurrent edits settle.

---

## File structure

- Create `vq2/live/map_control.py`: identity gate, corrected-state health, and source selection.
- Create `vq2/live/map_pose_worker.py`: CPU localizer/alignment worker.
- Create `vq2/live/map_route.py`: stable route-order and judge progression.
- Create `vq2/live/gpu_ownership.py`: stopped-hover recovery state machine.
- Create `vq2/map_control_approve.py`: shadow-report to control-approval artifact.
- Modify `vq2/live/vq2wp.py`: state fields, observe-only wiring, route lookup, and later approved source selection.
- Create `vq2/live/fly_servo_map_observe.bat`: shadow launcher.
- Create `vq2/live/fly_servo_map_control.bat`: approval-bound control launcher.
- Add focused tests under `vq2/tests/test_map_*_live.py` and source-wiring tests.

### Task 1: Verify runtime identity and make observe-only selection impossible

**Files:**
- Create: `vq2/live/map_control.py`
- Test: `vq2/tests/test_map_control_live.py`

**Interfaces:**
- Produces: `MapRuntimeIdentity`, `MapPoseState`, `load_runtime_identity(map_path, localization_approval, dpvo_identity)`.
- Produces: `select_control_position(legacy_p, map_state, *, observe_only, approval) -> SourceDecision`.

- [ ] **Step 1: Write identity and observe-only tests**

```python
import numpy as np
import pytest

from vq2.live.map_control import MapPoseState, select_control_position


def state(healthy=True, age_s=0.1):
    return MapPoseState(np.array([10.0, 2.0, -1.0]),
                        np.array([0.0, 0.0, 0.0, 1.0]),
                        healthy, "ok", age_s, "map-1", "dpvo-1", 0.1)


def test_observe_only_always_returns_legacy_position():
    legacy = np.array([1.0, 0.0, 0.0])
    decision = select_control_position(legacy, state(), observe_only=True,
                                       approval={"approved": True})
    assert np.array_equal(decision.position, legacy)
    assert decision.source == "legacy"


def test_unapproved_map_cannot_control():
    decision = select_control_position(np.zeros(3), state(), observe_only=False,
                                       approval={"approved": False})
    assert decision.position is None
    assert decision.reason == "approval"


def test_stale_map_state_is_rejected():
    decision = select_control_position(np.zeros(3), state(age_s=2.1),
                                       observe_only=False,
                                       approval={"approved": True})
    assert decision.position is None
    assert decision.reason == "stale"
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `python -m pytest vq2/tests/test_map_control_live.py -v`

Expected: FAIL importing `vq2.live.map_control`.

- [ ] **Step 3: Implement fail-closed source selection**

```python
# vq2/live/map_control.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MapPoseState:
    position: np.ndarray
    q_xyzw: np.ndarray
    healthy: bool
    reason: str
    age_s: float
    map_id: str
    dpvo_identity: str
    scale_uncertainty: float


@dataclass(frozen=True)
class SourceDecision:
    position: np.ndarray | None
    source: str
    reason: str


def select_control_position(legacy_p, map_state, *, observe_only, approval):
    legacy_p = np.asarray(legacy_p, float)
    if observe_only:
        return SourceDecision(legacy_p.copy(), "legacy", "observe_only")
    if not approval or not approval.get("approved", False):
        return SourceDecision(None, "none", "approval")
    if map_state is None or not map_state.healthy:
        return SourceDecision(None, "none", "unhealthy")
    if map_state.age_s > 2.0:
        return SourceDecision(None, "none", "stale")
    if not np.isfinite(map_state.position).all():
        return SourceDecision(None, "none", "nonfinite")
    if map_state.scale_uncertainty > 0.10:
        return SourceDecision(None, "none", "scale_uncertainty")
    return SourceDecision(np.asarray(map_state.position, float).copy(),
                          "map", "approved")
```

- [ ] **Step 4: Add exact digest verification**

`load_runtime_identity()` reads the map manifest and localization approval,
recomputes their SHA-256 values, and requires exact camera, feature, map,
policy, and DPVO identities. Return an immutable record; raise `ValueError`
with the mismatched field name on any difference.

- [ ] **Step 5: Run tests**

Run: `python -m pytest vq2/tests/test_map_control_live.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the control gate**

```bash
git add vq2/live/map_control.py vq2/tests/test_map_control_live.py
git commit -m "feat(vq2): gate map control by identity"
```

### Task 2: Publish corrected map state from an isolated CPU worker

**Files:**
- Create: `vq2/live/map_pose_worker.py`
- Test: `vq2/tests/test_map_pose_worker_live.py`

**Interfaces:**
- Produces: `MapPoseWorker(index, localizer, alignment, publish, log).submit(frame_ns, image, dpvo_pose)`.
- Worker publishes immutable `MapPoseState`; it never receives `KF` and never calls `update_position`.

- [ ] **Step 1: Write publish, rejection, and no-KF tests**

```python
from pathlib import Path
import numpy as np

from vq2.live.map_pose_worker import MapPoseWorker


def test_worker_publishes_corrected_state_from_accepted_fix():
    published = []
    worker = MapPoseWorker(fake_index(), accepting_localizer(), fake_alignment(),
                           published.append, lambda row: None)
    worker.process_one(100, np.zeros((360, 640, 3), np.uint8), fake_dpvo_pose(100))
    assert published[-1].healthy
    assert published[-1].map_id == "map-1"


def test_rejected_query_logs_reason_without_publishing_healthy():
    published, rows = [], []
    worker = MapPoseWorker(fake_index(), rejecting_localizer("inliers"),
                           fake_alignment(), published.append, rows.append)
    worker.process_one(100, np.zeros((360, 640, 3), np.uint8), fake_dpvo_pose(100))
    assert rows[-1]["reason"] == "inliers"
    assert not published[-1].healthy


def test_worker_source_has_no_kf_mutation():
    source = (Path(__file__).parents[2] / "vq2/live/map_pose_worker.py").read_text()
    assert "update_position" not in source
    assert "PosVelKF" not in source
```

- [ ] **Step 2: Run tests and verify missing worker**

Run: `python -m pytest vq2/tests/test_map_pose_worker_live.py -v`

Expected: FAIL importing `MapPoseWorker`.

- [ ] **Step 3: Implement bounded latest-frame worker**

Use a queue of size one; a new frame replaces an unprocessed older frame and
increments `dropped_query_count`. `process_one()` adds the raw pose to history,
runs normal localization using the current corrected prior, feeds an accepted
fix to `MapAlignment`, propagates to the submitted DPVO pose, and publishes
health, reason, fix age, map ID, DPVO identity, scale, uncertainty, inliers,
RMS, and latency.

```python
def process_one(self, frame_ns, image, dpvo_pose):
    self.alignment.history.add(frame_ns, dpvo_pose)
    fix, attempt = self.localizer.localize(
        image, frame_ns, self.alignment.predicted_map_pose(dpvo_pose))
    if fix is not None:
        self.alignment.add_fix(dpvo_pose, fix)
    corrected = self.alignment.correct(frame_ns, dpvo_pose)
    state = MapPoseState(corrected.p, corrected.q, corrected.healthy,
                         corrected.reason, corrected.fix_age_s,
                         self.index.manifest["map_id"],
                         self.alignment.dpvo_identity,
                         corrected.scale_uncertainty)
    self.publish(state)
    self.log({"kind": "map_query", "frame_ns": frame_ns,
              "accepted": fix is not None, "reason": attempt.reason,
              "inliers": attempt.inliers, "rms_px": attempt.rms_px,
              "latency_ms": attempt.latency_ms})
```

- [ ] **Step 4: Run worker and mapping regression tests**

Run: `python -m pytest vq2/tests/test_map_pose_worker_live.py vq2/tests/test_map_localizer.py vq2/tests/test_map_alignment.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit worker**

```bash
git add vq2/live/map_pose_worker.py vq2/tests/test_map_pose_worker_live.py
git commit -m "feat(vq2): publish corrected map poses"
```

### Task 3: Select route gates by map order and advance only on judge ticks

**Files:**
- Create: `vq2/live/map_route.py`
- Test: `vq2/tests/test_map_route_live.py`

**Interfaces:**
- Produces: `MapRoute.from_artifact(map_path) -> MapRoute`.
- Produces: `target(active_gate_index) -> RouteGate` and `observe_tick(old_index, new_index, tick_ns) -> RouteEvent`.
- Route order is explicit and never sorted by world x/y.

- [ ] **Step 1: Write route-order and regression tests**

```python
import pytest

from vq2.live.map_route import MapRoute, RouteGate


def route():
    return MapRoute((RouteGate("G1", 0, (10, 0, -1), (-1, 0, 0)),
                     RouteGate("G2", 1, (5, 20, -1), (0, -1, 0))))


def test_route_uses_explicit_order_not_world_x():
    assert route().target(0).gate_id == "G1"
    assert route().target(1).gate_id == "G2"


def test_gate_index_regression_is_rejected():
    with pytest.raises(ValueError, match="regression"):
        route().observe_tick(1, 0, 100)


def test_tick_records_aperture_event_without_forcing_pose():
    event = route().observe_tick(0, 1, 123)
    assert event.completed_gate_id == "G1"
    assert event.tick_ns == 123
    assert not hasattr(event, "camera_position")
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `python -m pytest vq2/tests/test_map_route_live.py -v`

Expected: FAIL importing `vq2.live.map_route`.

- [ ] **Step 3: Implement immutable route records and tick logic**

```python
# vq2/live/map_route.py
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteGate:
    gate_id: str
    route_order: int
    center_m: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class RouteEvent:
    completed_gate_id: str
    new_index: int
    tick_ns: int


class MapRoute:
    def __init__(self, gates):
        ordered = sorted(gates, key=lambda gate: gate.route_order)
        if [gate.route_order for gate in ordered] != list(range(len(ordered))):
            raise ValueError("route_order must be contiguous from zero")
        self.gates = tuple(ordered)

    def target(self, active_gate_index):
        return self.gates[int(active_gate_index)]

    def observe_tick(self, old_index, new_index, tick_ns):
        if new_index < old_index:
            raise ValueError("judge gate index regression")
        if new_index != old_index + 1:
            raise ValueError("judge gate index skipped route entry")
        return RouteEvent(self.target(old_index).gate_id, int(new_index), int(tick_ns))
```

- [ ] **Step 4: Run route tests**

Run: `python -m pytest vq2/tests/test_map_route_live.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit route manager**

```bash
git add vq2/live/map_route.py vq2/tests/test_map_route_live.py
git commit -m "feat(vq2): follow explicit metric map route"
```

### Task 4: Wire map localization into `vq2wp.py` as observe-only

**Files:**
- Modify: `vq2/live/vq2wp.py:359-365`
- Modify: `vq2/live/vq2wp.py:1163-1205`
- Modify: `vq2/live/vq2wp.py:1925-1950`
- Create: `vq2/live/fly_servo_map_observe.bat`
- Test: `vq2/tests/test_map_observe_wiring.py`

**Interfaces:**
- Consumes environment `MAP_LOCALIZE=1`, `MAP_OBSERVE=1`, `MAP_PATH`, and `LOCALIZATION_APPROVAL`.
- Publishes `state['map_pose_state']` under `MAP_LOCK` and logs `map_query`, `map_pose`, and `map_compare`.
- Observe-only source remains legacy for every controller phase.

- [ ] **Step 1: Write source-level non-interference tests**

```python
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_map_worker_is_started_only_for_explicit_mode():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert "os.environ.get('MAP_LOCALIZE') == '1'" in source
    assert "MapPoseWorker" in source
    assert "state['map_pose_state']" in source


def test_observe_mode_passes_legacy_position_to_selector():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert "os.environ.get('MAP_OBSERVE') == '1'" in source
    assert "map_compare" in source
    assert "select_control_position" in source


def test_launcher_is_observe_only_and_keeps_gatenet_unloaded():
    source = (ROOT / "vq2/live/fly_servo_map_observe.bat").read_text()
    assert "MAP_OBSERVE=1" in source
    assert "DPVO_ROUTE=1" in source
    assert "GN_RELOAD=0" in source
```

- [ ] **Step 2: Run tests and verify missing wiring**

Run: `python -m pytest vq2/tests/test_map_observe_wiring.py -v`

Expected: FAIL because no map worker is wired.

- [ ] **Step 3: Add state, lock, worker lifecycle, and logs**

Add `map_pose_state=None`, `map_worker_healthy=False`, and identity fields to
the central state. Start the worker only after map/localization identities
verify and DPVO tracking starts. Feed each new source JPEG and paired raw DPVO
pose. On every route loop, call `select_control_position(_pp, map_state,
observe_only=True, approval=None)`, keep its legacy result, and log both states.
Stop and join the worker in the existing `finally`/landing cleanup path.

- [ ] **Step 4: Add the observe launcher with concrete identities**

```bat
@echo off
setlocal
set MAP_LOCALIZE=1
set MAP_OBSERVE=1
set MAP_PATH=C:\Users\alexj\vq2_maps\map_A_candidate
set LOCALIZATION_APPROVAL=C:\Users\alexj\vq2_maps\validation_B1_001\localization-approval.json
set DPVO_ROUTE=1
set DPVO_OBSERVE=1
set GN_RELOAD=0
call C:\Users\alexj\g2fresh.bat
exit /b %ERRORLEVEL%
```

- [ ] **Step 5: Run source wiring and existing DPVO route tests**

Run: `python -m pytest vq2/tests/test_map_observe_wiring.py vq2/tests/test_vq2wp_dpvo_route_wiring.py vq2/tests/test_dpvo_control_gate.py -v`

Expected: all tests PASS and existing observe-only behavior remains intact.

- [ ] **Step 6: Commit observe-only integration**

```bash
git add vq2/live/vq2wp.py vq2/live/fly_servo_map_observe.bat vq2/tests/test_map_observe_wiring.py
git commit -m "feat(vq2): shadow metric map poses"
```

### Task 5: Generate a digest-bound control-approval artifact from shadow data

**Files:**
- Create: `vq2/map_control_approve.py`
- Test: `vq2/tests/test_map_control_approval.py`

**Interfaces:**
- Produces: `evaluate_shadow(log_path, map_path, localization_approval) -> ControlApprovalReport`.
- Produces `map-control-approval-v1.json`; it never edits map or localization approval.

- [ ] **Step 1: Write pass/fail and digest tests**

```python
import json

from vq2.map_control_approve import evaluate_shadow


def test_shadow_with_stale_or_unhealthy_rows_is_not_approved(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"kind": "map_compare", "healthy": False,
                               "age_s": 3.0, "jump_m": 0.0}) + "\n")
    report = evaluate_shadow(log, "map-digest", "localization-digest")
    assert not report.approved
    assert "healthy_fraction" in report.failed


def test_approval_binds_all_input_digests(tmp_path):
    log = write_passing_shadow_log(tmp_path / "log.jsonl", rows=100)
    report = evaluate_shadow(log, "map-digest", "localization-digest")
    assert report.approved
    assert report.map_digest == "map-digest"
    assert report.localization_approval_digest == "localization-digest"
```

- [ ] **Step 2: Run tests and verify missing approval tool**

Run: `python -m pytest vq2/tests/test_map_control_approval.py -v`

Expected: FAIL importing `vq2.map_control_approve`.

- [ ] **Step 3: Implement immutable shadow acceptance**

Require at least 100 post-GO compare rows, `>=95%` healthy corrected states,
zero identity changes, zero nonfinite states, p95 fix age `<=2.0 s`, p95
source-time residual `<=0.5 m`, scale uncertainty `<=10%`, zero correction
jumps above `0.25 m` per control tick, no map-worker/service death, and
`MAP_OBSERVE=1` for every row. Hash log, map, policy, localization approval,
camera, feature configuration, and DPVO identity into the output.

```python
@dataclass(frozen=True)
class ControlApprovalReport:
    approved: bool
    failed: tuple[str, ...]
    map_digest: str
    localization_approval_digest: str
    shadow_log_digest: str
```

- [ ] **Step 4: Run approval tests**

Run: `python -m pytest vq2/tests/test_map_control_approval.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit approval tool**

```bash
git add vq2/map_control_approve.py vq2/tests/test_map_control_approval.py
git commit -m "feat(vq2): approve shadow map control"
```

### Task 6: Enable corrected-state route control without changing terminal FastGate

**Files:**
- Modify: `vq2/live/vq2wp.py:1925-1950`
- Create: `vq2/live/fly_servo_map_control.bat`
- Test: `vq2/tests/test_map_control_wiring.py`

**Interfaces:**
- Consumes `MAP_OBSERVE=0` and `MAP_CONTROL_APPROVAL`.
- Route/transit position may come from `SourceDecision(source="map")`.
- Existing FastGate pursuit remains the terminal bearing controller.

- [ ] **Step 1: Write approval, unhealthy, and terminal-controller tests**

```python
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_control_launcher_requires_approval_and_disables_observe():
    source = (ROOT / "vq2/live/fly_servo_map_control.bat").read_text()
    assert "MAP_OBSERVE=0" in source
    assert "MAP_CONTROL_APPROVAL" in source
    assert "GN_RELOAD=0" in source


def test_route_aborts_when_approved_map_source_is_unhealthy():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    assert "Map route unhealthy" in source
    assert "state['_route_source'] = 'map'" in source


def test_fastgate_terminal_phase_is_not_replaced_by_map_pnp():
    source = (ROOT / "vq2/live/vq2wp.py").read_text(encoding="utf-8")
    pursuit = source.index("if phase == 'fgpursuit':")
    assert source.index("state.get('fg_wall'", pursuit) > pursuit
```

- [ ] **Step 2: Run tests and verify missing control path**

Run: `python -m pytest vq2/tests/test_map_control_wiring.py -v`

Expected: FAIL because the map source is still shadow-only.

- [ ] **Step 3: Add the narrow approved source branch**

During route/transit, load the verified approval at preflight and call:

```python
decision = select_control_position(
    _pp,
    state.get("map_pose_state"),
    observe_only=os.environ.get("MAP_OBSERVE", "1") == "1",
    approval=state.get("map_control_approval"),
)
if decision.position is None:
    aborted = f"Map route unhealthy: {decision.reason}"
    jlog("map_abort", reason=decision.reason)
    continue
_pp = decision.position
state["_route_source"] = decision.source
```

Do not run this branch inside `fgpursuit`; FastGate continues to center and
carry through the aperture. The map route supplies the next gate's metric
approach and transition geometry only.

- [ ] **Step 4: Add the approval-bound launcher**

```bat
@echo off
setlocal
set MAP_LOCALIZE=1
set MAP_OBSERVE=0
set MAP_PATH=C:\Users\alexj\vq2_maps\map_A_candidate
set LOCALIZATION_APPROVAL=C:\Users\alexj\vq2_maps\validation_B1_001\localization-approval.json
set MAP_CONTROL_APPROVAL=C:\Users\alexj\vq2_maps\validation_B1_001\map-control-approval.json
set DPVO_ROUTE=1
set DPVO_OBSERVE=0
set GN_RELOAD=0
call C:\Users\alexj\g2fresh.bat
exit /b %ERRORLEVEL%
```

- [ ] **Step 5: Run map, DPVO, and wiring tests**

Run: `python -m pytest vq2/tests/test_map_control_wiring.py vq2/tests/test_map_control_live.py vq2/tests/test_map_pose_worker_live.py vq2/tests/test_map_route_live.py vq2/tests/test_dpvo_*.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit approved control branch**

```bash
git add vq2/live/vq2wp.py vq2/live/fly_servo_map_control.bat vq2/tests/test_map_control_wiring.py
git commit -m "feat(vq2): control from approved map pose"
```

### Task 7: Add stopped-hover GateNet recovery without per-gate toggling

**Files:**
- Create: `vq2/live/gpu_ownership.py`
- Modify: `vq2/live/vq2wp.py`
- Test: `vq2/tests/test_map_gpu_recovery_live.py`

**Interfaces:**
- Produces: `GpuOwnership` states `DPVO_ACTIVE`, `HOLDING`, `DPVO_RELEASED`, `GATENET_ACTIVE`, `RECOVERED`, and `FAILED`.
- Recovery request is accepted only with commanded zero velocity, measured speed below `0.15 m/s`, and map-localization unhealthy.

- [ ] **Step 1: Write moving-recovery and ownership tests**

```python
from vq2.live.gpu_ownership import GpuOwnership, OwnershipState


def test_recovery_cannot_load_gatenet_while_moving():
    owner = GpuOwnership()
    event = owner.request_recovery(speed_mps=0.3, commanded_hold=True,
                                   map_healthy=False)
    assert event.state is OwnershipState.HOLDING
    assert not event.load_gatenet


def test_dpvo_release_precedes_gatenet_load():
    owner = GpuOwnership()
    owner.request_recovery(0.0, True, False)
    released = owner.observe_dpvo_released(cuda_free_mb=2800)
    assert released.state is OwnershipState.DPVO_RELEASED
    load = owner.next_action()
    assert load.load_gatenet


def test_normal_gate_progress_never_requests_recovery():
    owner = GpuOwnership()
    assert owner.observe_gate_tick(1).state is OwnershipState.DPVO_ACTIVE
```

- [ ] **Step 2: Run tests and verify missing ownership state machine**

Run: `python -m pytest vq2/tests/test_map_gpu_recovery_live.py -v`

Expected: FAIL importing `vq2.live.gpu_ownership`.

- [ ] **Step 3: Implement explicit resource transitions**

The state machine emits actions rather than loading models itself. Require a
stopped hold, signal DPVO worker shutdown, wait for worker confirmation and at
least `2500 MB` reported CUDA free memory, then permit GateNet load. GateNet
may produce a recovery pose/route association while the vehicle remains
stopped. Normal route continuation requires GateNet unload, DPVO restart with
a new identity, and fresh Sim3 initialization; failure lands.

- [ ] **Step 4: Wire recovery as an explicit operator-enabled fallback**

Require `MAP_GATENET_RECOVERY=1`. Do not invoke recovery on every stale query;
first hold and retry CPU PnP for the frozen recovery timeout. Log every state,
CUDA measurement, model transition, and new DPVO identity.

- [ ] **Step 5: Run recovery and existing handoff tests**

Run: `python -m pytest vq2/tests/test_map_gpu_recovery_live.py vq2/tests/test_dpvo_prewarm.py vq2/tests/test_vq2wp_dpvo_route_wiring.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit recovery state machine**

```bash
git add vq2/live/gpu_ownership.py vq2/live/vq2wp.py vq2/tests/test_map_gpu_recovery_live.py
git commit -m "feat(vq2): recover map localization at hover"
```

### Task 8: Execute replay, observe-only, and control promotion

**Files:**
- Create one unique corpus per live attempt outside git.
- Append experiment rows to the Obsidian vault or `C:\Users\alexj\obsidian_outbox.md`.

**Interfaces:**
- Produces immutable shadow logs, control approval, and staged judge evidence.

- [ ] **Step 1: Run full offline replay before a fresh flight**

Run: `python -m vq2.mapping.benchmark replay C:\Users\alexj\vq2_maps\map_A_candidate C:\Users\alexj\vq2_servo_fg62 --output C:\Users\alexj\vq2_maps\race_integration_replay\fg62`

Expected: healthy corrected state across both tick windows, zero false accepts,
bounded source-time residual, and no control output.

- [ ] **Step 2: Run no-arm live localization soak**

Start the simulator in a verified fresh state, set the observe launcher to a
unique record path, and run without arming. Expected: DPVO GPU ownership,
GateNet unloaded, CPU map fixes, bounded query queue, and no source-selection
change.

- [ ] **Step 3: Run one fresh observe-only flight**

Run: `C:\Users\alexj\algo_src\vq2\live\fly_servo_map_observe.bat`

Expected: `gidx2` verifies the intended fresh tick evidence; every compare row
shows legacy control; map worker/service survives; no identity mismatch or
unbounded correction.

- [ ] **Step 4: Generate control approval only from the accepted shadow run**

Run: `python -m vq2.map_control_approve C:\Users\alexj\vq2_map_observe_001\log.jsonl --map C:\Users\alexj\vq2_maps\map_A_candidate --localization-approval C:\Users\alexj\vq2_maps\validation_B1_001\localization-approval.json --output C:\Users\alexj\vq2_maps\validation_B1_001\map-control-approval.json`

Expected: `approved=true` and exact matching digests. A failed report remains
immutable and does not authorize another source.

- [ ] **Step 5: Promote one-gate, two-gate, then full-course control**

Use a fresh simulator window and a unique corpus for each attempt. First cap
the route at one judge tick, then two, then full course. Score only with
`gidx2.py`. Any missing fresh tick is invalid; any map/source/service failure,
unexpected GateNet load, collision, or identity change is a failure. Change
one flight variable between attempts.

---

## Completion gate

This plan is complete only when observe-only shadowing proves non-interference,
the exact shadow log generates a digest-bound control approval, one-gate and
two-gate control pass fresh judge scoring, and the full course is traversed
with DPVO resident, CPU map localization healthy, FastGate terminal steering,
and no normal per-gate GateNet reload.

