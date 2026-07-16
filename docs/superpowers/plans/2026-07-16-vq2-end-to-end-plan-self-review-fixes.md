# VQ2 End-to-End Plan Self-Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the concrete coverage, placeholder, and type-consistency gaps found while self-reviewing the four VQ2 end-to-end implementation plans.

**Architecture:** This is a normative companion to the four dated VQ2 plans. Each correction is executed at the named insertion point before the following task begins. Where a correction conflicts with the corresponding base plan, this document takes precedence.

**Tech Stack:** The same Python, NumPy, OpenCV, SciPy, pymavlink, DPVO bridge, and pytest stack declared by the four base plans; no new dependency.

## Global Constraints

- Read this document together with all four `2026-07-16-vq2-*` implementation plans.
- Keep the survey, mapping, localization, and race-integration promotion gates separate.
- All concrete paths below are examples for attempt `001`; every live retry changes only the final attempt number and uses a unique directory.
- Do not execute a later base-plan task until its applicable self-review correction passes.

---

### Task 1: Complete the survey state machine and safety guards

**Insertion point:** Execute after Task 6 and before Task 7 of
`2026-07-16-vq2-autonomous-survey-acquisition.md`.

**Files:**
- Modify: `vq2/survey/controller.py`
- Create: `vq2/tests/test_survey_controller_complete.py`

**Interfaces:**
- Completes every declared `SurveyPhase` transition.
- Adds `collision_serious`, `speed_mps`, `tilt_rad`, `gate_behind`, `fan_complete`, `bypass_complete`, and `route_end` inputs through `SurveySnapshot.extras`.
- Phase timeouts are explicit and a timeout always lands.

- [ ] **Step 1: Write missing transition and guard tests**

```python
import math

from vq2.survey.controller import SurveyController
from vq2.survey.types import RibbonObservation, SurveyPhase, profile_for
from vq2.tests.test_survey_controller_guidance import selected_track
from vq2.tests.test_survey_controller_lifecycle import snap


def test_reset_pad_lock_go_climb_reaches_ribbon_follow():
    controller = SurveyController(profile_for("A1"))
    assert controller.step(snap(now_s=0.0)).phase is SurveyPhase.RESET_SETTLE
    assert controller.step(snap(now_s=6.1, extras={"reset_verified": True})).phase is SurveyPhase.PAD_LOCK
    assert controller.step(snap(now_s=6.2, track=selected_track())).phase is SurveyPhase.WAIT_GO
    assert controller.step(snap(now_s=7.0, race_go=True)).phase is SurveyPhase.ARM_LEVEL_CLIMB
    assert controller.step(snap(now_s=9.0, race_go=True, armed=True,
                                extras={"climb_complete": True})).phase is SurveyPhase.FOLLOW_RIBBON


def test_observation_bypass_and_reacquire_complete_one_segment():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.OBSERVE_FAN
    assert controller.step(snap(race_go=True, armed=True, track=selected_track(),
                                extras={"fan_complete": True})).phase is SurveyPhase.BYPASS
    assert controller.step(snap(race_go=True, armed=True, track=selected_track(),
                                extras={"bypass_complete": True,
                                        "gate_behind": True})).phase is SurveyPhase.REACQUIRE_RIBBON
    ribbon = RibbonObservation(1, 1.0, (0.0, -1.0), ((320.0, 300.0),))
    assert controller.step(snap(race_go=True, armed=True, ribbon=ribbon)).phase is SurveyPhase.FOLLOW_RIBBON


def test_route_end_brakes_then_completes():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.FOLLOW_RIBBON
    decision = controller.step(snap(race_go=True, armed=True,
                                    extras={"route_end": True}))
    assert decision.phase is SurveyPhase.COMPLETE
    assert decision.command.forward_mps == 0.0


def test_collision_tilt_and_speed_each_abort():
    for extras, reason in (({"collision_serious": True}, "collision"),
                           ({"tilt_rad": math.radians(56)}, "tilt"),
                           ({"speed_mps": 6.1}, "speed")):
        controller = SurveyController(profile_for("A1"))
        controller.phase = SurveyPhase.FOLLOW_RIBBON
        decision = controller.step(snap(race_go=True, armed=True, extras=extras))
        assert decision.phase is SurveyPhase.ABORT_LAND
        assert decision.reason == reason
```

- [ ] **Step 2: Run tests and verify missing transitions**

Run: `python -m pytest vq2/tests/test_survey_controller_complete.py -v`

Expected: FAIL on the first unimplemented transition.

- [ ] **Step 3: Add exact safety and timeout gates**

```python
# Add to SurveyController
    PHASE_TIMEOUT_S = {
        SurveyPhase.RESET_SETTLE: 10.0,
        SurveyPhase.PAD_LOCK: 20.0,
        SurveyPhase.WAIT_GO: 20.0,
        SurveyPhase.ARM_LEVEL_CLIMB: 5.0,
        SurveyPhase.ACQUIRE_GATE: 8.0,
        SurveyPhase.OBSERVE_FAN: 8.0,
        SurveyPhase.BYPASS: 8.0,
        SurveyPhase.REACQUIRE_RIBBON: 8.0,
    }

    def _physical_reason(self, snapshot):
        if snapshot.extras.get("collision_serious", False):
            return "collision"
        if float(snapshot.extras.get("tilt_rad", 0.0)) > math.radians(55.0):
            return "tilt"
        if float(snapshot.extras.get("speed_mps", 0.0)) > 6.0:
            return "speed"
        timeout = self.PHASE_TIMEOUT_S.get(self.phase)
        if timeout is not None and snapshot.now_s - self._phase_started_s > timeout:
            return f"phase_timeout:{self.phase.name.lower()}"
        return None

    def _transition(self, phase, snapshot, reason):
        self.phase = phase
        self._phase_started_s = snapshot.now_s
        return self._decision(SurveyCommand(), reason, transition=phase.name)
```

Call `_physical_reason()` immediately after `_health_reason()`. Add explicit
branches matching the tests: reset verification to pad lock, stable pad lock to
wait GO, completed climb to ribbon follow, completed fan to bypass, completed
bypass with `gate_behind` to reacquisition, and `route_end` to complete.

- [ ] **Step 4: Run all survey-controller tests**

Run: `python -m pytest vq2/tests/test_survey_controller_*.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit completed survey behavior**

```bash
git add vq2/survey/controller.py vq2/tests/test_survey_controller_complete.py
git commit -m "fix(vq2): complete survey safety states"
```

### Task 2: Use concrete survey and map output paths

**Insertion point:** Treat these commands as replacements for angle-bracket
commands in Tasks 8-10 of the survey plan and Task 8 of the map plan.

**Files:**
- No source files; these are execution-command corrections.

- [ ] **Step 1: Use concrete survey directories**

```text
C:\Users\alexj\vq2_survey_A1_001
C:\Users\alexj\vq2_survey_A2_001
C:\Users\alexj\vq2_survey_A3_001
C:\Users\alexj\vq2_survey_B1_001
```

- [ ] **Step 2: Score the concrete A1 corpus**

Run: `python C:\Users\alexj\algo_src\vq2\gidx2.py C:\Users\alexj\vq2_survey_A1_001`

Expected: no fresh transition for a clean bypass survey.

- [ ] **Step 3: Freeze A1/A2 coverage**

Run: `python -m vq2.survey.replay coverage C:\Users\alexj\vq2_survey_A1_001\manifest.json C:\Users\alexj\vq2_survey_A2_001\manifest.json --output C:\Users\alexj\vq2_survey_coverage_A12.json`

Expected: only A1/A2 digests.

- [ ] **Step 4: Seal B1**

Run: `python -m vq2.survey.replay seal C:\Users\alexj\vq2_survey_B1_001\manifest.json --output C:\Users\alexj\vq2_survey_B1_001\B1.seal.json`

Expected: `opened=false`.

- [ ] **Step 5: Use the concrete map output**

All map-build commands write to `C:\Users\alexj\vq2_maps\map_A_candidate`;
the final content digest is stored inside that directory rather than embedded
in the directory name.

### Task 3: Add canonical rendered-gate geometry and normalization telemetry

**Insertion point:** Execute after Task 1 and before Task 2 of
`2026-07-16-vq2-offline-metric-map.md`; normalization changes join its Task 3
commit.

**Files:**
- Create: `vq2/mapping/gate_geometry.py`
- Modify: `vq2/live/bridge_dpvo.py`
- Test: `vq2/tests/test_mapping_gate_geometry.py`
- Modify test: `vq2/tests/test_mapping_dpvo_batch.py`

**Interfaces:**
- Produces one canonical front/back eight-corner gate model with outer width
  `2.741 m`, aperture width `1.457 m`, and depth `0.128 m`.
- Produces DPVO `normalization_factors` and keyframe indices in the final
  trajectory response without editing the external DPVO optimizer.

- [ ] **Step 1: Write geometry shape and dimension tests**

```python
import numpy as np

from vq2.mapping.gate_geometry import GATE_CORNERS_8, dimensions


def test_rendered_gate_geometry_is_eight_front_back_outer_corners():
    assert GATE_CORNERS_8.shape == (8, 3)
    assert np.isclose(np.ptp(GATE_CORNERS_8[:, 0]), 2.741)
    assert np.isclose(np.ptp(GATE_CORNERS_8[:, 2]), 2.741)
    assert np.isclose(np.ptp(GATE_CORNERS_8[:, 1]), 0.128)
    assert dimensions()["aperture_width_m"] == 1.457
```

- [ ] **Step 2: Implement the sole metric gate model**

```python
# vq2/mapping/gate_geometry.py
import numpy as np

OUTER_WIDTH_M = 2.741
APERTURE_WIDTH_M = 1.457
DEPTH_M = 0.128
_h, _d = OUTER_WIDTH_M / 2.0, DEPTH_M / 2.0
GATE_CORNERS_8 = np.array([
    [-_h, -_d, -_h], [_h, -_d, -_h], [_h, -_d, _h], [-_h, -_d, _h],
    [-_h, _d, -_h], [_h, _d, -_h], [_h, _d, _h], [-_h, _d, _h],
], dtype=float)


def dimensions():
    return {"outer_width_m": OUTER_WIDTH_M,
            "aperture_width_m": APERTURE_WIDTH_M,
            "depth_m": DEPTH_M}
```

- [ ] **Step 3: Wrap DPVO normalization for telemetry only**

Immediately after `_load_runtime()` creates `slam`, wrap its graph method and
store the factor before each original call:

```python
normalization_events = []
original_normalize = slam.pg.normalize


def normalize_with_telemetry():
    factor = float(slam.pg.patches_[:slam.n, :, 2].mean().detach().cpu())
    normalization_events.append({"keyframe_n": int(slam.n), "factor": factor})
    return original_normalize()


slam.pg.normalize = normalize_with_telemetry
```

Return `normalization_events` from `_load_runtime()` and include it in the
post-termination `trajectory` response. Do not change factor values or call
frequency.

- [ ] **Step 4: Test final response normalization provenance**

Add a fake runtime test asserting the final response contains an ordered list
of `{keyframe_n, factor}` rows and `DpvoTrajectory.normalization_factors`
preserves their factors.

Run: `python -m pytest vq2/tests/test_mapping_gate_geometry.py vq2/tests/test_mapping_dpvo_batch.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit geometry and telemetry**

```bash
git add vq2/mapping/gate_geometry.py vq2/live/bridge_dpvo.py vq2/tests/test_mapping_gate_geometry.py vq2/tests/test_mapping_dpvo_batch.py
git commit -m "feat(vq2): record canonical gate and DPVO gauge"
```

### Task 4: Define localization fixtures, orientation checks, and alignment records

**Insertion point:** Apply before running Tasks 2-6 of
`2026-07-16-vq2-prior-map-localization.md`.

**Files:**
- Modify: `vq2/tests/test_map_localizer.py`
- Modify: `vq2/mapping/localizer.py`
- Modify: `vq2/mapping/map_alignment.py`
- Modify: `vq2/tests/test_map_benchmark.py`

**Interfaces:**
- Defines every helper used by the base-plan tests.
- Enforces both position and orientation innovation.
- Defines `Similarity`, `AlignmentState`, and `DpvoPoseHistory` explicitly.

- [ ] **Step 1: Add concrete localizer test helpers**

```python
def fake_gate_only_correspondences(count):
    uv = np.column_stack([np.linspace(250, 390, count),
                          np.linspace(120, 240, count)])
    xyz = np.column_stack([np.linspace(-1, 1, count),
                           np.zeros(count), np.full(count, 8.0)])
    return {"uv": uv, "xyz": xyz, "gate_ids": np.ones(count, int),
            "background_groups": np.zeros(count, int)}


def shuffled_correspondences(seed):
    rng = np.random.default_rng(seed)
    data = fake_gate_only_correspondences(80)
    data["xyz"] = data["xyz"][rng.permutation(len(data["xyz"]))]
    data["gate_ids"][:] = -1
    data["background_groups"] = np.arange(80) % 2
    return data


@pytest.fixture
def query_image():
    return np.zeros((360, 640, 3), np.uint8)


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    path = write_map(tmp_path)
    monkeypatch.setattr("vq2.mapping.localizer.verify_artifact",
                        lambda root: json.loads((root / "manifest.json").read_text()))
    return MapIndex.from_artifact(path)
```

- [ ] **Step 2: Correct the synthetic PnP policy and add orientation innovation**

The synthetic origin test uses:

```python
PnPPolicy(min_inliers=10, min_grid_cells=1,
          max_innovation_m=2.0, max_innovation_deg=15.0)
```

Add `prior_q_xyzw` to `solve_absolute_pose()`. After inverting PnP, compute:

```python
angle_deg = np.degrees((Rotation.from_quat(prior_q_xyzw).inv()
                        * Rotation.from_quat(q_wc)).magnitude())
if angle_deg > policy.max_innovation_deg:
    return None, LocalizationAttempt(False, "orientation_innovation",
                                     len(indices), rms, cells, latency_ms)
```

- [ ] **Step 3: Define alignment records before `estimate_similarity()`**

```python
@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points):
        points = np.asarray(points, float)
        return self.scale * (self.rotation @ points.T).T + self.translation


@dataclass(frozen=True)
class AlignmentState:
    healthy: bool
    reason: str
    similarity: Similarity | None
    pair_count: int
    outlier_count: int
    baseline_m: float
    residual_rms_m: float
    scale_uncertainty: float


class DpvoPoseHistory:
    def __init__(self):
        self.rows = []
        self.healthy = True

    def add(self, frame_ns, pose):
        if self.rows and int(frame_ns) <= self.rows[-1][0]:
            self.healthy = False
            raise ValueError("timestamp regression")
        self.rows.append((int(frame_ns), pose))
```

- [ ] **Step 4: Add concrete holdout test helper**

```python
def matching_holdout_files(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"pass":"B1"}\n')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"pass": "B1", "opened": False,
                                "manifest_sha256": digest}))
    return manifest, seal
```

- [ ] **Step 5: Run localization, alignment, and benchmark suites**

Run: `python -m pytest vq2/tests/test_map_localizer.py vq2/tests/test_map_alignment.py vq2/tests/test_map_benchmark.py -v`

Expected: all tests PASS with no undefined fixture or helper.

- [ ] **Step 6: Commit self-contained localization tests and types**

```bash
git add vq2/mapping/localizer.py vq2/mapping/map_alignment.py vq2/tests/test_map_localizer.py vq2/tests/test_map_benchmark.py
git commit -m "fix(vq2): complete map localization contracts"
```

### Task 5: Define race-worker fixtures and rate-limit map corrections

**Insertion point:** Apply before Tasks 2 and 5 of
`2026-07-16-vq2-race-map-integration.md`.

**Files:**
- Modify: `vq2/live/map_control.py`
- Modify: `vq2/tests/test_map_pose_worker_live.py`
- Modify: `vq2/tests/test_map_control_approval.py`
- Modify: `vq2/tests/test_map_control_live.py`

**Interfaces:**
- Defines every fake used by worker tests.
- Produces `CorrectionLimiter(max_position_rate_mps=0.5)`; no accepted map fix
  may create a larger published-state step than elapsed time permits.

- [ ] **Step 1: Add concrete worker fakes**

```python
from types import SimpleNamespace


def fake_index():
    return SimpleNamespace(manifest={"map_id": "map-1"})


def fake_dpvo_pose(frame_ns):
    return SimpleNamespace(frame_ns=frame_ns, p=np.zeros(3),
                           q=np.array([0.0, 0.0, 0.0, 1.0]))


def accepting_localizer():
    attempt = SimpleNamespace(reason="accepted", inliers=40,
                              rms_px=1.0, latency_ms=10.0)
    fix = SimpleNamespace(frame_ns=100, p_map_m=(0.0, 0.0, 0.0))
    return SimpleNamespace(localize=lambda image, frame_ns, prior: (fix, attempt))


def rejecting_localizer(reason):
    attempt = SimpleNamespace(reason=reason, inliers=5,
                              rms_px=9.0, latency_ms=10.0)
    return SimpleNamespace(localize=lambda image, frame_ns, prior: (None, attempt))


def fake_alignment():
    corrected = SimpleNamespace(p=np.zeros(3),
                                q=np.array([0.0, 0.0, 0.0, 1.0]),
                                healthy=True, reason="ok", fix_age_s=0.1,
                                scale_uncertainty=0.01)
    history = SimpleNamespace(add=lambda frame_ns, pose: None)
    return SimpleNamespace(history=history, dpvo_identity="dpvo-1",
                           predicted_map_pose=lambda pose: np.zeros(3),
                           add_fix=lambda pose, fix: None,
                           correct=lambda frame_ns, pose: corrected)
```

- [ ] **Step 2: Add concrete passing-shadow helper**

```python
def write_passing_shadow_log(path, rows):
    payload = {"kind": "map_compare", "healthy": True, "age_s": 0.2,
               "jump_m": 0.01, "source_residual_m": 0.1,
               "scale_uncertainty": 0.02, "map_observe": True,
               "identity_changed": False, "service_alive": True}
    path.write_text("".join(json.dumps(payload) + "\n" for _ in range(rows)))
    return path
```

- [ ] **Step 3: Write the correction-limiter test**

```python
from vq2.live.map_control import CorrectionLimiter


def test_map_correction_is_rate_limited():
    limiter = CorrectionLimiter(max_position_rate_mps=0.5)
    assert np.allclose(limiter.update(np.zeros(3), 0.0), np.zeros(3))
    limited = limiter.update(np.array([10.0, 0.0, 0.0]), 0.1)
    assert np.allclose(limited, [0.05, 0.0, 0.0])
```

- [ ] **Step 4: Implement the limiter and apply it before publishing control state**

```python
class CorrectionLimiter:
    def __init__(self, max_position_rate_mps=0.5):
        self.max_position_rate_mps = float(max_position_rate_mps)
        self._position = None
        self._time_s = None

    def update(self, target, now_s):
        target = np.asarray(target, float)
        if self._position is None:
            self._position, self._time_s = target.copy(), float(now_s)
            return self._position.copy()
        dt = max(0.0, float(now_s) - self._time_s)
        delta = target - self._position
        distance = np.linalg.norm(delta)
        maximum = self.max_position_rate_mps * dt
        if distance > maximum > 0.0:
            delta *= maximum / distance
        self._position += delta
        self._time_s = float(now_s)
        return self._position.copy()
```

Use the limiter only for the published corrected control state; retain the raw
PnP/map estimate and residual in telemetry. Reset it on identity change or
unhealthy alignment.

- [ ] **Step 5: Run race helper and limiter tests**

Run: `python -m pytest vq2/tests/test_map_pose_worker_live.py vq2/tests/test_map_control_approval.py vq2/tests/test_map_control_live.py -v`

Expected: all tests PASS with no undefined helper and the `0.05 m` bound.

- [ ] **Step 6: Commit race self-review fixes**

```bash
git add vq2/live/map_control.py vq2/tests/test_map_pose_worker_live.py vq2/tests/test_map_control_approval.py vq2/tests/test_map_control_live.py
git commit -m "fix(vq2): bound live map corrections"
```

---

## Self-review completion gate

The four base plans are ready for execution only after these corrections are
included in task scheduling. This companion closes the spec-coverage gaps for
survey completion and physical guards, canonical gate geometry, DPVO gauge
telemetry, localization orientation/type consistency, concrete test fixtures,
concrete run paths, and live correction-rate limiting.

