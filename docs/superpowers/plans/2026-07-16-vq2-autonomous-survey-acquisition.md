# VQ2 Autonomous Survey Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replayable, fail-closed autonomous survey controller that follows the cyan course ribbon, observes and bypasses gates, and records immutable A1/A2/A3/B1 corpora with DPVO disabled.

**Architecture:** A pure state machine consumes timestamped health, ribbon, GateNet, FastGate, and inertial snapshots and emits expiring velocity/yaw commands. A single arbiter owns MAVLink actuation, while an independent recorder writes the source JPEG stream and versioned audit logs. The live script becomes thin wiring around components that are first exercised through corpus replay.

**Tech Stack:** Python 3, NumPy, OpenCV, SciPy, pymavlink, pytest; no new dependencies and no DPVO process during survey flights.

## Global Constraints

- Canonical camera identity is `640x360`, `fx=fy=226.0`, `cx=319.5`, `cy=179.5`, and 20-degree upward camera pitch from `vq2/camera.py`.
- Survey GPU ownership is GateNet only; assert DPVO is disabled before arming.
- A1 is left bypass, A2 is right bypass, A3 is coverage-directed, and B1 is sealed validation.
- Initial bypass centerline is outer half-width plus `1.0 m`, never less than `2.5 m` from gate center.
- Reserve `20 GB` free on C:, cap each pass at `1 GB` and `300 s`, and keep one source JPEG tree.
- A command has one owner, expires if stale, and may not move forward on stale or ambiguous perception.
- Do not modify `vq2/live/vq2wp.py` or `vq2/live/dpvo_odom.py` in this plan.
- Every live promotion changes one survey-profile variable and uses a unique recording directory.

---

## File structure

- Create `vq2/survey/__init__.py`: public survey API.
- Create `vq2/survey/types.py`: immutable records, phases, and pass profiles.
- Create `vq2/survey/recorder.py`: bounded corpus writer and atomic manifest.
- Create `vq2/survey/perception.py`: ribbon, GateNet, and FastGate adapters.
- Create `vq2/survey/tracker.py`: temporal association and target eligibility.
- Create `vq2/survey/controller.py`: pure mission state machine and guidance.
- Create `vq2/survey/arbiter.py`: expiring single-owner command output.
- Create `vq2/survey/replay.py`: corpus replay and coverage report.
- Modify `vq2/live/fastgate.py`: canonical, injectable camera calibration.
- Replace `vq2/live/vq2survey.py`: thin runtime composition only.
- Create `vq2/live/run_survey.bat`: explicit pass launcher.
- Create focused tests under `vq2/tests/test_survey_*.py`.

### Task 1: Freeze survey records and pass profiles

**Files:**
- Create: `vq2/survey/__init__.py`
- Create: `vq2/survey/types.py`
- Test: `vq2/tests/test_survey_types.py`

**Interfaces:**
- Produces: `SurveyPhase`, `SurveyPass`, `PassProfile`, `HealthSnapshot`, `GateCandidate`, `RibbonObservation`, `SurveySnapshot`, `SurveyCommand`, and `SurveyDecision`.
- Produces: `profile_for(name: str) -> PassProfile`.

- [ ] **Step 1: Write the failing profile tests**

```python
import pytest

from vq2.survey.types import SurveyPass, profile_for


def test_frozen_profiles_encode_collection_roles():
    assert profile_for("A1").survey_pass is SurveyPass.A1
    assert profile_for("A1").bypass_side == -1
    assert profile_for("A2").bypass_side == 1
    assert profile_for("A3").coverage_report_required
    assert profile_for("B1").sealed_holdout


def test_profiles_freeze_safety_and_disk_limits():
    profile = profile_for("A1")
    assert profile.bypass_margin_m == 1.0
    assert profile.min_gate_center_offset_m == 2.5
    assert profile.disk_reserve_bytes == 20 * 1024**3
    assert profile.max_run_bytes == 1024**3
    assert profile.max_run_s == 300.0


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown survey pass"):
        profile_for("B2")
```

- [ ] **Step 2: Run the test and verify the missing package failure**

Run: `python -m pytest vq2/tests/test_survey_types.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'vq2.survey'`.

- [ ] **Step 3: Add the immutable public types**

```python
# vq2/survey/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping


class SurveyPass(str, Enum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    B1 = "B1"


class SurveyPhase(Enum):
    PREFLIGHT = auto()
    RESET_SETTLE = auto()
    PAD_LOCK = auto()
    WAIT_GO = auto()
    ARM_LEVEL_CLIMB = auto()
    FOLLOW_RIBBON = auto()
    ACQUIRE_GATE = auto()
    OBSERVE_FAN = auto()
    BYPASS = auto()
    REACQUIRE_RIBBON = auto()
    COMPLETE = auto()
    ABORT_LAND = auto()


@dataclass(frozen=True)
class PassProfile:
    survey_pass: SurveyPass
    bypass_side: int
    coverage_report_required: bool = False
    sealed_holdout: bool = False
    bypass_margin_m: float = 1.0
    min_gate_center_offset_m: float = 2.5
    disk_reserve_bytes: int = 20 * 1024**3
    max_run_bytes: int = 1024**3
    max_run_s: float = 300.0


PROFILES = {
    "A1": PassProfile(SurveyPass.A1, -1),
    "A2": PassProfile(SurveyPass.A2, 1),
    "A3": PassProfile(SurveyPass.A3, -1, coverage_report_required=True),
    "B1": PassProfile(SurveyPass.B1, 1, sealed_holdout=True),
}


def profile_for(name: str) -> PassProfile:
    try:
        return PROFILES[str(name).upper()]
    except KeyError:
        raise ValueError(f"unknown survey pass: {name}") from None


@dataclass(frozen=True)
class HealthSnapshot:
    heartbeat_age_s: float
    imu_age_s: float
    camera_age_s: float
    command_age_s: float
    workers_alive: bool
    recorder_healthy: bool
    disk_free_bytes: int
    dpvo_disabled: bool


@dataclass(frozen=True)
class GateCandidate:
    detection_id: int
    frame_ns: int
    center_uv: tuple[float, float]
    t_cam_m: tuple[float, float, float]
    outer_half_width_m: float
    score: float
    clipped: bool = False


@dataclass(frozen=True)
class RibbonObservation:
    frame_ns: int
    confidence: float
    tangent_uv: tuple[float, float]
    points_uv: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class SurveySnapshot:
    now_s: float
    frame_ns: int
    race_clock_ms: int
    gate_index: int
    race_go: bool
    armed: bool
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    health: HealthSnapshot
    ribbon: RibbonObservation | None = None
    candidates: tuple[GateCandidate, ...] = ()
    track: object | None = None
    extras: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SurveyCommand:
    forward_mps: float = 0.0
    right_mps: float = 0.0
    up_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    arm: bool = False
    land: bool = False


@dataclass(frozen=True)
class SurveyDecision:
    phase: SurveyPhase
    command: SurveyCommand
    reason: str
    telemetry: Mapping[str, object] = field(default_factory=dict)
```

```python
# vq2/survey/__init__.py
from .types import PassProfile, SurveyDecision, SurveyPass, SurveyPhase, profile_for

__all__ = ["PassProfile", "SurveyDecision", "SurveyPass", "SurveyPhase", "profile_for"]
```

- [ ] **Step 4: Run the profile tests**

Run: `python -m pytest vq2/tests/test_survey_types.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit the frozen contracts**

```bash
git add vq2/survey/__init__.py vq2/survey/types.py vq2/tests/test_survey_types.py
git commit -m "feat(vq2): define autonomous survey contracts"
```

### Task 2: Canonicalize FastGate and extract the cyan ribbon

**Files:**
- Modify: `vq2/live/fastgate.py:1-85`
- Create: `vq2/survey/perception.py`
- Test: `vq2/tests/test_survey_perception.py`

**Interfaces:**
- Produces: `camera_vector(u, v, width_px, *, fx, fy, cx, cy) -> np.ndarray`.
- Produces: `RibbonExtractor.detect(image: np.ndarray, frame_ns: int) -> RibbonObservation | None`.
- Produces: `PerceptionEngine.detect(image, frame_ns) -> tuple[tuple[GateCandidate, ...], RibbonObservation | None]` with injected GateNet and FastGate callables.

- [ ] **Step 1: Write calibration and ribbon tests**

```python
import cv2
import numpy as np

from vq2 import camera
from vq2.live.fastgate import camera_vector
from vq2.survey.perception import RibbonExtractor


def test_fastgate_uses_canonical_camera_geometry():
    got = camera_vector(camera.CX, camera.CY, 113.0)
    assert np.allclose(got, [0.0, 0.0, 3.0], atol=1e-9)


def test_ribbon_extractor_returns_upward_image_tangent():
    image = np.zeros((camera.H, camera.W, 3), np.uint8)
    cv2.line(image, (320, 350), (340, 220), (255, 255, 0), 10)
    obs = RibbonExtractor().detect(image, 123)
    assert obs is not None
    assert obs.frame_ns == 123
    assert obs.confidence > 0.5
    assert obs.tangent_uv[1] < 0.0


def test_empty_image_has_no_ribbon():
    image = np.zeros((camera.H, camera.W, 3), np.uint8)
    assert RibbonExtractor().detect(image, 1) is None
```

- [ ] **Step 2: Run the test and verify the missing helper failure**

Run: `python -m pytest vq2/tests/test_survey_perception.py -v`

Expected: FAIL importing `camera_vector` or `vq2.survey.perception`.

- [ ] **Step 3: Move FastGate projection behind an injectable helper**

```python
# vq2/live/fastgate.py
from vq2 import camera

FX, FY, CX, CY = camera.FX, camera.FY, camera.CX, camera.CY


def camera_vector(u, v, width_px, *, fx=FX, fy=FY, cx=CX, cy=CY):
    z = fx * HOLE_W_M / max(float(width_px), 1.0)
    return np.array([(float(u) - cx) / fx * z,
                     (float(v) - cy) / fy * z,
                     z], dtype=float)
```

Replace both inline `t_cam` constructions in `detect()` with
`camera_vector(u, v, hw_est)` and `camera_vector(u, v, max(hw, hh))`.

- [ ] **Step 4: Implement deterministic ribbon extraction and adapters**

```python
# vq2/survey/perception.py
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .types import RibbonObservation


class RibbonExtractor:
    def detect(self, image: np.ndarray, frame_ns: int) -> RibbonObservation | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (80, 70, 110), (110, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.where(mask > 0)
        if len(xs) < 80:
            return None
        points = np.column_stack([xs, ys]).astype(float)
        mean = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
        tangent = vt[0]
        if tangent[1] > 0:
            tangent = -tangent
        confidence = min(1.0, len(points) / 1200.0)
        stride = max(1, len(points) // 64)
        sampled = tuple(map(tuple, points[::stride][:64]))
        return RibbonObservation(int(frame_ns), confidence,
                                 (float(tangent[0]), float(tangent[1])), sampled)


class PerceptionEngine:
    def __init__(self, gatenet, fastgate, ribbon=None):
        self._gatenet = gatenet
        self._fastgate = fastgate
        self._ribbon = ribbon or RibbonExtractor()

    def detect(self, image, frame_ns):
        gates = tuple(self._gatenet(image, frame_ns))
        self._fastgate(image)
        return gates, self._ribbon.detect(image, frame_ns)
```

- [ ] **Step 5: Run perception and existing camera tests**

Run: `python -m pytest vq2/tests/test_survey_perception.py vq2/tests/test_camera.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit canonical perception**

```bash
git add vq2/live/fastgate.py vq2/survey/perception.py vq2/tests/test_survey_perception.py
git commit -m "feat(vq2): add canonical survey perception"
```

### Task 3: Add the lossless bounded recorder

**Files:**
- Create: `vq2/survey/recorder.py`
- Test: `vq2/tests/test_survey_recorder.py`

**Interfaces:**
- Produces: `SurveyRecorder.create(root: Path, profile: PassProfile, identity: dict) -> SurveyRecorder`.
- Produces: `write_frame(frame_ns: int, jpeg: bytes, meta: dict)`, `write_row(stream: str, row: dict)`, and `finalize(status: str, reason: str) -> Path`.
- Manifest uses schema `vq2-survey-v1` and records pass, identities, counts, bytes, disk, and terminal status.

- [ ] **Step 1: Write recorder failure and finalization tests**

```python
import json
import pytest

from vq2.survey.recorder import SurveyRecorder
from vq2.survey.types import profile_for


def test_recorder_requires_unique_empty_directory(tmp_path):
    run = tmp_path / "A1-001"
    SurveyRecorder.create(run, profile_for("A1"), {"git": "abc"})
    with pytest.raises(FileExistsError):
        SurveyRecorder.create(run, profile_for("A1"), {"git": "abc"})


def test_frame_metadata_and_manifest_are_one_to_one(tmp_path, monkeypatch):
    monkeypatch.setattr("vq2.survey.recorder.shutil.disk_usage",
                        lambda path: type("D", (), {"free": 30 * 1024**3})())
    run = tmp_path / "A1-002"
    recorder = SurveyRecorder.create(run, profile_for("A1"), {"git": "abc"})
    recorder.write_frame(123, b"jpeg", {"rx_wall": 1.0})
    manifest_path = recorder.finalize("complete", "route_end")
    manifest = json.loads(manifest_path.read_text())
    assert (run / "frames/123.jpg").read_bytes() == b"jpeg"
    assert manifest["counts"]["frames"] == 1
    assert manifest["status"] == "complete"


def test_disk_reserve_fails_before_write(tmp_path, monkeypatch):
    monkeypatch.setattr("vq2.survey.recorder.shutil.disk_usage",
                        lambda path: type("D", (), {"free": 10 * 1024**3})())
    recorder = SurveyRecorder.create(tmp_path / "A1-003", profile_for("A1"), {})
    with pytest.raises(OSError, match="disk reserve"):
        recorder.write_frame(123, b"jpeg", {})
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_survey_recorder.py -v`

Expected: FAIL importing `vq2.survey.recorder`.

- [ ] **Step 3: Implement exclusive creation, bounded writes, and atomic finalize**

```python
# vq2/survey/recorder.py
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time


class SurveyRecorder:
    STREAMS = ("mavlink", "frames_dedup", "cmds", "detections", "tracks", "events")

    def __init__(self, root, profile, identity):
        self.root, self.profile, self.identity = root, profile, dict(identity)
        self.started_wall = time.time()
        self.start_free = shutil.disk_usage(root).free
        self.bytes_written = 0
        self.counts = {name: 0 for name in self.STREAMS}
        self.counts["frames"] = 0
        self._files = {name: (root / f"{name}.jsonl").open("x", encoding="utf-8")
                       for name in self.STREAMS}

    @classmethod
    def create(cls, root, profile, identity):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        (root / "frames").mkdir()
        return cls(root, profile, identity)

    def _check_budget(self, added):
        if shutil.disk_usage(self.root).free - added < self.profile.disk_reserve_bytes:
            raise OSError("disk reserve would be crossed")
        if self.bytes_written + added > self.profile.max_run_bytes:
            raise OSError("survey byte cap would be crossed")
        if time.time() - self.started_wall > self.profile.max_run_s:
            raise TimeoutError("survey duration cap reached")

    def write_row(self, stream, row):
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        size = len(encoded.encode("utf-8"))
        self._check_budget(size)
        self._files[stream].write(encoded)
        self._files[stream].flush()
        self.bytes_written += size
        self.counts[stream] += 1

    def write_frame(self, frame_ns, jpeg, meta):
        jpeg = bytes(jpeg)
        self._check_budget(len(jpeg))
        (self.root / "frames" / f"{int(frame_ns)}.jpg").write_bytes(jpeg)
        self.bytes_written += len(jpeg)
        self.counts["frames"] += 1
        self.write_row("frames_dedup", {"sim_ns": int(frame_ns), **meta,
                                        "sha256": hashlib.sha256(jpeg).hexdigest()})

    def finalize(self, status, reason):
        for handle in self._files.values():
            handle.close()
        manifest = {"schema": "vq2-survey-v1",
                    "pass": self.profile.survey_pass.value,
                    "profile": asdict(self.profile), "identity": self.identity,
                    "counts": self.counts, "bytes_written": self.bytes_written,
                    "disk_free_start": self.start_free,
                    "disk_free_end": shutil.disk_usage(self.root).free,
                    "status": str(status), "reason": str(reason)}
        tmp, final = self.root / "manifest.json.tmp", self.root / "manifest.json"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(final)
        return final
```

- [ ] **Step 4: Run recorder tests**

Run: `python -m pytest vq2/tests/test_survey_recorder.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit recorder**

```bash
git add vq2/survey/recorder.py vq2/tests/test_survey_recorder.py
git commit -m "feat(vq2): add bounded survey recorder"
```

### Task 4: Track gates and reject ambiguous route candidates

**Files:**
- Create: `vq2/survey/tracker.py`
- Test: `vq2/tests/test_survey_tracker.py`

**Interfaces:**
- Produces: `GateTrack`, `TrackDecision`, and `SurveyTracker.update(candidates, ribbon) -> TrackDecision`.

- [ ] **Step 1: Write continuity, decoy, and ambiguity tests**

```python
from vq2.survey.tracker import SurveyTracker
from vq2.survey.types import GateCandidate, RibbonObservation


def gate(det_id, u, z, score=0.9):
    return GateCandidate(det_id, det_id, (u, 180.0), (0.0, 0.0, z), 1.37, score)


def ribbon(u):
    return RibbonObservation(1, 1.0, (0.0, -1.0), ((u, 300.0), (u, 180.0)))


def test_tracker_keeps_identity_when_detection_order_changes():
    tracker = SurveyTracker(min_age_frames=2)
    tracker.update((gate(1, 320, 6), gate(2, 500, 5)), ribbon(320))
    decision = tracker.update((gate(3, 500, 4.8), gate(4, 322, 5.8)), ribbon(321))
    assert decision.selected.candidate.center_uv[0] == 322


def test_tracker_rejects_gate_not_supported_by_ribbon():
    decision = SurveyTracker(min_age_frames=1).update((gate(1, 500, 4),), ribbon(320))
    assert decision.selected is None
    assert decision.reason == "no_ribbon_supported_gate"


def test_near_equal_eligible_candidates_are_ambiguous():
    tracker = SurveyTracker(min_age_frames=1, ambiguity_margin=0.1)
    decision = tracker.update((gate(1, 315, 5), gate(2, 325, 5)), ribbon(320))
    assert decision.ambiguous and decision.selected is None
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_survey_tracker.py -v`

Expected: FAIL importing `vq2.survey.tracker`.

- [ ] **Step 3: Implement stable association and ribbon eligibility**

```python
# vq2/survey/tracker.py
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GateTrack:
    track_id: int
    candidate: object
    age_frames: int
    missed_frames: int
    ribbon_score: float


@dataclass(frozen=True)
class TrackDecision:
    selected: GateTrack | None
    ambiguous: bool
    reason: str


class SurveyTracker:
    def __init__(self, min_age_frames=3, max_pixel_jump=80.0,
                 ribbon_gate_px=80.0, ambiguity_margin=0.08):
        self.min_age_frames = int(min_age_frames)
        self.max_pixel_jump = float(max_pixel_jump)
        self.ribbon_gate_px = float(ribbon_gate_px)
        self.ambiguity_margin = float(ambiguity_margin)
        self._tracks, self._next_id = {}, 1

    def _ribbon_distance(self, candidate, ribbon):
        if ribbon is None or not ribbon.points_uv:
            return math.inf
        u, v = candidate.center_uv
        return min(math.hypot(u - x, v - y) for x, y in ribbon.points_uv)

    def update(self, candidates, ribbon):
        updated, unused = {}, set(range(len(candidates)))
        for track_id, old in self._tracks.items():
            if not unused:
                break
            index = min(unused, key=lambda i: math.dist(old.candidate.center_uv,
                                                        candidates[i].center_uv))
            if math.dist(old.candidate.center_uv,
                         candidates[index].center_uv) <= self.max_pixel_jump:
                candidate = candidates[index]
                unused.remove(index)
                distance = self._ribbon_distance(candidate, ribbon)
                updated[track_id] = GateTrack(track_id, candidate, old.age_frames + 1,
                                              0, max(0.0, 1.0 - distance / self.ribbon_gate_px))
        for index in unused:
            candidate = candidates[index]
            distance = self._ribbon_distance(candidate, ribbon)
            track_id, self._next_id = self._next_id, self._next_id + 1
            updated[track_id] = GateTrack(track_id, candidate, 1, 0,
                                          max(0.0, 1.0 - distance / self.ribbon_gate_px))
        self._tracks = updated
        eligible = [track for track in updated.values()
                    if track.age_frames >= self.min_age_frames
                    and track.ribbon_score > 0.0 and not track.candidate.clipped]
        eligible.sort(key=lambda track: track.candidate.score * track.ribbon_score,
                      reverse=True)
        if not eligible:
            return TrackDecision(None, False, "no_ribbon_supported_gate")
        if len(eligible) > 1:
            scores = [track.candidate.score * track.ribbon_score for track in eligible[:2]]
            if scores[0] - scores[1] < self.ambiguity_margin:
                return TrackDecision(None, True, "ambiguous_gate")
        return TrackDecision(eligible[0], False, "selected")
```

- [ ] **Step 4: Run tracker tests**

Run: `python -m pytest vq2/tests/test_survey_tracker.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit tracker**

```bash
git add vq2/survey/tracker.py vq2/tests/test_survey_tracker.py
git commit -m "feat(vq2): track survey gates by ribbon"
```

### Task 5: Implement fail-closed lifecycle transitions

**Files:**
- Create: `vq2/survey/controller.py`
- Test: `vq2/tests/test_survey_controller_lifecycle.py`

**Interfaces:**
- Produces: `SurveyController(profile).step(snapshot) -> SurveyDecision` and `abort(reason)`.

- [ ] **Step 1: Write preflight, GO, tick, and stale-input tests**

```python
from dataclasses import replace

from vq2.survey.controller import SurveyController
from vq2.survey.types import HealthSnapshot, SurveyPhase, SurveySnapshot, profile_for


HEALTHY = HealthSnapshot(0.01, 0.01, 0.01, 0.01, True, True,
                         30 * 1024**3, True)


def snap(**values):
    base = SurveySnapshot(0.0, 1, 0, 0, False, False, 0.0, 0.0, 0.0, HEALTHY)
    return replace(base, **values)


def test_preflight_refuses_to_arm_with_dpvo_enabled():
    controller = SurveyController(profile_for("A1"))
    decision = controller.step(snap(health=replace(HEALTHY, dpvo_disabled=False)))
    assert decision.phase is SurveyPhase.ABORT_LAND
    assert decision.command.land and decision.reason == "dpvo_enabled"


def test_wait_go_emits_zero_motion():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.WAIT_GO
    decision = controller.step(snap(race_go=False))
    assert decision.phase is SurveyPhase.WAIT_GO
    assert decision.command.forward_mps == 0.0 and not decision.command.arm


def test_unexpected_survey_tick_aborts():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.FOLLOW_RIBBON
    decision = controller.step(snap(race_go=True, armed=True, gate_index=1))
    assert decision.phase is SurveyPhase.ABORT_LAND
    assert decision.reason == "unexpected_tick"


def test_stale_camera_never_emits_forward_motion():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.FOLLOW_RIBBON
    decision = controller.step(snap(race_go=True, armed=True,
                                    health=replace(HEALTHY, camera_age_s=1.0)))
    assert decision.command.forward_mps <= 0.0
    assert decision.phase is SurveyPhase.ABORT_LAND
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_survey_controller_lifecycle.py -v`

Expected: FAIL importing `vq2.survey.controller`.

- [ ] **Step 3: Implement health gates and lifecycle shell**

```python
# vq2/survey/controller.py
from .types import SurveyCommand, SurveyDecision, SurveyPhase


class SurveyController:
    MAX_SENSOR_AGE_S = 0.25

    def __init__(self, profile):
        self.profile, self.phase = profile, SurveyPhase.PREFLIGHT
        self.initial_gate_index, self._phase_started_s = 0, 0.0

    def _decision(self, command, reason, **telemetry):
        return SurveyDecision(self.phase, command, reason, telemetry)

    def abort(self, reason):
        self.phase = SurveyPhase.ABORT_LAND
        return self._decision(SurveyCommand(land=True), reason)

    def _health_reason(self, health):
        if not health.dpvo_disabled:
            return "dpvo_enabled"
        if not health.workers_alive:
            return "worker_dead"
        if not health.recorder_healthy:
            return "recorder_unhealthy"
        if health.disk_free_bytes < self.profile.disk_reserve_bytes:
            return "disk_reserve"
        ages = (health.heartbeat_age_s, health.imu_age_s,
                health.camera_age_s, health.command_age_s)
        return "stale_input" if max(ages) > self.MAX_SENSOR_AGE_S else None

    def step(self, snapshot):
        reason = self._health_reason(snapshot.health)
        if reason:
            return self.abort(reason)
        if snapshot.gate_index != self.initial_gate_index:
            return self.abort("unexpected_tick")
        if self.phase is SurveyPhase.PREFLIGHT:
            self.phase = SurveyPhase.RESET_SETTLE
            self._phase_started_s = snapshot.now_s
            return self._decision(SurveyCommand(), "preflight_passed")
        if self.phase is SurveyPhase.WAIT_GO:
            if not snapshot.race_go:
                return self._decision(SurveyCommand(), "waiting_go")
            self.phase = SurveyPhase.ARM_LEVEL_CLIMB
            return self._decision(SurveyCommand(arm=True), "go")
        if self.phase is SurveyPhase.ABORT_LAND:
            return self._decision(SurveyCommand(land=True), "aborting")
        return self._decision(SurveyCommand(), "phase_hold")
```

- [ ] **Step 4: Run lifecycle tests**

Run: `python -m pytest vq2/tests/test_survey_controller_lifecycle.py -v`

Expected: `4 passed`.

- [ ] **Step 5: Commit lifecycle shell**

```bash
git add vq2/survey/controller.py vq2/tests/test_survey_controller_lifecycle.py
git commit -m "feat(vq2): add fail-closed survey lifecycle"
```

### Task 6: Add ribbon following, observation fan, and gate bypass guidance

**Files:**
- Modify: `vq2/survey/controller.py`
- Test: `vq2/tests/test_survey_controller_guidance.py`

**Interfaces:**
- Consumes: `TrackDecision` through `SurveySnapshot.track`.
- Produces: commands bounded to `0.6 m/s` and `0.6 rad/s`, plus `target_track_id`, `gate_offset_m`, and `blind_carry_s` telemetry.

- [ ] **Step 1: Write guidance and bypass-envelope tests**

```python
from vq2.survey.controller import SurveyController
from vq2.survey.tracker import GateTrack, TrackDecision
from vq2.survey.types import GateCandidate, SurveyPhase, profile_for
from vq2.tests.test_survey_controller_lifecycle import snap


def selected_track(outer_half_width=1.37):
    candidate = GateCandidate(1, 1, (320.0, 180.0), (0.0, 0.0, 6.0),
                              outer_half_width, 0.9)
    return TrackDecision(GateTrack(7, candidate, 4, 0, 0.9), False, "selected")


def test_ribbon_loss_holds_without_forward_motion():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.FOLLOW_RIBBON
    decision = controller.step(snap(race_go=True, armed=True, ribbon=None))
    assert decision.command.forward_mps == 0.0
    assert decision.reason == "ribbon_lost"


def test_stable_target_enters_observation_fan():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.ACQUIRE_GATE
    decision = controller.step(snap(race_go=True, armed=True, track=selected_track()))
    assert decision.phase is SurveyPhase.OBSERVE_FAN
    assert decision.command.forward_mps == 0.0


def test_bypass_offset_respects_outer_frame_and_floor():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.BYPASS
    decision = controller.step(snap(race_go=True, armed=True,
                                    track=selected_track(1.8)))
    assert decision.telemetry["gate_offset_m"] == -2.8
    assert decision.command.right_mps < 0.0


def test_ambiguous_target_holds():
    controller = SurveyController(profile_for("A1"))
    controller.phase = SurveyPhase.ACQUIRE_GATE
    decision = controller.step(snap(race_go=True, armed=True,
                                    track=TrackDecision(None, True, "ambiguous_gate")))
    assert decision.command.forward_mps == 0.0
    assert decision.reason == "ambiguous_gate"
```

- [ ] **Step 2: Run tests and verify guidance failures**

Run: `python -m pytest vq2/tests/test_survey_controller_guidance.py -v`

Expected: FAIL because the lifecycle shell returns `phase_hold`.

- [ ] **Step 3: Add bounded guidance methods and phase handlers**

```python
# Add inside SurveyController
    MAX_SPEED_MPS = 0.6
    MAX_YAW_RATE_RPS = 0.6

    def _ribbon_command(self, ribbon):
        if ribbon is None or ribbon.confidence < 0.5:
            return SurveyCommand(), "ribbon_lost"
        yaw = max(-self.MAX_YAW_RATE_RPS,
                  min(self.MAX_YAW_RATE_RPS, ribbon.tangent_uv[0] * 0.8))
        return SurveyCommand(forward_mps=0.35, yaw_rate_rps=yaw), "follow_ribbon"

    def _track_decision(self, snapshot):
        decision = snapshot.track
        if decision is None:
            return None, "no_gate"
        if decision.ambiguous:
            return None, "ambiguous_gate"
        return decision.selected, decision.reason

    def _bypass_command(self, track):
        offset = self.profile.bypass_side * max(
            self.profile.min_gate_center_offset_m,
            track.candidate.outer_half_width_m + self.profile.bypass_margin_m)
        right = max(-0.35, min(0.35, offset * 0.2))
        return SurveyCommand(forward_mps=0.2, right_mps=right), offset
```

Insert before the final `phase_hold` return:

```python
        if self.phase is SurveyPhase.FOLLOW_RIBBON:
            command, reason = self._ribbon_command(snapshot.ribbon)
            return self._decision(command, reason)
        if self.phase is SurveyPhase.ACQUIRE_GATE:
            track, reason = self._track_decision(snapshot)
            if track is None:
                return self._decision(SurveyCommand(), reason)
            self.phase = SurveyPhase.OBSERVE_FAN
            return self._decision(SurveyCommand(), "observe_gate",
                                  target_track_id=track.track_id)
        if self.phase is SurveyPhase.BYPASS:
            track, reason = self._track_decision(snapshot)
            if track is None:
                return self.abort(reason)
            command, offset = self._bypass_command(track)
            return self._decision(command, "bypass",
                                  target_track_id=track.track_id,
                                  gate_offset_m=offset, blind_carry_s=0.0)
        if self.phase is SurveyPhase.REACQUIRE_RIBBON:
            if snapshot.ribbon is None:
                return self._decision(SurveyCommand(yaw_rate_rps=0.2), "bounded_scan")
            self.phase = SurveyPhase.FOLLOW_RIBBON
            return self._decision(SurveyCommand(), "ribbon_reacquired")
```

- [ ] **Step 4: Run lifecycle and guidance tests together**

Run: `python -m pytest vq2/tests/test_survey_controller_lifecycle.py vq2/tests/test_survey_controller_guidance.py -v`

Expected: `8 passed`.

- [ ] **Step 5: Commit guidance**

```bash
git add vq2/survey/controller.py vq2/tests/test_survey_controller_guidance.py
git commit -m "feat(vq2): guide autonomous gate bypass"
```

### Task 7: Add the expiring command arbiter

**Files:**
- Create: `vq2/survey/arbiter.py`
- Test: `vq2/tests/test_survey_arbiter.py`

**Interfaces:**
- Produces: `CommandArbiter(sender, ttl_s=0.15).publish()`, `.tick()`, and `.force_land()`.

- [ ] **Step 1: Write owner and expiry tests**

```python
from vq2.survey.arbiter import CommandArbiter
from vq2.survey.types import SurveyCommand


def test_fresh_command_is_forwarded_once_per_tick():
    sent = []
    arbiter = CommandArbiter(sent.append, ttl_s=0.15)
    command = SurveyCommand(forward_mps=0.2)
    arbiter.publish(command, 1.0)
    assert arbiter.tick(1.1) == command and sent[-1] == command


def test_expired_command_brakes_then_lands():
    arbiter = CommandArbiter(lambda command: None, ttl_s=0.15, land_after_s=0.5)
    arbiter.publish(SurveyCommand(forward_mps=0.2), 1.0)
    assert arbiter.tick(1.2).forward_mps == 0.0
    assert arbiter.tick(1.6).land


def test_force_land_overrides_future_publications():
    arbiter = CommandArbiter(lambda command: None)
    arbiter.force_land("manual_abort")
    arbiter.publish(SurveyCommand(forward_mps=0.2), 2.0)
    assert arbiter.tick(2.0).land
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_survey_arbiter.py -v`

Expected: FAIL importing `vq2.survey.arbiter`.

- [ ] **Step 3: Implement the single-owner arbiter**

```python
# vq2/survey/arbiter.py
from .types import SurveyCommand


class CommandArbiter:
    def __init__(self, sender, ttl_s=0.15, land_after_s=0.5):
        self._sender, self.ttl_s, self.land_after_s = sender, float(ttl_s), float(land_after_s)
        self._command, self._published_s = SurveyCommand(), float("-inf")
        self._latched_land, self.reason = False, "startup"

    def publish(self, command, now_s):
        if not self._latched_land:
            self._command, self._published_s = command, float(now_s)

    def force_land(self, reason):
        self._latched_land, self.reason = True, str(reason)

    def tick(self, now_s):
        age = float(now_s) - self._published_s
        if self._latched_land or age > self.land_after_s:
            output = SurveyCommand(land=True)
        elif age > self.ttl_s:
            output = SurveyCommand()
        else:
            output = self._command
        self._sender(output)
        return output
```

- [ ] **Step 4: Run arbiter tests**

Run: `python -m pytest vq2/tests/test_survey_arbiter.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit arbiter**

```bash
git add vq2/survey/arbiter.py vq2/tests/test_survey_arbiter.py
git commit -m "feat(vq2): arbitrate expiring survey commands"
```

### Task 8: Replay corpora and enforce A/B separation

**Files:**
- Create: `vq2/survey/replay.py`
- Test: `vq2/tests/test_survey_replay.py`

**Interfaces:**
- Produces: `build_coverage_report(manifests, output) -> Path` accepting only A1/A2.
- Produces: `seal_holdout(manifest, output) -> Path` for B1 with `opened=false`.
- Produces: `replay_corpus(corpus, profile, engine, controller) -> ReplayReport`.

- [ ] **Step 1: Write training/holdout firewall tests**

```python
import json
import pytest

from vq2.survey.replay import build_coverage_report, seal_holdout


def manifest(path, name):
    path.write_text(json.dumps({"schema": "vq2-survey-v1", "pass": name,
                                "status": "complete", "counts": {"frames": 30}}))
    return path


def test_coverage_accepts_a1_and_a2(tmp_path):
    output = build_coverage_report([manifest(tmp_path / "a1.json", "A1"),
                                    manifest(tmp_path / "a2.json", "A2")],
                                   tmp_path / "coverage.json")
    assert json.loads(output.read_text())["training_passes"] == ["A1", "A2"]


def test_coverage_rejects_b1(tmp_path):
    with pytest.raises(ValueError, match="B1 is sealed"):
        build_coverage_report([manifest(tmp_path / "b1.json", "B1")],
                              tmp_path / "coverage.json")


def test_holdout_seal_starts_unopened(tmp_path):
    output = seal_holdout(manifest(tmp_path / "b1.json", "B1"),
                          tmp_path / "B1.seal.json")
    assert json.loads(output.read_text())["opened"] is False
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_survey_replay.py -v`

Expected: FAIL importing `vq2.survey.replay`.

- [ ] **Step 3: Implement deterministic firewall artifacts**

```python
# vq2/survey/replay.py
import hashlib
import json
from pathlib import Path


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_coverage_report(manifests, output):
    names = [_load(path)["pass"] for path in manifests]
    if "B1" in names:
        raise ValueError("B1 is sealed and cannot enter coverage training")
    if any(name not in {"A1", "A2"} for name in names):
        raise ValueError("coverage input must be A1/A2")
    report = {"schema": "vq2-survey-coverage-v1",
              "training_passes": sorted(names),
              "manifest_sha256": sorted(_digest(path) for path in manifests),
              "coverage_gaps": []}
    output = Path(output)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return output


def seal_holdout(manifest, output):
    if _load(manifest)["pass"] != "B1":
        raise ValueError("only B1 can be sealed as holdout")
    seal = {"schema": "vq2-survey-holdout-v1", "pass": "B1",
            "manifest_sha256": _digest(manifest), "opened": False}
    output = Path(output)
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return output
```

- [ ] **Step 4: Add `ReplayReport` and corpus-driven controller replay**

Use `vq2.corpus.load_corpus`, explicit manifest frame blocks, and the existing
clock bridge. For every source `sim_ns`, write phase, reason, command, target,
and input ages. The report must count stale-forward decisions and target
switches; its CLI exits nonzero if either is nonzero.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayReport:
    frame_count: int
    forward_on_stale_count: int
    target_switch_count: int
    decisions: tuple[dict, ...]
```

- [ ] **Step 5: Run replay and clock tests**

Run: `python -m pytest vq2/tests/test_survey_replay.py vq2/tests/test_clock_bridge.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit replay and sealing**

```bash
git add vq2/survey/replay.py vq2/tests/test_survey_replay.py
git commit -m "feat(vq2): replay and seal survey passes"
```

### Task 9: Replace the live monolith with thin wiring

**Files:**
- Modify: `vq2/live/vq2survey.py:1-end`
- Create: `vq2/live/run_survey.bat`
- Test: `vq2/tests/test_survey_runtime_wiring.py`

**Interfaces:**
- Produces: `main(env=os.environ) -> int` consuming `SURVEY_PASS`, `RECORD`, `SURVEY_NO_ARM`, and `SURVEY_COVERAGE`.
- Every exit path lands/disarms and finalizes the manifest.

- [ ] **Step 1: Write source-level safety wiring tests**

```python
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_runtime_composes_new_package_and_has_no_blind_95s_loop():
    source = (ROOT / "vq2/live/vq2survey.py").read_text(encoding="utf-8")
    assert "from vq2.survey.controller import SurveyController" in source
    assert "from vq2.survey.recorder import SurveyRecorder" in source
    assert "SURVEY_S = 95.0" not in source
    assert "CRUISE = 2.0" not in source


def test_runtime_requires_dpvo_disabled_and_finalizes_in_finally():
    source = (ROOT / "vq2/live/vq2survey.py").read_text(encoding="utf-8")
    assert "dpvo_disabled" in source
    assert "finally:" in source
    assert "recorder.finalize" in source
    assert "arbiter.force_land" in source


def test_launcher_requires_unique_record_and_explicit_pass():
    source = (ROOT / "vq2/live/run_survey.bat").read_text(encoding="utf-8")
    assert "SURVEY_PASS" in source and "RECORD" in source
    assert "DPVO=0" in source and "vq2survey.py" in source
```

- [ ] **Step 2: Run tests and verify the legacy-script failure**

Run: `python -m pytest vq2/tests/test_survey_runtime_wiring.py -v`

Expected: FAIL because the legacy script still contains the blind survey loop.

- [ ] **Step 3: Replace the script with thin composition**

```python
def main(env=os.environ):
    profile = profile_for(env["SURVEY_PASS"])
    if env.get("DPVO", "0") != "0":
        raise RuntimeError("survey requires DPVO=0")
    if profile.coverage_report_required and not env.get("SURVEY_COVERAGE"):
        raise RuntimeError("A3 requires SURVEY_COVERAGE")
    recorder = SurveyRecorder.create(Path(env["RECORD"]), profile,
                                     runtime_identity())
    arbiter = CommandArbiter(send_survey_command)
    status, reason = "aborted", "exception"
    try:
        runtime = SurveyRuntime(profile, recorder, arbiter,
                                no_arm=env.get("SURVEY_NO_ARM") == "1")
        status, reason = runtime.run()
        return 0 if status == "complete" else 2
    finally:
        arbiter.force_land(reason)
        send_land_and_disarm()
        recorder.finalize(status, reason)
```

Copy only the measured MAVLink decode, duplicate-IMU suppression, race-status
decode, raw JPEG passthrough, collision distinction, and bounded level-command
math from `vq2wp.py` into focused survey runtime helpers. Do not import or
execute `vq2wp.py` as a module.

- [ ] **Step 4: Add the explicit Windows launcher**

```bat
@echo off
setlocal
if "%SURVEY_PASS%"=="" exit /b 64
if "%RECORD%"=="" exit /b 64
set DPVO=0
set DPVO_ROUTE=0
set DPVO_OBSERVE=0
python C:\Users\alexj\algo_src\vq2\live\vq2survey.py
exit /b %ERRORLEVEL%
```

- [ ] **Step 5: Run the complete survey suite**

Run: `python -m pytest vq2/tests/test_survey_*.py vq2/tests/test_camera.py vq2/tests/test_clock_bridge.py -v`

Expected: all tests PASS without importing CUDA.

- [ ] **Step 6: Replay fg62 and fg75**

Run: `python -m vq2.survey.replay C:\Users\alexj\vq2_servo_fg62 --profile A1 --output C:\tmp\fg62-survey-replay.json`

Expected: exit `0`, `forward_on_stale_count=0`, and two judge transitions logged as replay evidence rather than survey progression.

Run: `python -m vq2.survey.replay C:\Users\alexj\vq2_servo_fg75 --profile A1 --output C:\tmp\fg75-survey-replay.json`

Expected: exit `0`, `forward_on_stale_count=0`, and one judge transition.

- [ ] **Step 7: Commit runtime wiring**

```bash
git add vq2/live/vq2survey.py vq2/live/run_survey.bat vq2/tests/test_survey_runtime_wiring.py
git commit -m "feat(vq2): wire autonomous survey runtime"
```

### Task 10: Execute the live promotion ladder

**Files:**
- Create per run outside git: `C:\Users\alexj\vq2_survey_<pass>_<attempt>\`
- Append flight results to `C:\Users\alexj\obsidian_outbox.md` if the Mac vault is unreachable.

**Interfaces:**
- Consumes the exact Task 9 launcher and profile.
- Produces one immutable corpus and manifest per attempt; no source changes.

- [ ] **Step 1: Verify disk and simulator freshness**

Run: `[math]::Round(([System.IO.DriveInfo]::new('C')).AvailableFreeSpace / 1GB, 2)`

Expected: at least `20.0`. Use the documented fresh-simulator restart procedure
and restart the app after any crash.

- [ ] **Step 2: Run the no-arm soak**

Run: `$env:SURVEY_PASS='A1'; $env:SURVEY_NO_ARM='1'; $env:RECORD='C:\Users\alexj\vq2_survey_A1_soak1'; & 'C:\Users\alexj\algo_src\vq2\live\run_survey.bat'`

Expected: finalized manifest, camera-rate metadata, GateNet/ribbon rows, zero
actuation commands, and no disk/queue/worker failure.

- [ ] **Step 3: Promote one live behavior at a time**

Run separate fresh attempts in this order: arm-hover-abort,
takeoff-stare-land, ribbon follow with bypass disabled, one-gate
observe/bypass, two-gate observe/bypass, A1, A2, A3, and sealed B1. After each
attempt inspect `events.jsonl`, `cmds.jsonl`, frame gaps, collision rows,
worker health, and the manifest before changing the next profile variable.

- [ ] **Step 4: Score accidental judge transitions correctly**

Run: `python C:\Users\alexj\algo_src\vq2\gidx2.py <corpus-path>`

Expected for a clean survey: no fresh gate transition. Any unexpected tick is
a survey failure and requires bypass-profile review.

- [ ] **Step 5: Freeze A1/A2 coverage and seal B1**

Run: `python -m vq2.survey.replay coverage <A1-manifest> <A2-manifest> --output <coverage.json>`

Expected: only A1/A2 digests and a deterministic A3 gap list.

Run: `python -m vq2.survey.replay seal <B1-manifest> --output <B1.seal.json>`

Expected: `opened=false` and a digest matching the finalized B1 manifest.

---

## Completion gate

This plan is complete only when the survey unit suite passes, fg62/fg75 replay
shows no stale forward actuation, the staged live ladder produces clean A1,
A2, and A3 corpora, and B1 is sealed without entering map construction. Do not
begin map implementation against an unstable corpus schema.

