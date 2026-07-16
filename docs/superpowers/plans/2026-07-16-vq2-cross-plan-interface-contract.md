# VQ2 Cross-Plan Interface Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the exact artifacts and method signatures shared by survey acquisition, metric mapping, prior-map localization, and live race integration.

**Architecture:** Small contract tests are implemented before the producer or consumer that depends on them. Survey manifests carry explicit frame blocks; all pose exchange uses source `frame_ns` and one `PoseSE3` record; the external alignment has one canonical high-level API; and every approval artifact forms an immutable digest chain.

**Tech Stack:** Python dataclasses, JSON/JSONL, compressed NPZ, SHA-256, NumPy, pytest; no new dependency.

## Global Constraints

- Execute Task 1 during survey-recorder Task 3 before any corpus is collected.
- Execute Task 2 before prior-map localization Task 4 or race-integration Task 2.
- Execute Task 3 before opening B1 or loading a map in a live process.
- Source timestamps are integer nanoseconds; wall time is telemetry and never a pose key.
- Quaternions are always `(x, y, z, w)` and map poses are camera-to-world.

---

### Task 1: Freeze the explicit survey-manifest frame block

**Files:**
- Modify: `vq2/survey/recorder.py`
- Modify: `vq2/tests/test_survey_recorder.py`
- Test: `vq2/tests/test_cross_plan_survey_manifest.py`

**Interfaces:**
- `manifest.json` contains `schema`, `pass`, `frame_block`, `status`, `identity`, `profile`, `counts`, `bytes_written`, and disk counters.
- `frame_block` is `[first_frame_ns, last_frame_ns]`, inclusive, and both endpoints have matching JPEG and `frames_dedup.jsonl` rows.

- [ ] **Step 1: Write producer/consumer contract test**

```python
import json

from vq2.mapping.extract import select_frame_rows
from vq2.mapping.schema import RunInput, sha256_file
from vq2.survey.recorder import SurveyRecorder
from vq2.survey.types import profile_for


def test_survey_manifest_frame_block_is_consumable_by_mapper(tmp_path, monkeypatch):
    monkeypatch.setattr("vq2.survey.recorder.shutil.disk_usage",
                        lambda path: type("D", (), {"free": 30 * 1024**3})())
    run_dir = tmp_path / "A1"
    recorder = SurveyRecorder.create(run_dir, profile_for("A1"), {"camera": "226"})
    recorder.write_frame(100, b"jpeg-100", {"rx_wall": 1.0})
    recorder.write_frame(200, b"jpeg-200", {"rx_wall": 2.0})
    manifest_path = recorder.finalize("complete", "route_end")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["frame_block"] == [100, 200]
    run = RunInput("A1", manifest_path, sha256_file(manifest_path), (100, 200))
    assert [row["sim_ns"] for row in select_frame_rows(run_dir, run)] == [100, 200]
```

- [ ] **Step 2: Run test and verify missing `frame_block`**

Run: `python -m pytest vq2/tests/test_cross_plan_survey_manifest.py -v`

Expected: FAIL with `KeyError: 'frame_block'` or failed equality.

- [ ] **Step 3: Track endpoints and write them atomically**

```python
# Add to SurveyRecorder.__init__
self.first_frame_ns = None
self.last_frame_ns = None

# Add to SurveyRecorder.write_frame after the budget check
frame_ns = int(frame_ns)
if self.last_frame_ns is not None and frame_ns <= self.last_frame_ns:
    raise ValueError("frame timestamps must be strictly increasing")
self.first_frame_ns = frame_ns if self.first_frame_ns is None else self.first_frame_ns
self.last_frame_ns = frame_ns

# Add to the manifest in finalize()
if self.first_frame_ns is None or self.last_frame_ns is None:
    raise ValueError("cannot finalize a survey corpus without frames")
manifest["frame_block"] = [self.first_frame_ns, self.last_frame_ns]
```

- [ ] **Step 4: Run recorder and cross-plan tests**

Run: `python -m pytest vq2/tests/test_survey_recorder.py vq2/tests/test_cross_plan_survey_manifest.py vq2/tests/test_mapping_extract.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit the manifest contract**

```bash
git add vq2/survey/recorder.py vq2/tests/test_survey_recorder.py vq2/tests/test_cross_plan_survey_manifest.py
git commit -m "fix(vq2): freeze survey frame blocks"
```

### Task 2: Freeze pose and alignment method signatures

**Files:**
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/mapping/map_alignment.py`
- Modify: `vq2/mapping/localizer.py`
- Modify: `vq2/live/map_pose_worker.py`
- Test: `vq2/tests/test_cross_plan_pose_api.py`

**Interfaces:**
- Produces: `PoseSE3(frame_ns, p, q_xyzw)`.
- Produces: `PriorMapLocalizer.localize(image, frame_ns, prior: PoseSE3) -> tuple[AbsolutePoseFix | None, LocalizationAttempt]`.
- Produces: `MapAlignment.add_dpvo_pose(pose: PoseSE3)`, `add_fix(dpvo_pose: PoseSE3, fix: AbsolutePoseFix)`, `predicted_pose(dpvo_pose: PoseSE3)`, `correct(dpvo_pose: PoseSE3)`, and `reset(dpvo_identity: str, reason: str)`.

- [ ] **Step 1: Write signature-level worker contract test**

```python
import inspect

from vq2.mapping.localizer import PriorMapLocalizer
from vq2.mapping.map_alignment import MapAlignment
from vq2.mapping.schema import PoseSE3


def test_pose_record_uses_ns_and_xyzw():
    pose = PoseSE3(123, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert pose.frame_ns == 123
    assert pose.q_xyzw[-1] == 1.0


def test_alignment_high_level_api_names_are_stable():
    assert tuple(inspect.signature(MapAlignment.add_dpvo_pose).parameters) == ("self", "pose")
    assert tuple(inspect.signature(MapAlignment.add_fix).parameters) == ("self", "dpvo_pose", "fix")
    assert tuple(inspect.signature(MapAlignment.predicted_pose).parameters) == ("self", "dpvo_pose")
    assert tuple(inspect.signature(MapAlignment.correct).parameters) == ("self", "dpvo_pose")


def test_localizer_accepts_one_pose_prior_record():
    assert tuple(inspect.signature(PriorMapLocalizer.localize).parameters) == (
        "self", "image", "frame_ns", "prior")
```

- [ ] **Step 2: Run test and verify inconsistent API**

Run: `python -m pytest vq2/tests/test_cross_plan_pose_api.py -v`

Expected: FAIL on missing `PoseSE3` or method-name mismatch.

- [ ] **Step 3: Add the shared pose record**

```python
# vq2/mapping/schema.py
@dataclass(frozen=True)
class PoseSE3:
    frame_ns: int
    p: tuple[float, float, float]
    q_xyzw: tuple[float, float, float, float]

    def __post_init__(self):
        if int(self.frame_ns) < 0:
            raise ValueError("frame_ns must be non-negative")
        if not np.isfinite(np.asarray(self.p, float)).all():
            raise ValueError("position must be finite")
        quaternion = np.asarray(self.q_xyzw, float)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("q_xyzw must contain four finite values")
        if abs(np.linalg.norm(quaternion) - 1.0) > 1e-3:
            raise ValueError("q_xyzw must be unit norm")
```

- [ ] **Step 4: Implement the canonical high-level alignment API**

`add_pair()` remains a private numeric helper named `_add_pair()`. The public
methods are exactly:

```python
def add_dpvo_pose(self, pose):
    self.history.add(pose.frame_ns, pose)


def add_fix(self, dpvo_pose, fix):
    if dpvo_pose.frame_ns != fix.frame_ns:
        raise ValueError("DPVO pose and absolute fix timestamps differ")
    covariance = np.asarray(fix.covariance_6x6, float).reshape(6, 6)[:3, :3]
    return self._add_pair(fix.frame_ns, dpvo_pose.p, fix.p_map_m, covariance)


def predicted_pose(self, dpvo_pose):
    if self.state.similarity is None:
        return None
    p = self.state.similarity.apply(np.asarray(dpvo_pose.p, float)[None])[0]
    return PoseSE3(dpvo_pose.frame_ns, tuple(p), dpvo_pose.q_xyzw)


def correct(self, dpvo_pose):
    return self._correct_from_history(dpvo_pose)


def reset(self, dpvo_identity, reason):
    self.dpvo_identity = str(dpvo_identity)
    self.history = DpvoPoseHistory()
    self.state = AlignmentState(False, str(reason), None, 0, 0, 0.0,
                                float("inf"), float("inf"))
```

- [ ] **Step 5: Update localizer and worker to the shared records**

`PriorMapLocalizer.localize()` reads `prior.p` and `prior.q_xyzw` for position
and orientation innovation. `MapPoseWorker.process_one()` calls
`add_dpvo_pose(dpvo_pose)`, `predicted_pose(dpvo_pose)`,
`add_fix(dpvo_pose, fix)`, and `correct(dpvo_pose)` in that order. No alternate
method names remain.

- [ ] **Step 6: Run cross-plan and subsystem suites**

Run: `python -m pytest vq2/tests/test_cross_plan_pose_api.py vq2/tests/test_map_localizer.py vq2/tests/test_map_alignment.py vq2/tests/test_map_pose_worker_live.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit the pose API**

```bash
git add vq2/mapping/schema.py vq2/mapping/map_alignment.py vq2/mapping/localizer.py vq2/live/map_pose_worker.py vq2/tests/test_cross_plan_pose_api.py
git commit -m "fix(vq2): unify map pose interfaces"
```

### Task 3: Freeze the artifact and approval digest chain

**Files:**
- Modify: `vq2/mapping/artifact.py`
- Modify: `vq2/mapping/benchmark.py`
- Modify: `vq2/live/map_control.py`
- Test: `vq2/tests/test_cross_plan_identity_chain.py`

**Interfaces:**
- Map manifest contains `map_id`, `camera_id`, `feature_id`, `input_manifest_sha256`, `dpvo_identity`, `policy_sha256`, and per-file hashes.
- Localization approval contains `map_sha256`, `policy_sha256`, `b1_manifest_sha256`, `camera_id`, `feature_id`, and `approved`.
- Control approval additionally contains `localization_approval_sha256`, `shadow_log_sha256`, `dpvo_identity`, and `approved`.

- [ ] **Step 1: Write mismatched-chain rejection test**

```python
import pytest

from vq2.live.map_control import verify_identity_chain


def test_runtime_rejects_localization_approval_for_another_map():
    map_manifest = {"map_sha256": "map-a", "policy_sha256": "policy-a",
                    "camera_id": "camera-226", "feature_id": "sift-v1",
                    "dpvo_identity": "dpvo-1"}
    localization = {"approved": True, "map_sha256": "map-b",
                    "policy_sha256": "policy-a", "camera_id": "camera-226",
                    "feature_id": "sift-v1"}
    control = {"approved": True, "localization_approval_sha256": "loc-1",
               "shadow_log_sha256": "shadow-1", "dpvo_identity": "dpvo-1"}
    with pytest.raises(ValueError, match="map_sha256"):
        verify_identity_chain(map_manifest, localization, control,
                              localization_sha256="loc-1")
```

- [ ] **Step 2: Run test and verify missing chain verifier**

Run: `python -m pytest vq2/tests/test_cross_plan_identity_chain.py -v`

Expected: FAIL importing `verify_identity_chain`.

- [ ] **Step 3: Implement exact field equality**

```python
def verify_identity_chain(map_manifest, localization, control,
                          localization_sha256):
    if not localization.get("approved") or not control.get("approved"):
        raise ValueError("approval is false")
    for field in ("map_sha256", "policy_sha256", "camera_id", "feature_id"):
        if localization.get(field) != map_manifest.get(field):
            raise ValueError(f"identity mismatch: {field}")
    if control.get("localization_approval_sha256") != localization_sha256:
        raise ValueError("identity mismatch: localization_approval_sha256")
    if control.get("dpvo_identity") != map_manifest.get("dpvo_identity"):
        raise ValueError("identity mismatch: dpvo_identity")
    return True
```

- [ ] **Step 4: Make B1 open receipt globally exclusive**

`open_holdout()` creates `B1.opened.json` adjacent to the seal with exclusive
mode `"x"` before reading query images. The receipt contains map, policy, B1,
Git, and session digests. If the receipt already exists, opening fails even if
a different output directory is requested. A failed benchmark retains the
receipt and requires a new holdout flight.

- [ ] **Step 5: Run identity, artifact, benchmark, and live-gate tests**

Run: `python -m pytest vq2/tests/test_cross_plan_identity_chain.py vq2/tests/test_mapping_artifact.py vq2/tests/test_map_benchmark.py vq2/tests/test_map_control_live.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the identity chain**

```bash
git add vq2/mapping/artifact.py vq2/mapping/benchmark.py vq2/live/map_control.py vq2/tests/test_cross_plan_identity_chain.py
git commit -m "fix(vq2): bind map approval identities"
```

---

## Contract completion gate

The end-to-end implementation may proceed only when the explicit survey frame
block is mapper-consumable, all pose exchange uses the shared `PoseSE3` API,
and the map/localization/control digest chain rejects every mismatched field.

