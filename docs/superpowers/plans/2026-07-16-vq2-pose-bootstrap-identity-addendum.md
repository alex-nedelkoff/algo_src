# VQ2 Pose Bootstrap and Identity Addendum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the final cross-plan edge cases for bootstrapping DPVO into the map frame, transforming pose orientation, and comparing the external map-artifact digest.

**Architecture:** The live worker uses strict global recovery until three metric fixes make the external Sim3 healthy, then changes to prior-guided localization. One shared pose record carries source nanoseconds and camera-to-world orientation. The verified map-package digest is loaded beside the manifest rather than embedded recursively inside the files it hashes.

**Tech Stack:** Python dataclasses, NumPy, SciPy Rotation, SHA-256, pytest; no new dependency.

## Global Constraints

- This addendum takes precedence over conflicting shared-interface examples in
  the four base plans, the self-review-fixes companion, and the cross-plan
  interface contract.
- Execute Task 1 before prior-map localization Task 4 or race-integration Task
  2. Execute Task 2 before B1 is opened or a map is loaded live.
- Quaternions are `(x, y, z, w)`; `PoseSE3` is camera-to-world; timestamps are
  source `frame_ns` integers.
- No pose is healthy before the frozen minimum pair count, baseline,
  conditioning, residual, and uncertainty gates all pass.

---

### Task 1: Bootstrap alignment and transform complete SE3 poses

**Files:**
- Modify: `vq2/mapping/schema.py`
- Modify: `vq2/mapping/map_alignment.py`
- Modify: `vq2/mapping/localizer.py`
- Modify: `vq2/live/map_pose_worker.py`
- Modify: `vq2/tests/test_map_alignment.py`
- Modify: `vq2/tests/test_map_localizer.py`
- Modify: `vq2/tests/test_map_pose_worker_live.py`
- Test: `vq2/tests/test_cross_plan_pose_bootstrap.py`

**Interfaces:**
- `PoseSE3(frame_ns, p, q_xyzw)` is the only pose exchanged between DPVO,
  localization, alignment, and the live worker.
- `PriorMapLocalizer.localize(image, frame_ns, prior, route_prior=None)` accepts
  `prior: PoseSE3 | None`. A missing prior invokes the already frozen strict
  recovery policy.
- `MapAlignment` exposes only `add_dpvo_pose(pose)`,
  `add_fix(dpvo_pose, fix)`, `predicted_pose(dpvo_pose)`, `correct(dpvo_pose)`,
  and `reset(dpvo_identity, reason)` to the worker.
- `Similarity.apply_pose(pose)` maps position and orientation from the DPVO
  world frame into the metric map frame.

- [ ] **Step 1: Write signature and orientation-transform tests**

```python
import inspect
import numpy as np
from scipy.spatial.transform import Rotation

from vq2.mapping.localizer import PriorMapLocalizer
from vq2.mapping.map_alignment import Similarity
from vq2.mapping.schema import PoseSE3


def test_localizer_has_optional_route_prior_contract():
    assert tuple(inspect.signature(PriorMapLocalizer.localize).parameters) == (
        "self", "image", "frame_ns", "prior", "route_prior")


def test_similarity_rotates_camera_to_world_orientation():
    rotation = Rotation.from_euler("z", 90, degrees=True)
    similarity = Similarity(2.0, rotation.as_matrix(), np.array([4., 5., 6.]))
    pose = PoseSE3(100, (1., 0., 0.), (0., 0., 0., 1.))
    mapped = similarity.apply_pose(pose)
    assert np.allclose(mapped.p, [4., 7., 6.], atol=1e-8)
    assert np.allclose(
        Rotation.from_quat(mapped.q_xyzw).as_matrix(),
        rotation.as_matrix(), atol=1e-8)
```

- [ ] **Step 2: Run tests and verify the incomplete contract**

Run: `python -m pytest vq2/tests/test_cross_plan_pose_bootstrap.py -k 'contract or rotates' -v`

Expected: FAIL on the missing optional argument, `apply_pose()`, or raw DPVO
orientation being returned unchanged.

- [ ] **Step 3: Implement complete pose transformation**

```python
def apply_pose(self, pose):
    p_map = self.apply(np.asarray(pose.p, float)[None])[0]
    q_map = (Rotation.from_matrix(self.rotation)
             * Rotation.from_quat(pose.q_xyzw)).as_quat()
    return PoseSE3(pose.frame_ns, tuple(p_map), tuple(q_map))
```

The multiplication is left-sided because the Sim3 rotation changes the world
basis of a camera-to-world pose. `predicted_pose()` returns
`self.state.similarity.apply_pose(dpvo_pose)` when a similarity exists and
`None` otherwise. `correct()` must use the same orientation convention and
must never copy a raw DPVO quaternion into a map-frame result.

- [ ] **Step 4: Write the unaligned bootstrap state-machine test**

Use four DPVO poses spanning more than `1.0 m`. Configure the localizer fake to
return three geometrically valid recovery fixes and one normal fix. Assert:

```python
assert calls[:3] == ["recovery", "recovery", "recovery"]
assert calls[3] == "normal"
assert published[0].healthy is False
assert published[1].healthy is False
assert published[2].healthy is True
assert all(row.frame_ns == pose.frame_ns
           for row, pose in zip(published, dpvo_poses))
```

Add separate cases in which a rejected recovery fix does not add a pair, a
weak baseline keeps alignment unhealthy, and an identity reset returns the
next query to recovery mode.

- [ ] **Step 5: Implement the canonical worker order**

```python
def process_one(self, frame_ns, image, dpvo_pose, route_prior=None):
    if frame_ns != dpvo_pose.frame_ns:
        raise ValueError("frame and DPVO timestamps differ")
    self.alignment.add_dpvo_pose(dpvo_pose)
    prior = self.alignment.predicted_pose(dpvo_pose)
    fix, attempt = self.localizer.localize(
        image, frame_ns, prior, route_prior=route_prior)
    if fix is not None:
        self.alignment.add_fix(dpvo_pose, fix)
    corrected = self.alignment.correct(dpvo_pose)
    self._publish_attempt(corrected, attempt, fix)
```

When `prior is None`, `localize()` dispatches to
`localize_recovery(image, frame_ns, route_prior)` and uses its stronger frozen
thresholds. When a prior exists, it uses normal local search. Do not keep the
older names `history.add`, `predicted_map_pose`, `add_pair`, or
`correct(frame_ns, pose)` in worker code or fakes.

- [ ] **Step 6: Update every worker fake to the canonical API**

The self-review companion's earlier `fake_alignment()` example is superseded
by:

```python
def fake_alignment():
    corrected = SimpleNamespace(
        p=np.zeros(3), q_xyzw=np.array([0., 0., 0., 1.]),
        healthy=False, reason="bootstrap", fix_age_s=float("inf"),
        scale_uncertainty=float("inf"))
    return SimpleNamespace(
        dpvo_identity="dpvo-1",
        add_dpvo_pose=lambda pose: None,
        predicted_pose=lambda pose: None,
        add_fix=lambda pose, fix: None,
        correct=lambda pose: corrected)
```

Accepting worker tests use a second fake whose third `add_fix()` changes
`predicted_pose()` and `correct()` to healthy map-frame poses. This prevents a
test double from bypassing the actual bootstrap gate.

- [ ] **Step 7: Run all pose/localization/worker tests**

Run: `python -m pytest vq2/tests/test_cross_plan_pose_api.py vq2/tests/test_cross_plan_pose_bootstrap.py vq2/tests/test_map_alignment.py vq2/tests/test_map_localizer.py vq2/tests/test_map_pose_worker_live.py -v`

Expected: all tests PASS, including orientation, weak-baseline, rejected-fix,
identity-reset, and recovery-to-normal transitions.

- [ ] **Step 8: Commit the bootstrap contract**

```bash
git add vq2/mapping/schema.py vq2/mapping/map_alignment.py vq2/mapping/localizer.py vq2/live/map_pose_worker.py vq2/tests/test_cross_plan_pose_bootstrap.py vq2/tests/test_map_alignment.py vq2/tests/test_map_localizer.py vq2/tests/test_map_pose_worker_live.py
git commit -m "fix(vq2): bootstrap complete map poses"
```

### Task 2: Separate verified artifact digest from map manifest fields

**Files:**
- Modify: `vq2/mapping/artifact.py`
- Modify: `vq2/mapping/benchmark.py`
- Modify: `vq2/live/map_control.py`
- Modify: `vq2/tests/test_mapping_artifact.py`
- Modify: `vq2/tests/test_map_control_live.py`
- Test: `vq2/tests/test_cross_plan_identity_chain.py`

**Interfaces:**
- Map `manifest.json` contains `map_id`, `camera_id`, `feature_id`,
  `input_manifest_sha256`, `dpvo_identity`, `policy_sha256`, and per-file
  hashes.
- `map_sha256` is the verified content of the external `digest.sha256`; it is
  not embedded in any file included by that digest.
- `load_verified_artifact(path) -> LoadedMapArtifact` returns the parsed map,
  manifest, and verified `map_sha256` together.
- `MapRuntimeIdentity` copies the verified `map_sha256` plus `policy_sha256`,
  `camera_id`, `feature_id`, and `dpvo_identity` for approval comparisons.

- [ ] **Step 1: Write self-reference and mismatch tests**

```python
def test_loaded_identity_uses_external_verified_digest(map_artifact):
    loaded = load_verified_artifact(map_artifact)
    assert loaded.map_sha256 == (map_artifact / "digest.sha256").read_text().strip()
    assert "map_sha256" not in loaded.manifest


def test_localization_approval_for_another_map_is_rejected():
    identity = runtime_identity(map_sha256="map-a")
    localization = localization_approval(map_sha256="map-b")
    with pytest.raises(ValueError, match="map_sha256"):
        verify_identity_chain(identity, localization, passing_control())
```

Also mutate each numeric NPZ file and each manifest identity field one at a
time; every mutation must fail verification before localization starts.

- [ ] **Step 2: Run tests and verify the digest ambiguity**

Run: `python -m pytest vq2/tests/test_mapping_artifact.py vq2/tests/test_cross_plan_identity_chain.py -k 'digest or identity or another_map' -v`

Expected: FAIL until the loader distinguishes manifest fields from the
external package digest.

- [ ] **Step 3: Implement the verified loaded-artifact record**

```python
@dataclass(frozen=True)
class LoadedMapArtifact:
    course_map: CourseMap
    manifest: dict
    map_sha256: str


def load_verified_artifact(path):
    path = Path(path)
    manifest = verify_artifact(path)
    expected = (path / "digest.sha256").read_text().strip()
    if artifact_digest(path) != expected:
        raise ValueError("map artifact digest mismatch")
    return LoadedMapArtifact(load_map_arrays(path, manifest), manifest,
                             expected)
```

Construct `MapRuntimeIdentity` only from this record. Localization and control
approvals compare exact equality for map digest, policy digest, camera,
feature, and DPVO identities. The localization-approval file itself is also
hashed, and the control approval must match that hash.

- [ ] **Step 4: Preserve globally exclusive B1 opening**

Before any B1 image is read, create `B1.opened.json` adjacent to the seal with
exclusive mode `"x"`. Include the verified map digest, frozen policy digest,
B1 manifest digest, Git SHA, and session ID. An existing receipt is a hard
failure even when a different output directory is requested. A failed B1 run
retains the receipt and requires a new sealed holdout flight.

- [ ] **Step 5: Run artifact, benchmark, and live identity suites**

Run: `python -m pytest vq2/tests/test_mapping_artifact.py vq2/tests/test_map_benchmark.py vq2/tests/test_cross_plan_identity_chain.py vq2/tests/test_map_control_live.py -v`

Expected: all tests PASS with mismatch rejection occurring before any map
query, B1 image read, or control-source change.

- [ ] **Step 6: Commit the identity clarification**

```bash
git add vq2/mapping/artifact.py vq2/mapping/benchmark.py vq2/live/map_control.py vq2/tests/test_mapping_artifact.py vq2/tests/test_map_benchmark.py vq2/tests/test_cross_plan_identity_chain.py vq2/tests/test_map_control_live.py
git commit -m "fix(vq2): verify external map identity"
```

---

## Addendum completion gate

The implementation may enter B1 validation or live map observation only when
unaligned frames bootstrap through strict recovery, the Sim3 transforms both
position and orientation, every worker and fake uses the canonical API, and
the runtime approval chain is bound to the externally verified map-package
digest.
