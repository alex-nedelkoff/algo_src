# VQ2 Prior-Map Localization and DPVO Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Localize live or replayed camera frames against the immutable metric map and maintain an external, uncertainty-aware DPVO-to-map Sim3 without modifying DPVO.

**Architecture:** A verified map index selects nearby keyframes from a pose prior, CPU SIFT matches query features to map 3D landmarks, and RANSAC PnP emits timestamped absolute SE3 fixes. A robust sliding-window Sim3 pairs those fixes with raw DPVO poses, applies corrections at source time, and propagates the later DPVO delta to now. B1 is opened exactly once for frozen-policy validation and can never tune the map or localizer.

**Tech Stack:** Python 3, NumPy, OpenCV SIFT/PnP, SciPy Rotation, pytest; CPU-only localization and no new dependencies.

## Global Constraints

- Depends on a verified, immutable `vq2-map-v1` artifact and its A-only `approval-policy.json`.
- Normal localization uses CPU SIFT and map landmarks; GateNet is not resident.
- PnP is 2D-to-3D and returns metric SE3. Sim3 is estimated separately from paired DPVO and map poses.
- Normal-mode defaults are at least `30` inliers, `4/9` occupied image cells, reprojection RMS at most `3 px`, and prior innovation at most `2 m` and `15 deg`.
- Sim3 scale requires at least `3` accepted fixes and `1.0 m` source baseline.
- B1 may be evaluated only after policy and map digests are frozen. Zero B1 results may modify either digest.
- Wrong-gate, shuffled-association, and repeated-appearance probes must yield zero accepted global fixes.
- This plan produces localization approval only; it does not enable race control or modify `vq2/live/vq2wp.py`.

---

## File structure

- Modify `vq2/mapping/schema.py`: absolute pose-fix and policy records.
- Create `vq2/mapping/localizer.py`: map index, SIFT query, PnP, and recovery search.
- Create `vq2/mapping/map_alignment.py`: robust sliding Sim3 and source-time propagation.
- Create `vq2/mapping/benchmark.py`: sealed-B1 and banked-corpus evaluation.
- Modify `vq2/mapping/validate.py`: localization approval artifact.
- Add `vq2/tests/test_map_localizer.py`, `test_map_alignment.py`, and `test_map_benchmark.py`.

### Task 1: Add pose-fix policy and verified map index

**Files:**
- Modify: `vq2/mapping/schema.py`
- Create: `vq2/mapping/localizer.py`
- Test: `vq2/tests/test_map_localizer.py`

**Interfaces:**
- Produces: `PnPPolicy`, `AbsolutePoseFix`, and `MapIndex.from_artifact(path) -> MapIndex`.
- Produces: `MapIndex.nearby_keyframes(position, radius_m, limit) -> tuple[int, ...]`.

- [ ] **Step 1: Write identity and spatial-selection tests**

```python
import json
import numpy as np
import pytest

from vq2.mapping.localizer import MapIndex


def write_map(tmp_path):
    np.savez_compressed(tmp_path / "poses.npz",
                        frame_ns=np.array([1, 2, 3]),
                        p=np.array([[0, 0, 0], [5, 0, 0], [20, 0, 0]], float))
    np.savez_compressed(tmp_path / "landmarks.npz",
                        xyz=np.array([[0, 0, 5]], float),
                        descriptor_index=np.array([0]))
    np.savez_compressed(tmp_path / "descriptors.npz",
                        descriptors=np.ones((1, 128), np.float32),
                        landmark_id=np.array([0]))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "vq2-map-v1", "map_id": "map-1",
        "camera_id": "vq2-camera-226", "digest": "verified-in-test"}))
    (tmp_path / "approval-policy.json").write_text(json.dumps({
        "min_inliers": 30, "max_rms_px": 3.0, "min_grid_cells": 4,
        "max_innovation_m": 2.0, "max_innovation_deg": 15.0}))
    return tmp_path


def test_nearby_keyframes_use_metric_pose_prior(tmp_path, monkeypatch):
    monkeypatch.setattr("vq2.mapping.localizer.verify_artifact",
                        lambda path: json.loads((path / "manifest.json").read_text()))
    index = MapIndex.from_artifact(write_map(tmp_path))
    assert index.nearby_keyframes([4.5, 0, 0], radius_m=6, limit=2) == (1, 0)


def test_camera_identity_mismatch_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("vq2.mapping.localizer.verify_artifact",
                        lambda path: json.loads((path / "manifest.json").read_text()))
    with pytest.raises(ValueError, match="camera identity"):
        MapIndex.from_artifact(write_map(tmp_path), camera_id="wrong-camera")
```

- [ ] **Step 2: Run tests and verify missing interfaces**

Run: `python -m pytest vq2/tests/test_map_localizer.py -v`

Expected: FAIL importing `MapIndex`.

- [ ] **Step 3: Add policy and absolute-fix records**

```python
# Append to vq2/mapping/schema.py
@dataclass(frozen=True)
class PnPPolicy:
    min_inliers: int = 30
    max_rms_px: float = 3.0
    min_grid_cells: int = 4
    max_innovation_m: float = 2.0
    max_innovation_deg: float = 15.0


@dataclass(frozen=True)
class AbsolutePoseFix:
    frame_ns: int
    p_map_m: tuple[float, float, float]
    q_map_camera_xyzw: tuple[float, float, float, float]
    covariance_6x6: tuple[float, ...]
    inliers: int
    rms_px: float
    grid_cells: int
    mode: str
    map_id: str
```

- [ ] **Step 4: Implement verified map loading and nearby selection**

```python
# vq2/mapping/localizer.py
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from .artifact import verify_artifact
from .schema import PnPPolicy


class MapIndex:
    def __init__(self, root, manifest, policy, frame_ns, positions,
                 landmark_xyz, descriptors, landmark_ids):
        self.root, self.manifest, self.policy = root, manifest, policy
        self.frame_ns, self.positions = frame_ns, positions
        self.landmark_xyz = landmark_xyz
        self.descriptors, self.landmark_ids = descriptors, landmark_ids

    @classmethod
    def from_artifact(cls, path, camera_id="vq2-camera-226"):
        path = Path(path)
        manifest = verify_artifact(path)
        if manifest["camera_id"] != camera_id:
            raise ValueError("map camera identity does not match runtime camera identity")
        raw_policy = json.loads((path / "approval-policy.json").read_text())
        names = {field.name for field in fields(PnPPolicy)}
        policy = PnPPolicy(**{key: raw_policy[key] for key in names})
        poses = np.load(path / "poses.npz")
        landmarks = np.load(path / "landmarks.npz")
        descriptor_file = np.load(path / "descriptors.npz")
        return cls(path, manifest, policy, poses["frame_ns"], poses["p"],
                   landmarks["xyz"], descriptor_file["descriptors"],
                   descriptor_file["landmark_id"])

    def nearby_keyframes(self, position, radius_m=12.0, limit=8):
        distance = np.linalg.norm(self.positions - np.asarray(position), axis=1)
        indices = np.flatnonzero(distance <= radius_m)
        order = indices[np.argsort(distance[indices])][:limit]
        return tuple(map(int, order))
```

- [ ] **Step 5: Run map-index tests**

Run: `python -m pytest vq2/tests/test_map_localizer.py -v`

Expected: initial index tests PASS.

- [ ] **Step 6: Commit contracts and index**

```bash
git add vq2/mapping/schema.py vq2/mapping/localizer.py vq2/tests/test_map_localizer.py
git commit -m "feat(vq2): index verified prior maps"
```

### Task 2: Produce normal-mode PnP fixes with fixed acceptance policy

**Files:**
- Modify: `vq2/mapping/localizer.py`
- Modify test: `vq2/tests/test_map_localizer.py`

**Interfaces:**
- Produces: `PriorMapLocalizer(index).localize(image, frame_ns, prior_pose) -> AbsolutePoseFix | None`.
- Produces: `LocalizationAttempt(accepted, reason, inliers, rms_px, grid_cells, latency_ms)` audit record for every query.

- [ ] **Step 1: Add a synthetic PnP acceptance test**

```python
import cv2
from scipy.spatial.transform import Rotation

from vq2.mapping.localizer import solve_absolute_pose
from vq2.mapping.schema import PnPPolicy


def test_synthetic_pnp_recovers_camera_pose():
    xyz = np.array([[x, y, 8 + (x + y) * 0.1]
                    for x in (-2, -1, 0, 1, 2) for y in (-1, 0, 1)], float)
    K = np.array([[226, 0, 319.5], [0, 226, 179.5], [0, 0, 1]], float)
    uv, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, None)
    fix, attempt = solve_absolute_pose(xyz, uv.reshape(-1, 2), 10, K,
                                       PnPPolicy(min_inliers=10), "map-1",
                                       prior_position=np.zeros(3))
    assert attempt.accepted
    assert np.linalg.norm(fix.p_map_m) < 1e-3
    assert fix.inliers == len(xyz)
```

- [ ] **Step 2: Run the test and verify missing solver**

Run: `python -m pytest vq2/tests/test_map_localizer.py::test_synthetic_pnp_recovers_camera_pose -v`

Expected: FAIL importing `solve_absolute_pose`.

- [ ] **Step 3: Implement RANSAC PnP, inversion, coverage, and policy checks**

```python
from dataclasses import dataclass
import math
import time

import cv2
from scipy.spatial.transform import Rotation

from .schema import AbsolutePoseFix


@dataclass(frozen=True)
class LocalizationAttempt:
    accepted: bool
    reason: str
    inliers: int
    rms_px: float
    grid_cells: int
    latency_ms: float


def _grid_cells(uv):
    columns = np.clip((np.asarray(uv)[:, 0] / (640 / 3)).astype(int), 0, 2)
    rows = np.clip((np.asarray(uv)[:, 1] / (360 / 3)).astype(int), 0, 2)
    return len(set(zip(columns.tolist(), rows.tolist())))


def solve_absolute_pose(xyz, uv, frame_ns, K, policy, map_id, prior_position):
    started = time.perf_counter()
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(xyz, np.float64), np.asarray(uv, np.float64),
        np.asarray(K, np.float64), None, iterationsCount=200,
        reprojectionError=policy.max_rms_px, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    indices = np.array([], int) if inliers is None else inliers.ravel()
    if not ok or len(indices) < policy.min_inliers:
        return None, LocalizationAttempt(False, "inliers", len(indices),
                                         float("inf"), 0,
                                         (time.perf_counter() - started) * 1000)
    rvec, tvec = cv2.solvePnPRefineLM(np.asarray(xyz)[indices],
                                      np.asarray(uv)[indices], K, None, rvec, tvec)
    projected, _ = cv2.projectPoints(np.asarray(xyz)[indices], rvec, tvec, K, None)
    residual = projected.reshape(-1, 2) - np.asarray(uv)[indices]
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    cells = _grid_cells(np.asarray(uv)[indices])
    R_cw = cv2.Rodrigues(rvec)[0]
    p_wc = (-R_cw.T @ tvec).reshape(3)
    if cells < policy.min_grid_cells or rms > policy.max_rms_px:
        return None, LocalizationAttempt(False, "geometry", len(indices), rms, cells,
                                         (time.perf_counter() - started) * 1000)
    if np.linalg.norm(p_wc - np.asarray(prior_position)) > policy.max_innovation_m:
        return None, LocalizationAttempt(False, "innovation", len(indices), rms, cells,
                                         (time.perf_counter() - started) * 1000)
    q_wc = Rotation.from_matrix(R_cw.T).as_quat()
    covariance = tuple((np.eye(6) * max(rms, 0.25) ** 2).ravel())
    fix = AbsolutePoseFix(int(frame_ns), tuple(p_wc), tuple(q_wc), covariance,
                          len(indices), rms, cells, "normal", str(map_id))
    return fix, LocalizationAttempt(True, "accepted", len(indices), rms, cells,
                                    (time.perf_counter() - started) * 1000)
```

- [ ] **Step 4: Implement query-to-map matching**

Extract SIFT through `vq2.mapping.features.extract_sift`, choose nearby
keyframes, collect their landmark descriptors, apply the `0.75` ratio test,
deduplicate by landmark ID, and pass the resulting 2D/3D pairs to
`solve_absolute_pose`. Record all rejection reasons; never return the prior as
a synthetic accepted fix.

- [ ] **Step 5: Run localizer tests**

Run: `python -m pytest vq2/tests/test_map_localizer.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit normal localization**

```bash
git add vq2/mapping/localizer.py vq2/tests/test_map_localizer.py
git commit -m "feat(vq2): localize against metric map"
```

### Task 3: Add recovery search and repeated-gate rejection

**Files:**
- Modify: `vq2/mapping/localizer.py`
- Modify test: `vq2/tests/test_map_localizer.py`

**Interfaces:**
- Produces: `localize_recovery(image, frame_ns, route_prior) -> AbsolutePoseFix | None`.
- Recovery requires at least `60` inliers, `6/9` image cells, RMS at most `2.5 px`, and either background landmarks spanning two spatial keyframe groups or a decisive route-association margin.

- [ ] **Step 1: Write repeated-gate and shuffled-landmark rejection tests**

```python
def test_single_repeated_gate_cannot_produce_recovery_fix(fake_index, query_image):
    localizer = PriorMapLocalizer(fake_index)
    localizer._match_recovery = lambda image: fake_gate_only_correspondences(80)
    fix, attempt = localizer.localize_recovery(query_image, 20, route_prior=None)
    assert fix is None
    assert attempt.reason == "single_repeated_gate"


def test_shuffled_3d_landmarks_are_rejected(fake_index, query_image):
    localizer = PriorMapLocalizer(fake_index)
    localizer._match_recovery = lambda image: shuffled_correspondences(seed=9)
    fix, attempt = localizer.localize_recovery(query_image, 21, route_prior=None)
    assert fix is None
    assert attempt.reason in {"inliers", "geometry"}
```

- [ ] **Step 2: Run tests and verify recovery behavior is absent**

Run: `python -m pytest vq2/tests/test_map_localizer.py -k 'recovery or shuffled or repeated' -v`

Expected: FAIL because `localize_recovery` is undefined.

- [ ] **Step 3: Implement wider retrieval with stronger verification**

Search all map keyframes in spatial chunks, rank candidates by descriptor
votes, and geometrically verify the best three. Reject fixes whose inliers are
all gate-mask features from one stable gate ID. If a route prior exists,
require the best gate association cost to beat the second-best by the frozen
map policy margin. Mark accepted fixes with `mode="recovery"`.

- [ ] **Step 4: Run all localizer tests**

Run: `python -m pytest vq2/tests/test_map_localizer.py -v`

Expected: all tests PASS, including zero false accepts for shuffled and
single-repeated-gate probes.

- [ ] **Step 5: Commit recovery localization**

```bash
git add vq2/mapping/localizer.py vq2/tests/test_map_localizer.py
git commit -m "feat(vq2): verify map recovery fixes"
```

### Task 4: Estimate a robust sliding-window DPVO-to-map Sim3

**Files:**
- Create: `vq2/mapping/map_alignment.py`
- Test: `vq2/tests/test_map_alignment.py`

**Interfaces:**
- Produces: `Similarity(scale, rotation, translation).apply(points)`.
- Produces: `MapAlignment.add_pair(frame_ns, p_dpvo, p_map, covariance) -> AlignmentState`.
- Publishes no healthy scale before three pairs and `1.0 m` DPVO/map baseline.

- [ ] **Step 1: Write known-Sim3, weak-baseline, and outlier tests**

```python
import numpy as np

from vq2.mapping.map_alignment import MapAlignment


def test_three_metric_pairs_recover_similarity():
    alignment = MapAlignment(min_pairs=3, min_baseline_m=1.0)
    source = [[0, 0, 0], [1, 0, 0], [2, 1, 0]]
    target = [[5, 2, 0], [7, 2, 0], [9, 4, 0]]
    for index, (a, b) in enumerate(zip(source, target)):
        state = alignment.add_pair(index, a, b, np.eye(3) * 0.01)
    assert state.healthy
    assert abs(state.similarity.scale - 2.0) < 1e-6
    assert np.allclose(state.similarity.translation, [5, 2, 0], atol=1e-6)


def test_stationary_pairs_do_not_publish_scale():
    alignment = MapAlignment(min_pairs=3, min_baseline_m=1.0)
    for index in range(5):
        state = alignment.add_pair(index, [0.01 * index, 0, 0],
                                   [0.02 * index, 0, 0], np.eye(3))
    assert not state.healthy and state.reason == "baseline"


def test_one_large_outlier_is_not_accepted():
    alignment = MapAlignment(min_pairs=3, min_baseline_m=1.0, max_residual_m=0.5)
    pairs = [([0, 0, 0], [0, 0, 0]), ([1, 0, 0], [2, 0, 0]),
             ([2, 0, 0], [4, 0, 0]), ([3, 0, 0], [100, 0, 0])]
    for index, (a, b) in enumerate(pairs):
        state = alignment.add_pair(index, a, b, np.eye(3) * 0.01)
    assert state.outlier_count == 1
    assert abs(state.similarity.scale - 2.0) < 1e-6
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `python -m pytest vq2/tests/test_map_alignment.py -v`

Expected: FAIL importing `vq2.mapping.map_alignment`.

- [ ] **Step 3: Implement weighted Umeyama similarity**

```python
def estimate_similarity(source, target, weights):
    source, target = np.asarray(source, float), np.asarray(target, float)
    weights = np.asarray(weights, float)
    weights /= weights.sum()
    mean_s, mean_t = weights @ source, weights @ target
    xs, xt = source - mean_s, target - mean_t
    covariance = (xt * weights[:, None]).T @ xs
    U, singular, Vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(U @ Vt))
    rotation = U @ np.diag(sign) @ Vt
    variance = np.sum(weights * np.sum(xs * xs, axis=1))
    scale = float(np.sum(singular * sign) / variance)
    translation = mean_t - scale * rotation @ mean_s
    return Similarity(scale, rotation, translation)
```

`MapAlignment` keeps the last 20 accepted pairs, fits the similarity, removes
pairs above `0.5 m` residual once, refits, and reports pair count, baseline,
scale uncertainty from bootstrap leave-one-out fits, residual RMS, and
outlier count. A DPVO identity change calls `reset("identity")`.

- [ ] **Step 4: Run alignment tests**

Run: `python -m pytest vq2/tests/test_map_alignment.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit Sim3 alignment**

```bash
git add vq2/mapping/map_alignment.py vq2/tests/test_map_alignment.py
git commit -m "feat(vq2): align DPVO to metric map"
```

### Task 5: Apply delayed fixes at source time and propagate to now

**Files:**
- Modify: `vq2/mapping/map_alignment.py`
- Modify test: `vq2/tests/test_map_alignment.py`

**Interfaces:**
- Produces: `DpvoPoseHistory.add(frame_ns, pose)` and `.delta(source_ns, current_ns)`.
- Produces: `MapAlignment.correct_current(fix, current_dpvo_pose) -> CorrectedPose`.

- [ ] **Step 1: Write delayed-fix and discontinuity tests**

```python
def test_delayed_fix_propagates_post_fix_dpvo_delta():
    history = DpvoPoseHistory()
    history.add(100, pose([0, 0, 0]))
    history.add(200, pose([1, 0, 0]))
    history.add(300, pose([2, 0, 0]))
    corrected = corrected_from_source_fix(history, 200, pose([11, 0, 0]), 300)
    assert np.allclose(corrected.p, [12, 0, 0])


def test_timestamp_regression_invalidates_history():
    history = DpvoPoseHistory()
    history.add(200, pose([0, 0, 0]))
    with pytest.raises(ValueError, match="timestamp regression"):
        history.add(100, pose([0, 0, 0]))
    assert not history.healthy
```

- [ ] **Step 2: Run delayed-fix tests and verify failure**

Run: `python -m pytest vq2/tests/test_map_alignment.py -k 'delayed or regression' -v`

Expected: FAIL because history APIs are undefined.

- [ ] **Step 3: Implement bounded pose history and SE3 delta propagation**

Store the last 10 seconds keyed by `frame_ns`. For a fix timestamp, require an
exact DPVO pose or a bracket no wider than `150 ms`; interpolate translation
and quaternion SLERP. Compute `T_source_to_now = inv(T_dpvo_source) @
T_dpvo_now`, then publish `T_map_source @ T_source_to_now`. Reject stale fixes,
missing brackets, timestamp regression, nonfinite poses, and identity changes.

- [ ] **Step 4: Run the full alignment suite**

Run: `python -m pytest vq2/tests/test_map_alignment.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit temporal alignment**

```bash
git add vq2/mapping/map_alignment.py vq2/tests/test_map_alignment.py
git commit -m "feat(vq2): propagate delayed map fixes"
```

### Task 6: Open B1 once and issue localization approval

**Files:**
- Create: `vq2/mapping/benchmark.py`
- Modify: `vq2/mapping/validate.py`
- Test: `vq2/tests/test_map_benchmark.py`

**Interfaces:**
- Produces: `benchmark_corpus(map_path, corpus_path, seal_path, output) -> BenchmarkReport`.
- Produces: `localization-approval.json` bound to map, policy, B1 manifest, camera, feature, and DPVO digests.
- Does not produce a control-approval latch.

- [ ] **Step 1: Write seal-consumption and immutability tests**

```python
import json
import pytest

from vq2.mapping.benchmark import open_holdout


def test_holdout_open_requires_matching_manifest_digest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"pass":"B1"}\n')
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"pass": "B1", "opened": False,
                                "manifest_sha256": "wrong"}))
    with pytest.raises(ValueError, match="seal digest"):
        open_holdout(manifest, seal, tmp_path / "session.json")


def test_holdout_cannot_be_opened_twice(tmp_path):
    manifest, seal = matching_holdout_files(tmp_path)
    session = tmp_path / "session.json"
    open_holdout(manifest, seal, session)
    with pytest.raises(ValueError, match="already opened"):
        open_holdout(manifest, seal, session)
```

- [ ] **Step 2: Run tests and verify missing benchmark module**

Run: `python -m pytest vq2/tests/test_map_benchmark.py -v`

Expected: FAIL importing `vq2.mapping.benchmark`.

- [ ] **Step 3: Implement content-addressed holdout session and metrics**

The session records the pre-open map and policy digests, B1 digest, Git SHA,
query frame list, and wall time. It creates its output path with exclusive
creation. Benchmark every configured query frame and report accepted-fix rate,
gap distribution, inliers, RMS, image coverage, normal/recovery counts,
cross-pass pose repeatability, wrong-association false accepts, latency, peak
RSS, and map size.

```python
@dataclass(frozen=True)
class BenchmarkReport:
    query_count: int
    accepted_count: int
    false_accept_count: int
    p95_gap_s: float
    max_gap_s: float
    p95_latency_ms: float
    repeatability_m: float
    repeatability_deg: float
```

- [ ] **Step 4: Freeze approval decision**

Approve localization only if B1 meets the already hashed policy: zero false
accepts, at least one accepted fix per second over mapped regions, maximum gap
at most `2.0 s`, repeatability at most `0.5 m` and `5 deg`, p95 CPU latency at
most `100 ms`, map size at most `250 MB`, and process RSS at most `1 GB`.
Failure records all missed criteria and leaves `approved=false`; it never
rewrites thresholds or map files.

- [ ] **Step 5: Run benchmark tests and mapping regression suite**

Run: `python -m pytest vq2/tests/test_map_benchmark.py vq2/tests/test_map_localizer.py vq2/tests/test_map_alignment.py vq2/tests/test_mapping_*.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit benchmark and approval**

```bash
git add vq2/mapping/benchmark.py vq2/mapping/validate.py vq2/tests/test_map_benchmark.py
git commit -m "feat(vq2): validate prior-map localization"
```

### Task 7: Run B1 and banked replay without flight control

**Files:**
- Create outside git: `C:\Users\alexj\vq2_maps\validation_B1_001\`
- Create outside git: `C:\Users\alexj\vq2_maps\wrapper_replay_001\`

**Interfaces:**
- Consumes the immutable map-A digest and sealed B1 manifest.
- Produces localization approval plus raw, normalization-compensated, and map-corrected trajectories.

- [ ] **Step 1: Verify map and seal before opening B1**

Run: `python -m vq2.mapping.validate verify C:\Users\alexj\vq2_maps\map_A_candidate`

Expected: digest match, input roles `A1,A2,A3`, and frozen approval policy.

Run: `python -m vq2.mapping.benchmark verify-seal C:\Users\alexj\vq2_survey_B1_001\manifest.json C:\Users\alexj\vq2_survey_B1_001\B1.seal.json`

Expected: matching digest and `opened=false`.

- [ ] **Step 2: Run the one-shot B1 benchmark**

Run: `python -m vq2.mapping.benchmark C:\Users\alexj\vq2_maps\map_A_candidate C:\Users\alexj\vq2_survey_B1_001 --seal C:\Users\alexj\vq2_survey_B1_001\B1.seal.json --output C:\Users\alexj\vq2_maps\validation_B1_001`

Expected: immutable report and explicit approval decision with unchanged map
and policy digests.

- [ ] **Step 3: Replay fg62 and fg75 through localizer and wrapper**

Run: `python -m vq2.mapping.benchmark replay C:\Users\alexj\vq2_maps\map_A_candidate C:\Users\alexj\vq2_servo_fg62 --output C:\Users\alexj\vq2_maps\wrapper_replay_001\fg62`

Run: `python -m vq2.mapping.benchmark replay C:\Users\alexj\vq2_maps\map_A_candidate C:\Users\alexj\vq2_servo_fg75 --output C:\Users\alexj\vq2_maps\wrapper_replay_001\fg75`

Expected: raw, normalization-compensated, and map-corrected trajectories;
source-time residuals; fix gaps; scale and uncertainty history; no false map
association. Judge transitions are decoded only with `gidx2` semantics.

- [ ] **Step 4: Stop if localization approval fails**

If B1 fails, retain the failed immutable report. Select one new A-only mapping
or algorithm change from the recorded failure class, collect a new sealed
validation flight for the next approval attempt, and do not begin race
integration against the failed map.

---

## Completion gate

This plan is complete only when the localizer and alignment suites pass, B1
produces zero false accepts under the frozen policy, replay demonstrates
bounded fix gaps and delayed-fix propagation, and a content-addressed
`localization-approval.json` exists. That approval does not authorize control.

