# VQ2 Offline Metric Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an immutable metric course map from survey flights A1/A2/A3 while keeping B1 sealed for independent validation.

**Architecture:** The pipeline validates explicit corpus frame blocks, runs GateNet and DPVO sequentially, extracts SIFT background tracks, associates persistent gates across runs, and jointly optimizes camera poses, metric gate poses, and sparse landmarks. Gate dimensions supply scale; final post-termination DPVO poses supply sequential priors. Every artifact is content-addressed and records all rejected observations and associations.

**Tech Stack:** Python 3, NumPy, OpenCV SIFT, SciPy `least_squares` and `linear_sum_assignment`, existing WSL DPVO bridge, pytest; no new dependencies.

## Global Constraints

- Depends on the frozen `vq2-survey-v1` corpus contract from the autonomous-survey plan.
- Map inputs are exactly A1, A2, and A3. B1 is rejected by build and tuning commands.
- Canonical camera identity is `640x360`, `fx=fy=226.0`, `cx=319.5`, `cy=179.5`, and the `vq2/camera.py` extrinsic.
- Run full-resolution GateNet, unload it, then run DPVO; never co-reside them on the 4 GB GPU.
- Export DPVO positions and quaternions only after `slam.terminate()` and map returned frame indices back to source `frame_ns`.
- GateNet per-frame `gate_id` is not a persistent map ID.
- Judge ticks are aperture-plane membership with uncertainty, never exact gate-center camera poses.
- Keep one JPEG source tree; store numeric derivatives in compressed NPZ and audit rows in JSON/JSONL.
- Do not modify `vq2/live/vq2wp.py` or enable any flight-control source in this plan.

---

## File structure

- Create `vq2/mapping/__init__.py`: public mapping API.
- Create `vq2/mapping/schema.py`: versioned observations, trajectories, build inputs, and map records.
- Create `vq2/mapping/extract.py`: explicit frame-block and GateNet extraction.
- Create `vq2/mapping/dpvo_batch.py`: final-trajectory bridge client and exporter.
- Create `vq2/mapping/features.py`: SIFT keyframes, tracks, and verified loops.
- Create `vq2/mapping/association.py`: within/across-run gate association.
- Create `vq2/mapping/optimizer.py`: sparse joint metric optimization.
- Create `vq2/mapping/artifact.py`: immutable map writer, reader, and digest.
- Create `vq2/mapping/validate.py`: map-A quality and sealed-B1 firewall.
- Modify `vq2/live/dpvo_bridge_protocol.py`: finalization packet.
- Modify `vq2/live/bridge_dpvo.py`: post-termination trajectory reply.
- Add focused `vq2/tests/test_mapping_*.py` tests.

### Task 1: Freeze mapping schemas and the B1 input firewall

**Files:**
- Create: `vq2/mapping/__init__.py`
- Create: `vq2/mapping/schema.py`
- Test: `vq2/tests/test_mapping_schema.py`

**Interfaces:**
- Produces: `RunInput`, `MapBuildInputs`, `GateObservation`, `DpvoPose`, `DpvoTrajectory`, `MapGate`, `MapLandmark`, and `CourseMap`.
- Produces: `MapBuildInputs.from_manifests(paths) -> MapBuildInputs`, requiring one complete A1/A2/A3 manifest each and rejecting B1.

- [ ] **Step 1: Write schema and firewall tests**

```python
import json
import pytest

from vq2.mapping.schema import GateObservation, MapBuildInputs


def write_manifest(path, role):
    path.write_text(json.dumps({"schema": "vq2-survey-v1", "pass": role,
                                "status": "complete", "frame_block": [10, 20]}))
    return path


def test_build_inputs_require_a1_a2_a3(tmp_path):
    inputs = MapBuildInputs.from_manifests([
        write_manifest(tmp_path / "a1.json", "A1"),
        write_manifest(tmp_path / "a2.json", "A2"),
        write_manifest(tmp_path / "a3.json", "A3"),
    ])
    assert tuple(run.role for run in inputs.runs) == ("A1", "A2", "A3")


def test_build_inputs_reject_b1(tmp_path):
    paths = [write_manifest(tmp_path / f"{role}.json", role)
             for role in ("A1", "A2", "A3", "B1")]
    with pytest.raises(ValueError, match="B1 is validation-only"):
        MapBuildInputs.from_manifests(paths)


def test_gate_observation_requires_eight_full_covariance_corners():
    with pytest.raises(ValueError, match="corners_uv"):
        GateObservation("A1", 1, 2, [[1.0, 2.0]] * 4,
                        [[[1.0, 0.0], [0.0, 1.0]]] * 4, 0.9)
```

- [ ] **Step 2: Run tests and verify the missing package failure**

Run: `python -m pytest vq2/tests/test_mapping_schema.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'vq2.mapping'`.

- [ ] **Step 3: Implement versioned immutable schemas**

```python
# vq2/mapping/schema.py
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RunInput:
    role: str
    manifest: Path
    manifest_sha256: str
    frame_block: tuple[int, int]


@dataclass(frozen=True)
class MapBuildInputs:
    runs: tuple[RunInput, ...]

    @classmethod
    def from_manifests(cls, paths):
        runs = []
        for path in map(Path, paths):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["pass"] == "B1":
                raise ValueError("B1 is validation-only")
            if row["pass"] not in {"A1", "A2", "A3"}:
                raise ValueError("map input must be A1/A2/A3")
            if row["status"] != "complete":
                raise ValueError("map input manifest is incomplete")
            runs.append(RunInput(row["pass"], path, sha256_file(path),
                                 tuple(map(int, row["frame_block"]))))
        runs.sort(key=lambda run: run.role)
        if [run.role for run in runs] != ["A1", "A2", "A3"]:
            raise ValueError("map build requires exactly A1, A2, and A3")
        return cls(tuple(runs))


@dataclass(frozen=True)
class GateObservation:
    run_id: str
    frame_ns: int
    track_id: int
    corners_uv: np.ndarray
    covariance_uv: np.ndarray
    score: float

    def __post_init__(self):
        corners = np.asarray(self.corners_uv, dtype=float)
        covariance = np.asarray(self.covariance_uv, dtype=float)
        if corners.shape != (8, 2):
            raise ValueError("corners_uv must have shape (8, 2)")
        if covariance.shape != (8, 2, 2):
            raise ValueError("covariance_uv must have shape (8, 2, 2)")
        object.__setattr__(self, "corners_uv", corners)
        object.__setattr__(self, "covariance_uv", covariance)


@dataclass(frozen=True)
class DpvoPose:
    frame_ns: int
    p: tuple[float, float, float]
    q_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class DpvoTrajectory:
    run_id: str
    identity: str
    final: bool
    poses: tuple[DpvoPose, ...]
    normalization_factors: tuple[float, ...] = ()


@dataclass(frozen=True)
class MapGate:
    gate_id: str
    route_order: int
    pose_se3: tuple[float, ...]
    covariance_6x6: tuple[float, ...]


@dataclass(frozen=True)
class MapLandmark:
    landmark_id: int
    xyz: tuple[float, float, float]
    descriptor_index: int


@dataclass(frozen=True)
class CourseMap:
    schema: str
    map_id: str
    gates: tuple[MapGate, ...]
    landmarks: tuple[MapLandmark, ...]
    approved: bool = False
```

```python
# vq2/mapping/__init__.py
from .schema import CourseMap, GateObservation, MapBuildInputs

__all__ = ["CourseMap", "GateObservation", "MapBuildInputs"]
```

- [ ] **Step 4: Run schema tests**

Run: `python -m pytest vq2/tests/test_mapping_schema.py -v`

Expected: `3 passed`.

- [ ] **Step 5: Commit schemas**

```bash
git add vq2/mapping/__init__.py vq2/mapping/schema.py vq2/tests/test_mapping_schema.py
git commit -m "feat(vq2): define metric map schemas"
```

### Task 2: Select explicit corpus blocks and export lossless GateNet observations

**Files:**
- Create: `vq2/mapping/extract.py`
- Test: `vq2/tests/test_mapping_extract.py`

**Interfaces:**
- Produces: `select_frame_rows(corpus: Path, run: RunInput) -> tuple[dict, ...]`.
- Produces: `extract_gatenet(corpus, run, detector, output) -> Path` containing all eight corners, full covariance, masks, source timestamp, track ID, PnP quality, and calibration digest.

- [ ] **Step 1: Write explicit-block and extra-frame rejection tests**

```python
import json
import pytest

from vq2.mapping.extract import select_frame_rows
from vq2.mapping.schema import RunInput


def test_selects_only_manifest_frame_block(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [{"sim_ns": value, "rx_wall": value / 1e9} for value in (5, 10, 15, 20, 25)]
    (corpus / "frames_dedup.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    run = RunInput("A1", tmp_path / "manifest.json", "abc", (10, 20))
    assert [row["sim_ns"] for row in select_frame_rows(corpus, run)] == [10, 15, 20]


def test_missing_block_endpoint_is_rejected(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "frames_dedup.jsonl").write_text('{"sim_ns":10}\n')
    run = RunInput("A1", tmp_path / "manifest.json", "abc", (10, 20))
    with pytest.raises(ValueError, match="frame block endpoint"):
        select_frame_rows(corpus, run)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_mapping_extract.py -v`

Expected: FAIL importing `vq2.mapping.extract`.

- [ ] **Step 3: Implement explicit selection and JPEG/hash validation**

```python
# vq2/mapping/extract.py
import json
from pathlib import Path

from .schema import GateObservation


def select_frame_rows(corpus, run):
    corpus = Path(corpus)
    rows = [json.loads(line) for line in
            (corpus / "frames_dedup.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    start, end = run.frame_block
    selected = tuple(row for row in rows if start <= int(row["sim_ns"]) <= end)
    if not selected or int(selected[0]["sim_ns"]) != start or int(selected[-1]["sim_ns"]) != end:
        raise ValueError("frame block endpoint missing from corpus")
    for row in selected:
        frame = corpus / "frames" / f"{int(row['sim_ns'])}.jpg"
        if not frame.is_file():
            raise ValueError(f"frame metadata has no JPEG: {row['sim_ns']}")
    return selected
```

- [ ] **Step 4: Add injected, sequential GateNet export**

For each selected source JPEG call the injected detector once. Convert every
decoded instance to `GateObservation`; retain `corners_uv`, full `(8,2,2)`
covariance, usable/suspect/visibility masks, `t_cam`, PnP residual, and survey
track ID. Write one JSONL row per instance plus an NPZ for numeric arrays.

```python
def observation_row(observation, masks, t_cam, pnp_rms, calibration_id):
    return {"run_id": observation.run_id, "frame_ns": observation.frame_ns,
            "track_id": observation.track_id, "score": observation.score,
            "corners_uv": observation.corners_uv.tolist(),
            "covariance_uv": observation.covariance_uv.tolist(),
            "usable": list(map(bool, masks["usable"])),
            "suspect": list(map(bool, masks["suspect"])),
            "visible": list(map(bool, masks["visible"])),
            "t_cam": list(map(float, t_cam)), "pnp_rms": float(pnp_rms),
            "calibration_id": str(calibration_id)}
```

- [ ] **Step 5: Run extraction and canonical camera tests**

Run: `python -m pytest vq2/tests/test_mapping_extract.py vq2/tests/test_camera.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit extraction**

```bash
git add vq2/mapping/extract.py vq2/tests/test_mapping_extract.py
git commit -m "feat(vq2): extract explicit map observations"
```

### Task 3: Export final post-termination DPVO trajectories

**Files:**
- Modify: `vq2/live/dpvo_bridge_protocol.py:13-112`
- Modify: `vq2/live/bridge_dpvo.py:85-175`
- Create: `vq2/mapping/dpvo_batch.py`
- Test: `vq2/tests/test_mapping_dpvo_batch.py`
- Modify test: `vq2/tests/test_dpvo_bridge_protocol.py`

**Interfaces:**
- Produces protocol payload `FIN1` through `encode_finalize_packet()` and `is_finalize_packet()`.
- Service replies with `type=trajectory`, `final=true`, source `frame_ns`, `p`, `q`, and identity.
- Produces `run_final_dpvo(corpus, run, config, output) -> DpvoTrajectory`.

- [ ] **Step 1: Write finalization protocol tests**

```python
from vq2.live.dpvo_bridge_protocol import encode_finalize_packet, is_finalize_packet


def test_finalize_packet_is_distinct_from_frame_packet():
    payload = encode_finalize_packet()
    assert payload == b"FIN1"
    assert is_finalize_packet(payload)
    assert not is_finalize_packet(b"FRM1" + b"x" * 12)
```

- [ ] **Step 2: Write post-termination index-to-source-time test**

```python
import numpy as np

from vq2.mapping.dpvo_batch import final_pose_rows


class FakeSlam:
    def terminate(self):
        poses = np.array([[1, 2, 3, 0, 0, 0, 1],
                          [4, 5, 6, 0, 0, 0, 1]], dtype=float)
        return poses, np.array([0.0, 2.0])


def test_final_pose_rows_map_dpvo_indices_to_frame_ns():
    rows = final_pose_rows(FakeSlam(), [100, 200, 300])
    assert [row["frame_ns"] for row in rows] == [100, 300]
    assert rows[1]["p"] == [4.0, 5.0, 6.0]
    assert rows[1]["q"] == [0.0, 0.0, 0.0, 1.0]
```

- [ ] **Step 3: Run tests and verify missing finalize API**

Run: `python -m pytest vq2/tests/test_mapping_dpvo_batch.py vq2/tests/test_dpvo_bridge_protocol.py -v`

Expected: FAIL importing `encode_finalize_packet` or `final_pose_rows`.

- [ ] **Step 4: Add the finalization protocol**

```python
# vq2/live/dpvo_bridge_protocol.py
FINALIZE_MAGIC = b"FIN1"


def encode_finalize_packet() -> bytes:
    return FINALIZE_MAGIC


def is_finalize_packet(payload: bytes) -> bool:
    return bytes(payload) == FINALIZE_MAGIC
```

- [ ] **Step 5: Implement deterministic post-termination export**

```python
# vq2/mapping/dpvo_batch.py
def final_pose_rows(slam, submitted_frame_ns):
    poses, indices = slam.terminate()
    output = []
    for pose, index in zip(poses, indices):
        source_index = int(round(float(index)))
        if source_index < 0 or source_index >= len(submitted_frame_ns):
            raise ValueError("DPVO final trajectory index is outside submitted frames")
        output.append({"frame_ns": int(submitted_frame_ns[source_index]),
                       "p": [float(value) for value in pose[:3]],
                       "q": [float(value) for value in pose[3:7]]})
    return output
```

In `serve_connection()`, append every submitted `frame_ns`. When `FIN1`
arrives, call the same helper after moving it to a CUDA-free shared module,
send one `trajectory` JSON message, and stop accepting frames for that session.
Do not call `terminate()` from the normal live-flight close path.

- [ ] **Step 6: Write the batch client and artifact**

The client submits the explicit frame block, sends `FIN1`, requires matching
session identity and `final=true`, and writes JSONL records plus a summary with
frame count, first/last `frame_ns`, model/config digest, and final provenance.

```python
def trajectory_summary(identity, rows):
    if not rows:
        raise ValueError("DPVO returned an empty final trajectory")
    return {"schema": "vq2-dpvo-final-v1", "identity": identity,
            "final": True, "pose_count": len(rows),
            "first_frame_ns": rows[0]["frame_ns"],
            "last_frame_ns": rows[-1]["frame_ns"]}
```

- [ ] **Step 7: Run bridge and batch tests**

Run: `python -m pytest vq2/tests/test_mapping_dpvo_batch.py vq2/tests/test_dpvo_bridge_protocol.py vq2/tests/test_dpvo_bridge_service.py vq2/tests/test_dpvo_bridge_replay.py -v`

Expected: all tests PASS without a real CUDA service.

- [ ] **Step 8: Commit final-batch export**

```bash
git add vq2/live/dpvo_bridge_protocol.py vq2/live/bridge_dpvo.py vq2/mapping/dpvo_batch.py vq2/tests/test_mapping_dpvo_batch.py vq2/tests/test_dpvo_bridge_protocol.py
git commit -m "feat(vq2): export final DPVO trajectories"
```

### Task 4: Extract SIFT keyframes and geometrically verified tracks

**Files:**
- Create: `vq2/mapping/features.py`
- Test: `vq2/tests/test_mapping_features.py`

**Interfaces:**
- Produces: `FrameFeatures(frame_ns, xy, descriptors)` and `FeatureTrack`.
- Produces: `extract_sift(image, gate_mask) -> FrameFeatures` and `verified_matches(a, b, K) -> np.ndarray`.

- [ ] **Step 1: Write gate-mask and geometric-verification tests**

```python
import cv2
import numpy as np

from vq2.mapping.features import extract_sift, verified_matches


def textured_image():
    image = np.zeros((360, 640), np.uint8)
    for x in range(40, 600, 60):
        cv2.circle(image, (x, 180 + (x % 40)), 8, 255, -1)
    return image


def test_sift_excludes_gate_mask():
    image = textured_image()
    mask = np.zeros_like(image)
    mask[:, :320] = 255
    features = extract_sift(image, mask, frame_ns=1)
    assert len(features.xy) > 0
    assert np.all(features.xy[:, 0] >= 320)


def test_repeated_unverified_matches_do_not_form_loop():
    image = textured_image()
    a = extract_sift(image, np.zeros_like(image), frame_ns=1)
    b = extract_sift(np.roll(image, 250, axis=1), np.zeros_like(image), frame_ns=2)
    K = np.array([[226, 0, 319.5], [0, 226, 179.5], [0, 0, 1]], float)
    assert len(verified_matches(a, b, K)) == 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_mapping_features.py -v`

Expected: FAIL importing `vq2.mapping.features`.

- [ ] **Step 3: Implement extraction, ratio matching, and essential verification**

```python
# vq2/mapping/features.py
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class FrameFeatures:
    frame_ns: int
    xy: np.ndarray
    descriptors: np.ndarray


def extract_sift(image, gate_mask, frame_ns):
    detector = cv2.SIFT_create(nfeatures=1600)
    allowed = cv2.bitwise_not(np.asarray(gate_mask, np.uint8))
    keypoints, descriptors = detector.detectAndCompute(image, allowed)
    xy = np.array([point.pt for point in keypoints], dtype=np.float32).reshape(-1, 2)
    descriptors = np.empty((0, 128), np.float32) if descriptors is None else descriptors
    return FrameFeatures(int(frame_ns), xy, descriptors)


def verified_matches(a, b, K):
    if len(a.descriptors) < 8 or len(b.descriptors) < 8:
        return np.empty((0, 2), np.int32)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.descriptors, b.descriptors, k=2)
    good = [(m.queryIdx, m.trainIdx) for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return np.empty((0, 2), np.int32)
    indices = np.asarray(good, np.int32)
    _, mask = cv2.findEssentialMat(a.xy[indices[:, 0]], b.xy[indices[:, 1]], K,
                                   cv2.RANSAC, 0.999, 1.5)
    if mask is None:
        return np.empty((0, 2), np.int32)
    return indices[mask.ravel().astype(bool)]
```

- [ ] **Step 4: Run feature tests**

Run: `python -m pytest vq2/tests/test_mapping_features.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit features**

```bash
git add vq2/mapping/features.py vq2/tests/test_mapping_features.py
git commit -m "feat(vq2): extract verified map features"
```

### Task 5: Associate gates across runs without silent merges

**Files:**
- Create: `vq2/mapping/association.py`
- Test: `vq2/tests/test_mapping_association.py`

**Interfaces:**
- Produces: `associate_gate_tracklets(tracklets, ambiguity_margin) -> AssociationResult`.
- Produces explicit `accepted`, `tentative`, and `rejected` pairs plus costs.
- Consumes an optional JSON override whose digest is included in the map identity.

- [ ] **Step 1: Write identical-gate ambiguity tests**

```python
import numpy as np

from vq2.mapping.association import GateTracklet, associate_gate_tracklets


def track(run, track_id, center, normal=(1, 0, 0), order=1):
    return GateTracklet(run, track_id, np.array(center, float),
                        np.array(normal, float), order, 12)


def test_clear_cross_run_match_is_accepted():
    result = associate_gate_tracklets([
        track("A1", 1, (10, 0, 0)), track("A2", 4, (10.2, 0.1, 0), order=1)
    ], ambiguity_margin=0.2)
    assert len(result.accepted) == 1


def test_equal_cost_identical_gates_remain_tentative():
    result = associate_gate_tracklets([
        track("A1", 1, (10, 0, 0)),
        track("A2", 4, (10.1, 1, 0)),
        track("A2", 5, (10.1, -1, 0)),
    ], ambiguity_margin=0.3)
    assert not result.accepted
    assert result.tentative
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_mapping_association.py -v`

Expected: FAIL importing `vq2.mapping.association`.

- [ ] **Step 3: Implement cost matrix, one-to-one assignment, and margin rejection**

Use `scipy.optimize.linear_sum_assignment`. The normalized cost is exactly:

```python
cost = (position_distance_m / 1.0
        + normal_angle_rad / 0.35
        + abs(route_order_a - route_order_b) * 4.0
        + max(0, 8 - min(observation_count_a, observation_count_b)) * 0.1)
```

Accept only if the assigned cost is below `3.0` and its difference from the
next-best candidate exceeds `ambiguity_margin`. Otherwise emit a tentative
record. Never use detector `gate_id` in the cost.

- [ ] **Step 4: Run association tests**

Run: `python -m pytest vq2/tests/test_mapping_association.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit association**

```bash
git add vq2/mapping/association.py vq2/tests/test_mapping_association.py
git commit -m "feat(vq2): associate metric gate tracklets"
```

### Task 6: Jointly optimize camera, gate, and landmark geometry

**Files:**
- Create: `vq2/mapping/optimizer.py`
- Test: `vq2/tests/test_mapping_optimizer.py`

**Interfaces:**
- Produces: `OptimizationProblem`, `OptimizationResult`, and `optimize_map(problem) -> OptimizationResult`.
- Variables: keyframe SE3, gate SE3, landmark XYZ, one run Sim3 per A pass; segmented DPVO log-scale is disabled unless the constant-scale residual gate fails.

- [ ] **Step 1: Write synthetic metric-scale and outlier tests**

```python
import numpy as np

from vq2.mapping.optimizer import synthetic_problem, optimize_map


def test_known_gate_width_recovers_metric_scale():
    problem = synthetic_problem(seed=7, trajectory_scale=0.2, pixel_noise=0.2)
    result = optimize_map(problem)
    assert result.success
    assert abs(result.run_scales["A1"] - 5.0) < 0.1
    assert result.background_rms_px < 1.0


def test_one_gross_corner_outlier_is_pruned():
    problem = synthetic_problem(seed=8, trajectory_scale=0.2, pixel_noise=0.2,
                                gross_corner_outlier=True)
    result = optimize_map(problem)
    assert result.success
    assert result.rejected_corner_count == 1
    assert np.isfinite(result.parameter_covariance).all()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_mapping_optimizer.py -v`

Expected: FAIL importing `vq2.mapping.optimizer`.

- [ ] **Step 3: Implement canonical projection and whitened residuals**

```python
def whitened_corner_residual(predicted_uv, observed_uv, covariance_uv):
    delta = np.asarray(predicted_uv) - np.asarray(observed_uv)
    chol = np.linalg.cholesky(np.asarray(covariance_uv) + np.eye(2) * 1e-6)
    return np.linalg.solve(chol, delta)


def dpvo_direction_residual(p_i, p_j, measured_delta):
    predicted = np.asarray(p_j) - np.asarray(p_i)
    predicted /= max(np.linalg.norm(predicted), 1e-9)
    measured = np.asarray(measured_delta)
    measured /= max(np.linalg.norm(measured), 1e-9)
    return predicted - measured
```

Pack variables deterministically by sorted run, keyframe, gate, and landmark
ID. Build one residual vector containing GateNet corner reprojection, SIFT
landmark reprojection, DPVO rotation/translation direction, IMU roll/pitch,
spawn prior, verified loop closure, and optional aperture-plane tick terms.
Call:

```python
solution = scipy.optimize.least_squares(
    residual_vector,
    x0,
    jac_sparsity=jacobian_sparsity(problem),
    loss="cauchy",
    f_scale=2.0,
    max_nfev=200,
)
```

After the first solve, reject corner residuals above `4.0` whitened sigma and
background residuals above `4.0 px`, then solve once more. Report every removed
factor. Estimate covariance from the finite normal matrix pseudo-inverse.

- [ ] **Step 4: Gate segmented scale behind a measured residual test**

Fit constant scale first. Enable one smooth log-scale knot per 20 retained
keyframes only when the constant-scale DPVO displacement residual has a
monotonic Spearman correlation magnitude above `0.6` with route time and the
segmented model reduces held-in A residual by at least `20%`. Record the model
choice and knot values in the artifact; do not inspect B1.

- [ ] **Step 5: Run optimizer tests**

Run: `python -m pytest vq2/tests/test_mapping_optimizer.py -v`

Expected: all tests PASS and produce the same parameters across two runs.

- [ ] **Step 6: Commit optimizer**

```bash
git add vq2/mapping/optimizer.py vq2/tests/test_mapping_optimizer.py
git commit -m "feat(vq2): optimize gate-constrained metric map"
```

### Task 7: Write immutable map artifacts and map-A validation policy

**Files:**
- Create: `vq2/mapping/artifact.py`
- Create: `vq2/mapping/validate.py`
- Test: `vq2/tests/test_mapping_artifact.py`
- Test: `vq2/tests/test_mapping_validation.py`

**Interfaces:**
- Produces: `write_map(result, inputs, directory) -> Path`, `load_map(directory) -> CourseMap`, and `validate_map_a(directory, policy) -> ValidationReport`.
- Produces immutable `manifest.json`, `poses.npz`, `gates.npz`, `landmarks.npz`, `descriptors.npz`, `audit.jsonl`, and `approval-policy.json`.

- [ ] **Step 1: Write digest and mutation tests**

```python
import json
import pytest

from vq2.mapping.artifact import artifact_digest, verify_artifact


def test_artifact_digest_changes_after_numeric_mutation(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema":"vq2-map-v1"}\n')
    before = artifact_digest(tmp_path)
    (tmp_path / "manifest.json").write_text('{"schema":"vq2-map-v2"}\n')
    assert artifact_digest(tmp_path) != before


def test_verify_rejects_b1_in_build_inputs(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "vq2-map-v1", "input_roles": ["A1", "A2", "A3", "B1"]}))
    with pytest.raises(ValueError, match="B1 contaminated"):
        verify_artifact(tmp_path)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest vq2/tests/test_mapping_artifact.py vq2/tests/test_mapping_validation.py -v`

Expected: FAIL importing artifact/validation modules.

- [ ] **Step 3: Implement sorted content digest and atomic directory finalize**

```python
def artifact_digest(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "digest.sha256":
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_artifact(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if "B1" in manifest["input_roles"]:
        raise ValueError("B1 contaminated map build inputs")
    expected = (Path(directory) / "digest.sha256").read_text().strip()
    if artifact_digest(directory) != expected:
        raise ValueError("map artifact digest mismatch")
    return manifest
```

- [ ] **Step 4: Freeze map-A approval policy before B1 opens**

Write `approval-policy.json` with numeric limits derived from A-only synthetic
and leave-one-run-out results: maximum normalized gate residual, background
RMS, per-gate minimum views and parallax, leave-one-anchor-out scale spread,
maximum map size, and localizer targets consumed by the next plan. Include the
policy digest in the map manifest and B1 seal-open event.

- [ ] **Step 5: Run artifact and validation tests**

Run: `python -m pytest vq2/tests/test_mapping_artifact.py vq2/tests/test_mapping_validation.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit artifacts**

```bash
git add vq2/mapping/artifact.py vq2/mapping/validate.py vq2/tests/test_mapping_artifact.py vq2/tests/test_mapping_validation.py
git commit -m "feat(vq2): seal immutable metric maps"
```

### Task 8: Build map A from the survey set without opening B1

**Files:**
- Create outside git: `C:\Users\alexj\vq2_maps\map_A_<digest>\`
- Keep source corpora in their unique A1/A2/A3 directories.

**Interfaces:**
- Produces one unapproved `vq2-map-v1` artifact and A-only quality report.
- Does not invoke the B1 validator.

- [ ] **Step 1: Validate input manifests and hashes**

Run: `python -m vq2.mapping.validate inputs C:\Users\alexj\vq2_survey_A1_001\manifest.json C:\Users\alexj\vq2_survey_A2_001\manifest.json C:\Users\alexj\vq2_survey_A3_001\manifest.json`

Expected: roles exactly `A1,A2,A3`, complete explicit frame blocks, matching JPEG hashes, and no B1 digest.

- [ ] **Step 2: Run sequential GateNet extraction**

Run: `python -m vq2.mapping.extract --runs C:\Users\alexj\vq2_survey_A1_001 C:\Users\alexj\vq2_survey_A2_001 C:\Users\alexj\vq2_survey_A3_001 --output C:\Users\alexj\vq2_map_work\gatenet`

Expected: one lossless observation artifact per run, canonical calibration digest, no copied JPEG tree, and GPU released on exit.

- [ ] **Step 3: Run final-batch DPVO after GateNet exits**

Run: `python -m vq2.mapping.dpvo_batch --runs C:\Users\alexj\vq2_survey_A1_001 C:\Users\alexj\vq2_survey_A2_001 C:\Users\alexj\vq2_survey_A3_001 --output C:\Users\alexj\vq2_map_work\dpvo`

Expected: `final=true`, positions and quaternions for all reconstructed source timestamps, matching DPVO identity, and no simulator process.

- [ ] **Step 4: Build features, associate, optimize, and seal map A**

Run: `python -m vq2.mapping.validate build --work C:\Users\alexj\vq2_map_work --output C:\Users\alexj\vq2_maps\map_A_candidate`

Expected: deterministic map digest, stable gate IDs plus separate route order,
finite covariances, zero silently merged tentative gates, and an A-only policy.

- [ ] **Step 5: Run the entire non-GPU mapping test suite**

Run: `python -m pytest vq2/tests/test_mapping_*.py vq2/tests/test_dpvo_bridge_*.py -v`

Expected: all tests PASS.

---

## Completion gate

This plan is complete when A1/A2/A3 produce one reproducible unapproved metric
map with a frozen approval policy, final DPVO trajectory provenance, explicit
association audit, and no B1 data or tuning influence. B1 remains sealed until
the prior-map localization plan is ready to evaluate it once.

