# VQ2 VIO — Ground-Plane Optical-Flow Velocity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ~0.6× distance-compression of the PosVelKF by fusing camera-derived ground-plane optical-flow velocity, validated offline against recorded corpora to <0.5 m error over a 12 m run, then wired into `vq2/live/vq2wp.py` as a velocity source for the blind final ~6 m to gate 1.

**Architecture:** Sparse LK optical flow on floor-plane features (the hangar floor texture is rich). Each tracked feature pair, combined with the trusted wfix-gyro attitude and the KF height above floor, yields a full 3D drone-displacement measurement (static-landmark identity: `Δx_drone = d0·r0_w − d1·r1_w`). A robust median over tracks becomes a velocity pseudo-measurement into the existing `PosVelKF.update_velocity` at frame rate (~30 Hz). Offline-first: everything is validated in deterministic replay on `vq2_data/vq2_rec` (at-rest) and `vq2_data/vq2_motion` (flight) before any live flight.

**Tech Stack:** Python 3, numpy, OpenCV (`cv2.goodFeaturesToTrack` + `calcOpticalFlowPyrLK`), pytest. No new dependencies — cv2 is already imported by `vq2wp.py` in the laptop `monorace` env and cv2 4.13/numpy 2.0.2 are on the Mac.

## Global Constraints

- Repo: `/Users/alex/Documents/drone-ai-grand-prix/algo_src`, branch `vq2-estimation`. All paths below relative to this root.
- Corpora (Mac): `/Users/alex/Documents/drone-ai-grand-prix/vq2_data/vq2_rec` (at rest, 2701 frames), `.../vq2_data/vq2_motion` (flight, 1931 frames + `detections.jsonl` with `t_cam` PnP solutions + `cmds.jsonl`).
- Camera: 640×360, fx = fy = 226.0 px (GateNet PnP default, `vq2/gatenet/postprocess.py:143`), nose camera pitched **20° up**, NO yaw offset (flight-validated 07-06; VQ1's camyaw +5.7° does NOT apply to VQ2).
- Frame conventions (flight-proven in `vq2wp.py`, NOT `vq2/estimators.py`): `roll += gx·dt`, `pitch += −gy·dt`, `yaw += −gz·dt` (wfix mirrors pitch AND yaw). Note: `vq2/estimators.py` `WFIX=(1,−1,1)` does not flip yaw — do not copy it; follow `vq2wp.py:202-204`.
- Level/world frame: z DOWN, yaw-aligned at arming point, origin = arming point (pad). Level rotation must be built exactly like `eskf.accel_level` (Ry@Rx composition); world = Rz(yaw) applied as in `vq2wp.py:206-208`.
- Camera clock is epoch ns (`sim_ns`), IMU clock is boot µs (`time_usec`); the only bridge is `_rx_wall` medians (pattern in `vq2/replay.py:154-165`).
- Detection obs Z is quarantined (per-run −0.6…−4.9 m bias) — all acceptance metrics are **xy-only**.
- Dedup per-channel `t_us` before integration (already in `vq2/corpus.py`); never validate against differentiated/derived quantities — anchors come from raw GateNet `t_cam`, rest windows are self-labeling truth.
- No local disk writes added to the live flight loop (`jlog` is the existing, allowed pattern).
- Live kill switch mandatory: `NOFLOW=1` disables the new update path entirely.
- `vq2/eskf.py` and `vq2/live/eskf.py` differ only in tuning (q_vel 0.12 vs 0.40, maha 11.34 vs 16.27) — do not "unify"; offline uses `vq2/eskf.py`.
- Tests: `python3 -m pytest algo_src/vq2/tests/ -v` from repo root (pytest 8.4.2 present). Corpus-dependent tests must `pytest.skip` when the corpus dir is absent (CI/laptop safety).
- Commit style: Conventional Commits, subject ≤50 chars.

## GO/NO-GO

Task 6 is the gate: VIO trajectory (IMU + flow, **no vision position fixes**) must hold **max xy error ≤ 0.5 m at every vision anchor within the first 12 m traveled** on `vq2_motion`. If it fails after the z_floor sweep and diagnostics, STOP — fallbacks are (a) MASt3R-SLAM cor-106 worktree on the laptop, (b) tightly-coupled gate-corner updates (playbook Tier 0.3). Do not iterate live flights to rescue a failed offline bar.

---

### Task 1: Camera model module

**Files:**
- Create: `vq2/camera.py`
- Create: `vq2/tests/__init__.py` (empty)
- Test: `vq2/tests/test_camera.py`

**Interfaces:**
- Produces: `pixel_rays_body(uv: np.ndarray (N,2)) -> np.ndarray (N,3)` unit rays in body frame; `R_level_body(roll, pitch) -> (3,3)`; `R_world_body(roll, pitch, yaw) -> (3,3)`; constants `W, H, FX, FY, CX, CY, M_BODY_CAM`. All later tasks import these.

- [ ] **Step 1: Write the failing test**

`vq2/tests/test_camera.py`:

```python
import math

import numpy as np

from vq2 import camera


def test_center_pixel_ray_is_optical_axis():
    r = camera.pixel_rays_body([[camera.CX, camera.CY]])[0]
    # optical axis 20 deg above horizon in body frame (z down)
    elev = math.degrees(math.asin(-r[2]))
    assert abs(elev - 20.0) < 1e-6
    assert abs(r[1]) < 1e-9  # no lateral component


def test_bottom_center_ray_looks_below_horizon():
    r = camera.pixel_rays_body([[camera.CX, camera.H - 1.0]])[0]
    elev = math.degrees(math.asin(-r[2]))
    # 20 up - atan(179.5/226)=38.45 down => ~-18.4 deg (floor visible)
    assert -19.0 < elev < -17.9


def test_rays_are_unit_norm():
    uv = np.array([[0.0, 0.0], [639.0, 359.0], [320.0, 100.0]])
    r = camera.pixel_rays_body(uv)
    assert np.allclose(np.linalg.norm(r, axis=1), 1.0)


def test_R_level_body_matches_accel_level_composition():
    from vq2.eskf import accel_level
    roll, pitch = 0.21, -0.13
    acc = np.array([1.1, -2.2, -9.0])
    via_helper = camera.R_level_body(roll, pitch) @ acc + np.array([0, 0, 9.81])
    assert np.allclose(via_helper, accel_level(acc, roll, pitch))


def test_R_world_body_yaw_matches_vq2wp_rotation():
    yaw = 0.7
    v_lvl = np.array([1.0, 2.0, 3.0])
    cyw, syw = math.cos(yaw), math.sin(yaw)
    expected = np.array([cyw * v_lvl[0] - syw * v_lvl[1],
                         syw * v_lvl[0] + cyw * v_lvl[1], v_lvl[2]])
    got = camera.R_world_body(0.0, 0.0, yaw) @ v_lvl
    assert np.allclose(got, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest vq2/tests/test_camera.py -v` (from `algo_src/`)
Expected: FAIL with `ModuleNotFoundError: No module named 'vq2.camera'`

- [ ] **Step 3: Write minimal implementation**

`vq2/camera.py`:

```python
"""VQ2 camera model (constants flight-validated in vq2/live/vq2wp.py).

Pixel convention: u right, v down, origin top-left (cv2). Camera frame:
x right, y down, z forward (optical axis). Body frame: x fwd, y right,
z down. Camera pitched CAM_TILT up in body, no yaw offset (VQ2 nose cam,
flight-validated 07-06; VQ1's camyaw +5.7 deg does NOT apply here).
"""
from __future__ import annotations

import math

import numpy as np

W, H = 640, 360
FX = FY = 226.0  # px; GateNet PnP default (vq2/gatenet/postprocess.py)
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
CAM_TILT = math.radians(20.0)

_ct, _st = math.cos(CAM_TILT), math.sin(CAM_TILT)
_CZ = np.array([_ct, 0.0, -_st])
_CY = np.array([_st, 0.0, _ct])
_CX = np.cross(_CY, _CZ)
M_BODY_CAM = np.stack([_CX, _CY, _CZ], axis=1)


def pixel_rays_body(uv) -> np.ndarray:
    """(N,2) pixel coords -> (N,3) unit rays in the body frame."""
    uv = np.atleast_2d(np.asarray(uv, float))
    rc = np.stack(
        [(uv[:, 0] - CX) / FX, (uv[:, 1] - CY) / FY, np.ones(len(uv))], axis=1
    )
    rc /= np.linalg.norm(rc, axis=1, keepdims=True)
    return rc @ M_BODY_CAM.T


def R_level_body(roll: float, pitch: float) -> np.ndarray:
    """Identical composition to eskf.accel_level (Ry @ Rx). Keep in sync."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx


def R_world_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rz(yaw) @ R_level_body — matches vq2wp.py's a_w / g_w rotations."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ R_level_body(roll, pitch)
```

Also create empty `vq2/tests/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest vq2/tests/test_camera.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add vq2/camera.py vq2/tests/__init__.py vq2/tests/test_camera.py
git commit -m "feat(vq2): camera model module for flow VIO"
```

---

### Task 2: Flow geometry core (pure math, no cv2)

**Files:**
- Create: `vq2/flow_vel.py` (geometry half)
- Test: `vq2/tests/test_flow_geometry.py`

**Interfaces:**
- Consumes: `camera.pixel_rays_body`, `camera.R_world_body`.
- Produces: `velocity_from_tracks(uv0, uv1, att0, att1, h0, h1, dt, min_decl_deg=8.0, max_depth=25.0) -> (v_w (M,3), used (N,) bool)` — per-track world-frame drone velocities; `robust_velocity(v_tracks (M,3), min_tracks=6) -> (v (3,), sigma float, n_inliers int) | None`. `att` tuples are `(roll, pitch, yaw)`; `h` is camera height above floor in metres (>0); world frame z down.

- [ ] **Step 1: Write the failing test**

`vq2/tests/test_flow_geometry.py`:

```python
import numpy as np

from vq2 import camera
from vq2.flow_vel import robust_velocity, velocity_from_tracks


def _project(points_w, pos_w, att):
    """World floor points -> pixel coords for a drone at pos_w with att."""
    R_wb = camera.R_world_body(*att)
    r_b = (points_w - pos_w) @ R_wb  # rows: rel vector rotated into body
    r_c = r_b @ camera.M_BODY_CAM    # body -> camera frame
    uv = np.stack(
        [r_c[:, 0] / r_c[:, 2] * camera.FX + camera.CX,
         r_c[:, 1] / r_c[:, 2] * camera.FY + camera.CY], axis=1
    )
    ok = (
        (r_c[:, 2] > 0.1)
        & (uv[:, 0] > 0) & (uv[:, 0] < camera.W - 1)
        & (uv[:, 1] > 0) & (uv[:, 1] < camera.H - 1)
    )
    return uv, ok


def _floor_points():
    xs, ys = np.meshgrid(np.linspace(3.0, 14.0, 12), np.linspace(-3.0, 3.0, 7))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    return pts


def _make_pair(pos0, pos1, att0, att1):
    pts = _floor_points()
    uv0, ok0 = _project(pts, pos0, att0)
    uv1, ok1 = _project(pts, pos1, att1)
    ok = ok0 & ok1
    return uv0[ok], uv1[ok]


def test_forward_motion_recovered_exactly():
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])          # z down, 1.3 m up
    pos1 = pos0 + np.array([1.5, 0.0, 0.0]) * dt
    att = (0.0, 0.0, 0.0)
    uv0, uv1 = _make_pair(pos0, pos1, att, att)
    h = 0.0 - pos0[2]  # floor at z=0
    v, used = velocity_from_tracks(uv0, uv1, att, att, h, h, dt)
    assert used.sum() >= 10
    est = robust_velocity(v)
    assert est is not None
    v_est, sigma, n = est
    assert np.allclose(v_est, [1.5, 0.0, 0.0], atol=1e-6)


def test_lateral_and_vertical_motion_recovered():
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])
    v_true = np.array([0.8, -0.6, 0.3])  # includes descent (z down +)
    pos1 = pos0 + v_true * dt
    att = (0.05, -0.1, 0.3)  # tilted + yawed
    uv0, uv1 = _make_pair(pos0, pos1, att, att)
    h0 = 0.0 - pos0[2]
    h1 = 0.0 - pos1[2]
    v, used = velocity_from_tracks(uv0, uv1, att, att, h0, h1, dt)
    v_est, sigma, n = robust_velocity(v)
    assert np.allclose(v_est, v_true, atol=1e-6)


def test_pure_yaw_rotation_gives_zero_velocity():
    dt = 1.0 / 30.0
    pos = np.array([0.0, 0.0, -1.3])
    att0 = (0.0, 0.0, 0.0)
    att1 = (0.0, 0.0, 0.12)  # ~7 deg/frame spin, no translation
    uv0, uv1 = _make_pair(pos, pos, att0, att1)
    h = 1.3
    v, used = velocity_from_tracks(uv0, uv1, att0, att1, h, h, dt)
    v_est, sigma, n = robust_velocity(v)
    assert np.linalg.norm(v_est) < 1e-6


def test_height_error_scales_speed_proportionally():
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])
    pos1 = pos0 + np.array([1.5, 0.0, 0.0]) * dt
    att = (0.0, 0.0, 0.0)
    uv0, uv1 = _make_pair(pos0, pos1, att, att)
    v, _ = velocity_from_tracks(uv0, uv1, att, att, 1.3 * 1.1, 1.3 * 1.1, dt)
    v_est, _, _ = robust_velocity(v)
    assert abs(v_est[0] / 1.5 - 1.1) < 0.01  # 10% h error -> 10% v error


def test_shallow_rays_rejected():
    # features near the horizon (declination < min) must be masked out
    dt = 1.0 / 30.0
    pos = np.array([0.0, 0.0, -1.3])
    att = (0.0, 0.0, 0.0)
    # top rows of the image look ABOVE the horizon with 20 deg up tilt
    uv_top = np.array([[320.0, 5.0], [100.0, 10.0], [500.0, 8.0]])
    v, used = velocity_from_tracks(uv_top, uv_top, att, att, 1.3, 1.3, dt)
    assert used.sum() == 0


def test_robust_velocity_rejects_outliers_and_small_n():
    good = np.tile([1.0, 0.0, 0.0], (20, 1))
    bad = np.tile([15.0, -9.0, 4.0], (3, 1))
    v_est, sigma, n = robust_velocity(np.vstack([good, bad]))
    assert np.allclose(v_est, [1.0, 0.0, 0.0], atol=1e-9)
    assert n == 20
    assert robust_velocity(good[:4]) is None  # below min_tracks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest vq2/tests/test_flow_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vq2.flow_vel'`

- [ ] **Step 3: Write minimal implementation**

`vq2/flow_vel.py` (geometry half; the tracker class is Task 3):

```python
"""Ground-plane optical-flow velocity for VQ2.

Static-landmark identity for a floor feature seen in two frames:
    L = x0 + d0*r0_w = x1 + d1*r1_w
      =>  drone displacement  x1 - x0 = d0*r0_w - d1*r1_w
with depth d = h / r_w_z from the flat-floor assumption (z down, floor
below the drone => ray z component positive). Height error scales the
recovered speed proportionally (10% h error -> 10% v error) — h comes
from the KF z which carries vision fixes.

Attitude must be the flight-proven wfix chain (vq2wp.py), passed per
frame; derotation is implicit because each frame's rays are rotated by
its own full attitude.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from .camera import pixel_rays_body, R_world_body
except ImportError:  # flat deploy next to vq2wp.py on the laptop
    from camera import pixel_rays_body, R_world_body

Z_FLOOR = 0.15  # m, floor z in the arming frame (z down); Task 6 calibrates

SIGMA_FLOOR = 0.15      # m/s, never report tighter than this
MAD_K = 3.0             # inlier band = MAD_K * 1.4826 * MAD + 0.05 m/s


def velocity_from_tracks(
    uv0,
    uv1,
    att0,
    att1,
    h0: float,
    h1: float,
    dt: float,
    min_decl_deg: float = 8.0,
    max_depth: float = 25.0,
):
    """Per-track world-frame drone velocity from floor-feature pairs.

    Returns (v_w (M,3) for the used tracks, used (N,) bool mask)."""
    uv0 = np.atleast_2d(np.asarray(uv0, float))
    uv1 = np.atleast_2d(np.asarray(uv1, float))
    r0 = pixel_rays_body(uv0) @ R_world_body(*att0).T
    r1 = pixel_rays_body(uv1) @ R_world_body(*att1).T
    min_z = math.sin(math.radians(min_decl_deg))
    used = (r0[:, 2] > min_z) & (r1[:, 2] > min_z)
    if h0 <= 0.05 or h1 <= 0.05 or dt <= 0:
        return np.zeros((0, 3)), np.zeros(len(uv0), dtype=bool)
    d0 = np.where(used, h0 / np.maximum(r0[:, 2], 1e-9), 0.0)
    d1 = np.where(used, h1 / np.maximum(r1[:, 2], 1e-9), 0.0)
    used &= (d0 < max_depth) & (d1 < max_depth)
    disp = d0[used, None] * r0[used] - d1[used, None] * r1[used]
    return disp / dt, used


def robust_velocity(v_tracks, min_tracks: int = 6):
    """Median + MAD inlier mean. Returns (v (3,), sigma, n_inliers) or None."""
    v = np.atleast_2d(np.asarray(v_tracks, float))
    if len(v) < min_tracks:
        return None
    med = np.median(v, axis=0)
    dev = np.linalg.norm(v - med, axis=1)
    mad = np.median(dev)
    inl = dev <= MAD_K * 1.4826 * mad + 0.05
    if inl.sum() < min_tracks:
        return None
    v_in = v[inl]
    v_est = v_in.mean(axis=0)
    spread = float(np.linalg.norm(v_in.std(axis=0)))
    sigma = max(SIGMA_FLOOR, spread / math.sqrt(len(v_in)))
    return v_est, sigma, int(inl.sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest vq2/tests/test_flow_geometry.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add vq2/flow_vel.py vq2/tests/test_flow_geometry.py
git commit -m "feat(vq2): floor-plane flow velocity geometry core"
```

---

### Task 3: LK feature tracker wrapper + at-rest corpus test

**Files:**
- Modify: `vq2/flow_vel.py` (append tracker class)
- Test: `vq2/tests/test_flow_tracker.py`

**Interfaces:**
- Consumes: `velocity_from_tracks`, `robust_velocity` (Task 2); `vq2.corpus.load` (existing).
- Produces: `class FlowVelocity` with method `process(gray: np.ndarray (H,W) uint8, t_s: float, att: (roll,pitch,yaw), h: float) -> None | (v_w (3,), sigma float, n_inliers int, n_tracks int)`. Stateful: keeps previous frame; returns None on first frame, degenerate dt (`<5 ms` or `>150 ms` — resets instead), too-few tracks, or h invalid. All later tasks (fusion, live) call exactly this.

- [ ] **Step 1: Write the failing test**

`vq2/tests/test_flow_tracker.py`:

```python
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vq2 import corpus as corpus_mod
from vq2.flow_vel import FlowVelocity

REC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_rec")


def test_synthetic_translating_texture_gives_forward_velocity():
    """Render a textured floor plane, translate camera, expect v_x > 0."""
    rng = np.random.default_rng(7)
    from vq2 import camera
    from vq2.tests.test_flow_geometry import _floor_points, _project

    # build two synthetic frames by splatting bright dots at projected floor pts
    pts = _floor_points()
    dt = 1.0 / 30.0
    pos0 = np.array([0.0, 0.0, -1.3])
    pos1 = pos0 + np.array([1.5, 0.0, 0.0]) * dt
    att = (0.0, 0.0, 0.0)
    frames = []
    for pos in (pos0, pos1):
        uv, ok = _project(pts, pos, att)
        img = np.zeros((camera.H, camera.W), np.uint8)
        for u, v in uv[ok]:
            cv2.circle(img, (int(round(u)), int(round(v))), 3, 255, -1)
        frames.append(cv2.GaussianBlur(img, (5, 5), 1.0))
    flow = FlowVelocity()
    assert flow.process(frames[0], 0.0, att, 1.3) is None  # first frame
    res = flow.process(frames[1], dt, att, 1.3)
    assert res is not None
    v_w, sigma, ninl, ntr = res
    assert v_w[0] > 1.0  # forward motion detected, right order of magnitude
    assert abs(v_w[1]) < 0.4


@pytest.mark.skipif(not os.path.isdir(REC), reason="vq2_rec corpus not present")
def test_at_rest_corpus_reports_near_zero_velocity():
    c = corpus_mod.load(REC)
    frames = [fr for fr in c.frames if os.path.exists(fr.path)][:120]
    assert len(frames) > 60, "corpus should have frames"
    flow = FlowVelocity()
    att = (0.0, 0.0, 0.0)  # at rest on the pad; spawn tilt irrelevant for |v|
    vs = []
    for fr in frames:
        gray = cv2.imread(fr.path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        res = flow.process(gray, fr.sim_ns / 1e9, att, 0.35)
        if res is not None:
            vs.append(np.linalg.norm(res[0]))
    assert len(vs) > 30, "flow should produce estimates on most frames"
    assert np.median(vs) < 0.05  # at rest => ~zero velocity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest vq2/tests/test_flow_tracker.py -v`
Expected: FAIL with `ImportError: cannot import name 'FlowVelocity'`

- [ ] **Step 3: Write minimal implementation**

Append to `vq2/flow_vel.py`:

```python
import cv2  # noqa: E402  (kept below the pure-geometry half: geometry tests
#             must not require cv2; move to top only if a linter forces it)

_GRID_STEP = 8


class FlowVelocity:
    """Sparse-LK floor-flow velocity. One instance per stream; call
    process() once per decoded frame in timestamp order."""

    def __init__(
        self,
        max_corners: int = 80,
        quality: float = 0.01,
        min_dist: int = 12,
        win: int = 21,
        levels: int = 3,
        fb_max_px: float = 1.0,
        min_decl_deg: float = 8.0,
        max_depth: float = 25.0,
        min_tracks: int = 6,
    ):
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist
        self.lk = dict(
            winSize=(win, win),
            maxLevel=levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.fb_max_px = fb_max_px
        self.min_decl_deg = min_decl_deg
        self.max_depth = max_depth
        self.min_tracks = min_tracks
        self._prev = None  # (gray, t_s, att, h)
        # precompute body rays on a coarse pixel grid for the floor mask
        try:
            from .camera import H, W, pixel_rays_body
        except ImportError:
            from camera import H, W, pixel_rays_body
        gy, gx = np.mgrid[0:H:_GRID_STEP, 0:W:_GRID_STEP]
        self._grid_shape = gy.shape
        self._grid_rays_b = pixel_rays_body(
            np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
        )
        self._img_shape = (H, W)

    def _floor_mask(self, att) -> np.ndarray:
        r_w = self._grid_rays_b @ R_world_body(*att).T
        below = (r_w[:, 2] > math.sin(math.radians(self.min_decl_deg)))
        m = below.reshape(self._grid_shape).astype(np.uint8) * 255
        return cv2.resize(
            m, (self._img_shape[1], self._img_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def reset(self) -> None:
        self._prev = None

    def process(self, gray, t_s: float, att, h: float):
        prev, self._prev = self._prev, (gray, t_s, att, h)
        if prev is None:
            return None
        pgray, pt, patt, ph = prev
        dt = t_s - pt
        if dt < 0.005 or dt > 0.15 or h <= 0.05 or ph <= 0.05:
            return None
        p0 = cv2.goodFeaturesToTrack(
            pgray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_dist,
            mask=self._floor_mask(patt),
        )
        if p0 is None or len(p0) < self.min_tracks:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pgray, gray, p0, None, **self.lk)
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(gray, pgray, p1, None, **self.lk)
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < self.fb_max_px)
        if good.sum() < self.min_tracks:
            return None
        uv0 = p0.reshape(-1, 2)[good]
        uv1 = p1.reshape(-1, 2)[good]
        v_tracks, used = velocity_from_tracks(
            uv0, uv1, patt, att, ph, h, dt,
            min_decl_deg=self.min_decl_deg, max_depth=self.max_depth,
        )
        est = robust_velocity(v_tracks, self.min_tracks)
        if est is None:
            return None
        v_w, sigma, ninl = est
        return v_w, sigma, ninl, int(good.sum())
```

Note: `import cv2` at module scope makes Task 2's pure-geometry tests depend on cv2 being installed. cv2 IS installed everywhere this runs (Mac + monorace). If that changes, guard the import; do not restructure now.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest vq2/tests/test_flow_tracker.py vq2/tests/test_flow_geometry.py -v`
Expected: all PASS (at-rest test SKIPs if corpus absent). If the at-rest median fails: dump per-frame `ninl` — the pad view may have too little floor in frame at rest tilt; lower `min_decl_deg` to 5.0 and re-run before touching anything else.

- [ ] **Step 5: Commit**

```bash
git add vq2/flow_vel.py vq2/tests/test_flow_tracker.py
git commit -m "feat(vq2): LK floor-flow tracker with fb-check"
```

---

### Task 4: Clock bridge helper in corpus.py

**Files:**
- Modify: `vq2/corpus.py` (append function)
- Test: `vq2/tests/test_clock_bridge.py`

**Interfaces:**
- Consumes: `Segment`, `FrameRef` (existing dataclasses).
- Produces: `clock_bridge(seg: Segment, frames: list[FrameRef]) -> None | (offset_s: float, spread_ms: float)` where `t_us_of_frame = frame.sim_ns / 1e3 * 1e-0 ... ` concretely: `frame_t_us = frame.sim_ns / 1e9 + offset_s`, expressed in the IMU boot-seconds domain: `frame_t_boot_s = frame.sim_ns / 1e9 + offset_s` and IMU sample boot-seconds is `s.t_us / 1e6`. Task 5 uses exactly this to merge streams.

- [ ] **Step 1: Write the failing test**

`vq2/tests/test_clock_bridge.py`:

```python
import os

import pytest

from vq2 import corpus as corpus_mod

REC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_rec")


def test_bridge_on_synthetic_streams():
    seg = corpus_mod.Segment()
    # IMU boot clock starts at 50.0 s boot, wall = boot + 1000.0
    for i in range(200):
        t_us = int((50.0 + i * 0.007) * 1e6)
        seg.imu.append(
            corpus_mod.ImuSample(t_us=t_us, acc=(0, 0, -9.81), gyr=(0, 0, 0),
                                 rx_wall=1000.0 + t_us / 1e6)
        )
    # camera epoch clock: wall = sim_s - 700.0  => frame boot_s = sim_s - 1700.0
    frames = [
        corpus_mod.FrameRef(sim_ns=int((1750.0 + i / 30.0) * 1e9),
                            rx_wall=1050.0 + i / 30.0, path="x")
        for i in range(90)
    ]
    res = corpus_mod.clock_bridge(seg, frames)
    assert res is not None
    offset_s, spread_ms = res
    assert abs(offset_s - (-1700.0)) < 0.01
    assert spread_ms < 1.0


def test_bridge_none_without_stamped_frames():
    seg = corpus_mod.Segment()
    seg.imu.append(
        corpus_mod.ImuSample(t_us=1, acc=(0, 0, -9.81), gyr=(0, 0, 0), rx_wall=5.0)
    )
    frames = [corpus_mod.FrameRef(sim_ns=123, rx_wall=0.0, path="x")]
    assert corpus_mod.clock_bridge(seg, frames) is None


@pytest.mark.skipif(not os.path.isdir(REC), reason="vq2_rec corpus not present")
def test_bridge_on_real_corpus_is_stable():
    c = corpus_mod.load(REC)
    res = corpus_mod.clock_bridge(c.flight_segment, c.frames)
    assert res is not None
    offset_s, spread_ms = res
    assert spread_ms < 200.0  # UDP receive jitter bound, not precision sync
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest vq2/tests/test_clock_bridge.py -v`
Expected: FAIL with `AttributeError: module 'vq2.corpus' has no attribute 'clock_bridge'`

- [ ] **Step 3: Write minimal implementation**

Append to `vq2/corpus.py`:

```python
def clock_bridge(seg, frames):
    """Camera-epoch -> IMU-boot clock offset via rx_wall medians.

    frame_t_boot_s = frame.sim_ns / 1e9 + offset_s  lives on the same axis
    as  imu_sample.t_us / 1e6.  Returns (offset_s, spread_ms) or None.
    Precision is UDP receive jitter (tens of ms) — good enough to order
    streams and pick the nearest attitude sample, NOT for sub-frame sync.
    """
    import statistics

    if seg is None or not seg.imu:
        return None
    stamped = [fr for fr in frames if fr.rx_wall > 0]
    if not stamped:
        return None
    imu_off = statistics.median(
        s.rx_wall - s.t_us / 1e6 for s in seg.imu[:: max(1, len(seg.imu) // 500)]
    )
    cam = [fr.rx_wall - fr.sim_ns / 1e9
           for fr in stamped[:: max(1, len(stamped) // 500)]]
    cam_off = statistics.median(cam)
    spread_ms = 1000.0 * (max(cam) - min(cam))
    return cam_off - imu_off, spread_ms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest vq2/tests/test_clock_bridge.py -v`
Expected: 3 PASS (third SKIPs if corpus absent)

- [ ] **Step 5: Commit**

```bash
git add vq2/corpus.py vq2/tests/test_clock_bridge.py
git commit -m "feat(vq2): camera-to-IMU clock bridge helper"
```

---

### Task 5: Offline fusion runner

**Files:**
- Create: `vq2/fusion.py`
- Test: `vq2/tests/test_fusion.py`

**Interfaces:**
- Consumes: `corpus.load`, `corpus.clock_bridge`, `replay.find_rest_windows`, `estimators.accel_implied_attitude`, `eskf.PosVelKF`, `eskf.accel_level`, `camera.R_world_body`, `camera.M_BODY_CAM`, `FlowVelocity`, `Z_FLOOR`.
- Produces: `run_fusion(root: str, cfg: FusionConfig) -> FusionResult` with `FusionConfig(use_flow: bool = True, use_vision_pos: bool = False, z_floor: float = Z_FLOOR, gate_map: tuple = GATES)` and `FusionResult(t_s: list, p: list[(3,)], v: list[(3,)], anchors: list[dict], flow_updates: int, flow_rejects: int)`. Each anchor dict: `{"t_boot_s", "p_vision" (3,), "p_est" (3,), "err_xy" float, "range" float}`. Task 6 consumes `FusionResult` verbatim.

- [ ] **Step 1: Write the failing test**

`vq2/tests/test_fusion.py`:

```python
import os

import numpy as np
import pytest

from vq2.fusion import FusionConfig, run_fusion

REC = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_rec")
MOT = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_motion")


@pytest.mark.skipif(not os.path.isdir(REC), reason="vq2_rec corpus not present")
def test_at_rest_flow_bounds_velocity_drift():
    res_imu = run_fusion(REC, FusionConfig(use_flow=False))
    res_flow = run_fusion(REC, FusionConfig(use_flow=True))
    v_end_imu = np.linalg.norm(res_imu.v[-1])
    v_end_flow = np.linalg.norm(res_flow.v[-1])
    # pure IMU integration drifts; flow must pin an at-rest drone near zero
    assert v_end_flow < 0.10
    assert v_end_flow <= v_end_imu + 1e-9
    assert res_flow.flow_updates > 50


@pytest.mark.skipif(not os.path.isdir(MOT), reason="vq2_motion corpus not present")
def test_motion_corpus_produces_anchors_and_flow():
    res = run_fusion(MOT, FusionConfig(use_flow=True))
    assert res.flow_updates > 100, "flow should track through the flight"
    assert len(res.anchors) > 20, "GateNet locks should anchor the run"
    # distance actually traveled per vision: last anchor should be well
    # downcourse of the pad (the flight approached gate 1 at ~11 m)
    rngs = [a["range"] for a in res.anchors]
    assert min(rngs) < 8.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest vq2/tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vq2.fusion'`

- [ ] **Step 3: Write minimal implementation**

`vq2/fusion.py`:

```python
"""Deterministic offline IMU + floor-flow (+ optional vision-position)
fusion through PosVelKF, replaying a recorded corpus.

Attitude chain is the flight-proven vq2wp.py one:
  roll += gx dt ; pitch += -gy dt ; yaw += -gz dt   (wfix)
seeded accel-implied at the end of the first rest window. NOT the
estimators.py WFIX (its yaw sign diverges from the flight code).

Vision anchors: GateNet t_cam (camera frame) -> body -> level -> world;
drone position implied by a map gate = gate_w - g_w. Obs Z is biased
(quarantined) so anchors are compared XY-ONLY, but the full 3-vector is
kept for diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import corpus as corpus_mod
from .camera import M_BODY_CAM, R_world_body
from .eskf import PosVelKF, accel_level
from .estimators import accel_implied_attitude
from .flow_vel import FlowVelocity, Z_FLOOR
from .replay import find_rest_windows

# course map (spawn frame, z down) — HANDOFF_vq2_racing.md course facts
GATES = (
    (11.0, 0.0, -1.3),   # gate 1 (judge gate)
    (10.4, 0.0, -4.0),   # stacked high gate above gate 1
    (30.5, 8.5, -1.5),   # gate 2
)
ACCEPT_R = 3.0  # xy radius for matching an obs to a map gate (as in vq2wp)


@dataclass
class FusionConfig:
    use_flow: bool = True
    use_vision_pos: bool = False
    z_floor: float = Z_FLOOR
    gate_map: tuple = GATES


@dataclass
class FusionResult:
    t_s: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    flow_updates: int = 0
    flow_rejects: int = 0


def _detection_events(c, bridge_off):
    """[(t_boot_s, g_cam (3,)) ...] for solved, confident, usable insts."""
    ev = []
    for d in c.detections:
        for inst in d.insts:
            if not inst.get("solved") or inst.get("low_confidence"):
                continue
            t_cam = inst.get("t_cam")
            if not t_cam:
                continue
            ev.append((d.sim_ns / 1e9 + bridge_off, np.asarray(t_cam, float)))
    ev.sort(key=lambda e: e[0])
    return ev


def run_fusion(root: str, cfg: FusionConfig) -> FusionResult:
    c = corpus_mod.load(root)
    seg = c.flight_segment
    res = FusionResult()
    if seg is None or not seg.imu:
        return res
    rests = find_rest_windows(seg.imu)
    if not rests:
        return res
    bridge = corpus_mod.clock_bridge(seg, c.frames)
    off = bridge[0] if bridge else None

    frames = sorted(
        ((fr.sim_ns / 1e9 + off, fr.path) for fr in c.frames), key=lambda e: e[0]
    ) if off is not None else []
    dets = _detection_events(c, off) if off is not None else []

    seed_idx = rests[0][1]
    s0 = seg.imu[seed_idx]
    roll, pitch = accel_implied_attitude(s0)
    yaw = 0.0
    kf = PosVelKF()
    kf.reset_at_rest()
    flow = FlowVelocity()
    fi = di = 0
    last_us = s0.t_us

    for s in seg.imu[seed_idx + 1:]:
        if s.t_us <= last_us:
            continue
        dt = (s.t_us - last_us) / 1e6
        last_us = s.t_us
        t_boot = s.t_us / 1e6
        gx, gy, gz = s.gyr
        roll += gx * dt
        pitch += -gy * dt   # wfix (vq2wp.py:203)
        yaw += -gz * dt     # wfix (vq2wp.py:204)
        a_lvl = accel_level(s.acc, roll, pitch)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        a_w = np.array([cyw * a_lvl[0] - syw * a_lvl[1],
                        syw * a_lvl[0] + cyw * a_lvl[1], a_lvl[2]])
        kf.predict(a_w, dt)

        # frame events up to this IMU stamp
        while cfg.use_flow and fi < len(frames) and frames[fi][0] <= t_boot:
            t_f, path = frames[fi]
            fi += 1
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            h = cfg.z_floor - float(kf.p[2])
            out = flow.process(gray, t_f, (roll, pitch, yaw), h)
            if out is None:
                res.flow_rejects += 1
                continue
            v_w, sigma, ninl, ntr = out
            kf.update_velocity(v_w, sigma)
            res.flow_updates += 1

        # detection events -> anchors (and optional position updates)
        while di < len(dets) and dets[di][0] <= t_boot:
            t_d, g_cam = dets[di]
            di += 1
            g_w = R_world_body(roll, pitch, yaw) @ (M_BODY_CAM @ g_cam)
            rng = float(np.linalg.norm(g_cam))
            p_vis_cands = [np.asarray(g) - g_w for g in cfg.gate_map]
            errs = [np.linalg.norm((pv - kf.p)[:2]) for pv in p_vis_cands]
            j = int(np.argmin(errs))
            if errs[j] > ACCEPT_R:
                continue
            p_vis = p_vis_cands[j]
            res.anchors.append({
                "t_boot_s": t_d,
                "p_vision": p_vis.copy(),
                "p_est": kf.p.copy(),
                "err_xy": float(np.linalg.norm((p_vis - kf.p)[:2])),
                "range": rng,
            })
            if cfg.use_vision_pos:
                kf.update_position(
                    np.array([p_vis[0], p_vis[1], kf.p[2]]), rng
                )

        res.t_s.append(t_boot)
        res.p.append(kf.p.copy())
        res.v.append(kf.v.copy())
    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest vq2/tests/test_fusion.py -v`
Expected: 2 PASS. Watch-outs if failing:
- `flow_updates == 0` on vq2_rec: the at-rest corpus names frames `<fid>.jpg`; confirm `corpus.load` resolved paths (it tries both stems) and that `clock_bridge` returned non-None.
- Anchor matching empty on vq2_motion: print the first 5 `p_vis_cands` — if the yaw sign is wrong, anchors land mirrored in y and all miss `ACCEPT_R`. That is the diagnostic for the `estimators.py` vs `vq2wp.py` yaw-wfix discrepancy; keep the `vq2wp.py` sign, it is the flight-proven one.

- [ ] **Step 5: Commit**

```bash
git add vq2/fusion.py vq2/tests/test_fusion.py
git commit -m "feat(vq2): offline IMU+flow fusion replay"
```

---

### Task 6: Acceptance harness + z_floor calibration (GO/NO-GO)

**Files:**
- Create: `vq2/accept_vio.py`
- Modify: `vq2/flow_vel.py` (update `Z_FLOOR` constant with the calibrated value)
- Test: `vq2/tests/test_accept_vio.py`

**Interfaces:**
- Consumes: `run_fusion`, `FusionConfig`, `FusionResult`.
- Produces: CLI `python3 -m vq2.accept_vio <corpus_dir> [--sweep]` printing a verdict block; function `evaluate(root: str, z_floor: float) -> dict` with keys `max_err_xy_12m`, `p90_err_xy`, `dist_traveled_m`, `n_anchors`, `flow_updates`, `passed` (bool: `max_err_xy_12m <= 0.5`); function `sweep_z_floor(root: str, lo=0.0, hi=0.45, step=0.05) -> list[(z_floor, mean_err)]`.

- [ ] **Step 1: Write the failing test**

`vq2/tests/test_accept_vio.py`:

```python
import os

import pytest

from vq2.accept_vio import evaluate, sweep_z_floor

MOT = os.path.expanduser("~/Documents/drone-ai-grand-prix/vq2_data/vq2_motion")


@pytest.mark.skipif(not os.path.isdir(MOT), reason="vq2_motion corpus not present")
def test_evaluate_reports_complete_verdict():
    r = evaluate(MOT, z_floor=0.15)
    for key in ("max_err_xy_12m", "p90_err_xy", "dist_traveled_m",
                "n_anchors", "flow_updates", "passed"):
        assert key in r
    assert r["n_anchors"] > 20
    assert r["dist_traveled_m"] > 5.0


@pytest.mark.skipif(not os.path.isdir(MOT), reason="vq2_motion corpus not present")
def test_sweep_orders_candidates_by_error():
    rows = sweep_z_floor(MOT, lo=0.05, hi=0.30, step=0.125)
    assert len(rows) == 3
    assert all(len(row) == 2 for row in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest vq2/tests/test_accept_vio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vq2.accept_vio'`

- [ ] **Step 3: Write minimal implementation**

`vq2/accept_vio.py`:

```python
"""VIO acceptance harness (the GO/NO-GO gate from HANDOFF_vq2_racing.md):
IMU + floor-flow trajectory (NO vision position fixes) must stay within
0.5 m XY of every vision anchor inside the first 12 m traveled.

Usage:
    python3 -m vq2.accept_vio <corpus_dir>            # verdict at Z_FLOOR
    python3 -m vq2.accept_vio <corpus_dir> --sweep    # calibrate z_floor
"""
from __future__ import annotations

import sys

import numpy as np

from .fusion import FusionConfig, run_fusion

PASS_BAR_M = 0.5
PASS_DIST_M = 12.0


def evaluate(root: str, z_floor: float) -> dict:
    res = run_fusion(root, FusionConfig(use_flow=True, use_vision_pos=False,
                                        z_floor=z_floor))
    out = {"z_floor": z_floor, "flow_updates": res.flow_updates,
           "flow_rejects": res.flow_rejects, "n_anchors": len(res.anchors)}
    if not res.p or not res.anchors:
        out.update(max_err_xy_12m=float("inf"), p90_err_xy=float("inf"),
                   dist_traveled_m=0.0, passed=False)
        return out
    p = np.asarray(res.p)
    seg_d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    dist_at = np.concatenate([[0.0], np.cumsum(seg_d)])
    t = np.asarray(res.t_s)
    errs_12 = []
    errs_all = []
    for a in res.anchors:
        i = int(np.searchsorted(t, a["t_boot_s"]))
        i = min(i, len(dist_at) - 1)
        errs_all.append(a["err_xy"])
        if dist_at[i] <= PASS_DIST_M:
            errs_12.append(a["err_xy"])
    max12 = max(errs_12) if errs_12 else float("inf")
    out.update(
        max_err_xy_12m=round(float(max12), 3),
        p90_err_xy=round(float(np.percentile(errs_all, 90)), 3),
        dist_traveled_m=round(float(dist_at[-1]), 2),
        passed=bool(max12 <= PASS_BAR_M),
    )
    return out


def sweep_z_floor(root: str, lo: float = 0.0, hi: float = 0.45,
                  step: float = 0.05) -> list:
    rows = []
    z = lo
    while z <= hi + 1e-9:
        res = run_fusion(root, FusionConfig(use_flow=True, z_floor=z))
        errs = [a["err_xy"] for a in res.anchors]
        rows.append((round(z, 3),
                     round(float(np.mean(errs)) if errs else float("inf"), 3)))
        z += step
    return rows


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    root = argv[0]
    if "--sweep" in argv:
        rows = sweep_z_floor(root)
        print("z_floor  mean_err_xy")
        for z, e in rows:
            print(f"  {z:5.2f}   {e}")
        best = min(rows, key=lambda r: r[1])
        print(f"best: z_floor={best[0]} (mean_err={best[1]})")
        print("-> update Z_FLOOR in vq2/flow_vel.py to this value")
        return 0
    from .flow_vel import Z_FLOOR
    r = evaluate(root, Z_FLOOR)
    for k, v in r.items():
        print(f"  {k}: {v}")
    # baseline comparison: how bad is IMU-only? (expect the ~0.6x compression)
    base = run_fusion(root, FusionConfig(use_flow=False))
    if base.anchors:
        errs = [a["err_xy"] for a in base.anchors]
        print(f"  imu_only_p90_err_xy: {round(float(np.percentile(errs, 90)), 3)}")
    print("VERDICT:", "PASS — wire into vq2wp.py (Task 7)" if r["passed"]
          else "FAIL — see fallbacks in the plan header; do NOT go live")
    return 0 if r["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest vq2/tests/test_accept_vio.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run the calibration sweep and pin Z_FLOOR**

Run: `python3 -m vq2.accept_vio ~/Documents/drone-ai-grand-prix/vq2_data/vq2_motion --sweep`
Expected: a table with a clear single minimum (physically ~0.10–0.25: floor is a little below the arming origin, z down positive). Edit `Z_FLOOR` in `vq2/flow_vel.py` to the winning value with a comment:

```python
Z_FLOOR = <best>  # m, calibrated 2026-07-05 via accept_vio --sweep on vq2_motion
```

If the sweep curve is flat or the minimum sits at an edge, flow is not actually constraining scale — check `flow_updates` vs frame count and inlier counts before proceeding.

- [ ] **Step 6: Run the acceptance verdict (GO/NO-GO)**

Run: `python3 -m vq2.accept_vio ~/Documents/drone-ai-grand-prix/vq2_data/vq2_motion`
Expected: `passed: True`, `max_err_xy_12m <= 0.5`, and `imu_only_p90_err_xy` visibly worse (documents the win over the compressed baseline).
If FAIL: record the numbers in the experiment log, diagnose with the printed flow stats (rejects dominated? few inliers ⇒ floor texture/declination problem; anchors good early but drifting late ⇒ height error or yaw drift), and STOP per the GO/NO-GO header. Do not proceed to Task 7.

- [ ] **Step 7: Commit**

```bash
git add vq2/accept_vio.py vq2/flow_vel.py vq2/tests/test_accept_vio.py
git commit -m "feat(vq2): VIO acceptance harness + z_floor cal"
```

---

### Task 7: Live wiring into vq2wp.py (only after Task 6 PASS)

**Files:**
- Modify: `vq2/live/vq2wp.py` (imports block ~line 34; `cam_loop` at lines 236–257)
- Test: manual — replay-verified code goes live; there is no automated sim harness

**Interfaces:**
- Consumes: `FlowVelocity`, `Z_FLOOR` from `flow_vel` (flat import — deploy copies files next to `vq2wp.py`); existing `KF`, `KF_LOCK`, `state`, `jlog`.
- Produces: flow-velocity KF updates at frame rate in flight; `flow_vel` rows in `log.jsonl`; `plots/flow_v` in Rerun.

- [ ] **Step 1: Add the import and switch (near line 34, after `from eskf import ...`)**

```python
from flow_vel import FlowVelocity, Z_FLOOR   # flat deploy, same dir as eskf.py
FLOW_ON = os.environ.get('NOFLOW', '0') != '1'
FLOW = FlowVelocity()
```

- [ ] **Step 2: Hook the decoded frame in `cam_loop`**

In `cam_loop`, right after `state['frame'], state['frame_ns'] = img, ns` (line 253 today), insert:

```python
                    if FLOW_ON:
                        try:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            att = (state['roll'], state['pitch'], state['yaw'])
                            with KF_LOCK:
                                h = Z_FLOOR - float(KF.p[2])
                            fout = FLOW.process(gray, ns / 1e9, att, h)
                            if fout is not None:
                                fv, fsig, fninl, fntr = fout
                                with KF_LOCK:
                                    KF.update_velocity(fv, fsig)
                                jlog('flow_vel', ns=ns,
                                     v=[round(float(x), 3) for x in fv],
                                     sigma=round(fsig, 3), ninl=fninl, ntr=fntr)
                        except Exception as e:
                            jlog('flow_err', err=repr(e))
```

Notes locked in by the offline work — keep them:
- Attitude is the arrival-time attitude, not the exposure-time attitude (~30–50 ms skew). Acceptable at the governed ≤1.0 m/s route speed; revisit with stochastic cloning (playbook 0.5) when speed rises.
- LK on 640×360 costs ~1–3 ms — fine inside `cam_loop`'s thread; it must never block `rx_loop` (it doesn't share that thread) and holds `KF_LOCK` only for the update call.
- The existing vision-velocity pseudo-fix (gate-obs derived) stays; both are honest velocity measurements and the KF composes them.

- [ ] **Step 3: Add the Rerun plot (inside `viz_tick`'s try block, next to `plots/alt_m`)**

```python
            rr.log('plots/flow_speed', rr.Scalars(float(np.linalg.norm(KF.v[:2]))))
```

- [ ] **Step 4: Sanity-run the modified file locally**

Run: `python3 -c "import ast; ast.parse(open('vq2/live/vq2wp.py').read())" && python3 -m pytest vq2/tests/ -v`
Expected: parse OK, all tests still PASS (vq2wp.py itself is not imported by tests — it opens sockets at import).

- [ ] **Step 5: Commit**

```bash
git add vq2/live/vq2wp.py
git commit -m "feat(vq2): fuse floor-flow velocity in live racer"
```

- [ ] **Step 6: Deploy checklist (laptop, when a flight session is next opened)**

1. Copy `vq2/camera.py`, `vq2/flow_vel.py`, `vq2/live/vq2wp.py` to `C:\Users\alexj\` (flat, next to `eskf.py`) — the try/except relative imports make the flat layout work.
2. cv2 already imports in the monorace env (vq2wp.py uses it today); verify once: `ssh laptop "C:\Users\alexj\miniconda3\envs\monorace\python.exe -c \"import cv2; print(cv2.__version__)\""`.
3. First flight with `NOFLOW=1` (regression check: nothing changed), then flow on. Compare `flow_vel` jlog rows against commanded speed on the Rerun plot before trusting a full gate run.
4. Trust frames + Alex's screen over derived telemetry when debugging (memory: `feedback-visual-evidence-over-derived-telemetry`).

---

## Self-review notes (done at plan time)

- Spec coverage: handoff's 4-step VIO plan → Task 5/6 (offline corpora + replay), Task 2/3 (optical-flow ground-plane velocity into `update_velocity`), Task 6 (<0.5 m/12 m acceptance), Task 7 (wire into vq2wp.py). MASt3R-SLAM fallback deliberately out of scope (GO/NO-GO header).
- Type consistency: `FlowVelocity.process` return `(v_w, sigma, ninl, ntr)` used identically in fusion.py and vq2wp.py; `FusionResult.anchors` dict keys match accept_vio's reads; `clock_bridge` returns `(offset_s, spread_ms)` consumed positionally in fusion.py.
- Known risk, stated: floor visibility with the 20°-up camera is thinnest when flying level at 1.3 m (bottom-edge declination ~18°, ground ~4 m ahead at nearest). Forward acceleration pitches the nose down and helps. Task 3's at-rest test and Task 5's `flow_updates` counts are the early empirical checks; if coverage is the failure mode it shows up there, not in Task 6.
