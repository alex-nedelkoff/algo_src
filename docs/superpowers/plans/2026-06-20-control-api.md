# Full Control API (`Drone` facade) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the proven `WaypointNavigator` into a full importable control API (`Drone` facade) with `lookat`/`face` camera modes, threaded interruptible missions, motion primitives (orbit/hover/takeoff/descend), and ergonomic polish.

**Architecture:** Keep `aigp/navigator.py` as the real-time control engine (additive edits only: `lookat`/`face` yaw + cooperative `abort`/`pause`/`status` hooks that default to no-op). Add `aigp/drone.py` with `Drone` (the public facade), `Mission` (a thread handle), `Status`/`Result`, and `FlightConfig`. Primitives compose existing navigator calls so they inherit the whole control stack + safe envelope.

**Tech Stack:** Python 3.9, numpy, `threading` (stdlib), pytest. No new deps.

## Global Constraints

- **Do not break the engine:** all `navigator.py` edits are additive; the interruption hooks default to `None` so existing behavior, `goto.py`, and the current 192 tests are unchanged. Tests run with `python3 -m pytest tests/test_aigp/ -q` (Python 3.9.6; no `python` on PATH).
- **Frames are canonical — reuse, don't re-derive.** `lookat`/`face`/`course` all use the same chart bridge: `yaw_ref_tf = yaw0_t + s_lat·signed_angle(cam_live, d)`.
- **Janahan owns the connection lifecycle:** `Drone` takes a live `store`/`commander`/`plant`; it does not open MAVLink or reset the sim.
- **One mission at a time:** starting a motion call auto-aborts the running mission (no raise). Explicit `Mission.abort()` → auto-start `hover()` so the drone holds, not falls.
- **Safe-envelope defaults** (from the 2026-06-20 envelope finding): `amax≈4`, `tilt_deg≤20`, `default_speed≈5` (clean, <1 m sag, no ground-crash). Faster is allowed but the docstring warns.
- **TDD, frequent commits.** Commit at the end of every task.

---

### Task 1: Engine — `lookat`/`face` camera modes (`_yaw_ref_tf`)

**Files:**
- Modify: `aigp/navigator.py` (`_yaw_ref_tf` ~706, `_is_tf_mode` ~727, `__init__` ~326, `_fly_strafe_course` ~565)
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `self._yaw0_t`, `self._s_lat`, `self._cam_live`, `self._look_point`, `self._cur_target`, `self._cur_travel`.
- Produces: module fn `_signed_angle(a, b) -> float`; `_yaw_ref_tf(yaw, ds)` handles `"hold"|"fixed"|"course"|"lookat"|"face"`; `_is_tf_mode(yaw)` returns True for all five; `WaypointNavigator.set_look_point(point_world)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_aigp/test_navigator.py`)

```python
from aigp.navigator import _signed_angle


def test_signed_angle_basic():
    assert abs(_signed_angle(np.array([1.0, 0.0]), np.array([0.0, 1.0])) - np.pi / 2) < 1e-9
    assert abs(_signed_angle(np.array([1.0, 0.0]), np.array([0.0, -1.0])) + np.pi / 2) < 1e-9


def test_yaw_ref_tf_lookat_points_camera_at_point():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL))
    nav.set_origin()
    nav._s_lat = 1.0
    nav._look_point = nav._origin_pos + np.array([5.0, 0.0, 0.0])  # 5 m along cam_live? compute angle
    ds = nav.store.get_drone()
    d = nav._unit_xy((nav._look_point - ds.pos_ned))
    expect = nav._yaw0_t + _signed_angle(nav._cam_live, d)
    assert abs(nav._yaw_ref_tf("lookat", ds) - expect) < 1e-6


def test_yaw_ref_tf_face_points_nose_at_target():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL))
    nav.set_origin()
    nav._s_lat = 1.0
    nav._cur_target = nav._origin_pos + np.array([4.0, 3.0, 0.0])
    ds = nav.store.get_drone()
    d = nav._unit_xy((nav._cur_target - ds.pos_ned))
    # face = nose at target = camera at the ANTI-direction
    expect = nav._yaw0_t + _signed_angle(nav._cam_live, -d)
    assert abs(nav._yaw_ref_tf("face", ds) - expect) < 1e-6


def test_is_tf_mode_includes_lookat_face():
    for m in ("hold", "fixed", "course", "lookat", "face"):
        assert WaypointNavigator._is_tf_mode(m)
    assert not WaypointNavigator._is_tf_mode("bogus")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -k "signed_angle or yaw_ref_tf_lookat or yaw_ref_tf_face or is_tf_mode_includes" -q`
Expected: FAIL (`ImportError: _signed_angle` / `NotImplementedError`).

- [ ] **Step 3: Implement** — add the module helper (near `_unit_xy` or top-level, after `_dr_vel`):

```python
def _signed_angle(a, b):
    """Signed angle (rad) rotating unit-ish vector a -> b in the XY plane (CCW positive)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], float(a @ b)))
```

In `__init__` (after `self._cur_travel = ...`, ~350) add:

```python
        self._cur_target = np.zeros(3)   # current waypoint (world) for yaw='face'
```

Replace `_yaw_ref_tf` (706-719) with:

```python
    def _yaw_ref_tf(self, yaw, ds):
        """NOSE heading in the true-cam frame. hold/fixed hold the spawn heading; course points the
        CAMERA along travel; lookat points the CAMERA at self._look_point; face points the NOSE at
        self._cur_target (= camera at the anti-direction). All use the same chart bridge
        yaw0_t + s_lat*signed_angle(cam_live, d) (s_lat=0 during the probe -> yaw0_t)."""
        if yaw in ("hold", "fixed"):
            return self._yaw0_t
        if yaw == "course":
            d = self._unit_xy(self._cur_travel)
        elif yaw == "lookat":
            d = self._unit_xy((np.asarray(self._look_point, float) - ds.pos_ned)
                              if self._look_point is not None else self._cur_travel)
        elif yaw == "face":
            d = -self._unit_xy(np.asarray(self._cur_target, float) - ds.pos_ned)
        else:
            raise NotImplementedError(yaw)
        return self._yaw0_t + (self._s_lat or 0.0) * _signed_angle(self._cam_live, d)
```

Update `_is_tf_mode` (727-731) body to:

```python
        return yaw in ("hold", "fixed", "course", "lookat", "face")
```

In `_fly_strafe_course`, where `self._cur_travel = self._unit_xy(target - leg_start)` is set (~586), add the line right after it:

```python
                self._cur_target = target   # for yaw='face'
```

Add the setter (near `set_origin`):

```python
    def set_look_point(self, point_world):
        """Set the world-NED point the camera tracks in yaw='lookat'."""
        self._look_point = np.asarray(point_world, float)
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `python3 -m pytest tests/test_aigp/ -q` → expect all pass (192 + 4 new = 196).

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): lookat/face camera modes via the course chart bridge"
```

---

### Task 2: Engine — cooperative abort/pause/status hooks

**Files:**
- Modify: `aigp/navigator.py` (`__init__` ~326, `_fly_strafe_course` ~565, `_probe_inflight` ~506)
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: existing loops.
- Produces: `WaypointNavigator` attrs `_abort_evt`, `_pause_evt`, `_status_cb` (default `None`); when set, `_fly_strafe_course` (a) returns `"abort"` immediately if `_abort_evt` is set, (b) holds position while `_pause_evt` is set, (c) calls `_status_cb(phase, ds, tilt, mode, target, progress)` each iteration. No-op when `None` → existing behavior unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_aigp/test_navigator.py`)

```python
import threading


class _RecCommander:
    def __init__(self): self.n = 0
    def send_attitude_target(self, rate, thr): self.n += 1


def test_abort_event_returns_immediately_no_commands():
    nav = WaypointNavigator(_FakeStore(_mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)),
                            _RecCommander(), _PLANT)
    nav.set_origin()
    nav._abort_evt = threading.Event(); nav._abort_evt.set()
    res = nav._fly_strafe_course([nav._origin_pos + np.array([5.0, 0, 0])], "course")
    assert res == "abort"
    assert nav.commander.n == 0      # never sent a command


def test_status_cb_invoked_with_phase():
    seen = []
    nav = WaypointNavigator(_FakeStore(_mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)),
                            _RecCommander(), _PLANT)
    nav.set_origin()
    nav._abort_evt = threading.Event()
    nav._status_cb = lambda **k: (seen.append(k["phase"]), nav._abort_evt.set())  # 1 iter then stop
    nav._fly_strafe_course([nav._origin_pos + np.array([5.0, 0, 0])], "course")
    assert seen and isinstance(seen[0], str)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -k "abort_event_returns or status_cb_invoked" -q`
Expected: FAIL (`AttributeError`/no abort handling — the loop flies/sends or errors).

- [ ] **Step 3: Implement** — in `__init__` (after `self._cur_target`):

```python
        self._abort_evt = None     # threading.Event injected by the Drone facade (None = no-op)
        self._pause_evt = None
        self._status_cb = None     # callable(phase=, ds=, tilt=, mode=, target=, progress=)
```

Add helpers (near `_world_vel`):

```python
    def _interrupted(self):
        return self._abort_evt is not None and self._abort_evt.is_set()

    def _emit(self, phase, ds, tilt, mode, target, progress):
        if self._status_cb is not None:
            self._status_cb(phase=phase, ds=ds, tilt=float(tilt), mode=mode,
                            target=None if target is None else np.asarray(target, float),
                            progress=float(progress))
```

In `_fly_strafe_course`, at the very top of the method (before the `_probe_inflight` call) add:

```python
        if self._interrupted():
            return "abort"
```

Inside the `while` loop, immediately after `ds = self.store.get_drone()` and the `if ds is not None:` guard opens, add the abort/pause checks and a status emit (use the existing `target`, `i`, `n`, `tilt`, `vw` once computed; place the emit right before `time.sleep(g.LOOP_DT)` at the loop end, and the abort/pause at the top of the iteration body):

```python
                if self._interrupted():
                    return "abort"
                while self._pause_evt is not None and self._pause_evt.is_set() \
                        and not self._interrupted():
                    rate, thr, tilt, dbg = self._strafe_attitude(
                        ds, 0.0, 0.0, float(ds.pos_ned[2]), self._yaw_ref_tf(yaw, ds))
                    self.commander.send_attitude_target(rate, thr)
                    self._emit("pause", ds, tilt, yaw, target, i / max(n, 1))
                    time.sleep(g.LOOP_DT)
                    ds = self.store.get_drone()
                    if ds is None:
                        break
                if ds is None:
                    time.sleep(g.LOOP_DT); continue
```

And at the end of the loop iteration, right before `time.sleep(g.LOOP_DT)`:

```python
                self._emit(f"leg {i + 1}/{n}", ds, tilt, yaw, target, i / max(n, 1))
```

In `_probe_inflight`'s loop, add at the top of the `while` body (after `ds = self.store.get_drone()`):

```python
            if self._interrupted():
                return
```

- [ ] **Step 4: Run to verify pass + no regression**

Run: `python3 -m pytest tests/test_aigp/ -q` → expect all pass (196 + 2 = 198).

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): cooperative abort/pause/status hooks (default no-op)"
```

---

### Task 3: Facade scaffolding — `Result`, `Status`, `FlightConfig`

**Files:**
- Create: `aigp/drone.py`
- Test: `tests/test_aigp/test_drone.py`

**Interfaces:**
- Consumes: `aigp.navigator.NavGains`.
- Produces: `class Result(Enum)` (`RUNNING, REACHED, TIMEOUT, ABORT, ERROR`); `@dataclass Status(result, phase, pos_ned, vel, tilt_deg, mode, target, progress, error)`; `@dataclass FlightConfig(vmax, amax, tilt_deg, zff, capture, kiz, c_max, default_speed)` with `.to_navgains() -> NavGains`.

- [ ] **Step 1: Write the failing test** (`tests/test_aigp/test_drone.py`, Create)

```python
import numpy as np
from aigp.drone import Result, Status, FlightConfig
from aigp.navigator import NavGains


def test_flightconfig_safe_defaults_within_envelope():
    c = FlightConfig()
    assert c.amax <= 4.5 and c.tilt_deg <= 20 and c.default_speed <= 6


def test_flightconfig_to_navgains_maps_fields():
    c = FlightConfig(vmax=7.0, amax=4.0, tilt_deg=20.0, zff=-2.4, capture=2.5, kiz=0.8, c_max=18.0)
    g = c.to_navgains()
    assert isinstance(g, NavGains)
    assert g.MAX_SPEED == 7.0 and g.VLAT_MAX == 7.0 and g.FWD_AMAX == 4.0
    assert g.DECEL_MAX >= 4.0 and g.TILT_MAX_DEG == 20.0 and g.Z_FF == -2.4
    assert g.CAPTURE == 2.5 and g.KI_Z == 0.8 and g.C_MAX == 18.0


def test_status_defaults_running():
    s = Status(result=Result.RUNNING, phase="x", pos_ned=np.zeros(3), vel=0.0,
               tilt_deg=0.0, mode="hold", target=None, progress=0.0, error=None)
    assert s.result is Result.RUNNING and s.error is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -q`
Expected: FAIL (`ModuleNotFoundError: aigp.drone`).

- [ ] **Step 3: Implement** (`aigp/drone.py`, Create)

```python
"""Drone — full importable control API over the proven WaypointNavigator.

Janahan wires + arms the MAVLink/sim himself and passes the live store/commander/plant in; this
facade adds threaded missions (abort/pause/status), motion primitives (orbit/hover/takeoff/descend),
and ergonomics. One mission at a time; a new motion call auto-aborts the running one.

Quickstart:
    drone = Drone(store, commander, plant)
    drone.set_origin()
    drone.takeoff(3).wait()                       # climb 3 m, block until done
    drone.orbit((10, 0, 0), radius=5, seconds=8)  # circle a point, camera locked, async
    ...
    drone.land(3).wait()
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from aigp.navigator import NavGains, WaypointNavigator


class Result(Enum):
    RUNNING = "running"
    REACHED = "reached"
    TIMEOUT = "timeout"
    ABORT = "abort"
    ERROR = "error"


@dataclass
class Status:
    result: Result
    phase: str
    pos_ned: np.ndarray
    vel: float
    tilt_deg: float
    mode: str
    target: "np.ndarray | None"
    progress: float
    error: "str | None" = None


@dataclass
class FlightConfig:
    """Flight envelope. Defaults are the proven-safe values (tilt <=20, ~5 m/s -> <1 m z-sag,
    no ground-crash). Raising amax/tilt past ~25 deg risks the high-tilt z-sag into terrain."""
    vmax: float = 6.0
    amax: float = 4.0
    tilt_deg: float = 20.0
    zff: float = -2.4
    capture: float = 2.5
    kiz: float = 0.8
    c_max: float = 18.0
    default_speed: float = 5.0

    def to_navgains(self) -> NavGains:
        g = NavGains()
        g.MAX_SPEED = self.vmax
        g.VLAT_MAX = self.vmax
        g.FWD_AMAX = self.amax
        g.DECEL_MAX = max(g.DECEL_MAX, self.amax)
        g.TILT_MAX_DEG = self.tilt_deg
        g.Z_FF = self.zff
        g.CAPTURE = self.capture
        g.KI_Z = self.kiz
        g.C_MAX = self.c_max
        return g
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -q` → PASS (3).

- [ ] **Step 5: Commit**

```bash
git add aigp/drone.py tests/test_aigp/test_drone.py
git commit -m "feat(drone): Result/Status/FlightConfig scaffolding"
```

---

### Task 4: Facade — `_StatusBox` + input validation helpers

**Files:**
- Modify: `aigp/drone.py`
- Test: `tests/test_aigp/test_drone.py`

**Interfaces:**
- Produces: `class _StatusBox` (thread-safe latest-`Status` holder: `.update(**kw)`, `.set_result(result, error=None)`, `.get() -> Status`); module fns `_check_frame(frame)`, `_check_yaw(yaw)`, `_check_positive(name, val)` raising `ValueError`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_aigp/test_drone.py`)

```python
import pytest
from aigp.drone import _StatusBox, _check_frame, _check_yaw, _check_positive


def test_statusbox_roundtrip_and_thread_safe_copy():
    box = _StatusBox()
    box.update(phase="leg 1/2", ds_pos=np.array([1.0, 2.0, -3.0]), vel=2.0, tilt=5.0,
               mode="course", target=np.array([4.0, 0.0, -3.0]), progress=0.5)
    s = box.get()
    assert s.phase == "leg 1/2" and s.vel == 2.0 and s.mode == "course"
    assert s.result is Result.RUNNING
    box.set_result(Result.REACHED)
    assert box.get().result is Result.REACHED


def test_validation_raises():
    with pytest.raises(ValueError):
        _check_frame("polar")
    with pytest.raises(ValueError):
        _check_yaw("spin")
    with pytest.raises(ValueError):
        _check_positive("radius", -1.0)
    _check_frame("body"); _check_yaw("lookat"); _check_positive("radius", 2.0)  # no raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -k "statusbox or validation_raises" -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement** (append to `aigp/drone.py`)

```python
_FRAMES = ("world", "body")
_YAWS = ("hold", "fixed", "course", "face", "lookat")


def _check_frame(frame):
    if frame not in _FRAMES:
        raise ValueError(f"frame must be one of {_FRAMES}, got {frame!r}")


def _check_yaw(yaw):
    if yaw not in _YAWS:
        raise ValueError(f"yaw must be one of {_YAWS}, got {yaw!r}")


def _check_positive(name, val):
    if not (isinstance(val, (int, float)) and val > 0):
        raise ValueError(f"{name} must be > 0, got {val!r}")


class _StatusBox:
    """Thread-safe holder of the latest Status. The worker thread calls update()/set_result();
    the caller reads get() (a copy)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._s = Status(result=Result.RUNNING, phase="init", pos_ned=np.zeros(3), vel=0.0,
                         tilt_deg=0.0, mode="", target=None, progress=0.0, error=None)

    def update(self, *, phase, ds_pos, vel, tilt, mode, target, progress):
        with self._lock:
            self._s.phase = phase
            self._s.pos_ned = np.asarray(ds_pos, float).copy()
            self._s.vel = float(vel)
            self._s.tilt_deg = float(tilt)
            self._s.mode = mode
            self._s.target = None if target is None else np.asarray(target, float).copy()
            self._s.progress = float(progress)

    def set_result(self, result, error=None):
        with self._lock:
            self._s.result = result
            self._s.error = error

    def get(self) -> Status:
        with self._lock:
            return Status(self._s.result, self._s.phase, self._s.pos_ned.copy(), self._s.vel,
                          self._s.tilt_deg, self._s.mode, None if self._s.target is None
                          else self._s.target.copy(), self._s.progress, self._s.error)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -q` → PASS (5).

- [ ] **Step 5: Commit**

```bash
git add aigp/drone.py tests/test_aigp/test_drone.py
git commit -m "feat(drone): _StatusBox + input validation helpers"
```

---

### Task 5: Facade — `Mission` (threaded run, abort/pause/status/wait)

**Files:**
- Modify: `aigp/drone.py`
- Test: `tests/test_aigp/test_drone.py`

**Interfaces:**
- Consumes: `Status`, `Result`, `_StatusBox`.
- Produces: `class Mission(run_fn, abort_evt, pause_evt, box)` where `run_fn(abort_evt, pause_evt, box) -> Result`. Methods: `.start()`, `.abort()`, `.pause()`, `.resume()`, `.wait(timeout=None) -> Status`, props `.done`, `.result`, `.status() -> Status`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_aigp/test_drone.py`)

```python
from aigp.drone import Mission


def _fake_run(abort_evt, pause_evt, box, n=50):
    # spins up to n ticks, honoring pause/abort, updating status; returns REACHED if it finishes
    for k in range(n):
        if abort_evt.is_set():
            return Result.ABORT
        while pause_evt.is_set() and not abort_evt.is_set():
            time.sleep(0.005)
        box.update(phase=f"tick {k}", ds_pos=np.zeros(3), vel=1.0, tilt=2.0,
                   mode="hold", target=None, progress=k / n)
        time.sleep(0.005)
    return Result.REACHED


def _mk_mission():
    a, p, box = threading.Event(), threading.Event(), _StatusBox()
    return Mission(lambda ae, pe, b: _fake_run(ae, pe, b), a, p, box)


def test_mission_runs_to_completion():
    m = _mk_mission(); m.start()
    s = m.wait(timeout=5)
    assert m.done and s.result is Result.REACHED


def test_mission_abort_stops_and_reports():
    m = _mk_mission(); m.start()
    time.sleep(0.02); m.abort()
    s = m.wait(timeout=5)
    assert s.result is Result.ABORT


def test_mission_pause_resume_gates_progress():
    m = _mk_mission(); m.start()
    time.sleep(0.02); m.pause()
    p1 = m.status().progress
    time.sleep(0.05)
    assert abs(m.status().progress - p1) < 1e-9     # frozen while paused
    m.resume()
    s = m.wait(timeout=5)
    assert s.result is Result.REACHED and s.progress > p1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -k "mission_" -q`
Expected: FAIL (`ImportError: Mission`).

- [ ] **Step 3: Implement** (append to `aigp/drone.py`)

```python
class Mission:
    """A flight running on a background thread. run_fn(abort_evt, pause_evt, box) -> Result must
    honor the events and write progress to box; its return value becomes the final result."""

    def __init__(self, run_fn, abort_evt, pause_evt, box):
        self._run_fn = run_fn
        self._abort_evt = abort_evt
        self._pause_evt = pause_evt
        self._box = box
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            result = self._run_fn(self._abort_evt, self._pause_evt, self._box)
        except Exception as e:                       # surface loop crashes as ERROR, don't kill thread
            self._box.set_result(Result.ERROR, error=repr(e))
            return
        self._box.set_result(result if isinstance(result, Result) else Result.REACHED)

    def start(self):
        self._thread.start()
        return self

    def abort(self):
        self._abort_evt.set()

    def pause(self):
        self._pause_evt.set()

    def resume(self):
        self._pause_evt.clear()

    def status(self) -> Status:
        return self._box.get()

    def wait(self, timeout=None) -> Status:
        self._thread.join(timeout)
        return self._box.get()

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    @property
    def result(self):
        return self._box.get().result
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -q` → PASS (8).

- [ ] **Step 5: Commit**

```bash
git add aigp/drone.py tests/test_aigp/test_drone.py
git commit -m "feat(drone): Mission threaded handle (abort/pause/resume/wait/status)"
```

---

### Task 6: Facade — `Drone` core (`goto`/`follow`/`look_at`, auto-abort preemption)

**Files:**
- Modify: `aigp/drone.py`
- Test: `tests/test_aigp/test_drone.py`

**Interfaces:**
- Consumes: `WaypointNavigator`, `Mission`, `_StatusBox`, validators, `FlightConfig`.
- Produces: `class Drone(store, commander, plant, *, config=None, flog=None)`; `.set_origin(...)`, `.look_at(point, frame="body")`, `.goto(wp, *, yaw="course", look_at=None, speed=None, frame="body", stop=False) -> Mission`, `.follow(wps, ...) -> Mission`. Internal `_start(run_fn) -> Mission` auto-aborts the active mission first; `_navigate(targets, yaw, look_at, speed, frame, stop)` wires nav hooks + runs the blocking nav call mapping its `str` result -> `Result`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_aigp/test_drone.py`)

```python
from aigp.drone import Drone
from aigp.navigator import WaypointNavigator


class _FakeStore2:
    def __init__(self, ds): self._ds = ds
    def get_drone(self): return self._ds
    def get_race_live(self): return True


class _RecCmd:
    def send_attitude_target(self, rate, thr): pass


def _drone():
    from aigp.state import DroneState
    ds = DroneState(np.array([0, 0, -2.0]), np.zeros(3), np.array([0.0, 1, 0, 0]), np.zeros(3), 0)
    return Drone(_FakeStore2(ds), _RecCmd(), (0.5, 13.0, np.array([1.0, 1.0, 1.0])))


def test_goto_validates_inputs():
    d = _drone()
    with pytest.raises(ValueError):
        d.goto((1, 0, 0), frame="polar").wait(timeout=1)
    with pytest.raises(ValueError):
        d.goto((1, 0, 0), yaw="spin").wait(timeout=1)


def test_goto_returns_mission_and_runs():
    d = _drone()
    # stub the navigator's blocking call so no real loop runs
    d.nav.goto = lambda *a, **k: "reached"
    m = d.goto((3, 0, 0), yaw="hold")
    s = m.wait(timeout=2)
    assert isinstance(m, Mission) and s.result is Result.REACHED


def test_new_mission_auto_aborts_previous():
    d = _drone()
    order = []
    def slow(*a, **k):
        for _ in range(100):
            if d.nav._abort_evt is not None and d.nav._abort_evt.is_set():
                order.append("aborted"); return "abort"
            time.sleep(0.005)
        order.append("finished"); return "reached"
    d.nav.goto = slow
    m1 = d.goto((3, 0, 0), yaw="hold")
    time.sleep(0.02)
    m2 = d.goto((4, 0, 0), yaw="hold")     # should auto-abort m1
    m2.wait(timeout=2)
    assert "aborted" in order and m1.result is Result.ABORT
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -k "goto_validates or goto_returns or auto_aborts" -q`
Expected: FAIL (`ImportError: Drone`).

- [ ] **Step 3: Implement** (append to `aigp/drone.py`)

```python
def _result_from_str(s):
    return {"reached": Result.REACHED, "timeout": Result.TIMEOUT,
            "abort": Result.ABORT}.get(s, Result.ERROR)


class Drone:
    """Public control facade. Janahan owns connect+arm; pass the live store/commander/plant in."""

    def __init__(self, store, commander, plant, *, config: FlightConfig = None, flog=None):
        self.config = config or FlightConfig()
        self.nav = WaypointNavigator(store, commander, plant,
                                     gains=self.config.to_navgains(), flog=flog)
        self._mission = None

    def set_origin(self, pos_ned=None, yaw=None):
        self.nav.set_origin(pos_ned, yaw)

    def look_at(self, point, frame="body"):
        _check_frame(frame)
        self.nav.set_look_point(self.nav._resolve(np.asarray(point, float), frame))

    # --- mission plumbing ---
    def _start(self, run_fn) -> Mission:
        if self._mission is not None and not self._mission.done:
            self._mission.abort()
            self._mission.wait(timeout=2.0)
        abort_evt, pause_evt, box = threading.Event(), threading.Event(), _StatusBox()
        self.nav._abort_evt = abort_evt
        self.nav._pause_evt = pause_evt
        self.nav._status_cb = lambda *, phase, ds, tilt, mode, target, progress: box.update(
            phase=phase, ds_pos=ds.pos_ned, vel=float(np.linalg.norm(ds.vel_ned[:2])),
            tilt=tilt, mode=mode, target=target, progress=progress)
        m = Mission(run_fn, abort_evt, pause_evt, box).start()
        self._mission = m
        return m

    def _navigate(self, targets, *, yaw, look_at, speed, frame, stop, single):
        _check_frame(frame); _check_yaw(yaw)
        if speed is not None:
            _check_positive("speed", speed)
        if yaw == "lookat" and look_at is None and self.nav._look_point is None:
            raise ValueError("yaw='lookat' requires look_at=... or a prior look_at()")
        v = speed if speed is not None else self.config.default_speed
        if look_at is not None:
            self.look_at(look_at, frame)
        if stop:
            self.nav._stop_each = True

        def run_fn(abort_evt, pause_evt, box):
            res = (self.nav.goto(targets[0], yaw=yaw, v_cruise=v, engine="legs", frame=frame)
                   if single else
                   self.nav.follow(targets, yaw=yaw, v_cruise=v, engine="legs", frame=frame))
            return _result_from_str(res)
        return self._start(run_fn)

    # --- public motion ---
    def goto(self, wp, *, yaw="course", look_at=None, speed=None, frame="body", stop=False) -> Mission:
        return self._navigate([wp], yaw=yaw, look_at=look_at, speed=speed, frame=frame,
                              stop=stop, single=True)

    def follow(self, wps, *, yaw="course", look_at=None, speed=None, frame="body", stop=False) -> Mission:
        return self._navigate(list(wps), yaw=yaw, look_at=look_at, speed=speed, frame=frame,
                              stop=stop, single=False)
```

Note: validation runs *before* `_start` (in `_navigate`), so bad input raises synchronously from
`goto()` — `test_goto_validates_inputs`'s `.wait()` is never reached. That is intended.

- [ ] **Step 4: Run to verify it passes + no regression**

Run: `python3 -m pytest tests/test_aigp/ -q` → expect all pass (drone 11 + navigator 198).

- [ ] **Step 5: Commit**

```bash
git add aigp/drone.py tests/test_aigp/test_drone.py
git commit -m "feat(drone): Drone goto/follow/look_at + auto-abort preemption"
```

---

### Task 7: Facade — motion primitives (orbit/hover/takeoff/descend)

**Files:**
- Modify: `aigp/drone.py`
- Test: `tests/test_aigp/test_drone.py`

**Interfaces:**
- Consumes: `Drone._navigate`/`_start`, `WaypointNavigator`.
- Produces: module fn `_orbit_ring(center, radius, n, direction) -> list[np.ndarray]`; `Drone.orbit(center, *, radius, speed=None, seconds, frame="body", direction="ccw") -> Mission`, `.hover(seconds=None) -> Mission`, `.takeoff(altitude) -> Mission`, `.descend(altitude) -> Mission`, `.land = descend`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_aigp/test_drone.py`)

```python
from aigp.drone import _orbit_ring


def test_orbit_ring_geometry():
    center = np.array([10.0, 0.0, -3.0])
    ring = _orbit_ring(center, radius=5.0, n=24, direction="ccw")
    assert len(ring) == 24
    for p in ring:
        assert abs(np.linalg.norm((p - center)[:2]) - 5.0) < 1e-6   # on the circle
        assert abs(p[2] - center[2]) < 1e-9                          # planar (center altitude)
    # ccw winding: cross of first two spokes is +z (NED down +, so signed area sign is consistent)
    a = (ring[0] - center)[:2]; b = (ring[1] - center)[:2]
    assert (a[0] * b[1] - a[1] * b[0]) > 0


def test_takeoff_and_descend_targets():
    d = _drone()
    captured = {}
    d._navigate = lambda targets, **k: captured.setdefault("t", np.asarray(targets[0], float))
    d.takeoff(3.0)
    assert abs(captured["t"][2] - (-2.0 - 3.0)) < 1e-9    # current z (-2) minus 3 = up
    d.descend(2.0)
    assert abs(captured["t"][2] - (-2.0 + 2.0)) < 1e-9    # down
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_drone.py -k "orbit_ring or takeoff_and_descend" -q`
Expected: FAIL (`ImportError: _orbit_ring`).

- [ ] **Step 3: Implement** (append to `aigp/drone.py`)

```python
def _orbit_ring(center, radius, n, direction):
    """n waypoints on the circle of `radius` around `center` (XY), at center's altitude."""
    center = np.asarray(center, float)
    sgn = 1.0 if direction == "ccw" else -1.0
    out = []
    for k in range(n):
        th = sgn * 2.0 * np.pi * k / n
        out.append(np.array([center[0] + radius * np.cos(th),
                             center[1] + radius * np.sin(th), center[2]]))
    return out
```

Add to `Drone` (after `follow`):

```python
    def orbit(self, center, *, radius, speed=None, seconds, frame="body", direction="ccw") -> Mission:
        _check_frame(frame); _check_positive("radius", radius); _check_positive("seconds", seconds)
        if speed is not None:
            _check_positive("speed", speed)
        if direction not in ("cw", "ccw"):
            raise ValueError("direction must be 'cw' or 'ccw'")
        v = speed if speed is not None else self.config.default_speed
        center_w = self.nav._resolve(np.asarray(center, float), frame)
        self.nav.set_look_point(center_w)
        ring = _orbit_ring(center_w, radius, 24, direction)

        def run_fn(abort_evt, pause_evt, box):
            t0 = time.time()
            while time.time() - t0 < seconds and not abort_evt.is_set():
                res = self.nav.follow(ring, yaw="lookat", v_cruise=v, engine="legs", frame="world")
                if res == "abort":
                    return Result.ABORT
            return Result.ABORT if abort_evt.is_set() else Result.REACHED
        return self._start(run_fn)

    def hover(self, seconds=None) -> Mission:
        if seconds is not None:
            _check_positive("seconds", seconds)

        def run_fn(abort_evt, pause_evt, box):
            t0 = time.time()
            while not abort_evt.is_set() and (seconds is None or time.time() - t0 < seconds):
                self.nav.settle()      # holds current position ~SETTLE_T; loop re-holds
            return Result.ABORT if abort_evt.is_set() else Result.REACHED
        return self._start(run_fn)

    def takeoff(self, altitude) -> Mission:
        _check_positive("altitude", altitude)
        pos = self.nav._current_pos()
        return self._navigate([np.array([pos[0], pos[1], pos[2] - altitude])], yaw="hold",
                              look_at=None, speed=None, frame="world", stop=True, single=True)

    def descend(self, altitude) -> Mission:
        _check_positive("altitude", altitude)
        pos = self.nav._current_pos()
        return self._navigate([np.array([pos[0], pos[1], pos[2] + altitude])], yaw="hold",
                              look_at=None, speed=None, frame="world", stop=True, single=True)

    land = descend
```

- [ ] **Step 4: Run to verify it passes + no regression**

Run: `python3 -m pytest tests/test_aigp/ -q` → expect all pass (drone 13 + navigator 198).

- [ ] **Step 5: Commit**

```bash
git add aigp/drone.py tests/test_aigp/test_drone.py
git commit -m "feat(drone): motion primitives orbit/hover/takeoff/descend"
```

---

### Task 8: Example script + API docstrings

**Files:**
- Create: `examples/drone_demo.py`
- Modify: `aigp/drone.py` (ensure each public method has a docstring)

**Interfaces:**
- Consumes: the full `Drone` API + the lifecycle pattern from `goto.py` (MAVLink connect + `fresh_start` + `arm`).
- Produces: a runnable demo + complete docstrings.

- [ ] **Step 1: Write the example** (`examples/drone_demo.py`, Create) — mirror `goto.py`'s lifecycle, then exercise the API:

```python
"""Demo: connect + arm the VQ sim, then drive the Drone control API.

Run on the laptop worktree (sim up):  python examples/drone_demo.py
"""
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil

import aigp.flight_telemetry as ftm
from aigp.commander import Commander
from aigp.drone import Drone, FlightConfig
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import load_plant
from aigp.state import Store

IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def idle(m, boot):
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start(s, c, m, boot):
    t = time.time()
    while time.time() - t < 1.0:
        idle(m, boot); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(m, boot); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(m, boot); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


def main():
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000); c = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, c, m, boot), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    yaw0 = float(np.arctan2(*(quat_to_R(ds0.quat_wxyz)[[1, 0], 0])))
    flog = ftm.from_args(sys.argv, plant[2], "drone_demo", store=s)

    drone = Drone(s, c, plant, config=FlightConfig(), flog=flog)
    drone.nav.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()

    drone.takeoff(2.0).wait()                                  # climb 2 m
    drone.orbit((10, 0, 0), radius=5.0, seconds=8.0).wait()    # circle a point, camera locked
    m_far = drone.goto((12, 8, 0), yaw="lookat", look_at=(0, 0, 0))   # fly while watching spawn
    time.sleep(2.0); print("status:", m_far.status().phase, m_far.status().tilt_deg)
    drone.goto((0, 0, 0), yaw="hold")                          # preempts m_far (auto-abort)
    drone.land(2.0).wait()
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses + docstrings present**

Run: `python3 -c "import ast; ast.parse(open('examples/drone_demo.py').read()); ast.parse(open('aigp/drone.py').read()); print('parse OK')"`
Then eyeball: each of `goto/follow/orbit/hover/takeoff/descend/look_at/set_origin` has a one-line docstring; add any missing.

- [ ] **Step 3: Run the full suite** (ensure nothing broke)

Run: `python3 -m pytest tests/test_aigp/ -q` → all pass.

- [ ] **Step 4: Commit**

```bash
git add examples/drone_demo.py aigp/drone.py
git commit -m "docs(drone): runnable demo + API docstrings"
```

---

### Task 9: LIVE validation (operator step — fresh sim)

**Files:** none (validation only). Requires the laptop worktree + a freshly restarted DCGame.

**Interfaces:**
- Consumes: the full API; the live VQ sim.
- Produces: confirmation + an AI-GP Experiment Log row.

- [ ] **Step 1: Sync + smoke-import on the laptop**

```
git push origin aigp-control-waypoints        # (or the gh token-URL form)
# laptop worktree:
git fetch <ssh-url> aigp-control-waypoints && git reset --hard FETCH_HEAD
python -c "from aigp.drone import Drone; print('import OK')"
```

- [ ] **Step 2: Run the demo, watch Mac Rerun**

```
python examples/drone_demo.py
```
Expected (fresh sim): takeoff climbs ~2 m; orbit circles (10,0,0) with the camera staying on it;
the lookat goto flies toward (12,8,0) with the camera on spawn; the second goto preempts it
(auto-abort); land descends. Keep within the safe envelope (default config). Stay bounded.

- [ ] **Step 3: Targeted checks** (separate quick runs if needed)

- `lookat`: camera frustum in Rerun stays pointed at the look-point through a square. If the camera
  points the *wrong* way, the chart-bridge sign is flipped — set `nav._s_lat *= -1` after the probe
  (or add a `--lookflip` toggle) and re-fly; translation is unaffected (heading-independent recompose).
- `abort`/`pause` mid-flight from a second thread (the demo's preemption covers abort).

- [ ] **Step 4: Log + commit any fix**

Append an AI-GP Experiment Log row (modes validated, `s_lat` sign, any toggle). Commit fixes.

---

## Self-Review

**1. Spec coverage:**
- Engine `lookat`/`face` + routing → Task 1. ✓
- Engine cooperative abort/pause/status hooks → Task 2. ✓
- `Result`/`Status`/`FlightConfig` + safe defaults → Task 3. ✓
- `_StatusBox` + validation → Task 4. ✓
- `Mission` threaded handle (abort/pause/resume/wait/status) → Task 5. ✓
- `Drone` goto/follow/look_at + **auto-abort preemption** → Task 6. ✓
- Primitives orbit/hover/takeoff/descend(=land) → Task 7. ✓
- Docs + example → Task 8. ✓
- Live validation (incl. `s_lat` sign toggle for lookat) → Task 9. ✓
- Out-of-scope items (moving lookat, real takeoff/land, network, multi-mission) → not implemented, as specified. ✓

**2. Placeholder scan:** every code step shows full code; no TBD/TODO. The one empirical unknown (lookat yaw-sign) has a concrete default + an exact live toggle (Task 9 Step 3), not a placeholder.

**3. Type consistency:** `Mission(run_fn, abort_evt, pause_evt, box)` with `run_fn(abort_evt, pause_evt, box) -> Result` is consistent across Tasks 5/6/7; `_StatusBox.update(phase, ds_pos, vel, tilt, mode, target, progress)` matches the `_status_cb` lambda in Task 6 and the navigator `_emit(phase, ds, tilt, mode, target, progress)` in Task 2 (the lambda adapts `ds -> ds_pos/vel`); `_navigate(..., single=)` consumed by goto/follow/takeoff/descend; `FlightConfig.to_navgains()` field names match `NavGains` (MAX_SPEED/VLAT_MAX/FWD_AMAX/DECEL_MAX/TILT_MAX_DEG/Z_FF/CAPTURE/KI_Z/C_MAX) verified against the engine. `_result_from_str` maps the navigator's `"reached"|"timeout"|"abort"` strings.

## Known risk (carried from spec)

- `lookat`/`face` yaw-sign (`s_lat` chart-bridge term) is validated live like `course`; a wrong sign points the camera the wrong way but does not destabilize (translation is heading-independent). Exact toggle in Task 9.
- Threading: all control commands stay on the worker thread; only `threading.Event`s + a lock-guarded `Status` cross threads.
