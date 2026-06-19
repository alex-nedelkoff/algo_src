# Waypoint Navigation API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `goto.py`'s proven waypoint control into an importable, unit-tested `WaypointNavigator` API, and rewrite `goto.py` as a thin CLI over it.

**Architecture:** A new `aigp/navigator.py` holds (a) three stateless control-law functions reused from `aigp/control_math.py` and (b) a `WaypointNavigator` class that owns the blocking flight loop and all IO. The control law is separated from `time`/`sleep`/IO so it is unit-testable; the loop is thin. `goto.py` is rewritten to do lifecycle (`fresh_start` + `arm`), build a `WaypointNavigator`, and call `nav.follow(...)`.

**Tech Stack:** Python 3, numpy, pymavlink (CLI only), pytest. No new dependencies.

## Global Constraints

- **Behavior parity:** the default path through `WaypointNavigator` must be numerically equivalent to current `goto.py` — same gains, same loop ordering (command sent BEFORE telemetry), same arrival/abort/timeout thresholds. The proven live result must be preserved.
- **No new dependencies.** Reuse `aigp/control_math.py`, `aigp/geometry.py`, `aigp/state.py`, `aigp/flight_telemetry.py`, `aigp/commander.py`.
- **Frames:** NED world, FRD body, quaternion `[w,x,y,z]`. Heading `yaw = atan2(R[1,0], R[0,0])`.
- **Pure law = no IO:** `cruise_accel`, `settle_accel`, `attitude_command` must not call `time`, `sleep`, the commander, or telemetry.
- **Status strings:** `'reached'`, `'timeout'`, `'abort'` (match `goto.py`).
- Tests live in `tests/test_aigp/`, run with `pytest tests/test_aigp/ -q`.

---

### Task 1: `NavGains` dataclass + `load_plant` helper

**Files:**
- Create: `aigp/navigator.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `NavGains` dataclass with fields (defaults = goto.py's proven values):
    `KP_ATT: np.ndarray=[0.5,1.6,1.0]`, `KP_YAW=3.0`, `KD_YAW=0.3`, `KP_CT=0.5`, `KD_CT=1.2`,
    `AL_MAX=0.5`, `KD_AL=1.2`, `KP_Z=1.8`, `KD_Z=3.0`, `WMAX=4.0`, `TILT_MAX_DEG=15.0`,
    `ABORT_TILT_DEG=80.0`, `WP_TIMEOUT=40.0`, `MAX_SPEED=1.2`, `ARRIVE=1.5`, `DECEL_MAX=2.0`,
    `SETTLE_T=2.5`, `SETTLE_V=0.4`, `LOOP_DT=0.004`.
  - `load_plant(path="sysid/sim_response.json") -> (hover: float, k_a: float, rg: np.ndarray(3,))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_navigator.py
import json
import numpy as np
from aigp.navigator import NavGains, load_plant


def test_navgains_defaults_match_goto():
    g = NavGains()
    assert g.MAX_SPEED == 1.2
    assert g.ARRIVE == 1.5
    assert g.TILT_MAX_DEG == 15.0
    assert g.WMAX == 4.0
    np.testing.assert_allclose(g.KP_ATT, [0.5, 1.6, 1.0])


def test_load_plant_reads_sim_response_schema(tmp_path):
    p = tmp_path / "sim_response.json"
    p.write_text(json.dumps({
        "hover_thrust": 0.42, "k_a": 13.0,
        "rate_gain_axes": {"roll": 9.0, "pitch": 9.5, "yaw": 4.0},
    }))
    hover, k_a, rg = load_plant(str(p))
    assert hover == 0.42 and k_a == 13.0
    np.testing.assert_allclose(rg, [9.0, 9.5, 4.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigp.navigator'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/navigator.py
"""Waypoint navigation API — importable, testable wrapper around goto.py's proven control.

Pure control law (cruise_accel / settle_accel / attitude_command) is separated from the
flight loop so it can be unit-tested with synthetic states. WaypointNavigator owns the
blocking loop + IO; the caller owns lifecycle (sim reset + arm) and hands in a live Store
and Commander.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from aigp.control_math import (accel_to_thrust_norm, attitude_error_quat,
                               collective_accel, desired_attitude, mat_to_quat)
from aigp.geometry import quat_to_R

G = 9.81


@dataclass
class NavGains:
    """All goto.py tuning constants; defaults are goto.py's proven live values."""
    KP_ATT: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.6, 1.0]))
    KP_YAW: float = 3.0
    KD_YAW: float = 0.3
    KP_CT: float = 0.5
    KD_CT: float = 1.2
    AL_MAX: float = 0.5
    KD_AL: float = 1.2
    KP_Z: float = 1.8
    KD_Z: float = 3.0
    WMAX: float = 4.0
    TILT_MAX_DEG: float = 15.0
    ABORT_TILT_DEG: float = 80.0
    WP_TIMEOUT: float = 40.0
    MAX_SPEED: float = 1.2
    ARRIVE: float = 1.5
    DECEL_MAX: float = 2.0
    SETTLE_T: float = 2.5
    SETTLE_V: float = 0.4
    LOOP_DT: float = 0.004


def load_plant(path: str = "sysid/sim_response.json"):
    """Probed plant params from sysid: (hover_thrust, k_a, rate_gain_axes[roll,pitch,yaw])."""
    r = json.load(open(path))
    hover = float(r["hover_thrust"])
    k_a = float(r["k_a"])
    rg = np.array([r["rate_gain_axes"]["roll"],
                   r["rate_gain_axes"]["pitch"],
                   r["rate_gain_axes"]["yaw"]], float)
    return hover, k_a, rg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): NavGains dataclass + load_plant helper"
```

---

### Task 2: Pure guidance — `cruise_accel` + `settle_accel`

**Files:**
- Modify: `aigp/navigator.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `aigp.state.DroneState` (`pos_ned`, `vel_ned` (3,) NED), `NavGains` from Task 1.
- Produces:
  - `cruise_accel(state, leg_start, target, gains) -> (a2: np.ndarray(2,), tangent: np.ndarray(2,), speed_cmd: float)`
  - `settle_accel(state, target, gains) -> a2: np.ndarray(2,)`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_navigator.py
from aigp.state import DroneState
from aigp.navigator import cruise_accel, settle_accel


def _mkstate(pos, vel, quat=(1, 0, 0, 0), omega=(0, 0, 0)):
    return DroneState(np.array(pos, float), np.array(vel, float),
                      np.array(quat, float), np.array(omega, float), 0)


def test_cruise_accel_drives_along_leg():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])
    a2, tv, spd = cruise_accel(st, leg_start=np.array([0, 0, -2.0]),
                               target=np.array([10, 0, -2.0]), gains=g)
    np.testing.assert_allclose(tv, [1.0, 0.0], atol=1e-9)
    assert spd == g.MAX_SPEED          # 0.6 * 10 clipped to 1.2
    assert a2[0] > 0 and abs(a2[1]) < 1e-9   # accel toward target, no cross term on the line


def test_cruise_accel_corrects_cross_track():
    g = NavGains()
    st = _mkstate([0, 1, -2], [0, 0, 0])   # 1 m to +E of the N-running line
    a2, tv, spd = cruise_accel(st, leg_start=np.array([0, 0, -2.0]),
                               target=np.array([10, 0, -2.0]), gains=g)
    assert a2[1] < 0                    # accel pushes back toward the line (-E)


def test_settle_accel_damps_and_pulls():
    g = NavGains()
    st = _mkstate([1, 0, -2], [0.5, 0, 0])  # 1 m past target in +N, moving +N
    a2 = settle_accel(st, target=np.array([0, 0, -2.0]), gains=g)
    assert a2[0] < 0                    # pulls back to target AND damps +N velocity
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: FAIL with `ImportError: cannot import name 'cruise_accel'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/navigator.py
def cruise_accel(state, leg_start, target, gains):
    """Horizontal accel for a leg: along-track ramp-to-stop + cross-track line-follow.
    Returns (a2, tangent, speed_cmd). Mirrors goto.py:fly_leg's inner law."""
    pos = state.pos_ned
    vel = state.vel_ned
    leg_start = np.asarray(leg_start, float)
    target = np.asarray(target, float)
    seg = (target - leg_start)[:2]
    seglen = float(np.linalg.norm(seg))
    rel = target - pos
    if seglen > 1e-3:
        tv = seg / seglen
    else:
        tv = rel[:2] / (np.linalg.norm(rel[:2]) + 1e-9)
    lat = np.array([-tv[1], tv[0]])
    spd = float(np.clip(0.6 * float(rel[:2] @ tv), 0.0, gains.MAX_SPEED))
    a_al = float(np.clip(gains.KD_AL * (spd - float(vel[:2] @ tv)),
                         -gains.DECEL_MAX, gains.AL_MAX))
    a_ct = -gains.KP_CT * float((pos - leg_start)[:2] @ lat) - gains.KD_CT * float(vel[:2] @ lat)
    a2 = a_al * tv + a_ct * lat
    return a2, tv, spd


def settle_accel(state, target, gains):
    """Horizontal accel to kill momentum at target: pull-to-point + velocity damp.
    Mirrors goto.py:settle."""
    target = np.asarray(target, float)
    err = (target - state.pos_ned)[:2]
    return np.clip(gains.KP_CT * err, -1.0, 1.0) - gains.KD_CT * state.vel_ned[:2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): pure cruise_accel + settle_accel guidance"
```

---

### Task 3: Pure inner law — `attitude_command`

**Files:**
- Modify: `aigp/navigator.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `DroneState` (`pos_ned`, `vel_ned`, `quat_wxyz`, `omega`), `NavGains`, `plant=(hover,k_a,rg)`.
- Produces:
  - `attitude_command(state, a2, z_sp, yaw_ref, plant, gains) -> (rate_cmd_norm: np.ndarray(3,), thrust: float, tilt_deg: float, dbg: dict)`
  - `dbg` keys: `{"a", "w_des", "q_des", "thr"}` (telemetry payload for `FlightLog.push`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_navigator.py
from aigp.navigator import attitude_command

_PLANT = (0.5, 13.0, np.array([1.0, 1.0, 1.0]))   # hover, k_a, rg (rg=1 -> rate_cmd == w_des clipped)


def test_attitude_command_level_hover():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])           # level, at rest, on altitude
    rate, thr, tilt, dbg = attitude_command(st, a2=np.zeros(2), z_sp=-2.0,
                                             yaw_ref=0.0, plant=_PLANT, gains=g)
    assert abs(thr - 0.5) < 1e-6                    # collective == hover when level + no accel
    np.testing.assert_allclose(rate, [0, 0, 0], atol=1e-9)
    assert abs(tilt) < 1e-6


def test_attitude_command_clamps_tilt():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])
    _, _, _, dbg = attitude_command(st, a2=np.array([100.0, 0.0]), z_sp=-2.0,
                                    yaw_ref=0.0, plant=_PLANT, gains=g)
    tilt_max_acc = np.tan(np.radians(g.TILT_MAX_DEG)) * G
    assert abs(np.linalg.norm(dbg["a"][:2]) - tilt_max_acc) < 1e-6


def test_attitude_command_clips_rate_to_wmax():
    g = NavGains(WMAX=0.1)
    st = _mkstate([0, 0, -2], [0, 0, 0])
    rate, _, _, _ = attitude_command(st, a2=np.array([5.0, 0.0]), z_sp=-2.0,
                                     yaw_ref=0.0, plant=_PLANT, gains=g)
    assert np.all(np.abs(rate) <= 0.1 + 1e-9)


def test_attitude_command_yaw_sign():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0])            # yaw_cur = 0
    rate_pos, *_ = attitude_command(st, np.zeros(2), -2.0, yaw_ref=+0.3, plant=_PLANT, gains=g)
    rate_neg, *_ = attitude_command(st, np.zeros(2), -2.0, yaw_ref=-0.3, plant=_PLANT, gains=g)
    assert rate_pos[2] > 0 and rate_neg[2] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: FAIL with `ImportError: cannot import name 'attitude_command'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/navigator.py
def attitude_command(state, a2, z_sp, yaw_ref, plant, gains):
    """Inner law (pure goto.py:cmd): horizontal accel -> tilt-clamp -> desired attitude ->
    normalized body-rate command (clipped) + thrust. Yaw rate steers toward yaw_ref.
    Returns (rate_cmd_norm, thrust, tilt_deg, dbg)."""
    hover, k_a, rg = plant
    a = np.zeros(3)
    a[:2] = np.asarray(a2, float)
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    n = float(np.linalg.norm(a[:2]))
    if n > tilt_max_acc:
        a[:2] = a[:2] / n * tilt_max_acc
    a[2] = gains.KP_Z * (z_sp - state.pos_ned[2]) + gains.KD_Z * (0.0 - state.vel_ned[2])

    R = quat_to_R(state.quat_wxyz)
    yaw_cur = float(np.arctan2(R[1, 0], R[0, 0]))
    q_des = mat_to_quat(desired_attitude(a, yaw_cur))
    w_des = gains.KP_ATT * attitude_error_quat(state.quat_wxyz, q_des)
    w_des[2] = (gains.KP_YAW * ((yaw_ref - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
                - gains.KD_YAW * float(state.omega[2]))
    thr = accel_to_thrust_norm(collective_accel(a, state.quat_wxyz), hover, k_a)
    rate_cmd_norm = np.clip(w_des / rg, -gains.WMAX, gains.WMAX)
    tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, R[2, 2])))))
    return rate_cmd_norm, thr, tilt, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): pure attitude_command inner law"
```

---

### Task 4: `WaypointNavigator` class (origin, frame/yaw resolution, flight loop)

**Files:**
- Modify: `aigp/navigator.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `cruise_accel`, `settle_accel`, `attitude_command`, `NavGains` (Tasks 1-3); a `store` exposing
  `get_drone() -> DroneState|None` and `get_race_live() -> bool`; a `commander` exposing
  `send_attitude_target(body_rates(3,), thrust_norm)`; optional `flog` exposing
  `push(t, ds, dbg, **kw)` and `set_path(arr)`.
- Produces:
  - `WaypointNavigator(store, commander, plant, *, gains=NavGains(), flog=None)`
  - `.set_origin(pos_ned=None, yaw=None) -> None`
  - `.goto(wp, *, frame='world', yaw='hold') -> str`
  - `.follow(wps, *, frame='world', yaw='hold', settle=True) -> str`
  - `.settle(target=None, yaw_ref=None) -> None`
  - frame `'body'` = (fwd,right,down) offset from origin pose; `'world'` = absolute NED.
  - yaw `'hold'` = origin heading; `'face'` = bearing `atan2(rel_E, rel_N)` to target.

The loop methods (`goto`/`follow`/`settle`) are NOT unit-tested (they require a live sim and real
time). This task unit-tests the pure, IO-free helpers: `_resolve`, `set_origin`, `_yaw_ref`, via a fake
store. The loop is verified live in Task 5.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_navigator.py
from aigp.navigator import WaypointNavigator


class _FakeStore:
    def __init__(self, ds):
        self._ds = ds
    def get_drone(self):
        return self._ds
    def get_race_live(self):
        return True


def _nav(ds):
    return WaypointNavigator(_FakeStore(ds), commander=None, plant=_PLANT)


def test_set_origin_captures_current_state():
    nav = _nav(_mkstate([1, 2, -3], [0, 0, 0]))   # level quat -> yaw 0
    nav.set_origin()
    np.testing.assert_allclose(nav._origin_pos, [1, 2, -3])
    assert abs(nav._origin_yaw) < 1e-9


def test_resolve_world_is_absolute():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)
    np.testing.assert_allclose(nav._resolve(np.array([3, 4, -2.0]), "world"), [3, 4, -2])


def test_resolve_body_offsets_from_origin():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)   # heading 0 -> fwd=-N, right=+E
    np.testing.assert_allclose(nav._resolve(np.array([6, 0, 0.0]), "body"), [-6, 0, -2])
    np.testing.assert_allclose(nav._resolve(np.array([0, 4, 0.0]), "body"), [0, 4, -2])


def test_resolve_rejects_bad_frame():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)
    try:
        nav._resolve(np.array([1, 0, 0.0]), "polar")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_yaw_ref_face_points_at_target():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0]))
    nav.set_origin(pos_ned=np.array([0, 0, -2.0]), yaw=0.0)
    yr = nav._yaw_ref("face", np.array([0, 5, -2.0]))   # target due +E
    assert abs(yr - np.pi / 2) < 1e-9
    assert nav._yaw_ref("hold", np.array([0, 5, -2.0])) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: FAIL with `ImportError: cannot import name 'WaypointNavigator'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/navigator.py
class WaypointNavigator:
    """Blocking waypoint navigation over the proven attitude-rate control. Caller owns lifecycle
    (sim reset + arm) and passes a live store + commander. See module docstring."""

    def __init__(self, store, commander, plant, *, gains=None, flog=None):
        self.store = store
        self.commander = commander
        self.plant = plant
        self.gains = gains if gains is not None else NavGains()
        self.flog = flog
        self._origin_pos = None
        self._origin_yaw = None
        self._t0 = None

    # --- pure helpers (IO-free) ---
    def set_origin(self, pos_ned=None, yaw=None):
        """Capture the body-frame reference pose. Missing fields are read from current drone state."""
        if pos_ned is None or yaw is None:
            ds = self.store.get_drone()
            if ds is None:
                raise RuntimeError("no drone state available to capture origin")
            if pos_ned is None:
                pos_ned = ds.pos_ned.copy()
            if yaw is None:
                R = quat_to_R(ds.quat_wxyz)
                yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self._origin_pos = np.asarray(pos_ned, float)
        self._origin_yaw = float(yaw)

    def _resolve(self, wp, frame):
        wp = np.asarray(wp, float)
        if frame == "world":
            return wp
        if frame == "body":
            if self._origin_pos is None:
                self.set_origin()
            cy, sy = np.cos(self._origin_yaw), np.sin(self._origin_yaw)
            fwd = np.array([-cy, -sy])
            right = np.array([-sy, cy])
            off = np.array([wp[0] * fwd[0] + wp[1] * right[0],
                            wp[0] * fwd[1] + wp[1] * right[1],
                            wp[2]])
            return self._origin_pos + off
        raise ValueError(f"frame must be 'world' or 'body', got {frame!r}")

    def _yaw_ref(self, yaw_mode, target):
        if yaw_mode == "hold":
            return self._origin_yaw if self._origin_yaw is not None else 0.0
        if yaw_mode == "face":
            ds = self.store.get_drone()
            pos = ds.pos_ned if ds is not None else (self._origin_pos
                                                     if self._origin_pos is not None else np.zeros(3))
            rel = np.asarray(target, float) - pos
            return float(np.arctan2(rel[1], rel[0]))
        raise ValueError(f"yaw must be 'hold' or 'face', got {yaw_mode!r}")

    def _current_pos(self):
        ds = self.store.get_drone()
        if ds is not None:
            return ds.pos_ned.copy()
        return self._origin_pos.copy() if self._origin_pos is not None else np.zeros(3)

    # --- blocking flight loop (live sim; not unit-tested) ---
    def goto(self, wp, *, frame="world", yaw="hold"):
        if self._origin_pos is None:
            self.set_origin()
        if self._t0 is None:
            self._t0 = time.time()
        target = self._resolve(wp, frame)
        return self._fly_leg(target, self._current_pos(), self._yaw_ref(yaw, target))

    def follow(self, wps, *, frame="world", yaw="hold", settle=True):
        if self._origin_pos is None:
            self.set_origin()
        if self._t0 is None:
            self._t0 = time.time()
        targets = [self._resolve(w, frame) for w in wps]
        if self.flog is not None:
            self.flog.set_path(np.vstack([self._current_pos()] + targets))
        leg_start = self._current_pos()
        for i, target in enumerate(targets):
            res = self._fly_leg(target, leg_start, self._yaw_ref(yaw, target))
            print(f"   {res.upper()} wp{i}", flush=True)
            if res != "reached":
                return res
            if settle:
                self.settle(target, self._yaw_ref(yaw, target))
            leg_start = self._current_pos()
        return "reached"

    def settle(self, target=None, yaw_ref=None):
        if target is None:
            target = self._current_pos()
        target = np.asarray(target, float)
        if yaw_ref is None:
            yaw_ref = self._origin_yaw if self._origin_yaw is not None else 0.0
        if self._t0 is None:
            self._t0 = time.time()
        t0 = time.time()
        while time.time() - t0 < self.gains.SETTLE_T:
            ds = self.store.get_drone()
            if ds is not None:
                if float(np.linalg.norm(ds.vel_ned[:2])) < self.gains.SETTLE_V:
                    return
                a2 = settle_accel(ds, target, self.gains)
                rate, thr, tilt, dbg = attitude_command(ds, a2, float(target[2]),
                                                         yaw_ref, self.plant, self.gains)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=target, cruise=0.0,
                                   running=self.store.get_race_live(), armed=True)
                if tilt > self.gains.ABORT_TILT_DEG:
                    return
            time.sleep(self.gains.LOOP_DT)

    def _fly_leg(self, target, leg_start, yaw_ref):
        g = self.gains
        t_leg = time.time()
        last = -1
        while time.time() - t_leg < g.WP_TIMEOUT:
            ds = self.store.get_drone()
            if ds is not None:
                rel = target - ds.pos_ned
                if float(np.linalg.norm(rel)) < g.ARRIVE:
                    return "reached"
                a2, tv, spd = cruise_accel(ds, leg_start, target, g)
                rate, thr, tilt, dbg = attitude_command(ds, a2, float(target[2]),
                                                         yaw_ref, self.plant, g)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=target, tangent=tv,
                                   cruise=spd, running=self.store.get_race_live(), armed=True)
                if tilt > g.ABORT_TILT_DEG:
                    print(f"  ABORT tilt={tilt:.0f}", flush=True)
                    return "abort"
                k = int((time.time() - t_leg) / 2.0)
                if k != last:
                    last = k
                    print(f"  dist={float(np.linalg.norm(rel)):5.1f} "
                          f"spd={float(np.linalg.norm(ds.vel_ned[:2])):4.1f} tilt={tilt:3.0f}",
                          flush=True)
            time.sleep(g.LOOP_DT)
        return "timeout"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aigp/test_navigator.py -q`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): WaypointNavigator loop + frame/yaw resolution"
```

---

### Task 5: Rewrite `goto.py` as a thin CLI over `WaypointNavigator`

**Files:**
- Modify: `goto.py` (full rewrite)
- Test: manual live run + diff review (no unit test — requires the live VQ sim).

**Interfaces:**
- Consumes: `WaypointNavigator`, `NavGains`, `load_plant` from `aigp.navigator`; `MavlinkIO`, `VisionIO`,
  `Store`, `Commander`, `aigp.flight_telemetry` (lifecycle + IO, unchanged from current goto.py).
- Produces: same CLI as before — `python goto.py [body|world] f r d  f r d ...`, `--no-viz`, `--rrd PATH`.
- Behavior parity requirement: `world` triples stay offsets-from-spawn (the CLI converts to absolute
  before calling `nav.follow(..., frame='world')`); `body` triples pass through as `frame='body'`.

- [ ] **Step 1: Rewrite goto.py**

```python
"""GOTO — high-level waypoint navigation CLI over the WaypointNavigator API.

Thin wrapper: does lifecycle (fresh_start + arm), then hands a live store + commander to
aigp.navigator.WaypointNavigator and calls follow(). The proven control loop now lives in
aigp/navigator.py (unit-tested); this file is just the CLI + sim lifecycle.

Usage (live dashboard ON by default -> --no-viz to disable, --rrd <path> to record):
  python goto.py                       # safe default: a small box out front (body frame)
  python goto.py body 6 0 0  6 4 0     # body-relative triples: (fwd, right, down) from spawn heading
  python goto.py world 8 0 0  0 8 0    # world-NED triples: (N, E, D) offsets from spawn

STATUS (live-tested on the VQ sim): single-waypoint go-to flies cleanly. Multi-waypoint paths with
sharp 90-deg turns / precise arrivals are MARGINAL — the weathervane instability fights stop-and-turn.
This platform wants to flow forward; precise waypoint following is better served by the RL policy.
"""
import sys
import time

import numpy as np
from pymavlink import mavutil

import aigp.flight_telemetry as ftm
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.navigator import WaypointNavigator, load_plant
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


def parse_args(argv):
    body = True
    if argv and argv[0] in ("body", "world"):
        body = (argv[0] == "body"); argv = argv[1:]
    if len(argv) >= 3:
        f = [float(x) for x in argv]
        return body, [tuple(f[i:i + 3]) for i in range(0, len(f) - len(f) % 3, 3)]
    return True, [(6.0, 0.0, 0.0), (6.0, 4.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 0.0)]


def main():
    body, wps = parse_args(ftm.strip_viz_args(sys.argv[1:]))
    s = Store()
    m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
    boot = int(time.time() * 1000)
    c = Commander(m.conn, boot)
    plant = load_plant("sysid/sim_response.json")
    assert fresh_start(s, c, m, boot), "not live"

    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
    R0 = quat_to_R(ds0.quat_wxyz)
    yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))

    # 'world' triples preserve goto.py's historical semantics (offsets from spawn) -> convert to
    # absolute NED here and call frame='world'; 'body' triples pass straight through.
    if body:
        targets = wps
        frame = "body"
    else:
        targets = [tuple((spawn + np.array(o, float)).tolist()) for o in wps]
        frame = "world"

    print(f"GOTO {len(wps)} waypoints ({'body fwd/right/down' if body else 'world NED'}): {wps}",
          flush=True)

    # real rate_gain (plant[2]) so FlightLog reconstructs rad/s for display (parity with old goto.py);
    # store=s streams the COLLISION flag too (dashboard hard rule).
    flog = ftm.from_args(sys.argv, plant[2], "goto", store=s)
    nav = WaypointNavigator(s, c, plant, flog=flog)
    nav.set_origin(pos_ned=spawn, yaw=yaw0)
    c.arm()
    res = nav.follow(targets, frame=frame, yaw="hold", settle=True)
    if flog is not None:
        flog.close()
    print(f"mission {res}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the rewrite imports cleanly and the test suite still passes**

Run: `python -c "import ast; ast.parse(open('goto.py').read()); print('goto.py parses')"`
Expected: `goto.py parses`

Run: `pytest tests/test_aigp/ -q`
Expected: PASS (all existing tests + the 14 navigator tests)

- [ ] **Step 3: Confirm behavior parity by reading the diff**

Run: `git diff origin/aigp-control-waypoints -- goto.py`
Confirm: the control constants moved verbatim into `NavGains` defaults (Task 1); the loop ordering
(command sent before telemetry) is preserved; `fresh_start`/`idle`/`parse_args` logic is byte-identical
except for being passed `s`/`c`/`m`/`boot` instead of using module globals; `world` triples are still
added to spawn before flying.

- [ ] **Step 4: Live smoke test (requires the VQ sim running — operator step)**

Run (on the laptop, sim up, dashboard on the Mac): `python goto.py body 6 0 0`
Expected: prints `GOTO 1 waypoints ...`, drone flies forward ~6 m, `REACHED wp0`, `mission reached`,
tilt stays low (~2°), telemetry streams to the Rerun dashboard. This reproduces goto.py's proven
single-waypoint result. Append a row to the AI-GP Experiment Log per project rules.

- [ ] **Step 5: Commit**

```bash
git add goto.py
git commit -m "refactor(goto): thin CLI over WaypointNavigator (control loop -> aigp/navigator.py)"
```

---

## Self-Review

**1. Spec coverage:**
- Pure law split (cruise/settle/attitude) → Tasks 2, 3. ✓
- `NavGains` + `load_plant` → Task 1. ✓
- `WaypointNavigator` surface (set_origin/goto/follow/settle, status strings) → Task 4. ✓
- World + body frames, hold/face yaw → Task 4 (`_resolve`, `_yaw_ref`). ✓
- Telemetry hook (optional flog, guarded, push after send, set_path) → Task 4 loop + Task 5 CLI wiring. ✓
- goto.py thin-CLI rewrite + parity verification → Task 5. ✓
- Out-of-scope items (async, server, takeoff/land, sign auto-cal) → not implemented (correct). ✓

**2. Placeholder scan:** no TBD/TODO/"add error handling"/"similar to Task N"; every code step shows full code. ✓

**3. Type consistency:** `cruise_accel` returns `(a2, tangent, speed_cmd)` and the loop consumes all three; `attitude_command` returns `(rate_cmd_norm, thrust, tilt_deg, dbg)` consumed identically in `_fly_leg`/`settle`; `plant=(hover,k_a,rg)` produced by `load_plant`, consumed by `attitude_command`; `NavGains` field names match between definition (Task 1) and all uses (Tasks 2-4). ✓

## Known limitations (carried from the spec)

- No lateral-sign auto-calibration: multi-waypoint sharp-turn paths remain marginal (goto.py's documented STATUS). A future enhancement could fold in `vq_waypoint2.py`'s `s_lat`/`s_yawb` calibration.
- `yaw='face'` points the **nose** (body-x) at the target. Camera-forward is `-body_x`; if vision needs the camera aimed at the target, add π to the `'face'` bearing — verify live before relying on it.
- The loop methods are integration-tested live (Task 5 Step 4), not unit-tested.
