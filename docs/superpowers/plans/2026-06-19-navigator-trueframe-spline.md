# Navigator True-Frame Port + Smooth Spline Trajectories — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `WaypointNavigator` fly smooth spline trajectories (default) and strafe correctly via the proven true-frame chain, with `course`/`fixed`/`lookat` camera modes.

**Architecture:** Reuse `GateTrajectory` (cubic spline + speed schedule) as the path engine; port `traj_track`'s drone-locked tracking loop for the desired accel; `course` uses the existing raw-frame `attitude_command` (nose-forward is raw-safe), `fixed`/`lookat` use a new true-frame `attitude_command_tf` (qfix + s_cam + WFIX). Keep the existing point-to-point loop as the `legs` fallback.

**Tech Stack:** Python 3.9, numpy, scipy (GateTrajectory), pymavlink (CLI only), pytest. No new deps.

## Global Constraints

- **Reuse, don't re-derive frames** (canonical rule): port `vq_waypoint2`'s true-frame block and `traj_track`'s loop verbatim; do not invent frame math.
- **course = raw frame** (existing `attitude_command`, unchanged + its 15 tests stay green). **fixed/lookat = true-frame** (`attitude_command_tf`). The strafe runaway lives in the raw frame; nose-forward is raw-safe.
- **Caller owns lifecycle**; navigator takes a live `Store` + `Commander`. Status strings `'reached'|'timeout'|'abort'`.
- **Telemetry** via injected `flog`, guarded (None = no viz); push AFTER send. Dashboard default on in `goto.py`.
- Camera = −body_x; camera-pointing yaw adds π. `WFIX = [1,-1,1]`. `qfix(q) = q[[1,2,3,0]]`.
- Plant `(hover, k_a, rg)` from `sysid/sim_response.json`. No `python` on PATH — tests run with `python3 -m pytest tests/test_aigp/ -q` (Python 3.9.6).
- **Live validation requires a fresh sim each flight** — the VQ sim wedges (race_live can read True with ODOMETRY frozen at |v|≈17/const pos); a full DCGame restart is needed. Laptop worktree: `C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-wp`; python `C:\Users\alexj\miniconda3\envs\aigp\python.exe`; fetch from `git@github.com:corvidx-drone-grand-prix/algo_src-alex.git aigp-control-waypoints`.

---

### Task 1: Lock the spline engine — pure tests for `GateTrajectory`

**Files:**
- Test: `tests/test_aigp/test_gate_traj.py` (Create)

**Interfaces:**
- Consumes: `gate_traj.GateTrajectory(gates, v_cruise=, tilt_budget_deg=, c_drag=, margin=)`; `.sample(s) -> {pos,tang,kappa,v,yaw,yaw_rate,a_lat}`; `.nearest_s(pos)`; `.s_max`; `.speed_at(kappa_abs)`.
- Produces: confidence the reused engine behaves (no code change to `gate_traj.py`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aigp/test_gate_traj.py
import numpy as np
from gate_traj import GateTrajectory


def _straight():
    # straight line along +N at constant z
    gates = np.array([[0, 0, -2.0], [10, 0, -2.0], [20, 0, -2.0]])
    return GateTrajectory(gates, v_cruise=2.5)


def test_straight_tangent_and_cruise():
    t = _straight()
    r = t.sample(t.s_max * 0.5)
    np.testing.assert_allclose(r["tang"][:2], [1.0, 0.0], atol=1e-3)   # points +N
    assert abs(r["kappa"]) < 1e-3                                       # straight
    assert abs(r["v"] - 2.5) < 1e-6                                     # full cruise on a straight


def test_sample_yaw_is_camera_forward():
    # body_x opposite travel (camera-forward): yaw = atan2(-tang)
    t = _straight()
    r = t.sample(t.s_max * 0.5)
    assert abs(((r["yaw"] - np.arctan2(-r["tang"][1], -r["tang"][0]) + np.pi) % (2*np.pi)) - np.pi) < 1e-6


def test_nearest_s_projects_along_path():
    t = _straight()
    s_near = t.nearest_s(np.array([5.0, 0.5, -2.0]))   # next to the line at ~5 m along
    assert 3.0 < s_near < 7.0


def test_curve_lowers_scheduled_speed():
    straight = _straight().sample(0.0)["v"]
    curve = GateTrajectory(np.array([[0,0,-2.0],[10,0,-2.0],[14,8,-2.0],[6,12,-2.0]]),
                           v_cruise=8.0, phi_max_deg=35.0)
    vmin = min(curve.sample(s)["v"] for s in np.linspace(0, curve.s_max, 40))
    assert vmin < 8.0                                   # curvature caps speed below cruise
```

- [ ] **Step 2: Run to verify they pass (no impl needed — engine exists)**

Run: `python3 -m pytest tests/test_aigp/test_gate_traj.py -q`
Expected: PASS (4 passed). If any fail, STOP and report — the engine behaves differently than assumed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_aigp/test_gate_traj.py
git commit -m "test(gate_traj): lock GateTrajectory behavior before reuse"
```

---

### Task 2: `qfix`/`WFIX` helpers + `s_cam`/`yaw0_t` in `set_origin`

**Files:**
- Modify: `aigp/navigator.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `aigp.geometry.quat_to_R`; existing `WaypointNavigator`, `NavGains`.
- Produces: module-level `WFIX = np.array([1.,-1.,1.])`, `_qfix(q) -> ndarray(4,)`; `WaypointNavigator.set_origin` also sets `self._s_cam` (±1.0) and `self._yaw0_t` (true-cam spawn yaw). `NavGains.KD_ATT = 0.3`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_navigator.py
from aigp.navigator import _qfix, WFIX


def test_qfix_shuffles_wxyz_to_xyzw():
    np.testing.assert_allclose(_qfix([0.7, 0.1, 0.2, 0.3]), [0.1, 0.2, 0.3, 0.7])


def test_wfix_is_pitch_mirror():
    np.testing.assert_allclose(WFIX, [1.0, -1.0, 1.0])


def test_set_origin_computes_scam_and_yaw0t():
    # spawn quat in sim-wire convention: identity true attitude is wire [0,0,0,1]
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=(0, 0, 0, 1)))
    nav.set_origin()
    assert nav._s_cam in (1.0, -1.0)
    assert np.isfinite(nav._yaw0_t)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -k "qfix or wfix or scam" -q`
Expected: FAIL with `ImportError: cannot import name '_qfix'`

- [ ] **Step 3: Implement**

In `aigp/navigator.py`, after the imports add:

```python
WFIX = np.array([1.0, -1.0, 1.0])   # sim rate-frame mirror (pitch axis), proven (fly_gate3/vq_waypoint2)


def _qfix(q):
    """Live sim-wire quat (wxyz) -> TRUE attitude quat (wxyz): q_true = q_wire[[1,2,3,0]]."""
    q = np.asarray(q, float)
    return q[[1, 2, 3, 0]]
```

Add to `NavGains` (after `KP_ATT`/yaw gains):

```python
    KD_ATT: float = 0.3
```

Extend `set_origin` — after it sets `self._origin_pos`/`self._origin_yaw`, append:

```python
        # true-frame calibration (verbatim from vq_waypoint2): camera-forward sign + true-cam spawn yaw
        ds = self.store.get_drone()
        if ds is not None:
            yaw0 = self._origin_yaw
            q_t0 = _qfix(ds.quat_wxyz)
            R_t0 = quat_to_R(q_t0)
            cam_live = -np.array([np.cos(yaw0), np.sin(yaw0)])
            self._s_cam = 1.0 if float(R_t0[:2, 0] @ cam_live) > 0 else -1.0
            self._yaw0_t = float(np.arctan2(self._s_cam * R_t0[1, 0], self._s_cam * R_t0[0, 0]))
        else:
            self._s_cam = 1.0
            self._yaw0_t = float(self._origin_yaw or 0.0)
```

Add `self._s_cam = 1.0` and `self._yaw0_t = 0.0` to `__init__` (defaults).

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -q`
Expected: PASS (existing 15 + 3 new = 18 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): qfix/WFIX helpers + s_cam/yaw0_t true-frame calibration"
```

---

### Task 3: True-frame strafe attitude — `attitude_command_tf`

**Files:**
- Modify: `aigp/navigator.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `_qfix`, `WFIX`, `quat_to_R`, control_math helpers, `NavGains` (with `KD_ATT`), plant `(hover,k_a,rg)`.
- Produces: `attitude_command_tf(state, a2, z_sp, yaw_ref, plant, gains, s_cam, ymirror=False) -> (rate_cmd_norm(3,), thrust, tilt_deg, dbg)`. `yaw_ref` is the desired NOSE heading in the **true-cam** frame; `a2` is desired horizontal accel in live-world (pos_ned). `ymirror` applies the world-y flip (default False; a live-validation toggle — see Task 5).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_navigator.py
from aigp.navigator import attitude_command_tf

# sim-wire quat whose TRUE attitude (qfix) is level + nose along +N: true wxyz [1,0,0,0] -> wire [0,0,0,1]
_WIRE_LEVEL = (0.0, 0.0, 0.0, 1.0)


def test_tf_level_hover_zero_rate():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)
    rate, thr, tilt, dbg = attitude_command_tf(st, np.zeros(2), -2.0, yaw_ref=0.0,
                                               plant=_PLANT, gains=g, s_cam=1.0)
    assert abs(thr - 0.5) < 1e-6 and abs(tilt) < 1e-6
    np.testing.assert_allclose(rate, [0, 0, 0], atol=1e-6)


def test_tf_clamps_tilt():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)
    _, _, _, dbg = attitude_command_tf(st, np.array([100.0, 0.0]), -2.0, 0.0, _PLANT, g, 1.0)
    assert abs(np.linalg.norm(dbg["a"][:2]) - np.tan(np.radians(g.TILT_MAX_DEG)) * 9.81) < 1e-6


def test_tf_yaw_sign_uses_truecam():
    g = NavGains()
    st = _mkstate([0, 0, -2], [0, 0, 0], quat=_WIRE_LEVEL)   # true-cam yaw_cur = 0
    rp, *_ = attitude_command_tf(st, np.zeros(2), -2.0, yaw_ref=+0.3, plant=_PLANT, gains=g, s_cam=1.0)
    rn, *_ = attitude_command_tf(st, np.zeros(2), -2.0, yaw_ref=-0.3, plant=_PLANT, gains=g, s_cam=1.0)
    # WFIX mirrors the yaw axis sign in the wire frame; assert the two are opposite + nonzero
    assert rp[2] * rn[2] < 0 and abs(rp[2]) > 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -k tf -q`
Expected: FAIL with `ImportError: cannot import name 'attitude_command_tf'`

- [ ] **Step 3: Implement** (verbatim port of `vq_waypoint2.py` lines 244-258)

```python
# append to aigp/navigator.py
def attitude_command_tf(state, a2, z_sp, yaw_ref, plant, gains, s_cam, ymirror=False):
    """TRUE-frame attitude for STRAFING (camera not along travel). Port of vq_waypoint2's
    true-frame block: qfix attitude, s_cam yaw basis, WFIX rate mirror, tilt-conditional
    collective cap. a2 = desired horizontal accel in live-world (pos_ned); yaw_ref = desired
    NOSE heading in the true-cam frame. Returns (rate_cmd_norm, thrust, tilt_deg, dbg)."""
    hover, k_a, rg = plant
    a = np.zeros(3)
    a[:2] = np.asarray(a2, float)
    a[2] = gains.KP_Z * (z_sp - state.pos_ned[2]) + gains.KD_Z * (0.0 - state.vel_ned[2])
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    n = float(np.linalg.norm(a[:2]))
    if n > tilt_max_acc:
        a[:2] = a[:2] / n * tilt_max_acc
    q_t = _qfix(state.quat_wxyz)
    R_t = quat_to_R(q_t)
    tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, R_t[2, 2])))))
    yaw_cur = float(np.arctan2(s_cam * R_t[1, 0], s_cam * R_t[0, 0]))
    yaw_body = yaw_ref if s_cam > 0 else yaw_ref + np.pi
    if ymirror:
        a[1] = -a[1]
    q_des = mat_to_quat(desired_attitude(a, yaw_body))
    om_t = np.asarray(state.omega, float) * WFIX
    w = gains.KP_ATT * attitude_error_quat(q_t, q_des)
    w[0] = float(np.clip(w[0] - gains.KD_ATT * om_t[0], -2.0, 2.0))
    w[1] = float(np.clip(w[1] - gains.KD_ATT * om_t[1], -2.0, 2.0))
    w[2] = float(np.clip(gains.KP_YAW * ((yaw_body - yaw_cur + np.pi) % (2 * np.pi) - np.pi)
                         - gains.KD_YAW * om_t[2], -1.5, 1.5))
    w = w * WFIX
    c_max = 10.0 if tilt > 40.0 else 18.0
    thr = accel_to_thrust_norm(min(collective_accel(a, q_t), c_max), hover, k_a)
    rate = np.clip(w / rg, -gains.WMAX, gains.WMAX)
    return rate, thr, tilt, {"a": a, "w_des": w, "q_des": q_des, "thr": thr}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -q`
Expected: PASS (18 + 3 = 21 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/navigator.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): true-frame strafe attitude_command_tf (qfix+s_cam+WFIX)"
```

---

### Task 4: Spline tracking + `course` mode (smooth) + `goto.py` CLI; LIVE-validate course

**Files:**
- Modify: `aigp/navigator.py`, `goto.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `GateTrajectory` (Task 1), `attitude_command` (existing raw), Task 2/3 helpers.
- Produces:
  - `WaypointNavigator.follow(wps, *, yaw='course', look_point=None, v_cruise=2.5, engine='spline', frame='body') -> str`
  - `WaypointNavigator.goto(wp, *, yaw='course', look_point=None, v_cruise=2.5, engine='spline', frame='body') -> str`
  - `engine='legs'` routes to the existing point-to-point loop (now renamed `_follow_legs`); `engine='spline'` routes to new `_follow_spline`.
  - new `_spline_accel(ds, ref, gains) -> (a2(2,), travel(2,))` (pure, testable): the `traj_track` cross/along law.

- [ ] **Step 1: Write the failing test** (pure `_spline_accel`)

```python
# append to tests/test_aigp/test_navigator.py
from aigp.navigator import _spline_accel


def test_spline_accel_corrects_cross_track_and_holds_speed():
    g = NavGains()
    # reference: on the +N line at the origin, tangent +N, scheduled speed 2.0
    ref = {"pos": np.array([0.0, 0.0, -2.0]), "tang": np.array([1.0, 0.0, 0.0]), "v": 2.0}
    st = _mkstate([0.0, 1.0, -2.0], [0.0, 0.0, 0.0])      # 1 m to +E of the line, at rest
    a2, travel = _spline_accel(st, ref, g)
    np.testing.assert_allclose(travel, [1.0, 0.0], atol=1e-9)
    assert a2[1] < 0      # cross-track accel pushes back toward the line (-E)
    assert a2[0] > 0      # along-track accel builds toward scheduled speed
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -k spline_accel -q`
Expected: FAIL with `ImportError: cannot import name '_spline_accel'`

- [ ] **Step 3: Implement** (`_spline_accel` ports `traj_track` lines 143-158; the loop wires GateTrajectory + raw attitude for course)

```python
# add near the other pure functions in aigp/navigator.py
from gate_traj import GateTrajectory   # add to imports at top

def _spline_accel(state, ref, gains):
    """traj_track's cross/along law to the spline reference. Returns (a2 horiz, travel unit)."""
    tang = np.asarray(ref["tang"], float)[:2]
    travel = tang / max(float(np.linalg.norm(tang)), 1e-9)
    lat_hat = np.array([-travel[1], travel[0]])
    d = (state.pos_ned - np.asarray(ref["pos"], float))[:2]
    v_al = float(state.vel_ned[:2] @ travel)
    v_ct = float(state.vel_ned[:2] @ lat_hat)
    p_ct = float(d @ lat_hat)
    tilt_max_acc = np.tan(np.radians(gains.TILT_MAX_DEG)) * G
    a_al = float(np.clip(gains.KD_AL * (ref["v"] - v_al), -4.0, gains.AL_MAX))
    a_ct_max = max(0.0, tilt_max_acc * gains.MARGIN - gains.C_DRAG * v_al * v_al)
    a_ct = float(np.clip(-gains.KP_CT * p_ct - gains.KD_CT * v_ct, -a_ct_max, a_ct_max))
    return a_al * travel + a_ct * lat_hat, travel
```

Add to `NavGains`: `AL_MAX: float = 4.0` (overridden by tilt budget at call sites is unnecessary — keep simple), `MARGIN: float = 0.6`, `C_DRAG: float = 0.057`. (Note: `AL_MAX` here is the spline along-track cap; the planner's `v(s)` already bounds speed. Keep `TILT_MAX_DEG=15.0` as-is — the spline planner uses its own `tilt_budget_deg`.)

Rename the existing loop body method `_fly_leg`-driven `follow` path to `_follow_legs` (keep its logic), then add the dispatcher + spline loop:

```python
    def follow(self, wps, *, yaw="course", look_point=None, v_cruise=2.5,
               engine="spline", frame="body"):
        if self._origin_pos is None:
            self.set_origin()
        if self._t0 is None:
            self._t0 = time.time()
        if look_point is not None:
            self._look_point = self._resolve(look_point, frame)
        targets = [self._resolve(w, frame) for w in wps]
        if engine == "legs":
            return self._follow_legs(targets, yaw=yaw, settle=True)
        return self._follow_spline(targets, yaw=yaw, v_cruise=v_cruise)

    def _follow_spline(self, targets, *, yaw, v_cruise):
        g = self.gains
        gates = np.vstack([self._origin_pos] + [np.asarray(t, float) for t in targets])
        traj = GateTrajectory(gates, v_cruise=v_cruise, tilt_budget_deg=25.0,
                              c_drag=g.C_DRAG, margin=g.MARGIN)
        if self.flog is not None:
            self.flog.set_path(traj._P)
        t_run = time.time()
        last = -1
        while time.time() - t_run < g.WP_TIMEOUT * max(2, len(targets)):
            ds = self.store.get_drone()
            if ds is not None:
                s_drone = traj.nearest_s(ds.pos_ned)
                ref = traj.sample(s_drone)
                a2, travel = _spline_accel(ds, ref, g)
                if yaw == "course":
                    rate, thr, tilt, dbg = attitude_command(ds, a2, float(ref["pos"][2]),
                                                            ref["yaw"], self.plant, g)
                else:
                    yref = self._yaw_ref_tf(yaw, ds)
                    rate, thr, tilt, dbg = attitude_command_tf(ds, a2, float(ref["pos"][2]),
                                                               yref, self.plant, g, self._s_cam,
                                                               ymirror=self._tf_ymirror)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=ref["pos"],
                                   tangent=travel, cruise=ref["v"],
                                   running=self.store.get_race_live(), armed=True)
                if tilt > g.ABORT_TILT_DEG:
                    print(f"  ABORT tilt={tilt:.0f}", flush=True)
                    return "abort"
                if s_drone >= traj.s_max - g.ARRIVE:
                    print("  REACHED end", flush=True)
                    return "reached"
                k = int((time.time() - t_run) / 1.0)
                if k != last:
                    last = k
                    print(f"  s={s_drone:5.1f}/{traj.s_max:.0f} v={float(np.linalg.norm(ds.vel_ned[:2])):4.1f}"
                          f"/{ref['v']:.1f} tilt={tilt:3.0f}", flush=True)
            time.sleep(g.LOOP_DT)
        return "timeout"
```

Add to `__init__`: `self._tf_ymirror = False`. Add stub `_yaw_ref_tf` (filled in Task 5/6):

```python
    def _yaw_ref_tf(self, yaw, ds):
        """NOSE heading in the true-cam frame for strafe modes. course handled in the raw path."""
        if yaw == "fixed":
            return self._yaw0_t
        raise NotImplementedError(yaw)   # 'lookat' added in Task 6
```

Update `goto()` similarly (dispatch to `_follow_spline([target])` for `engine='spline'`, else `_fly_leg`). Keep `_follow_legs`/`_fly_leg`/`settle`/`_align_yaw` as the legs engine.

In `goto.py`: extend `parse_opts` with `--vcruise V` (default 2.5), `--legs` (engine flag), and pass through; default `--yaw course`. Build `NavGains` as before; call `nav.follow(targets, yaw=opts["yaw"], look_point=look, v_cruise=opts["vcruise"], engine=("legs" if opts["legs"] else "spline"), frame=frame)`.

- [ ] **Step 4: Run unit tests + goto.py parse**

Run: `python3 -m pytest tests/test_aigp/ -q` → expect all pass (22 navigator + 4 gate_traj + rest).
Run: `python3 -c "import ast; ast.parse(open('goto.py').read()); ast.parse(open('aigp/navigator.py').read()); print('parse OK')"`

- [ ] **Step 5: LIVE-validate course (operator step — needs a freshly restarted sim)**

On the laptop worktree, fetch + run:
```
python goto.py body 12 0 0 12 12 0 0 12 0 0 0 0 --yaw course --vcruise 3
```
Expected: drone flies a smooth, corner-rounded path through the 4 corners nose-forward (camera follows tangent), low tilt, `REACHED end`. Streams to Mac Rerun. Confirm SMOOTH (no stop-and-go) vs the old legs run. Append an AI-GP Experiment Log row.

- [ ] **Step 6: Commit**

```bash
git add aigp/navigator.py goto.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): spline tracking + course mode (smooth) + goto.py engine/vcruise"
```

---

### Task 5: `fixed` strafe mode (true-frame + s_lat probe); LIVE-validate fixed square

**Files:**
- Modify: `aigp/navigator.py`, `goto.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `_follow_spline`, `attitude_command_tf`, `_yaw_ref_tf` (Task 4).
- Produces: `fixed` mode flies the spline with a held true-cam heading; an `s_lat` sign probe at motion start; `--heading DEG` CLI option; live-confirmed fixed-heading square.

- [ ] **Step 1: Write the failing test** (`_yaw_ref_tf` fixed = constant yaw0_t; `--heading` override)

```python
# append to tests/test_aigp/test_navigator.py
def test_yaw_ref_tf_fixed_is_constant_yaw0t():
    nav = _nav(_mkstate([0, 0, -2], [5, 0, 0], quat=(0, 0, 0, 1)))
    nav.set_origin()
    ds = nav.store.get_drone()
    assert nav._yaw_ref_tf("fixed", ds) == nav._yaw0_t
```

- [ ] **Step 2: Run to verify it fails / passes**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -k yaw_ref_tf_fixed -q`
Expected: PASS if Task 4's `_yaw_ref_tf` already returns `self._yaw0_t` for `fixed` (it does). If `--heading` support is added, extend below.

- [ ] **Step 3: Implement the s_lat probe + heading override**

Add to `__init__`: `self._s_lat = 0.0` (0 = unprobed). Add a probe run at the start of `_follow_spline` (only when `yaw != 'course'`):

```python
    def _probe_s_lat(self, traj):
        """Command a fixed lateral push for ~1.5 s; lock s_lat from the achieved world-lateral velocity
        sign (verbatim idea from vq_waypoint2). No-op if already locked."""
        if self._s_lat != 0.0:
            return
        ds0 = self.store.get_drone()
        ref0 = traj.sample(traj.nearest_s(ds0.pos_ned))
        tang = ref0["tang"][:2]; travel = tang / max(np.linalg.norm(tang), 1e-9)
        lat_hat = np.array([-travel[1], travel[0]])
        v0 = float(ds0.vel_ned[:2] @ lat_hat)
        t0 = time.time()
        while time.time() - t0 < 1.5:
            ds = self.store.get_drone()
            if ds is not None:
                a2 = 1.2 * lat_hat
                yref = self._yaw_ref_tf(self._yaw_mode, ds)
                rate, thr, tilt, dbg = attitude_command_tf(ds, a2, float(ref0["pos"][2]),
                                                           yref, self.plant, self.gains, self._s_cam,
                                                           ymirror=self._tf_ymirror)
                self.commander.send_attitude_target(rate, thr)
                if self.flog is not None:
                    self.flog.push(time.time() - self._t0, ds, dbg, nearest=ref0["pos"],
                                   cruise=0.0, running=self.store.get_race_live(), armed=True)
                if tilt > self.gains.ABORT_TILT_DEG:
                    return
            time.sleep(self.gains.LOOP_DT)
        ds = self.store.get_drone()
        v1 = float(ds.vel_ned[:2] @ lat_hat) if ds is not None else v0
        self._s_lat = 1.0 if (v1 - v0) > 0 else -1.0
        print(f"  s_lat locked: {self._s_lat:+.0f}", flush=True)
```

Apply `s_lat` to the strafe lateral command: in `_follow_spline`, for non-course modes multiply the cross/lateral part by `self._s_lat`. Simplest correct hook — scale `a2`'s lateral component:

```python
                if yaw != "course":
                    # decompose a2 onto (travel, lat); apply the calibrated lateral sign
                    lat_hat = np.array([-travel[1], travel[0]])
                    a_al = float(a2 @ travel); a_ct = float(a2 @ lat_hat) * (self._s_lat or 1.0)
                    a2 = a_al * travel + a_ct * lat_hat
```

Set `self._yaw_mode = yaw` at the top of `_follow_spline`; call `self._probe_s_lat(traj)` before the main loop when `yaw != 'course'`. Add `--heading DEG` to `goto.py` → if set, override `nav._yaw0_t` after `set_origin` (convert: `_yaw0_t` is true-cam; document that `--heading` is interpreted as a true-cam offset for now — full world-heading mapping is a follow-up).

- [ ] **Step 4: Run unit tests**

Run: `python3 -m pytest tests/test_aigp/ -q`
Expected: all pass.

- [ ] **Step 5: LIVE-validate fixed-heading square (operator step — fresh sim)**

```
python goto.py body 12 0 0 12 12 0 0 12 0 0 0 0 --yaw fixed --vcruise 2.5
```
Expected: 4/4 corners, fixed heading (camera does not rotate), strafe legs tracked (max tilt ≲ 20°), smoother than `vq_waypoint2 --square`. **If it runs away on a strafe leg**, the world-y mirror is needed: set `self._tf_ymirror = True` (one-line) and re-fly; that is the documented frame toggle. Cross-check against the working `vq_waypoint2 --square` (legs/reference). Append an Experiment Log row.

- [ ] **Step 6: Commit**

```bash
git add aigp/navigator.py goto.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): fixed-heading strafe over spline (true-frame + s_lat probe)"
```

---

### Task 6: `lookat` (camera about a point) + `goto.py --lookat`; LIVE-validate

**Files:**
- Modify: `aigp/navigator.py`, `goto.py`
- Test: `tests/test_aigp/test_navigator.py`

**Interfaces:**
- Consumes: `_yaw_ref_tf`, `_follow_spline`, `self._look_point`.
- Produces: `lookat` mode — camera (−body_x) steered to hold `self._look_point` while flying the spline.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_navigator.py
def test_yaw_ref_tf_lookat_points_camera_at_point():
    nav = _nav(_mkstate([0, 0, -2], [0, 0, 0], quat=(0, 0, 0, 1)))
    nav.set_origin()
    nav._look_point = np.array([0.0, 5.0, -2.0])    # 5 m to +E
    ds = nav.store.get_drone()
    yref = nav._yaw_ref_tf("lookat", ds)
    # camera (-body_x) should point at +E bearing pi/2 -> nose (body_x) heading = bearing + pi (true-cam)
    bearing = np.arctan2(5.0, 0.0)
    assert abs(((yref - (bearing + np.pi) + np.pi) % (2*np.pi)) - np.pi) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_aigp/test_navigator.py -k lookat -q`
Expected: FAIL with `NotImplementedError: lookat`

- [ ] **Step 3: Implement `lookat` in `_yaw_ref_tf`**

```python
    def _yaw_ref_tf(self, yaw, ds):
        if yaw == "fixed":
            return self._yaw0_t
        if yaw == "lookat":
            pos = ds.pos_ned if ds is not None else self._origin_pos
            pt = self._look_point if self._look_point is not None else self._origin_pos
            rel = np.asarray(pt, float) - pos
            if np.linalg.norm(rel[:2]) < 1.0:           # degenerate: hold true-cam heading
                R_t = quat_to_R(_qfix(ds.quat_wxyz))
                return float(np.arctan2(self._s_cam * R_t[1, 0], self._s_cam * R_t[0, 0]))
            return float(np.arctan2(rel[1], rel[0]) + np.pi)   # camera (-body_x) on the point
        raise ValueError(f"yaw must be 'course'|'fixed'|'lookat', got {yaw!r}")
```

`goto.py`: `--lookat X Y Z` already parsed (Task earlier); ensure it routes to `look_point` resolved in `frame`, and `--yaw lookat`.

- [ ] **Step 4: Run unit tests**

Run: `python3 -m pytest tests/test_aigp/ -q`
Expected: all pass.

- [ ] **Step 5: LIVE-validate lookat (operator step — fresh sim)**

```
python goto.py body 12 0 0 12 12 0 0 12 0 0 0 0 --yaw lookat 6 6 0 --vcruise 2.5
```
Expected: drone flies the square while the camera stays pointed at the body-frame point (6,6,0) (the square's center) — i.e., it circles the point with the camera locked on it. Tilt bounded; if a strafe leg diverges, the same `_tf_ymirror` / `s_lat` calibration applies. Append an Experiment Log row.

- [ ] **Step 6: Commit**

```bash
git add aigp/navigator.py goto.py tests/test_aigp/test_navigator.py
git commit -m "feat(navigator): lookat camera-about-a-point mode over the spline"
```

---

## Self-Review

**1. Spec coverage:**
- Smooth spline path engine (GateTrajectory + drone-locked tracking) → Tasks 1, 4. ✓
- True-frame strafe attitude (qfix+s_cam+WFIX+s_lat) → Tasks 2, 3, 5. ✓
- Camera modes course/fixed/lookat → Tasks 4, 5, 6. ✓
- Legs fallback retained (`engine='legs'`) → Task 4 dispatcher. ✓
- course uses raw attitude (existing tests preserved); strafe uses tf → Tasks 3, 4. ✓
- goto.py CLI (`--yaw`, `--lookat`, `--heading`, `--vcruise`, `--legs`) → Tasks 4, 5, 6. ✓
- Validation order course→fixed→lookat, fresh sim per flight → Tasks 4/5/6 Step 5. ✓

**2. Placeholder scan:** code given for every step; the one empirical unknown (world-y mirror in the strafe path) is a concrete default (`ymirror=False`) with an exact one-line live toggle documented in Task 5 Step 5 — not a placeholder.

**3. Type consistency:** `attitude_command_tf(...ns..., s_cam, ymirror=False)` signature consistent across Tasks 3/4/5; `_spline_accel -> (a2, travel)` consumed in Task 4; `_yaw_ref_tf(yaw, ds)` defined Task 4 (stub) → extended Tasks 5/6; `NavGains` fields (`KD_ATT`, `MARGIN`, `C_DRAG`, `AL_MAX`) added in Tasks 2/4 and used consistently; `GateTrajectory.sample` keys (`pos`,`tang`,`v`,`yaw`) match Task 1 + usage.

## Known risk (carried from spec)

The strafe frame detail (whether the world-y mirror belongs in `attitude_command_tf` when fed live-world `a2`) is resolved empirically in Task 5 Step 5 via the `_tf_ymirror` toggle + the `s_lat` probe, cross-checked against the proven `vq_waypoint2 --square`. course (raw frame) carries no such risk. Each live task needs a freshly restarted sim.
