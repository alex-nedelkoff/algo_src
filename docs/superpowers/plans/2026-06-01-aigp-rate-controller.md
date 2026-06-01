# AI-GP Sim-Response Probe + Body-Rate Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Characterize the AI-GP sim's thrust + body-rate response and build a stable body-rate cascade controller, so the drone flies controlled trajectories (hover, orbit) without the oscillation seen under the sim's own outer loop.

**Architecture:** Pure controller math + analysis fits are unit-tested offline; the excitation probe, the `send_attitude_target` command, and a 3-stage flight check run live against the sim. The controller is mass-agnostic: desired acceleration maps to normalized thrust via the empirically probed hover-thrust fraction and thrust→accel gain. All in the existing `aigp/` package.

**Tech Stack:** Python 3.13, numpy, pymavlink 2.4.49, in the `aigp` conda env. Linear: COR-125 (relates COR-96).

**Execution context (IMPORTANT):** Repo is a git worktree on the **Windows laptop** at `C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client` (branch `aigp-gate-data-collection`), over SSH (`alexj@100.120.233.90`). Tests/commits run there in the `aigp` env: `conda run -n aigp python -m pytest ...` from the worktree root. Tasks 1–5 are pure (no sim). Tasks 6–9 are live (need an active flight).

---

## File Structure

```
aigp/
  control_math.py     # pure: desired_accel, collective_accel, accel_to_thrust_norm,
                      #       desired_attitude, attitude_error, body_rate_cmd
  attitude_control.py # BodyRateController (composes control_math + probed constants)
  analyze_response.py # pure fits: finite_diff, fit_thrust_map, fit_rate_gain, summarize; + CLI
  probe_response.py   # build_schedule (pure) + live open-loop excitation runner
  commander.py        # MODIFY: add send_attitude_target (rate mode)
  fly_check.py        # live 3-stage verification (hover / orbit / resume capture)

tests/test_aigp/
  test_control_math.py
  test_attitude_control.py
  test_analyze_response.py
  test_probe_schedule.py
```

Reuses existing `aigp/geometry.py` (`quat_to_R`), `io_layer.py`, `state.py`, `guidance.py` (`OrbitPattern`), `acquire.py`.

---

### Task 1: control_math — thrust mapping

**Files:**
- Create: `aigp/control_math.py`
- Test: `tests/test_aigp/test_control_math.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_control_math.py
import numpy as np
from aigp.control_math import G, desired_accel, collective_accel, accel_to_thrust_norm

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_desired_accel_zero_at_setpoint():
    a = desired_accel(pos=np.zeros(3), vel=np.zeros(3),
                      pos_sp=np.zeros(3), vel_sp=np.zeros(3),
                      kp_pos=np.array([6, 6, 6]), kd_pos=np.array([4, 4, 4]))
    assert np.allclose(a, 0.0)


def test_desired_accel_points_to_setpoint():
    # setpoint 1 m "up" in NED (z more negative) -> a_des z negative (upward accel)
    a = desired_accel(np.zeros(3), np.zeros(3), np.array([0, 0, -1.0]), np.zeros(3),
                      np.array([6, 6, 6]), np.array([4, 4, 4]))
    assert a[2] < 0.0


def test_collective_accel_is_g_at_hover_level():
    # zero desired accel, level attitude -> collective magnitude ~ g
    c = collective_accel(np.zeros(3), IDENT)
    assert np.isclose(c, G)


def test_accel_to_thrust_norm_hover_and_climb():
    assert np.isclose(accel_to_thrust_norm(G, hover_thrust=0.5, k_a=20.0), 0.5)
    # one extra unit of k_a worth of accel -> +1.0 normalized (then clipped)
    assert np.isclose(accel_to_thrust_norm(G + 20.0, 0.5, 20.0), 1.0)  # 0.5+1.0 clipped to 1
    assert np.isclose(accel_to_thrust_norm(G + 4.0, 0.5, 20.0), 0.7)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_control_math.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.control_math'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/control_math.py
"""Pure body-rate controller math (NED world, FRD body). No sim deps."""
from __future__ import annotations

import numpy as np

from .geometry import quat_to_R

G = 9.81
_G_VEC = np.array([0.0, 0.0, G])  # NED gravity acceleration (down positive)


def desired_accel(pos, vel, pos_sp, vel_sp, kp_pos, kd_pos) -> np.ndarray:
    """PD on position + velocity error -> desired translational accel (NED)."""
    pos = np.asarray(pos, float); vel = np.asarray(vel, float)
    return (np.asarray(kp_pos, float) * (np.asarray(pos_sp, float) - pos)
            + np.asarray(kd_pos, float) * (np.asarray(vel_sp, float) - vel))


def collective_accel(a_des, quat, g: float = G) -> float:
    """Required thrust acceleration projected onto current body-up. ~g at hover."""
    t_vec = np.asarray(a_des, float) - np.array([0.0, 0.0, g])  # hover -> [0,0,-g]
    body_up = -quat_to_R(quat)[:, 2]                            # body up axis in world
    return float(t_vec @ body_up)


def accel_to_thrust_norm(c: float, hover_thrust: float, k_a: float, g: float = G) -> float:
    """Map collective accel magnitude -> normalized thrust [0,1] (mass-agnostic)."""
    return float(np.clip(hover_thrust + (c - g) / k_a, 0.0, 1.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_control_math.py -v -p no:cacheprovider`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/control_math.py tests/test_aigp/test_control_math.py
git commit -m "feat(aigp): control_math thrust mapping (COR-125)"
```

---

### Task 2: control_math — attitude + body-rate

**Files:**
- Modify: `aigp/control_math.py` (append)
- Test: `tests/test_aigp/test_control_math.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_control_math.py
from aigp.control_math import desired_attitude, attitude_error, body_rate_cmd


def _Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_desired_attitude_level_north_is_identity():
    R = desired_attitude(np.zeros(3), yaw_sp=0.0)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_desired_attitude_yaw_east_points_forward_east():
    R = desired_attitude(np.zeros(3), yaw_sp=np.pi / 2)
    # body x-axis (forward) should point east = world (0,1,0)
    assert np.allclose(R[:, 0], [0, 1, 0], atol=1e-9)


def test_attitude_error_zero_when_aligned():
    R = _Rx(0.3)
    assert np.allclose(attitude_error(R, R), 0.0, atol=1e-9)


def test_body_rate_cmd_corrects_roll():
    R_cur = np.eye(3)
    R_des = _Rx(0.2)            # desired rolled +0.2 about body-x
    w = body_rate_cmd(R_cur, R_des, kp_att=8.0)
    assert w[0] > 0.0 and abs(w[1]) < 1e-9 and abs(w[2]) < 1e-9


def test_body_rate_cmd_clamps():
    R_des = _Rx(1.0)
    w = body_rate_cmd(np.eye(3), R_des, kp_att=50.0, max_rate=2.0)
    assert np.all(np.abs(w) <= 2.0 + 1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_control_math.py -k "attitude or body_rate" -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'desired_attitude'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/control_math.py
def desired_attitude(a_des, yaw_sp: float, g: float = G) -> np.ndarray:
    """Desired body->world rotation: thrust along desired up, heading from yaw_sp."""
    t_vec = np.asarray(a_des, float) - np.array([0.0, 0.0, g])
    zb = -t_vec / np.linalg.norm(t_vec)              # body-down axis in world (hover ->[0,0,1])
    x_c = np.array([np.cos(yaw_sp), np.sin(yaw_sp), 0.0])  # desired heading (N,E)
    yb = np.cross(zb, x_c); yb = yb / np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])


def attitude_error(R_cur, R_des) -> np.ndarray:
    """Geometric attitude error (body frame), 0.5*vee(R_des^T R_cur - R_cur^T R_des)."""
    M = R_des.T @ R_cur - R_cur.T @ R_des
    return 0.5 * np.array([M[2, 1], M[0, 2], M[1, 0]])


def body_rate_cmd(R_cur, R_des, kp_att: float, max_rate: float | None = None) -> np.ndarray:
    """Body-rate setpoint that drives R_cur -> R_des."""
    w = -kp_att * attitude_error(R_cur, R_des)
    if max_rate is not None:
        w = np.clip(w, -max_rate, max_rate)
    return w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_control_math.py -v -p no:cacheprovider`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/control_math.py tests/test_aigp/test_control_math.py
git commit -m "feat(aigp): control_math attitude + body-rate (COR-125)"
```

---

### Task 3: BodyRateController

**Files:**
- Create: `aigp/attitude_control.py`
- Test: `tests/test_aigp/test_attitude_control.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_attitude_control.py
import numpy as np
from aigp.attitude_control import BodyRateController

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _ctl():
    return BodyRateController(hover_thrust=0.5, k_a=20.0, kp_pos=[6, 6, 6],
                             kd_pos=[4, 4, 4], kp_att=8.0, max_rate=4.0)


def test_hover_equilibrium_gives_hover_thrust_zero_rates():
    w, thrust = _ctl().update(
        pos=np.zeros(3), vel=np.zeros(3), quat=IDENT, omega=np.zeros(3),
        pos_sp=np.zeros(3), vel_sp=np.zeros(3), yaw_sp=0.0,
    )
    assert np.isclose(thrust, 0.5, atol=1e-6)
    assert np.allclose(w, 0.0, atol=1e-6)


def test_below_setpoint_increases_thrust():
    # drone at z=0, setpoint 1 m up (z=-1) -> climb -> thrust > hover
    w, thrust = _ctl().update(
        np.zeros(3), np.zeros(3), IDENT, np.zeros(3),
        pos_sp=np.array([0, 0, -1.0]), vel_sp=np.zeros(3), yaw_sp=0.0,
    )
    assert thrust > 0.5


def test_rates_clamped():
    w, _ = _ctl().update(
        np.zeros(3), np.zeros(3), IDENT, np.zeros(3),
        pos_sp=np.array([20.0, 0, 0]), vel_sp=np.zeros(3), yaw_sp=np.pi,
    )
    assert np.all(np.abs(w) <= 4.0 + 1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_attitude_control.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.attitude_control'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/attitude_control.py
"""Mass-agnostic body-rate cascade controller for the AI-GP sim."""
from __future__ import annotations

import numpy as np

from .control_math import (
    G, accel_to_thrust_norm, body_rate_cmd, collective_accel,
    desired_accel, desired_attitude,
)
from .geometry import quat_to_R


class BodyRateController:
    def __init__(self, hover_thrust, k_a, kp_pos=(6, 6, 6), kd_pos=(4, 4, 4),
                 kp_att=8.0, max_rate=4.0, g=G):
        self.hover_thrust = hover_thrust
        self.k_a = k_a
        self.kp_pos = np.asarray(kp_pos, float)
        self.kd_pos = np.asarray(kd_pos, float)
        self.kp_att = kp_att
        self.max_rate = max_rate
        self.g = g

    def update(self, pos, vel, quat, omega, pos_sp, vel_sp, yaw_sp):
        """Returns (body_rates (3,), thrust_norm) for send_attitude_target."""
        a_des = desired_accel(pos, vel, pos_sp, vel_sp, self.kp_pos, self.kd_pos)
        c = collective_accel(a_des, quat, self.g)
        thrust = accel_to_thrust_norm(c, self.hover_thrust, self.k_a, self.g)
        R_cur = quat_to_R(quat)
        R_des = desired_attitude(a_des, yaw_sp, self.g)
        w = body_rate_cmd(R_cur, R_des, self.kp_att, self.max_rate)
        return w, thrust
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_attitude_control.py -v -p no:cacheprovider`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/attitude_control.py tests/test_aigp/test_attitude_control.py
git commit -m "feat(aigp): BodyRateController cascade (COR-125)"
```

---

### Task 4: analyze_response — finite diff + thrust map fit

**Files:**
- Create: `aigp/analyze_response.py`
- Test: `tests/test_aigp/test_analyze_response.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_analyze_response.py
import numpy as np
from aigp.analyze_response import finite_diff, fit_thrust_map


def test_finite_diff_linear():
    t = np.linspace(0, 1, 11)
    v = 3.0 * t + 2.0
    assert np.allclose(finite_diff(v, t), 3.0, atol=1e-6)


def test_fit_thrust_map_recovers_constants():
    # thrust_up_accel = k_a*thrust_norm + b ; hover where thrust_up = g
    tn = np.array([0.3, 0.5, 0.7])
    thrust_up = 30.0 * tn - 6.0     # k_a=30, b=-6
    out = fit_thrust_map(tn, thrust_up, g=9.81)
    assert np.isclose(out["k_a"], 30.0, atol=1e-6)
    assert np.isclose(out["hover_thrust"], (9.81 + 6.0) / 30.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_analyze_response.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.analyze_response'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/analyze_response.py
"""Offline fits over probe logs -> sim_response constants. Pure numpy + CLI."""
from __future__ import annotations

import numpy as np

G = 9.81


def finite_diff(values, t) -> np.ndarray:
    return np.gradient(np.asarray(values, float), np.asarray(t, float))


def fit_thrust_map(thrust_norm, thrust_up_accel, g: float = G) -> dict:
    """Linear fit thrust_up_accel = k_a*thrust_norm + b; hover where thrust_up = g."""
    tn = np.asarray(thrust_norm, float)
    y = np.asarray(thrust_up_accel, float)
    A = np.vstack([tn, np.ones_like(tn)]).T
    k_a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"hover_thrust": float((g - b) / k_a), "k_a": float(k_a)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_analyze_response.py -v -p no:cacheprovider`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/analyze_response.py tests/test_aigp/test_analyze_response.py
git commit -m "feat(aigp): analysis finite-diff + thrust-map fit (COR-125)"
```

---

### Task 5: analyze_response — rate gain + summarize

**Files:**
- Modify: `aigp/analyze_response.py` (append)
- Test: `tests/test_aigp/test_analyze_response.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_analyze_response.py
from aigp.analyze_response import fit_rate_gain, summarize


def test_fit_rate_gain():
    cmd = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
    meas = 0.9 * cmd
    assert np.isclose(fit_rate_gain(cmd, meas), 0.9, atol=1e-9)


def test_summarize_from_samples():
    # thrust-sweep samples: vz linear in t so net accel constant per tn level
    samples = []
    t = 0.0
    # three thrust levels, each producing a known constant vertical accel
    # net_up = -dvz/dt ; thrust_up = net_up + g
    for tn, net_up in [(0.3, -3.0), (0.5, 0.0), (0.7, 3.0)]:
        for k in range(5):
            vz = -net_up * (t)        # vz_down; net_up = -dvz/dt
            samples.append({"segment": "thrust_sweep", "t": t, "thrust_norm": tn,
                            "cmd_rates": [0, 0, 0], "vel_ned": [0, 0, vz],
                            "omega": [0, 0, 0]})
            t += 0.1
    # one roll-doublet segment: measured omega = 0.8 * commanded
    for k in range(6):
        c = 1.0 if k < 3 else -1.0
        samples.append({"segment": "rate_roll", "t": t, "thrust_norm": 0.5,
                        "cmd_rates": [c, 0, 0], "vel_ned": [0, 0, 0],
                        "omega": [0.8 * c, 0, 0]})
        t += 0.1
    out = summarize(samples)
    assert 0.0 < out["hover_thrust"] < 1.0
    assert out["k_a"] > 0.0
    assert np.isclose(out["rate_gain"]["roll"], 0.8, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_analyze_response.py -k "rate_gain or summarize" -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'fit_rate_gain'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/analyze_response.py
def fit_rate_gain(cmd, meas) -> float:
    """Least-squares gain (through origin) of measured vs commanded body rate."""
    cmd = np.asarray(cmd, float); meas = np.asarray(meas, float)
    denom = float(cmd @ cmd)
    return float((cmd @ meas) / denom) if denom > 0 else 0.0


_AXIS = {"roll": 0, "pitch": 1, "yaw": 2}


def summarize(samples, g: float = G) -> dict:
    """Reduce tagged probe samples to sim_response constants."""
    sweep = [s for s in samples if s["segment"] == "thrust_sweep"]
    t = np.array([s["t"] for s in sweep])
    tn = np.array([s["thrust_norm"] for s in sweep])
    vz = np.array([s["vel_ned"][2] for s in sweep])
    net_up = -finite_diff(vz, t)            # upward net accel
    thrust_up = net_up + g
    tmap = fit_thrust_map(tn, thrust_up, g)

    rate_gain = {}
    for name, axis in _AXIS.items():
        seg = [s for s in samples if s["segment"] == f"rate_{name}"]
        if not seg:
            continue
        cmd = np.array([s["cmd_rates"][axis] for s in seg])
        meas = np.array([s["omega"][axis] for s in seg])
        rate_gain[name] = fit_rate_gain(cmd, meas)

    return {"hover_thrust": tmap["hover_thrust"], "k_a": tmap["k_a"],
            "rate_gain": rate_gain}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_analyze_response.py -v -p no:cacheprovider`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/analyze_response.py tests/test_aigp/test_analyze_response.py
git commit -m "feat(aigp): analysis rate-gain + summarize (COR-125)"
```

---

### Task 6: commander — send_attitude_target (rate mode)

**Files:**
- Modify: `aigp/commander.py`

**No unit test** (sends live MAVLink); import-checked here, exercised in Tasks 8–9.

- [ ] **Step 1: Add the import and method**

Add `import numpy as np` at the top of `aigp/commander.py` (below the existing imports), then append this method to the `Commander` class:

```python
    def send_attitude_target(self, body_rates, thrust_norm):
        """Body-rate setpoint + normalized thrust via SET_ATTITUDE_TARGET (rate mode)."""
        now_ms = int(time.time() * 1000) - self.system_boot_ms
        self.conn.mav.set_attitude_target_send(
            now_ms,
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
            [1.0, 0.0, 0.0, 0.0],   # quaternion ignored in rate mode
            float(body_rates[0]), float(body_rates[1]), float(body_rates[2]),
            float(np.clip(thrust_norm, 0.0, 1.0)),
        )
```

- [ ] **Step 2: Import-check**

Run: `conda run -n aigp python -c "import aigp.commander; from aigp.commander import Commander; print('ok', hasattr(Commander,'send_attitude_target'))"`
Expected: `ok True`

- [ ] **Step 3: Commit**

```bash
git add aigp/commander.py
git commit -m "feat(aigp): commander.send_attitude_target rate mode (COR-125)"
```

---

### Task 7: probe_response — excitation schedule + live runner

**Files:**
- Create: `aigp/probe_response.py`
- Test: `tests/test_aigp/test_probe_schedule.py`

- [ ] **Step 1: Write the failing test (pure schedule builder)**

```python
# tests/test_aigp/test_probe_schedule.py
from aigp.probe_response import build_schedule


def test_schedule_segments_and_bounds():
    sched = build_schedule(hover_guess=0.5, thrust_band=0.2, rate_amp=1.0,
                           seg_dur=2.0, dt=0.05)
    names = [s["name"] for s in sched]
    assert names == ["thrust_sweep", "rate_roll", "rate_pitch", "rate_yaw"]
    sweep = sched[0]
    # thrust stays within [hover-band, hover+band] and rates zero
    assert all(0.3 - 1e-9 <= c["thrust"] <= 0.7 + 1e-9 for c in sweep["commands"])
    assert all(c["rates"] == [0.0, 0.0, 0.0] for c in sweep["commands"])
    # roll doublet: |rate| <= rate_amp on axis 0 only
    roll = sched[1]
    assert all(abs(c["rates"][0]) <= 1.0 + 1e-9 for c in roll["commands"])
    assert all(c["rates"][1] == 0.0 and c["rates"][2] == 0.0 for c in roll["commands"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_probe_schedule.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.probe_response'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/probe_response.py
"""Open-loop excitation probe for sim-response characterization.

build_schedule() is pure (testable); run_probe() executes it live, logging
tagged samples and resetting the sim between segments / on safety breach.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


def build_schedule(hover_guess=0.5, thrust_band=0.2, rate_amp=1.0, seg_dur=2.0, dt=0.05):
    """Return a list of segments: {name, commands:[{thrust, rates}]}."""
    n = max(1, int(seg_dur / dt))
    sched = []

    # thrust sweep: ramp thrust across the band, rates zero
    sweep = []
    for k in range(n):
        frac = k / (n - 1) if n > 1 else 0.0
        thrust = hover_guess - thrust_band + 2 * thrust_band * frac
        sweep.append({"thrust": thrust, "rates": [0.0, 0.0, 0.0]})
    sched.append({"name": "thrust_sweep", "commands": sweep})

    # rate doublets per axis at hover thrust
    for name, axis in (("rate_roll", 0), ("rate_pitch", 1), ("rate_yaw", 2)):
        cmds = []
        for k in range(n):
            sign = 1.0 if k < n // 2 else -1.0
            rates = [0.0, 0.0, 0.0]
            rates[axis] = sign * rate_amp
            cmds.append({"thrust": hover_guess, "rates": rates})
        sched.append({"name": name, "commands": cmds})

    return sched


# ---- live runner (no unit test) ----

TILT_ABORT_DEG = 30.0
ALT_FLOOR_M = 0.5  # NED: abort if z (down) exceeds this below start


def _tilt_deg(quat):
    import numpy as np
    from .geometry import quat_to_R
    zb = quat_to_R(quat)[:, 2]              # body-down in world
    cos_tilt = float(-zb[2])               # vs world-down
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


def run_probe(store, commander, out_dir="sysid", run_id="probe", dt=0.05, **kw):
    from .state import DroneState
    sched = build_schedule(dt=dt, **kw)
    root = Path(out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "log.jsonl", "w")
    for seg in sched:
        commander.sim_reset()
        time.sleep(1.0)                    # let the sim respawn
        start = store.get_drone()
        z0 = start.pos_ned[2] if start else 0.0
        for cmd in seg["commands"]:
            commander.send_attitude_target(cmd["rates"], cmd["thrust"])
            time.sleep(dt)
            ds = store.get_drone()
            if ds is None:
                continue
            if _tilt_deg(ds.quat_wxyz) > TILT_ABORT_DEG or ds.pos_ned[2] - z0 > ALT_FLOOR_M:
                break                      # safety: bail this segment
            f.write(json.dumps({
                "segment": seg["name"], "t": time.time(),
                "thrust_norm": cmd["thrust"], "cmd_rates": list(cmd["rates"]),
                "pos_ned": [float(x) for x in ds.pos_ned],
                "vel_ned": [float(x) for x in ds.vel_ned],
                "quat_wxyz": [float(x) for x in ds.quat_wxyz],
                "omega": [float(x) for x in ds.omega],
            }) + "\n")
            f.flush()
    f.close()
    commander.sim_reset()
    print(f"probe log at {root/'log.jsonl'}", flush=True)
    return str(root / "log.jsonl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_probe_schedule.py -v -p no:cacheprovider`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/probe_response.py tests/test_aigp/test_probe_schedule.py
git commit -m "feat(aigp): excitation probe schedule + live runner (COR-125)"
```

---

### Task 8: analyze CLI + live characterization run

**Files:**
- Modify: `aigp/analyze_response.py` (append CLI)
- Create (output, not committed): `sysid/<run>/sim_response.json`

- [ ] **Step 1: Add the CLI**

Append to `aigp/analyze_response.py`:

```python
def _main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to probe log.jsonl")
    ap.add_argument("--out", default=None, help="output sim_response.json")
    args = ap.parse_args()
    samples = [json.loads(l) for l in open(args.log)]
    out = summarize(samples)
    dest = args.out or str(__import__("pathlib").Path(args.log).with_name("sim_response.json"))
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    _main()
```

- [ ] **Step 2: Run the live probe** (requires an active flight)

From the worktree root:
```bash
conda run -n aigp python -c "import time; from aigp.io_layer import MavlinkIO; from aigp.state import Store; from aigp.commander import Commander; from aigp.probe_response import run_probe; s=Store(); m=MavlinkIO(s); assert m.wait_heartbeat(10); boot=int(time.time()*1000); m.start(); c=Commander(m.conn, boot); run_probe(s, c, run_id='probe1')"
```
Expected: prints `probe log at sysid/probe1/log.jsonl`; the drone performs short thrust/rate bursts with resets between them.

- [ ] **Step 3: Run the analysis**

```bash
conda run -n aigp python -m aigp.analyze_response sysid/probe1/log.jsonl
```
Expected: prints a JSON block with `hover_thrust` in (0,1), `k_a` > 0, and `rate_gain` per axis ≈ O(1); writes `sysid/probe1/sim_response.json`.

- [ ] **Step 4: Commit (code only; sysid/ outputs stay untracked)**

```bash
git add aigp/analyze_response.py
git commit -m "feat(aigp): analyze_response CLI (COR-125)"
```

---

### Task 9: fly_check — staged live verification

**Files:**
- Create: `aigp/fly_check.py`

**No unit test** (live). This is the acceptance harness; it loads the probed constants and the `BodyRateController`.

- [ ] **Step 1: Write the implementation**

```python
# aigp/fly_check.py
"""Staged live verification of the body-rate controller against the sim."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .acquire import acquire_gates
from .attitude_control import BodyRateController
from .commander import Commander
from .guidance import OrbitPattern
from .io_layer import MavlinkIO, VisionIO
from .state import Store


def _ctl_from(path):
    r = json.load(open(path))
    return BodyRateController(hover_thrust=r["hover_thrust"], k_a=r["k_a"])


def _run_loop(store, cmd, ctl, sp_fn, duration, control_hz=50.0):
    dt = 1.0 / control_hz
    t_end = time.time() + duration
    pe_max = 0.0
    while time.time() < t_end:
        ds = store.get_drone()
        if ds is not None:
            pos_sp, vel_sp, yaw_sp = sp_fn(ds)
            w, thrust = ctl.update(ds.pos_ned, ds.vel_ned, ds.quat_wxyz, ds.omega,
                                   pos_sp, vel_sp, yaw_sp)
            cmd.send_attitude_target(w, thrust)
            pe_max = max(pe_max, float(np.linalg.norm(np.asarray(pos_sp) - ds.pos_ned)))
        time.sleep(dt)
    return pe_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="sysid/probe1/sim_response.json")
    ap.add_argument("--stage", choices=["hover", "orbit"], default="hover")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=2.0)
    args = ap.parse_args()

    store = Store(); mav = MavlinkIO(store)
    assert mav.wait_heartbeat(10), "no heartbeat — start a flight"
    boot = int(time.time() * 1000); mav.start(); VisionIO(store).start()
    cmd = Commander(mav.conn, boot)
    ctl = _ctl_from(args.response)
    cmd.arm()

    if args.stage == "hover":
        time.sleep(0.5)
        ds0 = store.get_drone()
        hold = ds0.pos_ned.copy()
        yaw0 = 0.0
        pe = _run_loop(store, cmd, ctl,
                       lambda ds: (hold, np.zeros(3), yaw0), args.duration)
        print(f"HOVER max position error = {pe:.2f} m "
              f"({'PASS' if pe < 1.5 else 'FAIL'})", flush=True)
    else:
        gates = acquire_gates(store, cmd)
        gate = gates[0]
        orbit = OrbitPattern(radius=args.radius, speed=args.speed,
                             target_z=float(gate.pos_ned[2]), max_speed=5.0)

        def sp(ds):
            s = orbit.update(ds.pos_ned, ds.vel_ned, gate.pos_ned)
            # convert velocity setpoint to a moving position setpoint one step ahead
            pos_sp = ds.pos_ned + np.array([s.vx, s.vy, s.vz]) * 0.2
            return pos_sp, np.array([s.vx, s.vy, s.vz]), s.yaw

        pe = _run_loop(store, cmd, ctl, sp, args.duration)
        print(f"ORBIT max tracking error = {pe:.2f} m", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Live — Stage 1 hover hold** (requires an active flight)

```bash
conda run -n aigp python -m aigp.fly_check --stage hover --duration 15
```
Expected: drone holds position (no oscillation / wild panning); prints `HOVER max position error = <x> m (PASS)` with x < 1.5. If it oscillates, lower `kp_pos`/`kp_att` (e.g. pass tuned gains) — see Step 4.

- [ ] **Step 3: Live — Stage 2 orbit**

```bash
conda run -n aigp python -m aigp.fly_check --stage orbit --duration 40
```
Expected: drone flies a smooth circle around gate 0 with the camera tracking it (no wild panning); prints a bounded tracking error.

- [ ] **Step 4: Tune if needed, then commit**

If hover or orbit is unstable, the fix is gain tuning (not structural): reduce `BodyRateController` defaults `kp_pos` (e.g. 6→3), `kp_att` (8→5), or `max_rate`. Edit `aigp/attitude_control.py` defaults, re-run Step 2/3 until stable, keeping all unit tests green (`pytest tests/test_aigp`).

```bash
git add aigp/fly_check.py aigp/attitude_control.py
git commit -m "feat(aigp): staged live flight verification + tuned gains (COR-125)"
```

---

### Task 10: Resume data capture with the stable controller

**Files:**
- Modify: `aigp/run.py` (swap velocity-setpoint commanding for the BodyRateController)

- [ ] **Step 1: Wire the controller into run.py**

In `aigp/run.py`, replace the control-loop body that calls `cmd.send_velocity_setpoint(...)` with the BodyRateController path. Add near the imports:

```python
from .attitude_control import BodyRateController
import json as _json
```

Add a CLI arg (next to the others):

```python
    ap.add_argument("--response", default="sysid/probe1/sim_response.json")
```

After `cmd = Commander(...)` and before the loop, build the controller:

```python
    _r = _json.load(open(args.response))
    ctl = BodyRateController(hover_thrust=_r["hover_thrust"], k_a=_r["k_a"])
```

Replace the in-loop command block:

```python
            ds = store.get_drone()
            if ds is not None:
                sp = pattern.update(ds.pos_ned, ds.vel_ned, gate.pos_ned)
                vel_sp = np.array([sp.vx, sp.vy, sp.vz])
                pos_sp = ds.pos_ned + vel_sp * 0.2
                w, thrust = ctl.update(ds.pos_ned, ds.vel_ned, ds.quat_wxyz, ds.omega,
                                       pos_sp, vel_sp, sp.yaw)
                cmd.send_attitude_target(w, thrust)
```

- [ ] **Step 2: Live — capture a dataset**

```bash
conda run -n aigp python -m aigp.run --pattern orbit --duration 40 --radius 4 --speed 3 --height 0 --run-id orbit_stable
```
Expected: stable orbit; prints `done. dataset at dataset/orbit_stable`. Verify:
```bash
conda run -n aigp python -c "import json,glob; n=len(glob.glob('dataset/orbit_stable/frames/*.jpg')); ls=[json.loads(l) for l in open('dataset/orbit_stable/labels.jsonl')]; print('frames',n,'in_frame',sum(x['in_frame'] for x in ls))"
```
Expected: nonzero frames, many `in_frame=True`.

- [ ] **Step 3: Overlay acceptance (COR-125 Task 14)**

```bash
conda run -n aigp python -m aigp.overlay_labels dataset/orbit_stable --limit 40
```
Copy a few overlays to the Mac and confirm the projected gate lands on the actual gate.

- [ ] **Step 4: Commit**

```bash
git add aigp/run.py aigp/acquire.py aigp/overlay_labels.py
git commit -m "feat(aigp): capture dataset with stable body-rate controller (COR-125)"
```

---

## Self-Review

**Spec coverage:** commander rate-mode send → Task 6. Excitation probe (thrust sweep + per-axis rate doublets, gentle bounds, auto-reset) → Task 7. Analysis (hover_thrust, k_a, rate tracking) → Tasks 4–5, run in Task 8. Body-rate cascade controller (mass-agnostic) → Tasks 1–3. Verification (hover → orbit → resume capture) → Tasks 9–10. Frame handling (desired-attitude transform) → Task 2. All spec components covered.

**Placeholder scan:** No TBD/TODO; every code step is complete; commands have expected output. The only judgement step (Task 9 Step 4 tuning) gives concrete gain directions and bounds, not a vague "handle it."

**Type consistency:** `BodyRateController(hover_thrust, k_a, kp_pos, kd_pos, kp_att, max_rate)` and `.update(pos, vel, quat, omega, pos_sp, vel_sp, yaw_sp) -> (w, thrust)` consistent across Tasks 3, 9, 10. `control_math` signatures (`desired_accel`, `collective_accel(a_des, quat, g)`, `accel_to_thrust_norm(c, hover_thrust, k_a, g)`, `desired_attitude(a_des, yaw_sp, g)`, `attitude_error(R_cur,R_des)`, `body_rate_cmd(R_cur,R_des,kp_att,max_rate)`) consistent across Tasks 1–3. `summarize(samples)->{hover_thrust,k_a,rate_gain}` matches the CLI (Task 8) and `_ctl_from` reader (Task 9) which reads `hover_thrust`/`k_a`. `send_attitude_target(body_rates, thrust_norm)` consistent across Tasks 6, 7, 9, 10. `run_probe(store, commander, ...)` and `build_schedule(...)` consistent across Task 7 and the Task 8 live call. Probe log fields (`segment, t, thrust_norm, cmd_rates, vel_ned, omega, ...`) match `summarize`'s reads.

**Assumptions validated during execution:** sim inner rate loop tracks well enough (Task 8 rate_gain); starting gains stable (Task 9 tuning); `SIM_RESET` respawn timing (1 s sleep in `run_probe`).
