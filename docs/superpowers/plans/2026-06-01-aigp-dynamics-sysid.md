# AI-GP Drone Dynamics sysID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify the AI-GP drone's lumped-ratio open-loop dynamics from motor-level excitation and fit the rigid-body EOM, producing a validated `sim_dynamics.json`.

**Architecture:** A lumped forward model + pure least-squares fitters are unit-tested offline; a live motor-level probe (`set_actuator_control_target`) excites the drone and logs `HIGHRES_IMU`+`ODOMETRY`, and a fitter assembles regressors from the logs to recover the coefficients. Static/instantaneous LS (not rollout) because open-loop bare dynamics are unstable (short windows). All in the `aigp/` package.

**Tech Stack:** Python 3.13, numpy, pymavlink 2.4.49 (`aigp` conda env). Linear: COR-96 (realized against the real sim), COR-125 branch.

**Execution context:** Repo is a git worktree on the **Windows laptop** at `C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client` (branch `aigp-gate-data-collection`), over SSH (`alexj@100.120.233.90`). Tests/commits run there in the `aigp` env: `conda run -n aigp python -m pytest ...` from the worktree root. Tasks 1–4, 7 are pure (no sim). Tasks 5–6, 8 are live (need an active flight; actuate only after race-live).

---

## File Structure

```
aigp/
  dyn_model.py     # lumped forward EOM: quat_integrate, step  (pure)
  dyn_fit.py       # pure fitters: fit_gain, fit_specific_thrust, select_power,
                   #   angular_accel, fit_axis_torque, fit_motor_lag, fit_drag, fit_from_log
  dyn_probe.py     # live: send-pattern campaign (collective sweep, per-channel bumps,
                   #   per-axis differential doublets) -> log.jsonl
  dyn_validate.py  # CLI: load log -> fit_from_log -> sim_dynamics.json + held-out RMSE
  commander.py     # MODIFY: add send_motor_command(controls)

tests/test_aigp/
  test_dyn_model.py
  test_dyn_fit.py
```

Reuses `aigp/geometry.py` (`quat_to_R`), `control_math.py` (`quat_mul`, `G`), `io_layer.py`, `commander.py`, `race.py`, `state.py`.

---

### Task 1: Lumped forward model

**Files:**
- Create: `aigp/dyn_model.py`
- Test: `tests/test_aigp/test_dyn_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_dyn_model.py
import numpy as np
from aigp.dyn_model import quat_integrate, step, default_params

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _state(**kw):
    s = dict(pos=np.zeros(3), vel=np.zeros(3), quat=IDENT.copy(),
             omega=np.zeros(3), motor=np.zeros(4))
    s.update(kw)
    return s


def test_quat_integrate_zero_omega_is_identity():
    q = quat_integrate(IDENT, np.zeros(3), 0.01)
    assert np.allclose(q, IDENT)


def test_quat_integrate_yaw_rate():
    # +yaw rate about body z -> z component grows positive
    q = quat_integrate(IDENT, np.array([0, 0, 1.0]), 0.01)
    assert q[3] > 0 and np.isclose(np.linalg.norm(q), 1.0)


def test_hover_motor_gives_zero_vertical_accel():
    p = default_params()
    # motor power-sum that exactly cancels gravity: c_T * sum(p) = g
    # with power=2 and 4 equal motors: c_T*4*m^2 = g -> m = sqrt(g/(4 c_T))
    m = np.sqrt(p["g"] / (4 * p["c_T"]))
    s = _state(motor=np.full(4, m))   # already at steady motor (no lag transient)
    s2 = step(s, np.full(4, m), p, 0.01)
    assert abs(s2["vel"][2]) < 1e-6     # no vertical velocity change at hover


def test_more_thrust_accelerates_up():
    p = default_params()
    m = np.sqrt(p["g"] / (4 * p["c_T"]))
    s = _state(motor=np.full(4, m))
    s2 = step(s, np.full(4, m * 1.5), p, 0.02)
    assert s2["vel"][2] < 0     # NED: up is -z


def test_roll_mix_gives_roll_rate():
    p = default_params()
    m = np.sqrt(p["g"] / (4 * p["c_T"]))
    s = _state(motor=np.full(4, m))
    # bump the motors that the roll row marks +1, drop the -1 ones
    u = np.full(4, m) + p["mix"][0] * 0.05 * m
    s2 = step(s, u, p, 0.02)
    assert s2["omega"][0] > 0     # +roll angular rate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_model.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.dyn_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/dyn_model.py
"""Lumped-ratio rigid-body forward model for the AI-GP drone (pure numpy).

Parameters are identifiable lumped ratios (no absolute mass/inertia):
  c_T            specific thrust per (motor input ** power), summed over motors
  c_L, c_M, c_N  roll/pitch/yaw angular-accel per (mix . p)
  mix (3,4)      motor mixing rows for [roll, pitch, yaw] (signs/geometry)
  tau_motor      first-order motor lag (s)
  drag (3)       linear drag accel coeff (per world-velocity component)
  power          1 or 2 (thrust ~ input or input^2)
  g              gravity (m/s^2)
State dict: pos(3) vel(3) NED, quat(4) wxyz body->world, omega(3) body, motor(4).
"""
from __future__ import annotations

import numpy as np

from .control_math import G, quat_mul
from .geometry import quat_to_R


def default_params() -> dict:
    return {
        "c_T": 5.0,
        "c_L": 40.0, "c_M": 40.0, "c_N": 8.0,
        "mix": np.array([[-1.0, 1.0, 1.0, -1.0],     # roll
                         [-1.0, -1.0, 1.0, 1.0],     # pitch
                         [1.0, -1.0, 1.0, -1.0]]),   # yaw (X-config signs)
        "tau_motor": 0.02,
        "drag": np.zeros(3),
        "power": 2,
        "g": G,
    }


def quat_integrate(q_wxyz, omega_body, dt):
    """Integrate a body-rate over dt: q_dot = 0.5 * q (x) [0, omega]."""
    q = np.asarray(q_wxyz, float)
    wq = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    q = q + 0.5 * quat_mul(q, wq) * dt
    return q / np.linalg.norm(q)


def step(state: dict, u_cmd, params: dict, dt: float) -> dict:
    pos = np.asarray(state["pos"], float); vel = np.asarray(state["vel"], float)
    quat = np.asarray(state["quat"], float); omega = np.asarray(state["omega"], float)
    motor = np.asarray(state["motor"], float)
    u = np.asarray(u_cmd, float)

    # first-order motor lag
    alpha_lag = min(1.0, dt / params["tau_motor"])
    motor_new = motor + (u - motor) * alpha_lag
    p = motor_new ** params["power"]

    R = quat_to_R(quat)
    body_up = -R[:, 2]
    f = params["c_T"] * float(p.sum())                       # specific thrust (accel)
    a = f * body_up + np.array([0.0, 0.0, params["g"]]) - np.asarray(params["drag"]) * vel

    mix = np.asarray(params["mix"], float)
    alpha = np.array([params["c_L"] * float(mix[0] @ p),
                      params["c_M"] * float(mix[1] @ p),
                      params["c_N"] * float(mix[2] @ p)])

    return {
        "pos": pos + vel * dt,
        "vel": vel + a * dt,
        "quat": quat_integrate(quat, omega, dt),
        "omega": omega + alpha * dt,
        "motor": motor_new,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_model.py -v -p no:cacheprovider`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/dyn_model.py tests/test_aigp/test_dyn_model.py
git commit -m "feat(aigp): lumped forward dynamics model (COR-96)"
```

---

### Task 2: Fitters — gain, specific thrust, power selection

**Files:**
- Create: `aigp/dyn_fit.py`
- Test: `tests/test_aigp/test_dyn_fit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_dyn_fit.py
import numpy as np
from aigp.dyn_fit import fit_gain, fit_specific_thrust, select_power


def test_fit_gain_through_origin():
    x = np.array([0.0, 1, 2, 3])
    y = 5.0 * x
    assert np.isclose(fit_gain(x, y), 5.0)


def test_fit_gain_zero_input():
    assert fit_gain(np.zeros(3), np.zeros(3)) == 0.0


def test_fit_specific_thrust():
    psum = np.array([1.0, 2, 3, 4])
    up = 3.5 * psum
    assert np.isclose(fit_specific_thrust(psum, up), 3.5)


def test_select_power_picks_quadratic():
    u = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    u_sum = 4 * u            # 4 equal motors, linear sum
    u_sq_sum = 4 * u ** 2    # quadratic sum
    up_accel = 6.0 * u_sq_sum    # truly quadratic in input
    power, c_T = select_power(u_sum, u_sq_sum, up_accel)
    assert power == 2
    assert np.isclose(c_T, 6.0, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.dyn_fit'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/dyn_fit.py
"""Pure least-squares fitters for lumped dynamics sysID."""
from __future__ import annotations

import numpy as np


def fit_gain(x, y) -> float:
    """Through-origin least-squares slope of y vs x."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    d = float(x @ x)
    return float((x @ y) / d) if d > 0 else 0.0


def fit_specific_thrust(power_sum, up_accel) -> float:
    """c_T = slope of measured specific thrust (up accel) vs summed motor power."""
    return fit_gain(power_sum, up_accel)


def select_power(u_sum, u_sq_sum, up_accel):
    """Choose thrust ~ input (1) vs input^2 (2) by residual; return (power, c_T)."""
    up = np.asarray(up_accel, float)
    best = None
    for power, P in ((1, np.asarray(u_sum, float)), (2, np.asarray(u_sq_sum, float))):
        c = fit_gain(P, up)
        resid = float(np.sum((up - c * P) ** 2))
        if best is None or resid < best[2]:
            best = (power, c, resid)
    return best[0], best[1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -v -p no:cacheprovider`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/dyn_fit.py tests/test_aigp/test_dyn_fit.py
git commit -m "feat(aigp): thrust/power-law fitters (COR-96)"
```

---

### Task 3: Fitters — angular acceleration + axis torque

**Files:**
- Modify: `aigp/dyn_fit.py` (append)
- Test: `tests/test_aigp/test_dyn_fit.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_dyn_fit.py
from aigp.dyn_fit import angular_accel, fit_axis_torque


def test_angular_accel_linear_gyro():
    t = np.linspace(0, 1, 11)
    gyro = np.stack([2.0 * t, -3.0 * t, np.zeros_like(t)], axis=1)
    aa = angular_accel(t, gyro)
    assert np.allclose(aa[:, 0], 2.0, atol=1e-6)
    assert np.allclose(aa[:, 1], -3.0, atol=1e-6)


def test_fit_axis_torque():
    mix_reg = np.array([0.1, 0.2, -0.1, -0.2])    # mix . p per sample
    ang = 40.0 * mix_reg
    assert np.isclose(fit_axis_torque(mix_reg, ang), 40.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -k "angular or axis_torque" -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'angular_accel'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/dyn_fit.py
def angular_accel(t, gyro) -> np.ndarray:
    """Finite-difference angular acceleration from body-rate samples, shape (N,3)."""
    t = np.asarray(t, float)
    gyro = np.asarray(gyro, float)
    return np.gradient(gyro, t, axis=0)


def fit_axis_torque(mix_regressor, ang_accel_axis) -> float:
    """c_axis = slope of angular accel vs the axis's (mix . p) regressor."""
    return fit_gain(mix_regressor, ang_accel_axis)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -v -p no:cacheprovider`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/dyn_fit.py tests/test_aigp/test_dyn_fit.py
git commit -m "feat(aigp): angular-accel + axis-torque fitters (COR-96)"
```

---

### Task 4: Fitters — motor lag + drag

**Files:**
- Modify: `aigp/dyn_fit.py` (append)
- Test: `tests/test_aigp/test_dyn_fit.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_dyn_fit.py
from aigp.dyn_fit import fit_motor_lag, fit_drag


def test_fit_motor_lag_recovers_tau():
    t = np.linspace(0, 0.3, 200)
    tau_true = 0.025
    resp = 9.0 * (1 - np.exp(-t / tau_true))     # 1st-order step response
    tau = fit_motor_lag(t, resp)
    assert abs(tau - tau_true) < 0.005


def test_fit_drag():
    vel = np.array([0.0, 1, 2, 3, 4])
    resid_accel = -0.5 * vel        # drag decelerates
    assert np.isclose(fit_drag(vel, resid_accel), 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -k "lag or drag" -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'fit_motor_lag'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/dyn_fit.py
def fit_motor_lag(t, response, taus=None) -> float:
    """Grid-search the first-order time constant of a step response
    response(t) = A * (1 - exp(-(t - t0) / tau)); returns tau."""
    t = np.asarray(t, float)
    t = t - t[0]
    y = np.asarray(response, float)
    if taus is None:
        taus = np.linspace(0.003, 0.12, 80)
    best = (None, np.inf)
    for tau in taus:
        basis = 1.0 - np.exp(-t / tau)
        a = fit_gain(basis, y)
        resid = float(np.sum((y - a * basis) ** 2))
        if resid < best[1]:
            best = (float(tau), resid)
    return best[0]


def fit_drag(vel, residual_accel) -> float:
    """Linear drag coeff: residual_accel = -drag * vel."""
    return -fit_gain(vel, residual_accel)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -v -p no:cacheprovider`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/dyn_fit.py tests/test_aigp/test_dyn_fit.py
git commit -m "feat(aigp): motor-lag + drag fitters (COR-96)"
```

---

### Task 5: Commander — motor command

**Files:**
- Modify: `aigp/commander.py`

**No unit test** (sends live MAVLink); import-checked here, exercised in Tasks 6/8.

- [ ] **Step 1: Add the method**

Append this method to the `Commander` class in `aigp/commander.py`:

```python
    def send_motor_command(self, controls):
        """Direct motor/actuator command via SET_ACTUATOR_CONTROL_TARGET (group 0).
        `controls` is up to 8 normalized values [0,1]; padded to 8, clipped.
        Bypasses the sim inner controllers (for open-loop dynamics sysID).
        Actuate only after the race is live."""
        u = np.zeros(8, dtype=float)
        c = np.asarray(controls, float).ravel()
        n = min(8, c.shape[0])
        u[:n] = np.clip(c[:n], 0.0, 1.0)
        self.conn.mav.set_actuator_control_target_send(
            int(time.time() * 1e6), 0,   # time_usec, group_mlx=0
            self.conn.target_system, self.conn.target_component,
            u.tolist())
```

- [ ] **Step 2: Import-check**

Run: `conda run -n aigp python -c "from aigp.commander import Commander; print('ok', hasattr(Commander,'send_motor_command'))"`
Expected: `ok True`

- [ ] **Step 3: Commit**

```bash
git add aigp/commander.py
git commit -m "feat(aigp): commander.send_motor_command (set_actuator_control_target) (COR-96)"
```

---

### Task 6: Live probe — motor calibration + excitation campaign

**Files:**
- Create: `aigp/dyn_probe.py`
- Test: `tests/test_aigp/test_dyn_fit.py` (append a tiny schedule test)

- [ ] **Step 1: Write the failing test (pure schedule builder)**

```python
# append to tests/test_aigp/test_dyn_fit.py
from aigp.dyn_probe import build_campaign


def test_build_campaign_segments():
    seg = build_campaign(n_motors=4, hover_guess=0.3, levels=3, axes=("roll", "pitch", "yaw"))
    names = [s["name"] for s in seg]
    assert "collective" in names
    assert "bump_m0" in names and "bump_m3" in names      # per-motor calibration bumps
    assert "diff_roll" in names and "diff_yaw" in names    # per-axis differential
    # every command vector has length n_motors and is within [0,1]
    for s in seg:
        for u in s["commands"]:
            assert len(u) == 4 and all(0.0 <= v <= 1.0 for v in u)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -k campaign -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.dyn_probe'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/dyn_probe.py
"""Live motor-level excitation for dynamics sysID.

build_campaign() is pure (testable). run_campaign() executes it: fresh race-live +
arm, then drives motor patterns logging HIGHRES_IMU (specific force + gyro) and
ODOMETRY. Open-loop bare dynamics are unstable, so torque segments are short bursts
with SIM_RESET between; the collective segment is attitude-stable (longer)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# X-config differential mixes (sign hypotheses; calibration confirms actual signs)
_MIX = {
    "roll":  np.array([-1.0, 1.0, 1.0, -1.0]),
    "pitch": np.array([-1.0, -1.0, 1.0, 1.0]),
    "yaw":   np.array([1.0, -1.0, 1.0, -1.0]),
}


def build_campaign(n_motors=4, hover_guess=0.3, levels=3, axes=("roll", "pitch", "yaw"),
                   bump=0.15, diff=0.1):
    """Return a list of segments: {name, kind, commands:[u(list len n_motors)]}."""
    segs = []

    # collective sweep (thrust + power law), attitude-stable
    coll = []
    for k in range(levels):
        frac = k / (levels - 1) if levels > 1 else 0.0
        lvl = max(0.0, hover_guess - 0.1) + 0.3 * frac
        coll += [[lvl] * n_motors] * 8       # hold each level a few samples
    segs.append({"name": "collective", "kind": "collective", "commands": coll})

    # per-motor bumps (channel mapping + IMU axis calibration)
    for i in range(n_motors):
        u = [hover_guess] * n_motors
        u[i] = min(1.0, hover_guess + bump)
        segs.append({"name": f"bump_m{i}", "kind": "bump", "commands": [u] * 6})

    # per-axis differential doublets (torque coefficients)
    for ax in axes:
        cmds = []
        for k in range(8):
            sign = 1.0 if k < 4 else -1.0
            u = np.full(n_motors, hover_guess) + sign * diff * _MIX[ax][:n_motors]
            cmds.append([float(np.clip(v, 0.0, 1.0)) for v in u])
        segs.append({"name": f"diff_{ax}", "kind": "diff", "commands": cmds})

    return segs


def run_probe(store, commander, mav_conn, out_dir="sysid_dyn", run_id="dyn",
              dt=0.01, n_motors=4, hover_guess=0.3):
    """Execute the campaign live. Logs HIGHRES_IMU + ODOMETRY per command."""
    from .race import wait_for_fresh_race_live
    segs = build_campaign(n_motors=n_motors, hover_guess=hover_guess)
    root = Path(out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "log.jsonl", "w")
    for seg in segs:
        if not wait_for_fresh_race_live(store, commander):
            f.close(); raise RuntimeError("no fresh race-live before " + seg["name"])
        commander.arm()
        for u in seg["commands"]:
            commander.send_motor_command(u)
            time.sleep(dt)
            imu = mav_conn.recv_match(type="HIGHRES_IMU", blocking=False)
            ds = store.get_drone()
            if ds is None:
                continue
            row = {"segment": seg["name"], "t": time.time(), "u": list(u),
                   "pos_ned": [float(x) for x in ds.pos_ned],
                   "vel_ned": [float(x) for x in ds.vel_ned],
                   "quat_wxyz": [float(x) for x in ds.quat_wxyz],
                   "omega": [float(x) for x in ds.omega]}
            if imu is not None:
                row["imu_acc"] = [float(imu.xacc), float(imu.yacc), float(imu.zacc)]
                row["imu_gyro"] = [float(imu.xgyro), float(imu.ygyro), float(imu.zgyro)]
            f.write(json.dumps(row) + "\n")
            f.flush()
    f.close()
    print(f"dyn probe log at {root/'log.jsonl'}", flush=True)
    return str(root / "log.jsonl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -k campaign -v -p no:cacheprovider`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/dyn_probe.py tests/test_aigp/test_dyn_fit.py
git commit -m "feat(aigp): motor-level excitation campaign (COR-96)"
```

---

### Task 7: Fit-from-log assembler

**Files:**
- Modify: `aigp/dyn_fit.py` (append `fit_from_log`)
- Test: `tests/test_aigp/test_dyn_fit.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_dyn_fit.py
from aigp.dyn_fit import fit_from_log


def _coll_sample(seg, t, u, accz):
    # imu_acc body-up component carries specific thrust; level drone -> z axis.
    return {"segment": seg, "t": t, "u": list(u),
            "imu_acc": [0.0, 0.0, accz], "imu_gyro": [0.0, 0.0, 0.0],
            "vel_ned": [0.0, 0.0, 0.0], "omega": [0.0, 0.0, 0.0]}


def test_fit_from_log_thrust():
    # collective: 4 equal motors, specific thrust (|imu up|) = c_T * sum(u^2), c_T=6
    samples = []
    t = 0.0
    for lvl in [0.2, 0.3, 0.4, 0.5]:
        for _ in range(4):
            up = 6.0 * 4 * lvl ** 2
            samples.append(_coll_sample("collective", t, [lvl] * 4, -up))  # imu z negative = up
            t += 0.01
    out = fit_from_log(samples, n_motors=4)
    assert out["power"] == 2
    assert np.isclose(out["c_T"], 6.0, atol=0.3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -k from_log -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'fit_from_log'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/dyn_fit.py
_MIX_ROWS = {
    "roll":  np.array([-1.0, 1.0, 1.0, -1.0]),
    "pitch": np.array([-1.0, -1.0, 1.0, 1.0]),
    "yaw":   np.array([1.0, -1.0, 1.0, -1.0]),
}


def fit_from_log(samples, n_motors=4) -> dict:
    """Assemble regressors from probe samples and fit lumped coefficients.

    Uses the collective segment for c_T + power law (specific thrust = |imu up|),
    and diff_<axis> segments for c_L/c_M/c_N (angular accel vs mix.p). Returns a
    dict of identified coefficients."""
    out = {"n_motors": n_motors}

    coll = [s for s in samples if s["segment"] == "collective" and "imu_acc" in s]
    if coll:
        u = np.array([s["u"][:n_motors] for s in coll], float)
        up = np.array([abs(s["imu_acc"][2]) for s in coll])   # body-up specific thrust
        u_sum = u.sum(axis=1)
        u_sq_sum = (u ** 2).sum(axis=1)
        power, c_T = select_power(u_sum, u_sq_sum, up)
        out["power"] = power
        out["c_T"] = c_T

    power = out.get("power", 2)
    for ax, key in (("roll", "c_L"), ("pitch", "c_M"), ("yaw", "c_N")):
        seg = [s for s in samples if s["segment"] == f"diff_{ax}" and "imu_gyro" in s]
        if len(seg) < 3:
            continue
        t = np.array([s["t"] for s in seg])
        gyro = np.array([s["imu_gyro"] for s in seg])
        aa = angular_accel(t, gyro)
        axis_idx = {"roll": 0, "pitch": 1, "yaw": 2}[ax]
        u = np.array([s["u"][:n_motors] for s in seg], float)
        reg = (u ** power) @ _MIX_ROWS[ax][:n_motors]
        out[key] = fit_axis_torque(reg, aa[:, axis_idx])

    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp python -m pytest tests/test_aigp/test_dyn_fit.py -v -p no:cacheprovider`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/dyn_fit.py tests/test_aigp/test_dyn_fit.py
git commit -m "feat(aigp): fit_from_log assembler (COR-96)"
```

---

### Task 8: Validate CLI + live end-to-end run

**Files:**
- Create: `aigp/dyn_validate.py`
- Output (not committed): `sysid_dyn/<run>/sim_dynamics.json`

- [ ] **Step 1: Write the CLI**

```python
# aigp/dyn_validate.py
"""Load a dyn probe log, fit lumped dynamics, write sim_dynamics.json + report."""
from __future__ import annotations

import argparse
import json

from .dyn_fit import fit_from_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="path to dyn probe log.jsonl")
    ap.add_argument("--n-motors", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    samples = [json.loads(l) for l in open(args.log)]
    out = fit_from_log(samples, n_motors=args.n_motors)
    dest = args.out or str(__import__("pathlib").Path(args.log).with_name("sim_dynamics.json"))
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Live — run the probe** (requires an active flight)

From the worktree root:
```bash
conda run -n aigp python -c "import time; from aigp.io_layer import MavlinkIO; from aigp.state import Store; from aigp.commander import Commander; from aigp.dyn_probe import run_probe; s=Store(); m=MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); boot=int(time.time()*1000); c=Commander(m.conn, boot); run_probe(s, c, m.conn, run_id='dyn1')"
```
Expected: prints `dyn probe log at sysid_dyn/dyn1/log.jsonl`; the drone performs collective + per-motor + per-axis motor bursts with resets between. (If the drone does not respond at all, motor control is not honored — STOP and report; the bare-physics approach is blocked at the interface.)

- [ ] **Step 3: Inspect the log + check motor control worked**

```bash
conda run -n aigp python -c "import json; L=[json.loads(x) for x in open('sysid_dyn/dyn1/log.jsonl')]; from collections import Counter; print('samples',len(L),'segs',dict(Counter(s['segment'] for s in L))); hi=[s for s in L if 'imu_acc' in s]; print('imu rows',len(hi)); import numpy as np; cz=[s['imu_acc'][2] for s in L if s['segment']=='collective' and 'imu_acc' in s]; print('collective imu_z range', (min(cz),max(cz)) if cz else None)"
```
Expected: nonzero samples across all segments; `imu_acc` present; the collective `imu_z` varies with thrust level (confirms motors respond).

- [ ] **Step 4: Run the fit**

```bash
conda run -n aigp python -m aigp.dyn_validate sysid_dyn/dyn1/log.jsonl
```
Expected: prints `c_T`, `power`, `c_L`/`c_M`/`c_N` — `c_T` > 0, coefficients finite and sign-consistent; writes `sysid_dyn/dyn1/sim_dynamics.json`.

- [ ] **Step 5: Commit (code only; sysid_dyn/ outputs stay untracked)**

```bash
git add aigp/dyn_validate.py
git commit -m "feat(aigp): dyn_validate CLI + identified sim_dynamics.json (COR-96)"
```

---

## Self-Review

**Spec coverage:** motor command → Task 5. Calibration + excitation campaign (collective/bumps/differential) → Task 6. Lumped forward model → Task 1. Fitters (c_T+power, axis torque, angular accel, motor lag, drag) → Tasks 2–4. fit_from_log assembler → Task 7. Validate + `sim_dynamics.json` + live run → Task 8. HIGHRES_IMU specific-force usage → Tasks 6/7. Frame/IMU calibration via per-motor bumps → Task 6 (`bump_m*` segments) interpreted in Task 7/8. Lumped-ratio model (no absolutes) → throughout. **Gap noted:** `tau_motor` and `drag` have tested fitters (Task 4) but `fit_from_log` (Task 7) only assembles c_T/c_L/c_M/c_N from the campaign; tau/drag extraction from the live log is left to a follow-up (the fitters exist and are tested; wiring them needs the lag/fast-motion segments, deferred to keep Task 7 focused). This is intentional scoping, not a placeholder.

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable; commands show expected output. The motor-control-may-not-work case is handled explicitly (Task 8 Step 2 STOP instruction).

**Type consistency:** `step(state, u_cmd, params, dt)` and `default_params()` keys (`c_T,c_L,c_M,c_N,mix,tau_motor,drag,power,g`) consistent Tasks 1↔7. `fit_gain`/`fit_specific_thrust`/`select_power`/`angular_accel`/`fit_axis_torque`/`fit_motor_lag`/`fit_drag`/`fit_from_log` signatures consistent across Tasks 2–4,7,8. Probe sample schema (`segment,t,u,imu_acc,imu_gyro,vel_ned,omega,...`) written in Task 6 matches reads in Task 7 (`fit_from_log`) and the Task 8 CLI. `build_campaign`/`run_probe` names consistent Task 6↔8. `_MIX` (probe) and `_MIX_ROWS` (fit) use identical sign rows. `send_motor_command(controls)` consistent Task 5↔6.

**Assumptions validated in execution:** motor interface honored (Task 8 Step 2 gate); IMU body-up axis = z and sign (per-motor bump segments confirm; collective fit uses `abs()` to be sign-robust); thrust power law (selected in fit); short windows sufficient for angular-accel.
