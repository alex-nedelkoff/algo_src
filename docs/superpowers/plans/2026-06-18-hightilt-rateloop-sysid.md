# High-tilt rate-loop sysID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline tool that refits the VQ body-rate loop in the >45° "ring" regime from live recordings (+ optional targeted excitation), with a command-replay holdout that quantifies the tilt/rate envelope where the model is trustworthy.

**Architecture:** Four small pure-Python units (numpy/scipy only) — a rate-loop *model* (linear + nonlinear 2nd-order, ZOH-propagated), a *loader* (mine `vq_data` recordings → a binned dataset), a *fitter* (train/holdout split, least-squares replay fit, per-bin R² → validity envelope), and a CLI *orchestrator* that emits a refit model + envelope JSON + a go/no-go report. Plus a pure *excitation generator* (open-loop chirp/step schedule) wired into the deploy behind a `--ringid` flag for filling regime gaps.

**Tech Stack:** Python 3, numpy, scipy (`scipy.optimize.least_squares`, `scipy.signal.cont2discrete`), pytest. No casadi (that is Phase 1). Lives in `scripts/sysid/` + `tests/sysid/`.

## Global Constraints

- Pure numpy/scipy only in `scripts/sysid/*` — no sim/mavlink/torch imports (offline + unit-testable). Exact value: the four model/loader/fit/orchestrator modules must import only `numpy`, `scipy`, stdlib.
- Validate against **measured/IMU body rate**, never differentiated odometry (the latter inflated holdout 5–8×).
- Discrete-time propagation must use **exact ZOH** at the data's sample dt (matches the deploy rate-loop integration).
- Go/no-go metric: per-axis, per-tilt-bin **command-replay holdout R² ≥ 0.9** = valid; **< 0.85** in the >45° regime = no-go for that bin. Target envelope: usable to ~60° tilt.
- Preserve existing `vq_model.json` low-tilt defaults byte-identical; the refit is written to a **sibling** `sysid/vq_model_hightilt.json`, never overwriting `vq_model.json`.
- Live excitation (`--ringid`) must ramp amplitude + keep the existing `ABORT_TILT` auto-level guard; runs hover/safe-area, never the race course.

---

### Task 1: Linear 2nd-order rate-loop model (`ring_model.py`)

**Files:**
- Create: `scripts/sysid/ring_model.py`
- Create: `scripts/sysid/__init__.py` (empty)
- Test: `tests/sysid/test_ring_model.py`
- Create: `tests/sysid/__init__.py` (empty)

**Interfaces:**
- Produces:
  - `RateLoopParams` dataclass: fields `k: float`, `wn: float`, `zeta0: float`, `zeta1: float = 0.0` (zeta1 unused until Task 2).
  - `propagate(cmd: np.ndarray, dt: float, p: RateLoopParams) -> np.ndarray` — given a 1-D command series (rad/s) and timestep, returns the predicted actual-rate series (same length), 2nd-order ZOH, zero initial state.

- [ ] **Step 1: Write the failing test**

```python
# tests/sysid/test_ring_model.py
import numpy as np
from scripts.sysid.ring_model import RateLoopParams, propagate

def test_underdamped_step_overshoot():
    # Unit-gain 2nd-order, wn=30 rad/s, zeta=0.14 -> analytic overshoot exp(-z*pi/sqrt(1-z^2))
    p = RateLoopParams(k=1.0, wn=30.0, zeta0=0.14)
    dt = 1.0 / 720.0
    n = int(1.0 / dt)
    cmd = np.ones(n)
    y = propagate(cmd, dt, p)
    overshoot = (y.max() - 1.0)
    expected = np.exp(-p.zeta0 * np.pi / np.sqrt(1 - p.zeta0**2))
    assert abs(overshoot - expected) < 0.03           # within 3% of analytic
    assert abs(y[-1] - 1.0) < 0.02                     # settles to unit gain

def test_gain_scales_output():
    p = RateLoopParams(k=2.0, wn=20.0, zeta0=0.7)
    dt = 1.0 / 720.0
    y = propagate(np.ones(int(0.5 / dt)), dt, p)
    assert abs(y[-1] - 2.0) < 0.05                     # steady state = k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_model.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.sysid.ring_model`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sysid/ring_model.py
"""VQ body-rate loop model (command omega_cmd -> measured omega), per axis.
2nd-order continuous LTI discretized by exact ZOH. Task 2 adds amplitude-dependent
damping (the nonlinear "ring"). Pure numpy/scipy, no sim deps."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import cont2discrete


@dataclass
class RateLoopParams:
    k: float          # steady-state gain (omega/omega_cmd)
    wn: float         # natural frequency (rad/s)
    zeta0: float      # base damping ratio
    zeta1: float = 0.0  # amplitude slope (Task 2); 0 = linear


def _ABCD(wn: float, zeta: float, k: float):
    # x' = A x + B u ; y = C x + D u  for  Y/U = k wn^2 / (s^2 + 2 zeta wn s + wn^2)
    A = np.array([[0.0, 1.0], [-wn * wn, -2.0 * zeta * wn]])
    B = np.array([[0.0], [wn * wn]])
    C = np.array([[k, 0.0]])
    D = np.array([[0.0]])
    return A, B, C, D


def propagate(cmd: np.ndarray, dt: float, p: RateLoopParams) -> np.ndarray:
    cmd = np.asarray(cmd, float)
    Ad, Bd, Cd, Dd, _ = cont2discrete(_ABCD(p.wn, p.zeta0, p.k), dt, method="zoh")
    x = np.zeros((2, 1))
    out = np.empty(len(cmd))
    for i, u in enumerate(cmd):
        out[i] = float(Cd @ x + Dd * u)
        x = Ad @ x + Bd * u
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_model.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/__init__.py scripts/sysid/ring_model.py tests/sysid/__init__.py tests/sysid/test_ring_model.py
git commit -m "feat(sysid): linear 2nd-order rate-loop model with ZOH propagate"
```

---

### Task 2: Amplitude-dependent damping (the nonlinear ring)

**Files:**
- Modify: `scripts/sysid/ring_model.py`
- Test: `tests/sysid/test_ring_model.py` (add cases)

**Interfaces:**
- Produces: `propagate_nl(cmd, dt, p) -> np.ndarray` — quasi-LPV: per step, `zeta = p.zeta0 + p.zeta1 * abs(omega_current)` (clamped to `[0.02, 2.0]`), re-discretized when zeta changes materially. `propagate` (Task 1) remains the linear special case (`zeta1 == 0`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/sysid/test_ring_model.py
from scripts.sysid.ring_model import propagate_nl

def test_negative_zeta1_rings_more_at_high_amplitude():
    dt = 1.0 / 720.0
    n = int(1.0 / dt)
    p = RateLoopParams(k=1.0, wn=30.0, zeta0=0.5, zeta1=-0.08)  # damping drops as |omega| grows
    small = propagate_nl(0.2 * np.ones(n), dt, p)
    large = propagate_nl(5.0 * np.ones(n), dt, p)
    os_small = small.max() / small[-1] - 1.0
    os_large = large.max() / large[-1] - 1.0
    assert os_large > os_small + 0.05                  # bigger relative overshoot at high amplitude

def test_nl_reduces_to_linear_when_zeta1_zero():
    dt = 1.0 / 720.0
    cmd = np.ones(int(0.5 / dt))
    p = RateLoopParams(k=1.0, wn=25.0, zeta0=0.3, zeta1=0.0)
    assert np.allclose(propagate_nl(cmd, dt, p), propagate(cmd, dt, p), atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'propagate_nl'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/sysid/ring_model.py
def propagate_nl(cmd: np.ndarray, dt: float, p: RateLoopParams) -> np.ndarray:
    """Quasi-LPV: damping varies with the current rate magnitude (the nonlinear ring)."""
    cmd = np.asarray(cmd, float)
    x = np.zeros((2, 1))
    out = np.empty(len(cmd))
    zeta_cur = None
    Ad = Bd = Cd = Dd = None
    for i, u in enumerate(cmd):
        omega = float(x[0, 0])
        zeta = float(np.clip(p.zeta0 + p.zeta1 * abs(omega), 0.02, 2.0))
        if zeta_cur is None or abs(zeta - zeta_cur) > 0.005:
            Ad, Bd, Cd, Dd, _ = cont2discrete(_ABCD(p.wn, zeta, p.k), dt, method="zoh")
            zeta_cur = zeta
        out[i] = float(Cd @ x + Dd * u)
        x = Ad @ x + Bd * u
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_model.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/ring_model.py tests/sysid/test_ring_model.py
git commit -m "feat(sysid): amplitude-dependent damping (nonlinear ring) in rate-loop model"
```

---

### Task 3: Recording loader → time-series dataset (`ring_loader.py`)

**Files:**
- Create: `scripts/sysid/ring_loader.py`
- Test: `tests/sysid/test_ring_loader.py`

**Interfaces:**
- Produces:
  - `RunSeries` dataclass: `dt: float`, `cmd: np.ndarray (T,3)` (omega_cmd rad/s), `omega: np.ndarray (T,3)` (measured), `tilt_deg: np.ndarray (T,)`.
  - `load_npz(path, keymap) -> RunSeries` — `keymap` is a dict mapping the four logical fields to the npz array keys (so the real schema is supplied explicitly, not guessed).
  - `DEFAULT_KEYMAP: dict` — the best-guess mapping for the current `vq_data` recorder/flightlog schema, to be confirmed against a real file in Step 1.

**Discovery (do first — the real npz schema must be confirmed):**

- [ ] **Step 0: Inspect one real recording to confirm the key names**

Run (laptop has the recordings; copy one locally or inspect over ssh):
```bash
ssh alexj@100.120.233.90 'C:\Users\alexj\miniconda3\envs\aigp\python.exe -c "import numpy as np,glob; f=sorted(glob.glob(r\"C:\Users\alexj\Documents\vq_data\*teacher*real*.npz\"))[-1]; d=np.load(f,allow_pickle=True); print(f); [print(k, getattr(d[k],\"shape\",None), getattr(d[k],\"dtype\",None)) for k in d.files]"'
```
Expected: a list of array keys with shapes. Set `DEFAULT_KEYMAP` in `ring_loader.py` to the real keys for omega_cmd, measured omega, quaternion (for tilt), and timestamps. If a recording stores commands and telemetry in *separate* files, `load_npz` takes both paths — adjust the signature accordingly and note it in the docstring. (This step produces the concrete keymap the test below encodes.)

- [ ] **Step 1: Write the failing test** (synthetic npz with the confirmed schema)

```python
# tests/sysid/test_ring_loader.py
import numpy as np
from scripts.sysid.ring_loader import load_npz, RunSeries

def _quat_from_tilt(tilt_deg):
    # roll-only quat [w,x,y,z] giving the requested tilt about body-x
    a = np.radians(tilt_deg) / 2.0
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a), np.zeros_like(a)], axis=1)

def test_load_npz_maps_fields_and_computes_tilt(tmp_path):
    T = 100
    t = np.linspace(0, 1.0, T)
    cmd = np.random.default_rng(0).standard_normal((T, 3))
    omega = cmd * 0.9
    tilt = np.linspace(0, 50, T)
    quat = _quat_from_tilt(tilt)
    f = tmp_path / "run.npz"
    np.savez(f, t=t, wcmd=cmd, omega=omega, quat=quat)
    km = {"t": "t", "cmd": "wcmd", "omega": "omega", "quat": "quat"}
    rs = load_npz(str(f), km)
    assert isinstance(rs, RunSeries)
    assert rs.cmd.shape == (T, 3) and rs.omega.shape == (T, 3)
    assert abs(rs.dt - (t[1] - t[0])) < 1e-9
    assert abs(rs.tilt_deg[-1] - 50.0) < 0.5          # tilt recovered from quat
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.sysid.ring_loader`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sysid/ring_loader.py
"""Mine vq_data recordings into a (cmd, omega, tilt) time series for rate-loop ID.
Pure numpy. The npz key mapping is passed explicitly (keymap) so the real recorder
schema is supplied, not guessed; DEFAULT_KEYMAP is the confirmed current schema."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Confirmed against a real vq_data recording in Step 0. Update if the schema differs.
DEFAULT_KEYMAP = {"t": "t", "cmd": "wcmd", "omega": "omega", "quat": "quat"}


@dataclass
class RunSeries:
    dt: float
    cmd: np.ndarray     # (T,3) omega_cmd rad/s
    omega: np.ndarray   # (T,3) measured rad/s
    tilt_deg: np.ndarray  # (T,)


def _tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    # tilt = angle of body-z from world-up = acos(R[2,2]); R[2,2] = 1 - 2(x^2+y^2)
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(r22, -1.0, 1.0)))


def load_npz(path: str, keymap: dict = None) -> RunSeries:
    km = keymap or DEFAULT_KEYMAP
    d = np.load(path, allow_pickle=True)
    t = np.asarray(d[km["t"]], float).ravel()
    cmd = np.asarray(d[km["cmd"]], float).reshape(len(t), 3)
    omega = np.asarray(d[km["omega"]], float).reshape(len(t), 3)
    quat = np.asarray(d[km["quat"]], float).reshape(len(t), 4)
    dt = float(np.median(np.diff(t)))
    return RunSeries(dt=dt, cmd=cmd, omega=omega, tilt_deg=_tilt_deg(quat))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_loader.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/ring_loader.py tests/sysid/test_ring_loader.py
git commit -m "feat(sysid): load vq_data recordings into (cmd,omega,tilt) RunSeries"
```

---

### Task 4: Regime binning + coverage report

**Files:**
- Modify: `scripts/sysid/ring_loader.py`
- Test: `tests/sysid/test_ring_loader.py` (add cases)

**Interfaces:**
- Produces: `bin_coverage(runs, axis, tilt_edges, amp_edges) -> np.ndarray` — a 2-D int count matrix (samples per `[tilt_bin, |omega_cmd| bin]` cell) for one axis (0=roll,1=pitch,2=yaw), aggregated over a list of `RunSeries`. Empty cells (count 0) are the gaps the `--ringid` campaign must fill.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/sysid/test_ring_loader.py
from scripts.sysid.ring_loader import bin_coverage, RunSeries

def test_bin_coverage_counts_cells():
    T = 50
    rs = RunSeries(dt=1/720,
                   cmd=np.column_stack([np.full(T, 3.0), np.zeros(T), np.zeros(T)]),
                   omega=np.zeros((T, 3)),
                   tilt_deg=np.full(T, 47.0))
    tilt_edges = np.array([0, 45, 60, 90])
    amp_edges = np.array([0, 2, 4, 10])
    cov = bin_coverage([rs], axis=0, tilt_edges=tilt_edges, amp_edges=amp_edges)
    assert cov.shape == (3, 3)
    assert cov[1, 1] == T          # tilt in [45,60), |cmd|=3 in [2,4)
    assert cov.sum() == T
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_loader.py::test_bin_coverage_counts_cells -v`
Expected: FAIL with `ImportError: cannot import name 'bin_coverage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/sysid/ring_loader.py
def bin_coverage(runs, axis, tilt_edges, amp_edges) -> np.ndarray:
    tilt_edges = np.asarray(tilt_edges, float)
    amp_edges = np.asarray(amp_edges, float)
    cov = np.zeros((len(tilt_edges) - 1, len(amp_edges) - 1), dtype=int)
    for rs in runs:
        ti = np.digitize(rs.tilt_deg, tilt_edges) - 1
        ai = np.digitize(np.abs(rs.cmd[:, axis]), amp_edges) - 1
        ok = (ti >= 0) & (ti < cov.shape[0]) & (ai >= 0) & (ai < cov.shape[1])
        np.add.at(cov, (ti[ok], ai[ok]), 1)
    return cov
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_loader.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/ring_loader.py tests/sysid/test_ring_loader.py
git commit -m "feat(sysid): regime binning + coverage matrix for rate-loop dataset"
```

---

### Task 5: Least-squares replay fit (`ring_fit.py`)

**Files:**
- Create: `scripts/sysid/ring_fit.py`
- Test: `tests/sysid/test_ring_fit.py`

**Interfaces:**
- Consumes: `RunSeries` (Task 3), `RateLoopParams` + `propagate_nl` (Tasks 1–2).
- Produces: `fit_axis(runs, axis, dt, x0=None) -> RateLoopParams` — concatenates the per-run command/measured series for one axis and minimizes replay error (`propagate_nl(cmd) - omega_meas`) via `scipy.optimize.least_squares` over `[k, wn, zeta0, zeta1]`. Per-run series are propagated independently (state resets per run) and stacked for the residual.

- [ ] **Step 1: Write the failing test**

```python
# tests/sysid/test_ring_fit.py
import numpy as np
from scripts.sysid.ring_model import RateLoopParams, propagate_nl
from scripts.sysid.ring_loader import RunSeries
from scripts.sysid.ring_fit import fit_axis

def _make_run(p, dt, T, seed):
    rng = np.random.default_rng(seed)
    cmd1 = np.cumsum(rng.standard_normal(T)) * 0.3          # smooth-ish excitation on roll
    cmd = np.column_stack([cmd1, np.zeros(T), np.zeros(T)])
    om1 = propagate_nl(cmd1, dt, p) + rng.standard_normal(T) * 0.02
    omega = np.column_stack([om1, np.zeros(T), np.zeros(T)])
    return RunSeries(dt=dt, cmd=cmd, omega=omega, tilt_deg=np.zeros(T))

def test_fit_recovers_known_params():
    dt, T = 1/720, 1500
    true = RateLoopParams(k=1.0, wn=28.0, zeta0=0.20, zeta1=-0.03)
    runs = [_make_run(true, dt, T, s) for s in range(3)]
    est = fit_axis(runs, axis=0, dt=dt)
    assert abs(est.wn - true.wn) / true.wn < 0.10
    assert abs(est.zeta0 - true.zeta0) < 0.05
    assert abs(est.zeta1 - true.zeta1) < 0.03
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_fit.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.sysid.ring_fit`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sysid/ring_fit.py
"""Fit the rate-loop model to recordings by command-replay least squares, per axis.
Pure numpy/scipy."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scripts.sysid.ring_model import RateLoopParams, propagate_nl


def _residual(theta, runs, axis, dt):
    k, wn, z0, z1 = theta
    p = RateLoopParams(k=k, wn=wn, zeta0=z0, zeta1=z1)
    res = []
    for rs in runs:
        pred = propagate_nl(rs.cmd[:, axis], dt, p)
        res.append(pred - rs.omega[:, axis])
    return np.concatenate(res)


def fit_axis(runs, axis, dt, x0=None) -> RateLoopParams:
    x0 = x0 or [1.0, 25.0, 0.3, 0.0]
    lb = [0.2, 3.0, 0.02, -0.2]
    ub = [3.0, 120.0, 2.0, 0.2]
    sol = least_squares(_residual, x0, bounds=(lb, ub), args=(runs, axis, dt),
                        method="trf", max_nfev=400)
    k, wn, z0, z1 = sol.x
    return RateLoopParams(k=float(k), wn=float(wn), zeta0=float(z0), zeta1=float(z1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_fit.py -v`
Expected: PASS (1 passed). (If `wn` is slightly outside 10%, widen the excitation: this is a fit-quality smoke test, not a precision claim.)

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/ring_fit.py tests/sysid/test_ring_fit.py
git commit -m "feat(sysid): least-squares command-replay fit of the rate loop"
```

---

### Task 6: Holdout R² + validity envelope

**Files:**
- Modify: `scripts/sysid/ring_fit.py`
- Test: `tests/sysid/test_ring_fit.py` (add cases)

**Interfaces:**
- Produces:
  - `holdout_r2(p, runs, axis, dt, tilt_edges) -> np.ndarray` — per-tilt-bin R² of `propagate_nl` prediction vs measured omega on held-out runs (1 value per tilt bin; `nan` where a bin has no samples). R² = `1 - SS_res/SS_tot` computed over the samples whose tilt falls in the bin.
  - `envelope(r2_by_bin, tilt_edges, thresh=0.9) -> dict` — returns `{"max_tilt_deg": float, "valid_bins": list[tuple]}`: the contiguous tilt range from 0 up to the first bin below `thresh`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/sysid/test_ring_fit.py
from scripts.sysid.ring_fit import holdout_r2, envelope
from scripts.sysid.ring_model import propagate_nl, RateLoopParams
from scripts.sysid.ring_loader import RunSeries

def test_holdout_r2_high_for_matching_model_and_envelope():
    dt, T = 1/720, 1500
    p = RateLoopParams(k=1.0, wn=28.0, zeta0=0.2, zeta1=-0.03)
    rng = np.random.default_rng(7)
    cmd1 = np.cumsum(rng.standard_normal(T)) * 0.3
    om1 = propagate_nl(cmd1, dt, p) + rng.standard_normal(T) * 0.01
    rs = RunSeries(dt=dt,
                   cmd=np.column_stack([cmd1, np.zeros(T), np.zeros(T)]),
                   omega=np.column_stack([om1, np.zeros(T), np.zeros(T)]),
                   tilt_deg=np.linspace(0, 60, T))
    edges = np.array([0, 30, 45, 60, 90])
    r2 = holdout_r2(p, [rs], axis=0, dt=dt, tilt_edges=edges)
    assert np.nanmin(r2[:3]) > 0.9                  # well-modelled where samples exist
    env = envelope(r2, edges, thresh=0.9)
    assert env["max_tilt_deg"] >= 45.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_fit.py -v`
Expected: FAIL with `ImportError: cannot import name 'holdout_r2'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to scripts/sysid/ring_fit.py
def holdout_r2(p, runs, axis, dt, tilt_edges) -> np.ndarray:
    tilt_edges = np.asarray(tilt_edges, float)
    nb = len(tilt_edges) - 1
    pred_all, meas_all, tilt_all = [], [], []
    for rs in runs:
        pred_all.append(propagate_nl(rs.cmd[:, axis], dt, p))
        meas_all.append(rs.omega[:, axis])
        tilt_all.append(rs.tilt_deg)
    pred = np.concatenate(pred_all); meas = np.concatenate(meas_all); tilt = np.concatenate(tilt_all)
    out = np.full(nb, np.nan)
    bi = np.digitize(tilt, tilt_edges) - 1
    for b in range(nb):
        m = bi == b
        if m.sum() < 10:
            continue
        ss_res = float(np.sum((meas[m] - pred[m]) ** 2))
        ss_tot = float(np.sum((meas[m] - meas[m].mean()) ** 2)) + 1e-12
        out[b] = 1.0 - ss_res / ss_tot
    return out


def envelope(r2_by_bin, tilt_edges, thresh=0.9) -> dict:
    tilt_edges = np.asarray(tilt_edges, float)
    valid, max_tilt = [], 0.0
    for b, r2 in enumerate(r2_by_bin):
        if np.isnan(r2) or r2 < thresh:
            break
        valid.append((float(tilt_edges[b]), float(tilt_edges[b + 1])))
        max_tilt = float(tilt_edges[b + 1])
    return {"max_tilt_deg": max_tilt, "valid_bins": valid}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_fit.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/ring_fit.py tests/sysid/test_ring_fit.py
git commit -m "feat(sysid): per-tilt holdout R2 + validity envelope"
```

---

### Task 7: CLI orchestrator → refit model + envelope + report (`ring_id.py`)

**Files:**
- Create: `scripts/sysid/ring_id.py`
- Test: `tests/sysid/test_ring_id.py`

**Interfaces:**
- Consumes: all of Tasks 1–6.
- Produces: `run_id(npz_paths, keymap, out_dir, tilt_edges, amp_edges, holdout_frac=0.3) -> dict` — loads runs, splits train/holdout by run, fits each axis on train, computes per-axis holdout R² + envelope, writes `vq_model_hightilt.json` (the three axes' params), `ring_envelope.json` (per-axis envelope + coverage), and `ring_report.md` (coverage table, per-axis R², the go/no-go verdict per the Global-Constraints thresholds). Returns the report dict. A `__main__` block parses `--npz`, `--out`.

- [ ] **Step 1: Write the failing test**

```python
# tests/sysid/test_ring_id.py
import json, numpy as np
from scripts.sysid.ring_model import RateLoopParams, propagate_nl
from scripts.sysid.ring_id import run_id

def _write_run(path, p, dt, T, seed, tilt_hi):
    rng = np.random.default_rng(seed)
    c = np.cumsum(rng.standard_normal(T)) * 0.3
    cmd = np.column_stack([c, c * 0.5, np.zeros(T)])
    omega = np.column_stack([propagate_nl(cmd[:, 0], dt, p),
                             propagate_nl(cmd[:, 1], dt, p), np.zeros(T)]) + rng.standard_normal((T, 3)) * 0.01
    a = np.radians(np.linspace(0, tilt_hi, T)) / 2
    quat = np.column_stack([np.cos(a), np.sin(a), np.zeros(T), np.zeros(T)])
    t = np.arange(T) * dt
    np.savez(path, t=t, wcmd=cmd, omega=omega, quat=quat)

def test_run_id_writes_outputs_and_verdict(tmp_path):
    dt, T = 1/720, 1500
    p = RateLoopParams(k=1.0, wn=28.0, zeta0=0.2, zeta1=-0.03)
    paths = []
    for s in range(4):
        f = tmp_path / f"r{s}.npz"; _write_run(str(f), p, dt, T, s, 60.0); paths.append(str(f))
    km = {"t": "t", "cmd": "wcmd", "omega": "omega", "quat": "quat"}
    rep = run_id(paths, km, str(tmp_path), tilt_edges=[0, 30, 45, 60, 90],
                 amp_edges=[0, 2, 4, 10], holdout_frac=0.25)
    assert (tmp_path / "vq_model_hightilt.json").exists()
    assert (tmp_path / "ring_envelope.json").exists()
    assert (tmp_path / "ring_report.md").exists()
    env = json.loads((tmp_path / "ring_envelope.json").read_text())
    assert env["roll"]["max_tilt_deg"] >= 45.0
    assert rep["verdict"] in ("go", "partial", "no-go")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_id.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.sysid.ring_id`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sysid/ring_id.py
"""Orchestrate high-tilt rate-loop sysID: mine recordings -> fit -> holdout -> refit
model + envelope + go/no-go report. Pure numpy/scipy + stdlib."""
from __future__ import annotations
import argparse, json, os
import numpy as np
from dataclasses import asdict
from scripts.sysid.ring_loader import load_npz, bin_coverage
from scripts.sysid.ring_fit import fit_axis, holdout_r2, envelope

AXES = ["roll", "pitch", "yaw"]


def run_id(npz_paths, keymap, out_dir, tilt_edges, amp_edges, holdout_frac=0.3) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    runs = [load_npz(p, keymap) for p in npz_paths]
    dt = float(np.median([r.dt for r in runs]))
    n_hold = max(1, int(round(len(runs) * holdout_frac)))
    train, hold = runs[n_hold:], runs[:n_hold]
    if not train:                       # tiny dataset: reuse for both
        train = runs
    params, env, r2s, cov = {}, {}, {}, {}
    for ax in range(3):
        p = fit_axis(train, ax, dt)
        r2 = holdout_r2(p, hold or train, ax, dt, tilt_edges)
        params[AXES[ax]] = asdict(p)
        r2s[AXES[ax]] = [None if np.isnan(v) else round(float(v), 3) for v in r2]
        env[AXES[ax]] = envelope(r2, tilt_edges, thresh=0.9)
        cov[AXES[ax]] = bin_coverage(runs, ax, tilt_edges, amp_edges).tolist()
    max_tilts = [env[a]["max_tilt_deg"] for a in AXES]
    verdict = "go" if min(max_tilts) >= 55.0 else ("partial" if max(max_tilts) >= 45.0 else "no-go")
    (open(os.path.join(out_dir, "vq_model_hightilt.json"), "w")
        .write(json.dumps({"rate_loop": params, "dt": dt}, indent=2)))
    env_out = {a: {**env[a], "r2_by_tilt_bin": r2s[a], "coverage": cov[a]} for a in AXES}
    open(os.path.join(out_dir, "ring_envelope.json"), "w").write(json.dumps(env_out, indent=2))
    report = {"verdict": verdict, "max_tilt_deg": dict(zip(AXES, max_tilts)),
              "tilt_edges": list(tilt_edges), "r2": r2s, "n_runs": len(runs)}
    md = [f"# Ring sysID report\n", f"**Verdict: {verdict}**  (per-axis max valid tilt: "
          + ", ".join(f"{a}={env[a]['max_tilt_deg']:.0f}°" for a in AXES) + ")\n",
          "## Per-tilt holdout R²\n", "| axis | " + " | ".join(
              f"{tilt_edges[i]:.0f}-{tilt_edges[i+1]:.0f}°" for i in range(len(tilt_edges) - 1)) + " |",
          "|" + "---|" * (len(tilt_edges)) ]
    for a in AXES:
        md.append(f"| {a} | " + " | ".join("—" if v is None else f"{v:.2f}" for v in r2s[a]) + " |")
    open(os.path.join(out_dir, "ring_report.md"), "w").write("\n".join(md) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", default="sysid")
    a = ap.parse_args()
    from scripts.sysid.ring_loader import DEFAULT_KEYMAP
    rep = run_id(a.npz, DEFAULT_KEYMAP, a.out, tilt_edges=[0, 30, 45, 52, 60, 90],
                 amp_edges=[0, 2, 4, 6, 12])
    print(json.dumps(rep, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_id.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/ring_id.py tests/sysid/test_ring_id.py
git commit -m "feat(sysid): ring_id CLI -> refit model + envelope + go/no-go report"
```

---

### Task 8: Open-loop excitation generator (`ring_excite.py`)

**Files:**
- Create: `scripts/sysid/ring_excite.py`
- Test: `tests/sysid/test_ring_excite.py`

**Interfaces:**
- Produces: `excite_schedule(axis, dt, tilt_setpoints, chirp_f0, chirp_f1, seg_s, amp) -> np.ndarray (N,4)` — a `[wx,wy,wz, hold_tilt_flag]` command schedule: for each tilt setpoint, a `seg_s`-second body-rate **chirp** (f0→f1 Hz) of amplitude `amp` on the chosen axis, ramped in over the first 20%. Column 3 carries the desired hold-tilt for the (live) outer loop. Pure numpy — the live deploy consumes this; the generator is unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/sysid/test_ring_excite.py
import numpy as np
from scripts.sysid.ring_excite import excite_schedule

def test_schedule_shape_axis_and_ramp():
    dt = 1/720
    sched = excite_schedule(axis=0, dt=dt, tilt_setpoints=[20.0, 40.0],
                            chirp_f0=1.0, chirp_f1=12.0, seg_s=2.0, amp=3.0)
    assert sched.shape[1] == 4
    assert sched.shape[0] == int(2.0 / dt) * 2          # two segments
    assert np.all(sched[:, 1] == 0) is np.bool_(False)  # roll axis is driven...
    assert np.allclose(sched[:, 2], 0.0)                # ...yaw is not
    assert abs(sched[0, 0]) < abs(sched[int(0.5/dt), 0]) # ramp-in: amplitude grows
    assert abs(sched[:, 0]).max() <= 3.0 + 1e-6          # respects amp cap
    assert sched[int(1.0/dt), 3] == 20.0                 # hold-tilt of segment 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_excite.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.sysid.ring_excite`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/sysid/ring_excite.py
"""Open-loop body-rate excitation for high-tilt rate-loop ID: ramped chirp per tilt
setpoint. Pure numpy; the live --ringid deploy mode consumes this schedule."""
from __future__ import annotations
import numpy as np


def excite_schedule(axis, dt, tilt_setpoints, chirp_f0, chirp_f1, seg_s, amp) -> np.ndarray:
    seg_n = int(seg_s / dt)
    t = np.arange(seg_n) * dt
    # linear chirp phase
    phase = 2 * np.pi * (chirp_f0 * t + 0.5 * (chirp_f1 - chirp_f0) / seg_s * t ** 2)
    base = np.sin(phase)
    ramp = np.clip(t / (0.2 * seg_s), 0.0, 1.0)             # ramp-in over first 20%
    seg_cmd = amp * ramp * base
    rows = []
    for tilt in tilt_setpoints:
        block = np.zeros((seg_n, 4))
        block[:, axis] = seg_cmd
        block[:, 3] = float(tilt)
        rows.append(block)
    return np.vstack(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && python -m pytest tests/sysid/test_ring_excite.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/sysid/ring_excite.py tests/sysid/test_ring_excite.py
git commit -m "feat(sysid): open-loop chirp excitation schedule generator"
```

---

### Task 9: Wire `--ringid` into the deploy (live; manual verification)

**Files:**
- Modify: `sysid/vq_deploy_teacher_real.py` (the laptop deploy — edit a local copy, scp + run as in this repo's deploy workflow)

**Interfaces:**
- Consumes: `excite_schedule` (Task 8).
- Produces: a `--ringid` flag that, after `fresh_start`, drives the `excite_schedule` open-loop on the wire-rate output (bypassing the trajectory tracker) while the existing collective/level loop holds the commanded tilt, recording via the existing `Recorder` + `FlightLog`. The existing `ABORT_TILT` auto-level guard stays active.

This task is **live-only** (no unit test — the deploy imports mavlink/sim and runs on hardware). It has an explicit manual-verification procedure instead of pytest.

- [ ] **Step 1: Add the flag + excitation branch**

In `sysid/vq_deploy_teacher_real.py`, near the other flag parses add:
```python
RINGID = "--ringid" in sys.argv      # COR-136 Phase 0: open-loop chirp excitation for rate-loop sysID (hover/safe-area only)
RINGAX = int(argf("--ringax", 0)); RINGAMP = argf("--ringamp", 3.0)
```
After `fresh_start()` and arm, before the main race loop, add a guarded branch:
```python
if RINGID:
    from scripts.sysid.ring_excite import excite_schedule
    sched = excite_schedule(RINGAX, 1.0/72.0, tilt_setpoints=[15, 25, 35, 45, 52],
                            chirp_f0=1.0, chirp_f1=14.0, seg_s=2.5, amp=RINGAMP)
    for k in range(len(sched)):
        ds = s.get_drone()
        if ds is None: time.sleep(0.005); continue
        tilt = float(np.degrees(np.arccos(np.clip(quat_to_R(qfix(ds.quat_wxyz))[2,2], -1, 1))))
        if tilt > ABORT_TILT:          # safety auto-level + stop
            print(f"  RINGID ABORT tilt={tilt:.0f}", flush=True); break
        rates = sched[k, :3].copy()    # open-loop wire rates on the excited axis
        thr = accel_to_thrust_norm(collective_accel(np.array([0,0,1.8*(ds0.pos_ned[2]-ds.pos_ned[2])-3.0*ds.vel_ned[2]]), ds.quat_wxyz), 0.2675, 62.0)
        c.send_attitude_target(rates, thr)
        rec.log([float(rates[0]), float(rates[1]), float(rates[2]), float(thr)])
        if flog is not None: flog.push(k/72.0, ds, {"thr": thr}, cruise=0.0, running=s.get_race_live(), armed=True)
        time.sleep(1.0/72.0)
    idle(); rec.close()
    if flog is not None: flog.close()
    sys.exit(0)
```

- [ ] **Step 2: Syntax check on the laptop**

Run: `ssh alexj@100.120.233.90 'cd /d C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src & set "PYTHONPATH=...\.claude\worktrees\aigp-client" & C:\Users\alexj\miniconda3\envs\aigp\python.exe -c "import ast;ast.parse(open(r\"sysid/vq_deploy_teacher_real.py\").read());print(\"OK\")"'`
Expected: `OK`.

- [ ] **Step 3: Manual live verification (hover/safe-area)**

Run one short, low-amplitude pass: `teacher_canon.bat`-style invocation with `--ringid --ringax 0 --ringamp 2.0`. Observe on the Rerun dashboard that the drone holds roughly level, the roll-rate command chirps, and tilt stays well under `ABORT_TILT` (auto-aborts if not). Confirm a `vq_data/*.npz` is written.
Expected: a recording exists with a clean chirp on the excited axis and no runaway.

- [ ] **Step 4: Feed the new recording through the pipeline**

Run: `python -m scripts.sysid.ring_id --npz <the new npz> <session race npzs...> --out sysid`
Expected: `ring_report.md` verdict + an envelope that now covers the excited tilt bins (coverage gaps from Task 4 reduced).

- [ ] **Step 5: Commit**

```bash
git add sysid/vq_deploy_teacher_real.py
git commit -m "feat(sysid): --ringid open-loop excitation mode for high-tilt rate-loop ID"
```

---

## Self-Review

**Spec coverage:**
- Mine existing recordings → Tasks 3–4 (load + bin/coverage). ✓
- Fit (linear 2nd-order + nonlinear amplitude-dependent) → Tasks 1, 2, 5. ✓
- Command-replay holdout vs measured rate → Task 6. ✓
- `--ringid` excitation (chirp/step sweep, ramp + auto-abort) → Tasks 8, 9. ✓
- Refit model + envelope descriptor + go/no-go report → Task 7. ✓
- Sibling `vq_model_hightilt.json`, low-tilt defaults untouched → Task 7 (writes the sibling). ✓
- Validate against measured/IMU rate, not differentiated odometry → loader uses recorded `omega` (measured); noted in Global Constraints. ✓
- Go/no-go thresholds (R² ≥ 0.9 valid, target ~60°) → Task 6 `envelope(thresh=0.9)` + Task 7 verdict. ✓

**Placeholder scan:** Task 3 Step 0 is a real discovery command (confirm npz keys), not a TBD — its output sets `DEFAULT_KEYMAP`. No "TODO"/"handle edge cases" steps remain. Task 9 is explicitly live-only with a manual procedure (no fake pytest). ✓

**Type consistency:** `RateLoopParams(k,wn,zeta0,zeta1)`, `RunSeries(dt,cmd,omega,tilt_deg)`, `propagate`/`propagate_nl(cmd,dt,p)`, `fit_axis(runs,axis,dt)`, `holdout_r2(p,runs,axis,dt,tilt_edges)`, `envelope(r2,edges,thresh)`, `run_id(npz_paths,keymap,out_dir,tilt_edges,amp_edges,holdout_frac)` — names/signatures consistent across tasks. ✓

**Note:** tasks import as `scripts.sysid.*`, so run pytest from the repo root (`cd algo_src`); add an empty `scripts/__init__.py` if `scripts` is not already a package (check in Task 1 Step 2 — if the import error is about `scripts` not `scripts.sysid`, create `scripts/__init__.py` and re-run).
