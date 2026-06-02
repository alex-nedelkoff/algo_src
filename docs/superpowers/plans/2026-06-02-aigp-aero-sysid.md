# AI-GP Aerodynamic SysID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify the AI-GP sim drone's aerodynamic forces and moments as a parametric model + learned residual, validated against held-out flight data (`sim_aero.json` + report).

**Architecture:** Feature-based grey-box: the parametric aero terms are *linear in their coefficients*, so identification is a convex least-squares solve over regressor features (drag, thrust-droop, weathervane, damping); a small differentiable residual MLP then mops up the leftover. Pure-math modules (`aero_model`, `aero_dataset`, `aero_fit`, `aero_validate`) are unit-tested offline with synthetic data; data capture + final validation run on the live sim.

**Tech Stack:** Python 3.13, NumPy, SciPy (lstsq), PyTorch (residual MLP), pandas + pyarrow (parquet), pymavlink; `aigp` conda env on the Windows sim host. Spec: `docs/superpowers/specs/2026-06-02-aigp-aero-sysid-design.md`.

---

## Working environment (read first)
- Code + tests run in the **`aigp` conda env on the Windows host** (`ssh alexj@100.120.233.90`).
- Edit on Mac → `scp` to `C:/Users/alexj/Documents/algo_src/.claude/worktrees/aigp-client/<path>` → run over ssh. Do **NOT** pipe (`| tail`, `| grep`, `| wc`) *inside* the ssh command — it runs in Windows cmd and breaks. Pipe on the Mac side, outside the closing quote.
  - Run: `ssh alexj@100.120.233.90 "cd C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client && conda run -n aigp <cmd> 2>&1" | tail -30`
- Tests live in **`tests/test_aigp/`** (pytest runs from the worktree root).
- Commit on the worktree branch `aigp-gate-data-collection`; end commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Live flight scripts require a **live qualifier flight** (only the user drives the GUI); gate actuation on race-live via `fresh_start()` (copy the one in `race_cruise.py`). Reference: `aigp/io_layer.py` (HIGHRES_IMU specific force + gyro; ODOMETRY world vel/attitude), `aigp/geometry.py::quat_to_R`, `aigp/dyn_model.py` (lumped EOM), `aigp/commander.py`, `capture_diag.py` (telemetry-capture pattern to extend).

## Conventions (used throughout)
- Body frame **FRD** (x-fwd, y-right, z-down). Thrust accel acts along **body −z**: `thrust_body = [0, 0, -T]`.
- IMU `HIGHRES_IMU` gives **specific force** `f_body = (thrust + aero)/m` (gravity excluded) and gyro `ω`.
- So **aero force accel** `a_aero = f_body − thrust_body`; on a **coast** (T≈0) it's just `f_body`.
- World velocity → body: `v_body = R.T @ v_world` (R = body→world from `quat_to_R`).
- No wind in sim → airspeed = ground velocity; air density folds into the drag coefficients.

## File structure
- `aigp/aero_model.py` — regressor features + predict (pure). One job: parametric aero force/moment as `features·θ`.
- `aigp/aero_dataset.py` — raw log → `(features, targets, coast_mask)` (pure). One job: physics bookkeeping.
- `aigp/aero_fit.py` — lstsq parametric fit + residual MLP (pure). One job: estimate θ + residual.
- `aigp/aero_validate.py` — held-out metrics + `sim_aero.json` + report (pure math, file I/O).
- `aero_capture.py` (root) — live data-collection driver (maneuver battery + coast probes) → parquet + manifest.
- Tests: `tests/test_aigp/test_aero_model.py`, `test_aero_dataset.py`, `test_aero_fit.py`, `test_aero_validate.py`.

---

## Task 0: Setup — deps + data dir

**Files:** none (environment only)

- [ ] **Step 1: Install deps into the aigp env**

Run: `ssh alexj@100.120.233.90 "conda run -n aigp pip install torch pandas pyarrow scipy 2>&1" | tail -5`
Expected: all install (numpy/matplotlib already present from earlier work).

- [ ] **Step 2: Create the data directory**

Run: `ssh alexj@100.120.233.90 "cd C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client && mkdir aero_data 2>nul & echo ok"`
Expected: `aero_data/` exists (will hold per-run parquet + the fitted `sim_aero.json`).

- [ ] **Step 3: Add `aero_data/` to .gitignore (logs are large, not source)**

Append `aero_data/` to `.gitignore` (create if absent). Run: `ssh ... "cd ... && git add .gitignore && git commit -m 'chore: ignore aero_data/ logs' -m 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>' 2>&1" | tail -3`

---

## Task 1: `aero_model.py` — regressor features + predict (TDD)

**Files:** Create `aigp/aero_model.py`; Test `tests/test_aigp/test_aero_model.py`

The model is **linear in θ**: `a_aero = Φ_F(v) · θ_F`, `α_aero = Φ_M(v, ω) · θ_M`. Φ are the feature matrices; θ the coefficients. This is what makes the fit a clean lstsq.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aigp/test_aero_model.py
import numpy as np
from aigp.aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS, predict

def test_force_features_shape_and_values():
    v = np.array([2.0, -3.0, 1.0])
    phi = force_features(v)                      # (3 axes, n_force_params)
    assert phi.shape == (3, len(FORCE_COLS))
    # linear-drag block: diagonal -v ; quadratic block: diagonal -v|v|
    iD = FORCE_COLS.index("D_x"); iC = FORCE_COLS.index("C_x")
    assert np.isclose(phi[0, iD], -2.0)
    assert np.isclose(phi[0, iC], -2.0 * 2.0)

def test_force_features_are_linear_in_theta():
    v = np.array([1.0, 0.0, -2.0])
    theta = np.zeros(len(FORCE_COLS)); theta[FORCE_COLS.index("C_z")] = 0.5
    a = force_features(v) @ theta
    assert np.isclose(a[2], 0.5 * (-(-2.0) * 2.0))   # -C_z * v_z|v_z| = -0.5*(-2*2)=2.0

def test_moment_features_damping_and_weathervane():
    v = np.array([5.0, 0.0, 0.0]); w = np.array([0.0, 0.0, 0.3])
    phi = moment_features(v, w)
    assert phi.shape == (3, len(MOMENT_COLS))
    idw = MOMENT_COLS.index("d_z")               # damping on yaw
    assert np.isclose(phi[2, idw], -0.3)
    assert "w_x" in MOMENT_COLS and "w_z" in MOMENT_COLS   # weathervane velocity-coupling cols exist

def test_predict_roundtrip():
    v = np.array([3.0, 1.0, 0.0]); w = np.array([0.1, 0.0, 0.0])
    thF = np.zeros(len(FORCE_COLS)); thF[FORCE_COLS.index("D_x")] = 0.2
    thM = np.zeros(len(MOMENT_COLS)); thM[MOMENT_COLS.index("d_x")] = 0.05
    aF, aM = predict(v, w, thF, thM)
    assert np.isclose(aF[0], -0.2 * 3.0)
    assert np.isclose(aM[0], -0.05 * 0.1)
```

- [ ] **Step 2: Run, verify FAIL**

Run: `ssh ... "cd ... && conda run -n aigp pytest tests/test_aigp/test_aero_model.py -q 2>&1" | tail -20`
Expected: FAIL (`ModuleNotFoundError: aigp.aero_model`).

- [ ] **Step 3: Implement `aigp/aero_model.py`**

```python
"""Parametric aero model as linear-in-coefficients regressor features (pure numpy, body frame).
Forces:  a_aero = -D*v - C*v|v|             (per-axis D, C; accel units, lumped /mass)
Moments: alpha_aero = -d*w + weathervane(v) (per-axis damping d; velocity-coupling 'w_*' terms)
The weathervane block lets a body-velocity component drive a yaw/pitch moment (the destabilizing
tail-first term); kept linear so the whole fit is a least-squares solve. Coefficients live in two
flat vectors theta_F (FORCE_COLS) and theta_M (MOMENT_COLS)."""
from __future__ import annotations
import numpy as np

FORCE_COLS = ["D_x", "D_y", "D_z", "C_x", "C_y", "C_z"]
# moment: per-axis rotational damping d_* + weathervane coupling w_* (axial/lateral velocity -> moment)
MOMENT_COLS = ["d_x", "d_y", "d_z", "w_x", "w_y", "w_z"]


def force_features(v_body) -> np.ndarray:
    """(3, len(FORCE_COLS)) so that force_features(v) @ theta_F = -D*v - C*v|v| (per axis)."""
    v = np.asarray(v_body, float)
    phi = np.zeros((3, len(FORCE_COLS)))
    for i in range(3):
        phi[i, i] = -v[i]                 # -D_i * v_i
        phi[i, 3 + i] = -v[i] * abs(v[i])  # -C_i * v_i|v_i|
    return phi


def moment_features(v_body, omega) -> np.ndarray:
    """(3, len(MOMENT_COLS)) so that moment_features(v,w) @ theta_M = -d*w + weathervane(v).
    Weathervane: axial body velocity (v_x, flying fore/aft) couples into the lateral moments
    (pitch about y, yaw about z) — the standard tail-first destabilizer. Encoded as w_y, w_z on
    v_x, and w_x reserved for a roll-coupling term."""
    v = np.asarray(v_body, float); w = np.asarray(omega, float)
    phi = np.zeros((3, len(MOMENT_COLS)))
    for i in range(3):
        phi[i, i] = -w[i]                 # -d_i * w_i (damping)
    # weathervane velocity-coupling (linear in the w_* coefficients):
    phi[0, 3] = v[1]                       # roll moment from lateral velocity   (coef w_x)
    phi[1, 4] = v[0]                       # pitch moment from axial velocity     (coef w_y)
    phi[2, 5] = v[0]                       # yaw moment from axial velocity       (coef w_z)
    return phi


def predict(v_body, omega, theta_F, theta_M):
    """Return (a_aero(3,), alpha_aero(3,))."""
    return force_features(v_body) @ np.asarray(theta_F, float), \
           moment_features(v_body, omega) @ np.asarray(theta_M, float)
```

- [ ] **Step 4: Run, verify PASS** (`4 passed`). Run the pytest command from Step 2.

- [ ] **Step 5: Commit**
```bash
git add aigp/aero_model.py tests/test_aigp/test_aero_model.py
git commit -m "feat(aero): linear-in-theta parametric aero feature model" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `aero_dataset.py` — log → features + targets (TDD)

**Files:** Create `aigp/aero_dataset.py`; Test `tests/test_aigp/test_aero_dataset.py`

Turns a raw per-sample log into the regression problem: stacked feature rows + target accels + a coast mask. Targets: `a_aero = f_body − thrust_body`; `alpha_aero = filtered(dω/dt) + ω×(I_ratio·ω) − motor_torque`. For the first cut we rely on **coast samples** (thrust=0, motor=0) where targets are clean, plus powered samples with the known thrust/torque models.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aigp/test_aero_dataset.py
import numpy as np
import pandas as pd
from aigp.aero_dataset import build_targets, world_to_body_vel, finite_diff_filtered

def test_world_to_body_vel_identity():
    R = np.eye(3); v_world = np.array([1.0, 2.0, 3.0])
    assert np.allclose(world_to_body_vel(v_world, R), v_world)

def test_finite_diff_filtered_constant_rate():
    t = np.linspace(0, 1, 101)
    w = np.stack([2.0 * t, 0 * t, -1.0 * t], axis=1)   # constant angular accel [2,0,-1]
    a = finite_diff_filtered(t, w)
    assert np.allclose(np.median(a[5:-5], axis=0), [2.0, 0.0, -1.0], atol=0.1)

def test_build_targets_coast_force_is_specific_force():
    # one coast sample: thrust off -> aero force target == f_body
    df = pd.DataFrame({
        "t": [0.0, 0.004], "vx": [5.0, 5.0], "vy": [0, 0], "vz": [0, 0],
        "qw": [1, 1], "qx": [0, 0], "qy": [0, 0], "qz": [0, 0],
        "wx": [0, 0], "wy": [0, 0], "wz": [0, 0],
        "fx": [-1.2, -1.2], "fy": [0, 0], "fz": [0, 0],   # specific force (aero only, coasting)
        "thrust_accel": [0.0, 0.0], "coast": [True, True],
    })
    out = build_targets(df, I_ratio=np.array([3.7, 1.0, 1.0]))
    assert out["a_aero"].shape[1] == 3
    assert np.allclose(out["a_aero"][0], [-1.2, 0.0, 0.0])
    assert out["coast"].all()
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: aigp.aero_dataset`).

- [ ] **Step 3: Implement `aigp/aero_dataset.py`**

```python
"""Raw flight log -> regression dataset for aero sysID (pure numpy/pandas).
Input columns (per sample): t, vx/vy/vz (world NED), qw/qx/qy/qz (body->world), wx/wy/wz (gyro),
fx/fy/fz (HIGHRES_IMU specific force, body), thrust_accel (commanded thrust accel magnitude),
coast (bool). Output: aero force/moment targets + the coast mask. Feature stacking is done in
aero_fit so the same targets serve both the lstsq backbone and the residual."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .geometry import quat_to_R


def world_to_body_vel(v_world, R) -> np.ndarray:
    return np.asarray(R, float).T @ np.asarray(v_world, float)


def finite_diff_filtered(t, w, win: int = 9) -> np.ndarray:
    """Angular acceleration = d(omega)/dt, central difference + moving-average smoothing."""
    t = np.asarray(t, float); w = np.asarray(w, float)
    a = np.gradient(w, t, axis=0)
    if win > 1 and len(a) > win:
        k = np.ones(win) / win
        a = np.stack([np.convolve(a[:, j], k, mode="same") for j in range(a.shape[1])], axis=1)
    return a


def build_targets(df: pd.DataFrame, I_ratio) -> dict:
    """Compute per-sample body velocity, aero force target, aero moment target, coast mask.
    a_aero  = f_body - thrust_body        (thrust along body -z)
    al_aero = alpha + omega x (I_ratio*omega) - motor_torque  (motor_torque omitted here: rely on
              coast/zero-motor samples; powered-moment fitting can add it later)."""
    t = df["t"].to_numpy()
    Ir = np.asarray(I_ratio, float)
    vb = np.zeros((len(df), 3)); aero_f = np.zeros((len(df), 3))
    w = df[["wx", "wy", "wz"]].to_numpy()
    f = df[["fx", "fy", "fz"]].to_numpy()
    for i in range(len(df)):
        R = quat_to_R(df.loc[df.index[i], ["qw", "qx", "qy", "qz"]].to_numpy())
        vb[i] = world_to_body_vel(df.loc[df.index[i], ["vx", "vy", "vz"]].to_numpy(), R)
        thrust_body = np.array([0.0, 0.0, -float(df["thrust_accel"].iloc[i])])
        aero_f[i] = f[i] - thrust_body
    alpha = finite_diff_filtered(t, w)
    gyro_coupling = np.cross(w, Ir * w)
    aero_m = alpha + gyro_coupling                       # coast/zero-motor: motor_torque = 0
    return {"v_body": vb, "omega": w, "a_aero": aero_f, "al_aero": aero_m,
            "coast": df["coast"].to_numpy().astype(bool)}
```

- [ ] **Step 4: Run, verify PASS** (`3 passed`).

- [ ] **Step 5: Commit**
```bash
git add aigp/aero_dataset.py tests/test_aigp/test_aero_dataset.py
git commit -m "feat(aero): log->targets (body vel, aero force/moment, coast mask)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `aero_fit.py` — least-squares parametric backbone (TDD)

**Files:** Create `aigp/aero_fit.py`; Test `tests/test_aigp/test_aero_fit.py`

Stack the per-sample feature matrices into one big regression and solve with `lstsq`. Synthetic round-trip = the COR-96 recovery test: generate data from known θ, recover it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aigp/test_aero_fit.py
import numpy as np
from aigp.aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS, predict
from aigp.aero_fit import fit_parametric

def _make_synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    thF = np.zeros(len(FORCE_COLS)); thF[FORCE_COLS.index("C_x")] = 0.05; thF[FORCE_COLS.index("D_y")] = 0.2
    thM = np.zeros(len(MOMENT_COLS)); thM[MOMENT_COLS.index("d_z")] = 0.04; thM[MOMENT_COLS.index("w_z")] = 0.01
    V = rng.uniform(-20, 20, (n, 3)); W = rng.uniform(-3, 3, (n, 3))
    aF = np.array([force_features(V[i]) @ thF for i in range(n)])
    aM = np.array([moment_features(V[i], W[i]) @ thM for i in range(n)])
    return V, W, aF, aM, thF, thM

def test_recovers_force_and_moment_params():
    V, W, aF, aM, thF, thM = _make_synth()
    res = fit_parametric(V, W, aF, aM)
    assert np.allclose(res["theta_F"], thF, atol=1e-6)
    assert np.allclose(res["theta_M"], thM, atol=1e-6)
    assert res["r2_force"] > 0.999 and res["r2_moment"] > 0.999

def test_robust_to_noise():
    V, W, aF, aM, thF, thM = _make_synth(n=2000)
    aF += np.random.default_rng(1).normal(0, 0.05, aF.shape)
    res = fit_parametric(V, W, aF, aM)
    assert np.allclose(res["theta_F"], thF, atol=0.02)
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: aigp.aero_fit`).

- [ ] **Step 3: Implement `fit_parametric` in `aigp/aero_fit.py`**

```python
"""Fit the parametric aero model by least-squares (linear in theta), then (Task 4) a residual MLP."""
from __future__ import annotations
import numpy as np
from .aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS


def _stack(feature_fn, V, W=None):
    rows = []
    for i in range(len(V)):
        rows.append(feature_fn(V[i]) if W is None else feature_fn(V[i], W[i]))
    return np.vstack(rows)                       # (3n, n_params)


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean(0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_parametric(V, W, a_force, a_moment, weights=None) -> dict:
    """V,W: (n,3) body vel & rates. a_force,a_moment: (n,3) targets. weights: optional (n,) per-sample
    (e.g. up-weight coast samples). Returns theta_F, theta_M, per-term R^2, residual arrays."""
    V = np.asarray(V, float); W = np.asarray(W, float)
    PhiF = _stack(force_features, V)             # (3n, |FORCE_COLS|)
    PhiM = _stack(moment_features, V, W)         # (3n, |MOMENT_COLS|)
    yF = np.asarray(a_force, float).reshape(-1); yM = np.asarray(a_moment, float).reshape(-1)
    if weights is not None:
        sw = np.sqrt(np.repeat(np.asarray(weights, float), 3))
        PhiF, yF = PhiF * sw[:, None], yF * sw
        PhiM, yM = PhiM * sw[:, None], yM * sw
    thF, *_ = np.linalg.lstsq(PhiF, yF, rcond=None)
    thM, *_ = np.linalg.lstsq(PhiM, yM, rcond=None)
    return {
        "theta_F": thF, "theta_M": thM,
        "force_cols": FORCE_COLS, "moment_cols": MOMENT_COLS,
        "r2_force": _r2(np.asarray(a_force).reshape(-1), _stack(force_features, V) @ thF),
        "r2_moment": _r2(np.asarray(a_moment).reshape(-1), _stack(moment_features, V, W) @ thM),
    }
```

- [ ] **Step 4: Run, verify PASS** (`2 passed`).

- [ ] **Step 5: Commit**
```bash
git add aigp/aero_fit.py tests/test_aigp/test_aero_fit.py
git commit -m "feat(aero): least-squares parametric backbone fit + R^2" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: residual MLP — fit the leftover (TDD)

**Files:** Modify `aigp/aero_fit.py`; Test add to `tests/test_aigp/test_aero_fit.py`

After the parametric fit, the residual `r = target − parametric` is what physics missed. Fit a small torch MLP on `(v_body, ω) → r`; report the **residual variance share** (the "how fully known" number = `1 − var(residual_after_MLP)/var(target)`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_aigp/test_aero_fit.py
import numpy as np
from aigp.aero_fit import fit_residual

def test_residual_mlp_reduces_leftover():
    rng = np.random.default_rng(3); n = 1500
    V = rng.uniform(-10, 10, (n, 3)); W = rng.uniform(-2, 2, (n, 3))
    # a nonlinear residual the linear model cannot capture
    r = np.stack([0.1 * np.sin(V[:, 0]) * np.abs(V[:, 1]), 0 * V[:, 0], 0.05 * V[:, 2] ** 2], axis=1)
    model, info = fit_residual(V, W, r, epochs=300)
    assert info["residual_var_share"] < 0.5     # MLP explains >50% of the leftover variance
    pred = model.predict(V, W)
    assert pred.shape == (n, 3)
```

- [ ] **Step 2: Run, verify FAIL** (`fit_residual` undefined).

- [ ] **Step 3: Add `fit_residual` + `ResidualMLP` to `aigp/aero_fit.py`**

```python
# --- append to aigp/aero_fit.py ---
import torch
import torch.nn as nn


class ResidualMLP:
    """Small MLP r(v_body, omega) -> (dF or dTau) 3-vector. Differentiable, drops into a replica later."""
    def __init__(self, net):
        self._net = net

    def predict(self, V, W):
        x = torch.tensor(np.hstack([np.asarray(V, float), np.asarray(W, float)]), dtype=torch.float32)
        with torch.no_grad():
            return self._net(x).numpy()


def fit_residual(V, W, residual, hidden=64, epochs=400, lr=1e-3):
    """Fit an MLP to the parametric residual. Returns (ResidualMLP, info{residual_var_share})."""
    X = torch.tensor(np.hstack([np.asarray(V, float), np.asarray(W, float)]), dtype=torch.float32)
    Y = torch.tensor(np.asarray(residual, float), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, 3))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net(X) - Y) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        leftover = (net(X) - Y).numpy()
    share = float(np.var(leftover) / (np.var(np.asarray(residual)) + 1e-12))
    return ResidualMLP(net), {"residual_var_share": share}
```

- [ ] **Step 4: Run, verify PASS** (now 3 tests in the file pass). If torch training is flaky, bump `epochs` to 600.

- [ ] **Step 5: Commit**
```bash
git add aigp/aero_fit.py tests/test_aigp/test_aero_fit.py
git commit -m "feat(aero): residual MLP + residual-variance-share metric" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `aero_validate.py` — held-out metrics + deliverable (TDD)

**Files:** Create `aigp/aero_validate.py`; Test `tests/test_aigp/test_aero_validate.py`

Single-step prediction error on held-out data (force & moment, per axis), residual share, the per-term contribution breakdown, and serialize `sim_aero.json`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aigp/test_aero_validate.py
import json, numpy as np
from aigp.aero_model import force_features, moment_features, FORCE_COLS, MOMENT_COLS
from aigp.aero_validate import single_step_metrics, term_contributions, save_model

def _synth(n=300, seed=5):
    rng = np.random.default_rng(seed)
    thF = np.zeros(len(FORCE_COLS)); thF[FORCE_COLS.index("C_x")] = 0.05
    V = rng.uniform(-15, 15, (n, 3)); W = rng.uniform(-2, 2, (n, 3))
    aF = np.array([force_features(V[i]) @ thF for i in range(n)])
    return V, W, aF, thF

def test_single_step_metrics_perfect_fit():
    V, W, aF, thF = _synth()
    thM = np.zeros(len(MOMENT_COLS))
    m = single_step_metrics(V, W, aF, np.zeros_like(aF), thF, thM, residual=None)
    assert m["rmse_force"] < 1e-9 and m["r2_force"] > 0.999

def test_term_contributions_identifies_dominant():
    V, W, aF, thF = _synth()
    contrib = term_contributions(V, thF, force_features, FORCE_COLS)
    assert max(contrib, key=contrib.get) == "C_x"

def test_save_model_roundtrip(tmp_path):
    p = tmp_path / "sim_aero.json"
    save_model(str(p), theta_F=np.array([1.0, 2.0]), theta_M=np.array([3.0]),
               force_cols=["D_x", "D_y"], moment_cols=["d_x"], meta={"envelope_max_speed": 22.0})
    d = json.load(open(p))
    assert d["theta_F"] == [1.0, 2.0] and d["meta"]["envelope_max_speed"] == 22.0
```

- [ ] **Step 2: Run, verify FAIL** (`ModuleNotFoundError: aigp.aero_validate`).

- [ ] **Step 3: Implement `aigp/aero_validate.py`**

```python
"""Held-out validation + deliverable serialization for the aero sysID."""
from __future__ import annotations
import json
import numpy as np
from .aero_model import force_features, moment_features


def _rmse(y, yhat): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))
def _r2(y, yhat):
    y = np.asarray(y); ss = float(np.sum((y - yhat) ** 2)); tot = float(np.sum((y - y.mean(0)) ** 2))
    return 1.0 - ss / tot if tot > 0 else 0.0


def _predict_stack(feature_fn, V, theta, W=None):
    return np.array([(feature_fn(V[i]) if W is None else feature_fn(V[i], W[i])) @ theta
                     for i in range(len(V))])


def single_step_metrics(V, W, a_force, a_moment, theta_F, theta_M, residual=None) -> dict:
    """Per-axis force/moment RMSE + R^2 on held-out data. residual: optional ResidualMLP."""
    pf = _predict_stack(force_features, V, theta_F)
    pm = _predict_stack(moment_features, V, theta_M, W)
    if residual is not None:
        pf = pf + residual.predict(V, W)
    return {
        "rmse_force": _rmse(a_force, pf), "r2_force": _r2(a_force, pf),
        "rmse_moment": _rmse(a_moment, pm), "r2_moment": _r2(a_moment, pm),
        "rmse_force_axes": [_rmse(a_force[:, j], pf[:, j]) for j in range(3)],
    }


def term_contributions(V, theta, feature_fn, cols) -> dict:
    """Variance each parametric column contributes to the predicted force/moment (per-term breakdown)."""
    out = {}
    for k, name in enumerate(cols):
        tk = np.zeros_like(theta); tk[k] = theta[k]
        out[name] = float(np.var(_predict_stack(feature_fn, V, tk)))
    return out


def save_model(path, theta_F, theta_M, force_cols, moment_cols, meta=None):
    json.dump({"theta_F": list(map(float, theta_F)), "theta_M": list(map(float, theta_M)),
               "force_cols": list(force_cols), "moment_cols": list(moment_cols),
               "meta": meta or {}}, open(path, "w"), indent=2)
```

- [ ] **Step 4: Run, verify PASS** (`3 passed`). Then run the full aero suite: `pytest tests/test_aigp/test_aero_*.py -q` → all green.

- [ ] **Step 5: Commit**
```bash
git add aigp/aero_validate.py tests/test_aigp/test_aero_validate.py
git commit -m "feat(aero): held-out metrics, term contributions, sim_aero.json serialize" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `aero_capture.py` — live data-collection driver (run-and-verify; SIM LIVE)

**Files:** Create `aero_capture.py` (root)

Logs the per-sample row (Conventions section) to parquet while flying a chosen maneuver. **Requires the user to confirm a live qualifier flight.** Reuse `fresh_start()` + the rate-control inner loop from `race_cruise.py`, and the IMU/odometry reads from `io_layer`.

- [ ] **Step 1: Implement the capture loop + the 6 maneuvers**

Build `aero_capture.py` with: a `Logger` that appends rows `{t, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz, fx,fy,fz, thrust_accel, coast}` (specific force + gyro from `s.get_imu()`, world vel/quat from `s.get_drone()`), and one function per maneuver selected by `argv[1]`:
- `sweep <axis> <vmax>` — race_cruise toward ±axis ramping speed; powered.
- `circle <radius> <speed>` / `lemniscate <speed>` — parametric horizontal path via the velocity setpoint into the rate loop; powered.
- `doublet <axis> <speed>` — reach speed, inject a ±rate doublet on the chosen body axis; powered.
- `thrustchirp <speed>` — at a held speed, sinusoidally vary collective; powered.
- `coast <dir> <orient> <entry_speed>` — build speed (controller or short motor burst), set `coast=True`, send **idle/zero thrust**, log the ~300–500 ms decel/twist window (`thrust_accel=0`).

Each run: `fresh_start()` → `arm()` → maneuver → write `aero_data/<maneuver>_<stamp>.parquet` + a one-line manifest append.

- [ ] **Step 2: Smoke a single safe run (sim live)**

Ask the user to confirm a live qualifier. Run a hover capture:
`ssh ... "cd ... && conda run -n aigp python aero_capture.py sweep x 4 2>&1" | tail -15`
Expected: a parquet appears in `aero_data/`; print the row count + speed range reached.

- [ ] **Step 3: Verify the parquet has clean targets**

Run a tiny check that loads the parquet and runs `aero_dataset.build_targets` on it; print `a_aero` range on the coast rows (should be small at low speed, growing with speed). Confirms the capture→targets path end-to-end.

- [ ] **Step 4: Commit**
```bash
git add aero_capture.py
git commit -m "feat(aero): live data-capture driver (sweeps/circle/lemniscate/doublet/chirp/coast)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Collect → fit → validate (run-and-verify; SIM LIVE)

**Files:** Create `aero_run.py` (root) — orchestration glue

- [ ] **Step 1: Collect to coverage (sim live)**

Run the maneuver battery, **weighting coasts** and the hard corners (high speed × off-axis × tail-first). Track a coverage map over (speed-bin × velocity-direction × orientation). Keep collecting until the convergence rule trips (Step 3). Log everything to `aero_data/`.

- [ ] **Step 2: Fit (offline)**

`aero_run.py` loads all `aero_data/*.parquet` (minus the held-out runs), runs `build_targets` → `fit_parametric` (coast-weighted) → `fit_residual` on the leftover.
Run: `ssh ... "cd ... && conda run -n aigp python aero_run.py fit 2>&1" | tail -20`
Expected: prints `theta_F`/`theta_M`, per-term R², residual-variance-share.

- [ ] **Step 3: Convergence check (offline, drives Step 1)**

Re-fit on increasing data fractions; stop collecting when (a) `theta` deltas are within noise, (b) held-out single-step RMSE plateaus, (c) residual-variance-share stops dropping. Print the three curves; declare convergence.

- [ ] **Step 4: Validate on held-out + write deliverable (offline)**

Run `single_step_metrics` + a short-horizon rollout (integrate force+moment+residual over held-out windows from recorded initial state + commands; report position/velocity RMSE) + `term_contributions`; `save_model("aero_data/sim_aero.json", ..., meta={"envelope_max_speed": <max v reached>})`; write a markdown report (RMSE tables, residual share, term breakdown, coverage/envelope note).
Run: `ssh ... "cd ... && conda run -n aigp python aero_run.py validate 2>&1" | tail -25`
Expected: held-out force/moment R² high, residual share small + bounded, `sim_aero.json` written.

- [ ] **Step 5: Commit the deliverable + report**
```bash
git add aero_run.py docs/aero_sysid_report.md
git commit -m "feat(aero): collect/fit/validate orchestration + identified sim_aero.json report" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(`aero_data/` is gitignored; `sim_aero.json` is the small deliverable — copy it out of `aero_data/` or add an explicit exception so it's tracked.)

---

## Self-review notes
- **Spec coverage:** model terms (T1 force/moment features incl. weathervane), targets incl. thrust/coast + gyro-coupling (T2), lstsq backbone (T3), residual + share (T4), held-out single-step + term-breakdown + serialize (T5), rollout (T7-S4), data-collection battery + coast probe + coverage/convergence (T6, T7), pitfalls (coast-weighting in fit T3/T7, filtered α T2). The **thrust-vs-speed `k_h` term** and the **μ-multinomial moment refinement** are noted in the spec; T1's feature model uses the simpler velocity-coupling weathervane + leaves `k_h`/BEMT as a documented extension (add a `v_h^2` force column if the thrust residual stays large — YAGNI per spec).
- **Types consistent:** `FORCE_COLS`/`MOMENT_COLS`, `force_features`/`moment_features`/`predict`, `build_targets`→dict keys (`v_body`,`omega`,`a_aero`,`al_aero`,`coast`), `fit_parametric`→`theta_F`/`theta_M`/`r2_*`, `fit_residual`→`ResidualMLP.predict`, `single_step_metrics`/`term_contributions`/`save_model` — used consistently across T1–T7.
- **Live vs unit:** T1–T5 are pure/offline TDD; T6–T7 are live run-and-verify (need the sim + user-confirmed qualifier), with explicit expected observations.

---

## Phase 2 — literature-grounded refinement (added 2026-06-02 after NotebookLM review)

**Phase 1 outcome:** full grey-box pipeline built + validated; **`D_x≈0.52/s` rotor drag identified and held-out-validated (0.16 m/s² RMSE) — matches published ground-truth 0.49–0.54/s** (NeuroBEM/Faessler/grey-box circle+lemniscate trials). Moments-from-coast and vertical force did NOT converge. Literature (this notebook) explains why and points the way:

### What the literature changed
- **Free tumbles are the wrong regime for moments.** Rotor-coupled aero moments (hub moment ∝ √T·ω, H-force/weathervane) physically require *spinning rotors* — motors-off removes the very effect we want. Standard practice (NeuroBEM, Faessler) is **powered flight with the rotor/control torque modeled**.
- **Off-axis coast ≠ clean drag.** Empirically confirmed here: a 45° diagonal coast carries a non-diagonal bluff-body force that `−D·v` can't represent and it corrupts `D_x`. Lateral drag must come from **powered** maneuvers (as the published `d_y` did).
- **Vertical force**: free-fall coast = vortex-ring/windmill-brake state (momentum theory fails). `d_z` is minor / often set to zero — **pinned to 0** in the deliverable.
- **Residual**: a memoryless MLP overfits. NeuroBEM-style residuals need **temporal context** (~20-state history, often a **TCN**).

### Phase 2 maneuver curriculum (all powered; coast retained only for `D_x` cross-check)
- **`lemniscate` / `circle`** (powered, ~3–5 m/s, held heading → continuous sideslip sweep): identify `D_x` AND `D_y` together. Primary drag maneuver.
- **Steady tail-first cruise + 2-1-1 doublets near trim** (powered): excite the weathervane/damping moments without leaving trim. Safer than tumbles; keeps rotors loaded.
- **`thrustchirp`** (powered, held speed): identify the thrust-vs-airspeed droop (`k_h·v_h²`) / feed a BEM anchor.
- **High-speed sweeps** (as the controller envelope allows): excite the quadratic parasitic drag `C` (dominates >15–20 m/s) toward the 33 m/s course speed.

### Phase 2 estimator changes
- **Moments via virtual mixer + rollout**: predict control torque from commanded rate/thrust (rate-loop gain ≈ −2.5), attribute `I·ω̇ − τ_control` to aero; fit by **trajectory rollout + gradient-free optimization (Nelder-Mead)** minimizing orientation error — NOT lstsq on finite-differenced ω̇. Use cubic-spline derivatives where needed.
- **Forces at high speed**: add the `k_h·v_h²` thrust term and/or a **BEM anchor** (NeuroBEM hybrid); the empirical droop term alone under-predicts >15% at race speed.
- **Residual**: temporal-window features (history of v, ω, commanded thrust/torque), TCN over MLP, 70/20/10 split with full-envelope coverage, regularized to avoid unstable feedback.
- **Data split discipline**: held-out *runs* (not random samples) covering the speed/direction envelope.
