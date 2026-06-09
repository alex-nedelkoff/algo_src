# Tilt-budget Speed-scheduled Trajectory Controller — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `traj_track.py` fly the translated/curve courses on the live VQ sim without tumbling by replacing the planner's fixed lateral-accel budget with a tilt-budget speed law and adding a budget-aware cross-track cap.

**Architecture:** `gate_traj.GateTrajectory` (pure numpy/scipy planner) gets a speed law `v=√(g·tan(φ)·margin/(c_drag+κ))` that unifies the straight-line drag ceiling and the lateral wall (both = shared tilt budget). `traj_track.py` (the proven race_cruise tracking loop) uses the same φ budget for its tilt cap and clamps the cross-track command to the residual budget `g·tan(φ)·margin − c_drag·v²`. The proven loop, frames, ZVD yaw, and drone-locked reference are unchanged.

**Tech Stack:** Python (numpy, scipy), pymavlink ACRO rate interface, the VQ DCL sim. Code lives in the laptop `aigp-client` git worktree; the planner is unit-tested with pytest, the tracker is validated live.

---

## Environment & conventions (read before any task)

All target files are in the laptop `aigp-client` worktree. Define:
- `WT = C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src\.claude\worktrees\aigp-client`
- `PY = C:\Users\alexj\miniconda3\envs\aigp\python.exe`
- SSH alias `laptop` (Tailscale). `scp -O` required (Windows OpenSSH).

**Edit pattern (files are remote):** fetch the file to `/tmp`, edit with the Edit tool, `scp -O` it back:
```bash
scp -O 'laptop:C:/Users/alexj/Documents/drone-ai-grand-prix/algo_src/.claude/worktrees/aigp-client/<file>' /tmp/<file>
# ...Edit /tmp/<file>...
scp -O /tmp/<file> 'laptop:C:/Users/alexj/Documents/drone-ai-grand-prix/algo_src/.claude/worktrees/aigp-client/<file>'
```
**Run python/pytest on the laptop:**
```bash
ssh laptop "cd /d \"$WT\" && \"$PY\" -m pytest <file> -v"
```
**Commit on the laptop worktree (its own branch):**
```bash
ssh laptop "cd /d \"$WT\" && git add <files> && git commit -m \"<msg>\""
```
**Before any LIVE sim run (Tasks 4–5):** kill stray python on MAVLink udp 14550, run unbuffered to a log, poll the log (ssh `| tail` block-buffers until exit):
```bash
ssh laptop "powershell -NoProfile -Command \"Get-NetUDPEndpoint -LocalPort 14550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }\""
```
Judge stability by **TILT**, never a hand-rolled β/heading metric (frame-buggy — this session's bug). The user must have the VQ sim up for Tasks 4–5; the Rerun dashboard is at `100.101.13.126:9876`.

---

## Task 1: Planner tilt-budget speed law (TDD)

**Files:**
- Create: `$WT/test_tilt_budget.py`
- Modify: `$WT/gate_traj.py` (constructor `__init__` + `speed_at`)

- [ ] **Step 1: Write the failing test**

Create `/tmp/test_tilt_budget.py` and scp to `$WT/test_tilt_budget.py`:
```python
import numpy as np
from gate_traj import GateTrajectory, G

GATES = np.array([[0, 0, -3], [12, 0, -3], [24, 0, -3], [36, 0, -3]], float)


def _traj(v_cruise=12.0, **kw):
    return GateTrajectory(GATES, v_cruise=v_cruise, **kw)


def test_speed_at_straight_is_drag_ceiling():
    tr = _traj(tilt_budget_deg=25.0, c_drag=0.057, margin=0.6)
    expected = np.sqrt(G * np.tan(np.radians(25.0)) * 0.6 / 0.057)   # ~6.94
    assert abs(tr.speed_at(0.0) - expected) < 1e-6
    assert 6.5 < tr.speed_at(0.0) < 7.5


def test_curvature_lowers_speed():
    tr = _traj(tilt_budget_deg=25.0, c_drag=0.057, margin=0.6)
    assert tr.speed_at(0.5) < tr.speed_at(0.0)
    assert tr.speed_at(2.0) < tr.speed_at(0.5)


def test_capped_at_v_cruise():
    tr = _traj(v_cruise=3.0, tilt_budget_deg=25.0, c_drag=0.057, margin=0.6)
    assert tr.speed_at(0.0) == 3.0          # 6.9 budget ceiling > 3.0 cap


def test_higher_budget_higher_ceiling():
    lo = _traj(tilt_budget_deg=18.0); hi = _traj(tilt_budget_deg=30.0)
    ratio = hi.speed_at(0.0) / lo.speed_at(0.0)
    assert abs(ratio - np.sqrt(np.tan(np.radians(30)) / np.tan(np.radians(18)))) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" -m pytest test_tilt_budget.py -v"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'tilt_budget_deg'`.

- [ ] **Step 3: Write the minimal implementation**

In `$WT/gate_traj.py`, change the constructor signature + body. Replace:
```python
    def __init__(self, gates, v_cruise: float = 2.5, phi_max_deg: float = 35.0,
                 samples_per_seg: int = 80):
```
with:
```python
    def __init__(self, gates, v_cruise: float = 2.5, phi_max_deg: float = 35.0,
                 samples_per_seg: int = 80, tilt_budget_deg: float = 25.0,
                 c_drag: float = 0.057, margin: float = 0.6):
```
And immediately after the existing line `self.a_lat_max = G * np.tan(np.radians(phi_max_deg))` add:
```python
        self.tilt_budget = G * np.tan(np.radians(tilt_budget_deg))   # horizontal accel budget
        self.c_drag = float(c_drag)                                  # REFIT-02 quadratic drag (/m)
        self.margin = float(margin)                                  # reserve for cross-track + gusts
```
Then replace the whole `speed_at` method:
```python
    def speed_at(self, kappa_abs: float) -> float:
        if kappa_abs < 1e-4:
            return self.v_cruise
        return float(min(self.v_cruise, np.sqrt(self.a_lat_max / kappa_abs)))
```
with:
```python
    def speed_at(self, kappa_abs: float) -> float:
        """Tilt-budget speed law: forward drag (c*v^2) + centripetal (v^2*kappa) <= budget*margin,
        solved for v. Straight (kappa->0) -> sqrt(budget*margin/c_drag) = the drag-saturation ceiling;
        curvature lowers it. Unifies the straight-line and lateral walls (REFIT-02 / LATERAL-WALL)."""
        denom = self.c_drag + max(float(kappa_abs), 0.0)
        v_budget = np.sqrt(self.tilt_budget * self.margin / denom)
        return float(min(self.v_cruise, v_budget))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" -m pytest test_tilt_budget.py -v"`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
ssh laptop "cd /d \"$WT\" && git add gate_traj.py test_tilt_budget.py && git commit -m \"feat(cor-127): tilt-budget speed law in GateTrajectory\""
```

---

## Task 2: Offline schedule sanity check

**Files:** none modified (runs the planner demo).

- [ ] **Step 1: Run the planner demo and inspect the schedule**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" gate_traj.py"`
Expected: the `curving corridor` rows show `v` dropping below `v_cruise` where bank/κ is high; the straight segments sit near the budget ceiling. (Default `v_cruise=2.5` caps it — this only confirms curvature lowers v relative to the cap.)

- [ ] **Step 2: Run the tracker dry schedule at racing v_cruise**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" traj_track.py curve --dry"` (after Task 3 raises CRUISE; if run before Task 3 it uses CRUISE=1.8).
Expected after Task 3: straight sections `v ≈ 6–7`, curved sections visibly lower. This is a visual sanity gate, no assertion.

---

## Task 3: Tracker — budget-aware cross-track + consistent tilt cap + recorder

**Files:**
- Modify: `$WT/traj_track.py`

- [ ] **Step 1: Replace the tilt-cap/gains constants**

In `$WT/traj_track.py`, replace this line:
```python
WMAX = 4.0; LOOP_DT = 0.004; TILT_MAX_ACC = np.tan(np.radians(35)) * G
LEAD = 2.5; ARRIVE = 1.5; CRUISE = 1.8; MAX_T = 90.0
```
with:
```python
WMAX = 4.0; LOOP_DT = 0.004
# --- tilt budget (single source of truth, shared with the planner speed law) ---
TILT_BUDGET_DEG = 25.0; C_DRAG = 0.057; MARGIN = 0.6
TILT_MAX_ACC = np.tan(np.radians(TILT_BUDGET_DEG)) * G
AL_MAX = TILT_MAX_ACC                         # forward governor = tilt budget (was 0.6); planner caps v
LEAD = 2.5; ARRIVE = 1.5; CRUISE = 8.0; MAX_T = 90.0
```
The along-track governor cap (`AL_MAX`) is now the tilt budget so the scheduled `ref["v"]` governs (the old hardcoded 0.6 capped forward accel and prevented reaching racing speed). Remove the old `AL_MAX` from the gains line — replace:
```python
KP_CT = 0.5; KD_CT = 1.2; KD_AL = 1.2; AL_MAX = 0.6; KP_AL = 0.6
```
with:
```python
KP_CT = 0.5; KD_CT = 1.2; KD_AL = 1.2; KP_AL = 0.6
```

- [ ] **Step 2: Pass the budget params to the planner**

Replace:
```python
    traj = GateTrajectory(np.array(gates), v_cruise=CRUISE, phi_max_deg=35.0)
```
with:
```python
    traj = GateTrajectory(np.array(gates), v_cruise=CRUISE, tilt_budget_deg=TILT_BUDGET_DEG,
                          c_drag=C_DRAG, margin=MARGIN)
```

- [ ] **Step 3: Add the budget-aware cross-track cap**

Replace:
```python
            a_al = float(np.clip(KD_AL * (ref["v"] - v_al), -4.0, AL_MAX))     # pure speed governor
            a_ct = -KP_CT * p_ct - KD_CT * v_ct
```
with:
```python
            a_al = float(np.clip(KD_AL * (ref["v"] - v_al), -4.0, AL_MAX))     # pure speed governor
            # budget-aware cross-track: clamp lateral accel to the tilt budget LEFT after holding speed
            # (drag = C_DRAG*v_al^2). Limits the transient line-acquisition that triggered CRUISE-04.
            a_ct_max = max(0.0, TILT_MAX_ACC * MARGIN - C_DRAG * v_al * v_al)
            a_ct = float(np.clip(-KP_CT * p_ct - KD_CT * v_ct, -a_ct_max, a_ct_max))
```

- [ ] **Step 4: Add durable recording (hard rule: never fly blind)**

Add to the imports block (after `from aigp.flight_telemetry import sideslip_deg`):
```python
from aigp.recorder import Recorder
```
After `c.arm()` and the existing `flog = ftm.from_args(sys.argv, RG, "traj_track")` / `flog.set_path(...)` block, add:
```python
    rec = Recorder(s, script="traj_track", mode="rate",
                   notes=f"tilt-budget speed-scheduled tracker, course={course}",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "base_controller": "race_cruise",
                               "tilt_budget_deg": TILT_BUDGET_DEG, "c_drag": C_DRAG, "margin": MARGIN,
                               "course": course})
```
Replace the send line:
```python
            c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)
```
with:
```python
            w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
```
Replace the cleanup:
```python
    if flog is not None:
        flog.close()
```
with:
```python
    rec.close()
    if flog is not None:
        flog.close()
```

- [ ] **Step 5: Compile-check + commit**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" -m py_compile traj_track.py && echo COMPILE_OK"`
Expected: `COMPILE_OK`.
Then:
```bash
ssh laptop "cd /d \"$WT\" && git add traj_track.py && git commit -m \"feat(cor-127): budget-aware cross-track cap + tilt budget + recorder in traj_track\""
```

---

## Task 4: Live validation — translated course

**Files:** none (runs the controller; user must have the VQ sim up).

- [ ] **Step 1: Kill stray python + launch the run (backgrounded)**

```bash
ssh laptop "powershell -NoProfile -Command \"Get-NetUDPEndpoint -LocalPort 14550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }\" & cd /d \"$WT\" && \"$PY\" -u traj_track.py translated > tt_translated.log 2>&1"
```
Run this with `run_in_background: true` (the run holds the ssh until the controller exits).

- [ ] **Step 2: Poll the log and judge**

Poll: `ssh laptop "type \"$WT\\tt_translated.log\""` (repeat until `DONE`/`REACHED END`/`ABORT`).
Expected PASS: a `REACHED END` line; final `DONE ... max_tilt=<N> ... OK` with **max_tilt < ~45** and not `TUMBLED`.
If `ABORT`/`TUMBLED` or max_tilt high: lower `MARGIN` to 0.5 (Task 3 Step 1) or `TILT_BUDGET_DEG` to 22, recompile, re-run. Record what was tried.

---

## Task 5: Live validation — curve course

**Files:** none.

- [ ] **Step 1: Kill stray python + launch the curve run (backgrounded)**

```bash
ssh laptop "powershell -NoProfile -Command \"Get-NetUDPEndpoint -LocalPort 14550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }\" & cd /d \"$WT\" && \"$PY\" -u traj_track.py curve > tt_curve.log 2>&1"
```
Run with `run_in_background: true`.

- [ ] **Step 2: Poll the log and judge**

Poll: `ssh laptop "type \"$WT\\tt_curve.log\""` until terminal line.
Expected PASS: `REACHED END`, `DONE ... OK`, **max_tilt < ~45**, `peak_spd` competitive (the curve sections will show lower `v=actual/ref` than the straights). Contrast: the pre-change controller TUMBLED here (CRUISE-04→06).

---

## Task 6: Record the result

**Files:**
- Modify: `/Users/alex/Documents/obsidian-vault/01 Projects/CorvidX/AI-GP Experiment Log.md` (append a row under the COR-127 section)
- Modify: `/Users/alex/Documents/drone-ai-grand-prix/algo_src/HANDOFF.md` (update the "remaining BUILD" line)

- [ ] **Step 1: Append an exp-log entry**

Append a `**TRACK-01 (06-09).**` paragraph: the tilt-budget speed law + budget-aware cross-track cap, the translated/curve live results (reached-end? max_tilt? peak_spd?), φ_budget/margin used, and whether the CRUISE-04→06 divergence is now fixed. Include the run_ids from the recorder.

- [ ] **Step 2: Update HANDOFF.md**

Update the `(c)` "remaining BUILD" bullet: mark the speed-scheduled controller built + its live result; note any remaining tuning.

- [ ] **Step 3: Commit the spec/plan tree (Mac repo)**

```bash
cd /Users/alex/Documents/drone-ai-grand-prix/algo_src && git add docs/superpowers HANDOFF.md && git commit -m "docs(cor-127): tilt-budget controller plan + result"
```

---

## Self-review notes
- **Spec coverage:** C1 speed law → Task 1; C2 tilt cap + a_ct cap + recorder → Task 3; C3 offline → Task 2, live → Tasks 4–5; params/defaults → Task 1/3 constants; acceptance tests → Tasks 1,2,4,5. All covered.
- **Type consistency:** `tilt_budget_deg`, `c_drag`, `margin` identical across `gate_traj.__init__`, the test, and the `traj_track` `GateTrajectory(...)` call; `TILT_BUDGET_DEG`/`C_DRAG`/`MARGIN`/`TILT_MAX_ACC`/`a_ct_max`/`w_cmd` defined before use.
- **No placeholders:** every code/command step is concrete.
