# Cornering at Speed — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fly one coordinated corner at ~6 m/s on live VQ, stabilized by an explicit weathervane feedforward, with an FF on/off A/B that proves the FF is what makes it possible.

**Architecture:** `corner_ff.py` holds two pure, unit-tested helpers (body-frame velocity + the weathervane-FF rate-command offset). `corner_speed.py` is a copy of the proven `coord_turn.py` (coordinated bank + yaw-rate + pitch-comp FF, velocity-tracking yaw) with the weathervane-FF wired into its `cmd()`, a tilt-budget speed cap, and `--no-wvff`/`--wvff_sign`/`--wvff_gain` flags for live sign-verification.

**Tech Stack:** Python (numpy, pytest), pymavlink ACRO rate interface, the VQ DCL sim, `aigp.recorder`. Code + data on the laptop `aigp-client` worktree.

---

## Environment & conventions (read before any task)
- `WT = C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src\.claude\worktrees\aigp-client`
- `PY = C:\Users\alexj\miniconda3\envs\aigp\python.exe` (NEVER bare `python`)
- SSH alias `laptop`; `scp -O` required; forward slashes in scp remote paths.
- Edit remote: `scp -O 'laptop:C:/Users/alexj/Documents/drone-ai-grand-prix/algo_src/.claude/worktrees/aigp-client/<f>' /tmp/<f>` → Read → Edit → scp back.
- Run: `ssh -o ConnectTimeout=25 laptop "cd /d \"<WT>\" && \"<PY>\" <cmd>"`. Commit on the worktree.
- **Before LIVE runs (Task 3):** sim must be up; kill stray python on udp 14550; `-u > log`, poll the log. **Judge by TILT + β on the dashboard; verify the FF sign empirically — never assume the body/world velocity frame** (this session's β bug).

---

## Task 1: Pure helpers `corner_ff.py` (TDD)

**Files:** Create `$WT/corner_ff.py`, `$WT/test_corner_ff.py`

- [ ] **Step 1: Write the failing test** — write `/tmp/test_corner_ff.py`, scp to `test_corner_ff.py`:
```python
import numpy as np
from corner_ff import body_vel, weathervane_ff


def test_body_vel_identity():
    assert np.allclose(body_vel(np.eye(3), [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])


def test_body_vel_yaw90():
    th = np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    # R is body->world (+90deg about z): a body +x velocity shows as world +y.
    # body_vel(R, world=+y) must recover body +x.
    assert np.allclose(body_vel(R, [0.0, 1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-9)


def test_weathervane_ff_formula():
    out = weathervane_ff(vbx=-7.0, vby=2.0, roll_wv0=-0.105, roll_wv1=-0.019,
                         yaw_wv=-0.149, g_roll=-2.54, g_yaw=-2.29, gain=1.0, sign=1.0)
    exp_roll = -1.0 * 1.0 * (-0.105 + -0.019 * -7.0) * 2.0 / -2.54
    exp_yaw = -1.0 * 1.0 * -0.149 * 2.0 / -2.29
    assert abs(out[0] - exp_roll) < 1e-12 and abs(out[1] - exp_yaw) < 1e-12


def test_sign_flips():
    a = weathervane_ff(-5.0, 1.5, -0.105, -0.019, -0.149, -2.54, -2.29, sign=1.0)
    b = weathervane_ff(-5.0, 1.5, -0.105, -0.019, -0.149, -2.54, -2.29, sign=-1.0)
    assert np.allclose(a, [-x for x in b])


def test_zero_sideslip_zero_ff():
    assert weathervane_ff(-5.0, 0.0, -0.105, -0.019, -0.149, -2.54, -2.29) == (0.0, 0.0)
```

- [ ] **Step 2: Run, confirm FAIL** — `ssh ... "... -m pytest test_corner_ff.py -v"` → `ImportError: cannot import name 'body_vel'`.

- [ ] **Step 3: Implement** — write `/tmp/corner_ff.py`, scp to `corner_ff.py`:
```python
"""corner_ff.py -- pure helpers for the cornering controller: body-frame velocity + the weathervane
feedforward (cancel wv*v_body_y in the rate command, WV-DYNAMIC-validated coeffs). Pure numpy ->
unit-testable; the live frame/sign is verified on the dashboard (--wvff_sign, --no-wvff)."""
import numpy as np


def body_vel(R, vel_world):
    """World velocity -> body frame: v_body = R^T @ vel_world. R = body->world (quat_to_R)."""
    return np.asarray(R, float).T @ np.asarray(vel_world, float)


def weathervane_ff(vbx, vby, roll_wv0, roll_wv1, yaw_wv, g_roll, g_yaw, gain=1.0, sign=1.0):
    """Rate-command offset (wcmd units) that cancels the weathervane moment.
    Model: om_roll += (roll_wv0+roll_wv1*vbx)*vby*dt ; om_yaw += yaw_wv*vby*dt. Command goes through
    om=G*wcmd, so divide by the per-axis rate-loop gain. gain/sign tuned + verified live."""
    roll_ff = -sign * gain * (roll_wv0 + roll_wv1 * vbx) * vby / g_roll
    yaw_ff = -sign * gain * yaw_wv * vby / g_yaw
    return float(roll_ff), float(yaw_ff)
```

- [ ] **Step 4: Run, confirm PASS** — `... -m pytest test_corner_ff.py -v` → 5 passed.

- [ ] **Step 5: Commit** — `git add corner_ff.py test_corner_ff.py && git commit -m "feat(cor-127): pure weathervane-FF + body-velocity helpers (TDD)"`

---

## Task 2: `corner_speed.py` (coord_turn + weathervane-FF)

**Files:** Create `$WT/corner_speed.py` (from `coord_turn.py`)

- [ ] **Step 1: Copy coord_turn.py → corner_speed.py**

`ssh ... "cd /d \"<WT>\" && copy coord_turn.py corner_speed.py"` → expect `1 file(s) copied.`
Then fetch `corner_speed.py` to /tmp and Read it (it has `cmd(ds, a2, z_sp, yaw_sp, yaw_ff, pitch_ff)` ending in `c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)`, and a `main()` with `PHI_SAFE`/`V_MAX`).

- [ ] **Step 2: Add the weathervane-FF imports + constants**

After the existing import block (the `import aigp.flight_telemetry as ftm` line), add:
```python
from corner_ff import body_vel, weathervane_ff
```
After the `G = 9.81` line, add the weathervane constants + flags:
```python
# --- weathervane FF (WV-DYNAMIC-validated coeffs; sign/gain verified live) ---
ROLL_WV0, ROLL_WV1, YAW_WV = -0.105, -0.019, -0.149
WVFF = "--no-wvff" not in sys.argv
WVFF_SIGN = float(sys.argv[sys.argv.index("--wvff_sign") + 1]) if "--wvff_sign" in sys.argv else 1.0
WVFF_GAIN = float(sys.argv[sys.argv.index("--wvff_gain") + 1]) if "--wvff_gain" in sys.argv else 1.0
WVFF_CLIP = 1.0     # rad/s: clamp the FF so a bad sideslip estimate can't command a huge rate
```

- [ ] **Step 3: Wire the FF into `cmd()`**

In `cmd()`, replace:
```python
    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
    c.send_attitude_target(np.clip(w_des / RG, -WMAX, WMAX), thr)
    return float(np.degrees(np.arccos(max(-1, min(1, Rc[2, 2]))))), {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}
```
with:
```python
    wcmd = w_des / RG
    if WVFF:                                               # weathervane FF: cancel wv*v_body_y
        vb = body_vel(Rc, ds.vel_ned)                      # Rc = quat_to_R(ds.quat_wxyz), already computed above
        roll_ff, yaw_ff_wv = weathervane_ff(float(vb[0]), float(vb[1]), ROLL_WV0, ROLL_WV1, YAW_WV,
                                            RG[0], RG[2], WVFF_GAIN, WVFF_SIGN)
        wcmd[0] += float(np.clip(roll_ff, -WVFF_CLIP, WVFF_CLIP))
        wcmd[2] += float(np.clip(yaw_ff_wv, -WVFF_CLIP, WVFF_CLIP))
    thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
    c.send_attitude_target(np.clip(wcmd, -WMAX, WMAX), thr)
    return float(np.degrees(np.arccos(max(-1, min(1, Rc[2, 2]))))), {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr}
```

- [ ] **Step 4: Tilt-budget speed cap (drag-aware)**

In `main()`, replace:
```python
    V_MAX = float(np.sqrt(R * G * np.tan(PHI_SAFE)))   # fastest coordinated speed for R (govern speed to this)
```
with:
```python
    C_DRAG = 0.057                                     # REFIT-02 quadratic drag
    V_MAX = float(np.sqrt(G * np.tan(PHI_SAFE) / (C_DRAG + 1.0 / R)))   # tilt budget: drag + centripetal <= budget
```

- [ ] **Step 5: Durable recording + banner flag note**

In `main()`, after the existing `flog = ftm.from_args(sys.argv, RG, "coord_turn")` line, add:
```python
    from aigp.recorder import Recorder
    rec_d = Recorder(s, script="corner_speed", mode="rate",
                     notes=f"coordinated corner + weathervane-FF (wvff={WVFF} sign={WVFF_SIGN} gain={WVFF_GAIN})",
                     extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "R": R, "wvff": WVFF,
                                 "wvff_sign": WVFF_SIGN, "wvff_gain": WVFF_GAIN})
```
In the loop, immediately after the `tilt, dbg = cmd(...)` line, add:
```python
            rec_d.log([float(dbg["w_des"][0] / RG[0]), float(dbg["w_des"][1] / RG[1]),
                       float(dbg["w_des"][2] / RG[2]), float(dbg["thr"])])
```
Before the final `print("REC ...")`, add `rec_d.close()`.
Update the banner print to include the FF state — change the `print(f"EXP-28 coord_turn ...` line's f-string prefix from `EXP-28 coord_turn` to `corner_speed wvff={WVFF} sign={WVFF_SIGN}`.

- [ ] **Step 6: Compile-check + commit**

`ssh ... "... -m py_compile corner_speed.py && echo COMPILE_OK"` → `COMPILE_OK`.
`git add corner_speed.py && git commit -m "feat(cor-127): cornering controller (coord_turn + weathervane-FF + tilt-budget cap)"`

---

## Task 3: Live validation — one corner, FF on/off A/B

**Files:** none (live runs; sim must be up).

- [ ] **Step 1: Baseline — FF OFF at speed**

Kill stray python, launch (run_in_background):
```bash
ssh laptop "powershell -NoProfile -Command \"Get-NetUDPEndpoint -LocalPort 14550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }\" & cd /d \"<WT>\" && \"<PY>\" -u corner_speed.py 10 6 --no-wvff > corner_off.log 2>&1"
```
Poll `corner_off.log` to a terminal line. Record `turned`, `peak|beta|`, `peak_spd`, ABORT/SUCCESS. Expect the bare coordinated turn at v6 to struggle — high `peak|beta|` and/or ABORT (the failure mode the FF must fix).

- [ ] **Step 2: FF ON, verify the sign**

Launch `corner_speed.py 10 6 > corner_on.log` (FF on, sign +1, gain 1). Poll.
- If `peak|beta|` is LOWER than Step 1 and the turn progresses → sign is correct.
- If β grows faster / aborts sooner than Step 1 → wrong sign: re-run with `--wvff_sign -1`. Record which sign reduces β. (This is the empirical frame/sign verification the spec requires — do NOT assume.)

- [ ] **Step 3: Judge success**

With the correct sign, success = **β bounded < ~20°, tilt bounded (no ABORT), ≥90° of turn (`turned` ≥ 90)**. If β is bounded but the turn is slow/incomplete, try `--wvff_gain 1.5` (stronger cancellation) or lower speed to 5. Record the best result + the A/B contrast (FF off vs on `peak|beta|`).

- [ ] **Step 4: Note the data**

`ssh ... "powershell -NoProfile -Command \"Get-ChildItem 'C:\Users\alexj\Documents\vq_data' -Directory -Filter '*corner_speed*' | Select-Object -ExpandProperty Name\""` → record the run ids.

---

## Task 4: Record results

**Files:** Modify the obsidian exp log, `$MAC/HANDOFF.md`; commit Mac docs.

- [ ] **Step 1: Append an exp-log entry**

Append a `**CORNER-01 (06-09).**` paragraph to `/Users/alex/Documents/obsidian-vault/01 Projects/CorvidX/AI-GP Experiment Log.md`: the coordinated-turn + weathervane-FF controller, the A/B result (FF off vs on `peak|beta|`, the verified sign), whether one corner at ~6 m/s flew with bounded β, and the run ids. Be honest if the FF didn't help (→ escalate to INDI, Approach 2).

- [ ] **Step 2: Update HANDOFF.md**

Update item (e)/(f): cornering at speed — coordinated turn + weathervane-FF built; one-corner result; next = gate-to-gate chaining (downstream spec) then DAgger/RL.

- [ ] **Step 3: Commit Mac docs**

`cd $MAC && git add docs/superpowers HANDOFF.md && git commit -m "docs(cor-127): cornering-at-speed plan + one-corner result"`

---

## Self-review notes
- **Spec coverage:** C1 coordinated-turn FF → reused (Task 2 copy); C2 weathervane-FF → Task 1 (pure) + Task 2 Step 3 (wired); C3 tilt-budget cap → Task 2 Step 4; C4 validation/A-B → Task 3; frame/sign verification → Task 3 Step 2; results → Task 4. All covered.
- **Type consistency:** `body_vel(R, vel_world)` and `weathervane_ff(vbx, vby, roll_wv0, roll_wv1, yaw_wv, g_roll, g_yaw, gain, sign)` signatures identical in test (Task 1.1), impl (Task 1.3), and caller (Task 2.3). `ROLL_WV0/ROLL_WV1/YAW_WV/WVFF/WVFF_SIGN/WVFF_GAIN/WVFF_CLIP` defined (Task 2.2) before use (2.3). `Rc` is the `quat_to_R(ds.quat_wxyz)` already computed earlier in `cmd()` — reused, not recomputed.
- **No placeholders:** every code/command step is concrete; the only conditionals (Task 3 sign) are explicit empirical branches.
- **Frame discipline:** `v_body_y` via `body_vel` (defined op), sign verified by the live A/B (β must shrink); FF clipped; judged by tilt + β on the dashboard.
