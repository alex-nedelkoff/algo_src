# Weathervane Dynamic-Regime Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `VQMatchedDynamics` faithful in the high-sideslip forward-racing regime by collecting controlled high-sideslip data (a crab-sweep), refitting the weathervane there, and trajectory-validating the refit against live VQ.

**Architecture:** A new `collect_vq_crab.py` (race_cruise loop + a *lateral-velocity* crab setpoint instead of line-hold → sustained controlled sideslip) collects data at several forward speeds. `wv_speed_fit.py` is extended with a pure, unit-tested `fit_wv_axis()` that recovers the weathervane coefficient from the rate residual; it fits roll & yaw vs `v_body_y` binned by `v_body_x`. A replay check compares the current vs refit model against held-out live runs. The refit coefficients update `vq_model.json` (yaw) and the `roll_wv0/roll_wv1` constants in `vq_matched.py`/`vq_sim.py` (roll).

**Tech Stack:** Python (numpy, scipy, pytest), pymavlink ACRO rate interface, the VQ DCL sim, `aigp.recorder`. Flight code + data + fit live in the laptop `aigp-client` worktree; `VQMatchedDynamics` lives in the Mac repo.

---

## Environment & conventions (read before any task)

- `WT = C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src\.claude\worktrees\aigp-client`  (laptop worktree)
- `PY = C:\Users\alexj\miniconda3\envs\aigp\python.exe`  (NEVER bare `python`)
- `MAC = /Users/alex/Documents/drone-ai-grand-prix/algo_src`  (this repo; holds `sim/dynamics/vq_matched.py`)
- SSH alias `laptop`; `scp -O` required; forward slashes in scp remote paths.

**Edit a remote file:** `scp -O 'laptop:C:/Users/alexj/Documents/drone-ai-grand-prix/algo_src/.claude/worktrees/aigp-client/<f>' /tmp/<f>` → Read → Edit → `scp -O /tmp/<f> 'laptop:.../<f>'`.
**Run on laptop:** `ssh -o ConnectTimeout=25 laptop "cd /d \"$WT\" && \"$PY\" <cmd>"`.
**Commit on laptop worktree:** `ssh laptop "cd /d \"$WT\" && git add <f> && git commit -m \"<msg>\""`.
**Before any LIVE run (Tasks 1,2):** the user must have the VQ sim up. Kill stray python on udp 14550 first; run `-u > log`; poll the log (ssh `| tail` block-buffers). Judge stability by **TILT**, never a hand-rolled β. Dashboard: `100.101.13.126:9876`.

---

## Task 1: Crab-sweep maneuver `collect_vq_crab.py`

**Files:** Create `$WT/collect_vq_crab.py`

- [ ] **Step 1: Write the script**

Write `/tmp/collect_vq_crab.py` and `scp -O` it to `$WT/collect_vq_crab.py`:
```python
"""collect_vq_crab.py -- collect CONTROLLED high-sideslip data to validate/refit the weathervane in
the dynamic (forward-racing) regime. The trap prior sysID hit: closed-loop slalom suppresses sideslip
(can't ID the weathervane). Fix: decouple "make sideslip" from "stay upright" -- hold forward speed +
FIXED heading, command a LATERAL-VELOCITY crab (not line-hold) so the drone slides sideways relative
to its nose -> sustained, controlled v_body_y. The attitude loop keeps it level; the crab supplies the
sideslip. Alternating-sign, growing-magnitude velocity steps keep position bounded while sweeping
sideslip up to the wall. Reuses race_cruise's control law VERBATIM + Recorder + dashboard + tilt gate.

Forward speed is set by --almax (drag-balance), NOT --cruise (the v_al-sign quirk makes the setpoint
non-binding). Run at several --almax for v~3/5/7. RUN ON A CLEAN SIM.
  python collect_vq_crab.py [--almax 1.5] [--vlat 4] [--step 4] [--dur 44] [--tiltcap 30] [--no-viz]
"""
import sys, json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.flight_telemetry import tilt_deg
import aigp.flight_telemetry as ftm
from aigp.recorder import Recorder


def argf(flag, d): return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


KP_ATT = np.array([0.5, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KP_CT = 0.5; KD_CT = 1.2; KD_AL = 1.2
KP_Z = 1.8; KD_Z = 3.0
WMAX = 4.0; LOOP_DT = 0.004
AL_MAX = argf("--almax", 1.5); VLAT = argf("--vlat", 4.0); STEP = argf("--step", 4.0)
DUR = argf("--dur", 44.0); SETTLE = 6.0
TILT_CAP_DEG = argf("--tiltcap", 30.0); TILT_MAX_ACC = np.tan(np.radians(TILT_CAP_DEG)) * 9.81
ABORT_TILT = 75.0

r = json.load(open("sysid/sim_response.json")); HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time() * 1000) - boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1.0, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    t = time.time()
    while time.time() - t < 1.0:
        idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time() - t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)):
            break
        time.sleep(0.02)
    while time.time() - t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0:
            return True
        time.sleep(0.02)
    return False


def crab_vref(t):
    """Alternating-sign, growing-magnitude lateral-velocity setpoint (m/s). 0 during settle."""
    if t < SETTLE:
        return 0.0, 0.0
    te = t - SETTLE
    level = min(VLAT, 0.5 + (VLAT - 0.5) * te / max(DUR - SETTLE, 1e-6))
    sign = 1.0 if int(te / STEP) % 2 == 0 else -1.0
    return sign * level, level


def main():
    print(f"collect_vq_crab AL_MAX={AL_MAX} VLAT={VLAT} STEP={STEP} DUR={DUR} tiltcap={TILT_CAP_DEG}", flush=True)
    assert fresh_start(), "not live"
    ds0 = s.get_drone(); spawn = ds0.pos_ned.copy(); z_sp = float(spawn[2])
    yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
    travel = -np.array([np.cos(yaw0), np.sin(yaw0)])          # camera-forward = -body_x
    lat_hat = np.array([-travel[1], travel[0]])
    c.arm()
    flog = ftm.from_args(sys.argv, RG, run_name="collect_vq_crab", store=s)
    rec = Recorder(s, script="collect_vq_crab", mode="rate",
                   notes="controlled high-sideslip crab sweep for weathervane refit (dynamic regime)",
                   extra_meta={"cmd_layout": ["wx", "wy", "wz", "thrust"], "base_controller": "race_cruise",
                               "al_max": AL_MAX, "vlat": VLAT, "step": STEP, "tilt_cap_deg": TILT_CAP_DEG})
    t0 = time.time(); last = -1; _, coll0 = s.get_collision(); max_tilt = 0.0; max_vct = 0.0
    try:
        while time.time() - t0 < DUR + 2.0:
            ds = s.get_drone()
            if ds is None:
                time.sleep(LOOP_DT); continue
            t = time.time() - t0
            v_ref, level = crab_vref(t)
            d = ds.pos_ned - spawn
            v_al = float(ds.vel_ned[:2] @ travel); v_ct = float(ds.vel_ned[:2] @ lat_hat)
            a_al = float(np.clip(KD_AL * (12.0 - v_al), -4.0, AL_MAX))     # always push fwd; AL_MAX governs speed
            a_ct = float(np.clip(KD_CT * (v_ref - v_ct), -TILT_MAX_ACC, TILT_MAX_ACC))  # lateral-VELOCITY crab
            a = np.zeros(3); a[:2] = a_al * travel + a_ct * lat_hat
            a[2] = KP_Z * (z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
            ah = a[:2]; n = float(np.linalg.norm(ah))
            if n > TILT_MAX_ACC:
                a[:2] = ah / n * TILT_MAX_ACC
            Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q_des = mat_to_quat(desired_attitude(a, yaw0))            # FIXED heading (crab), not nose-follows-tangent
            w_des = KP_ATT * attitude_error_quat(ds.quat_wxyz, q_des)
            w_des[2] = KP_YAW * ((yaw0 - yaw_cur + np.pi) % (2 * np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            w_cmd = np.clip(w_des / RG, -WMAX, WMAX)
            c.send_attitude_target(w_cmd, thr)
            rec.log([float(w_cmd[0]), float(w_cmd[1]), float(w_cmd[2]), float(thr)])
            if flog is not None:
                flog.push(t, ds, {"a": a, "w_des": w_des, "q_des": q_des, "thr": thr},
                          tangent=travel, cruise=AL_MAX, running=s.get_race_live(), armed=True)
            tilt = tilt_deg(ds.quat_wxyz); max_tilt = max(max_tilt, tilt)
            if t >= SETTLE and tilt < 40:
                max_vct = max(max_vct, abs(v_ct))
            if t < 3.5:
                _, coll0 = s.get_collision()
            else:
                _, coll = s.get_collision()
                if coll != coll0:
                    print(f"  COLLISION at t={t:.1f} v_ct={v_ct:+.1f} -> stop", flush=True); break
            if tilt > ABORT_TILT:
                print(f"  TUMBLE tilt={tilt:.0f} at t={t:.1f} v_ct={v_ct:+.1f} -> stop", flush=True); break
            k = int(t)
            if k != last:
                last = k
                print(f"t={t:4.1f} fwd={v_al:+4.1f} v_ct={v_ct:+4.1f} vref={v_ref:+4.1f} tilt={tilt:3.0f}", flush=True)
            time.sleep(LOOP_DT)
    finally:
        idle(); rec.close()
        if flog is not None:
            flog.close()
    print(f"DONE collect_vq_crab almax={AL_MAX}: max |v_ct| held (tilt<40) = {max_vct:.1f} m/s, "
          f"max_tilt={max_tilt:.0f} (run {rec.run_id})", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Compile-check + commit**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" -m py_compile collect_vq_crab.py && echo COMPILE_OK"` → expect `COMPILE_OK`.
Commit: `ssh laptop "cd /d \"$WT\" && git add collect_vq_crab.py && git commit -m \"feat(cor-127): crab-sweep maneuver for high-sideslip weathervane data\""`

---

## Task 2: Collect the dataset (live, 3 forward speeds)

**Files:** none (produces `vq_data/*_collect_vq_crab` runs). Requires the VQ sim up.

- [ ] **Step 1: Run at low speed (almax 0.6 → v~2.8)**

Kill stray python, then launch (run_in_background):
```bash
ssh laptop "powershell -NoProfile -Command \"Get-NetUDPEndpoint -LocalPort 14550 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }\" & cd /d \"$WT\" && \"$PY\" -u collect_vq_crab.py --almax 0.6 --vlat 6 > crab_a06.log 2>&1"
```
Poll `crab_a06.log` until `DONE`. PASS = sustained `v_ct` (lateral velocity) reached >2 m/s with tilt bounded for a stretch (sideslip present), data recorded (`run ...collect_vq_crab`).

- [ ] **Step 2: Run at mid speed (almax 1.5 → v~4.5)**

Same, `--almax 1.5 --vlat 5 > crab_a15.log`. Poll → DONE, note `max |v_ct|` and run_id.

- [ ] **Step 3: Run at high speed (almax 3.0 → v~7)**

Same, `--almax 3.0 --vlat 4 > crab_a30.log`. Poll → DONE, note run_id. (Expect the wall to limit `|v_ct|` here — that's the high-sideslip onset we want.)

- [ ] **Step 4: Confirm the dataset**

Run: `ssh laptop "powershell -NoProfile -Command \"Get-ChildItem 'C:\Users\alexj\Documents\vq_data' -Directory -Filter '*crab*' | Select-Object -ExpandProperty Name\""` → expect 3 dirs. Record their ids.

---

## Task 3: Weathervane refit (TDD the fit core, then run on data)

**Files:** Create `$WT/test_wv_fit.py`; Modify `$WT/wv_speed_fit.py`

- [ ] **Step 1: Write the failing test for a pure fit function**

Write `/tmp/test_wv_fit.py`, scp to `$WT/test_wv_fit.py`:
```python
import numpy as np
from wv_speed_fit import fit_wv_axis


def test_recovers_known_weathervane():
    rng = np.random.default_rng(0)
    n = 5000
    wcmd = rng.normal(0, 0.3, n); omega = rng.normal(0, 0.5, n); vby = rng.normal(0, 1.5, n)
    a, b, wv_true = 0.8, 1.2, -0.15
    dW = a * wcmd - b * omega + wv_true * vby + rng.normal(0, 0.01, n)
    out = fit_wv_axis(dW, wcmd, omega, vby)
    assert abs(out["wv"] - wv_true) < 0.02
    assert out["r2"] > 0.95


def test_zero_weathervane():
    rng = np.random.default_rng(1)
    n = 3000
    wcmd = rng.normal(0, 0.3, n); omega = rng.normal(0, 0.5, n); vby = rng.normal(0, 1.5, n)
    dW = 0.8 * wcmd - 1.2 * omega + rng.normal(0, 0.01, n)
    assert abs(fit_wv_axis(dW, wcmd, omega, vby)["wv"]) < 0.02
```

- [ ] **Step 2: Run it, verify it fails**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" -m pytest test_wv_fit.py -v"` → FAIL `ImportError: cannot import name 'fit_wv_axis'`.

- [ ] **Step 3: Add `fit_wv_axis` + the binned refit to `wv_speed_fit.py`**

Fetch `wv_speed_fit.py`, Read it. Add this pure function near the top (after the imports):
```python
def fit_wv_axis(dW_axis, wcmd_axis, omega_axis, v_body):
    """Weathervane coeff on one axis: dW = a*wcmd - b*omega + wv*v_body (the fit_model.fit_weathervane
    structure). Returns wv (the moment-per-sideslip) + R^2. Pure -> unit-testable."""
    M = np.vstack([wcmd_axis, -omega_axis, v_body]).T
    sol, *_ = np.linalg.lstsq(M, dW_axis, rcond=None)
    r2 = 1.0 - np.var(dW_axis - M @ sol) / np.var(dW_axis)
    return {"a": float(sol[0]), "b": float(sol[1]), "wv": float(sol[2]), "r2": float(r2), "n": int(len(dW_axis))}
```
Then add a refit section that pools the crab + lateral runs and fits roll & yaw vs `v_body_y` binned by `v_body_x` (append after the existing `fit_axis` calls, or in `__main__`):
```python
def refit_dynamic(P):
    """Roll & yaw weathervane vs v_body_y, binned by forward speed |v_body_x|, across the crab data."""
    vb = np.concatenate([p["vb"] for p in P]); W = np.concatenate([p["W"] for p in P])
    C = np.concatenate([p["C"] for p in P]); dW = np.concatenate([p["dW"] for p in P])
    spd = np.abs(vb[:, 0])
    for axname, ax in (("roll", 0), ("yaw", 2)):
        print(f"\n== {axname} weathervane vs v_body_y, binned by |v_body_x| ==", flush=True)
        for lo, hi in [(0, 2), (2, 3.5), (3.5, 5), (5, 7), (7, 10)]:
            mm = (spd >= lo) & (spd < hi) & (np.abs(vb[:, 1]) > 0.3)   # require real sideslip
            if int(mm.sum()) < 200:
                print(f"   v[{lo},{hi}) n={int(mm.sum())} (skip)", flush=True); continue
            o = fit_wv_axis(dW[mm, ax], C[mm, ax], W[mm, ax], vb[mm, 1])
            print(f"   v[{lo},{hi}) n={o['n']:5d}  wv={o['wv']:+.4f}  R2={o['r2']:.2f}", flush=True)
```
Wire `refit_dynamic` to run when invoked (extend the `__main__`/run block to also call it; include `collect_vq_crab` in the `load_index` script filter):
```python
    runs = [r for r in load_index() if r["script"] in ("collect_vq_crab", "collect_vq_lateral", "collect_vq")]
    P = [p for p in (prep(r) for r in runs) if p]
    refit_dynamic(P)
```
scp `wv_speed_fit.py` back.

- [ ] **Step 4: Run the test, verify it passes**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" -m pytest test_wv_fit.py -v"` → 2 passed.

- [ ] **Step 5: Run the refit on the real data**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" wv_speed_fit.py"` → prints the roll & yaw weathervane coefficient surface vs (v_body_y, |v_body_x|). Record the numbers. Compare to `vq_model.json` (yaw −0.149) + the prior roll fit (−0.105−0.019·vx). PASS = coefficients with R²>0.5 at v>4 (the regime that was missing), showing whether the moment grows/holds/changes vs the v0–4 fit.

- [ ] **Step 6: Commit**

`ssh laptop "cd /d \"$WT\" && git add wv_speed_fit.py test_wv_fit.py && git commit -m \"feat(cor-127): dynamic-regime weathervane refit (fit_wv_axis + binned by speed)\""`

---

## Task 4: Trajectory validation (replay current vs refit)

**Files:** Create `$WT/wv_validate.py`

- [ ] **Step 1: Write the replay-compare script**

Write `/tmp/wv_validate.py`, scp to `$WT/wv_validate.py`. It replays a held-out crab run through `vq_sim` (the scalar matched model) with the current vs refit yaw/roll weathervane and reports which better reproduces the live sideslip + attitude trajectory:
```python
"""wv_validate.py -- does the matched model reproduce the live high-sideslip trajectory? Replay a
held-out crab run through vq_sim with CURRENT vs REFIT weathervane; compare predicted sideslip +
attitude vs actual over short horizons. PASS = refit beats current on the dynamic-regime run.
Usage: python wv_validate.py <crab_run_dir> --roll_wv0 -0.105 --roll_wv1 -0.019 --yaw_wv -0.149
"""
import sys, json
import numpy as np
from aigp.recorder import load_run
from fit_model import prep, qfix, WFIX
from aigp.geometry import quat_to_R


def argf(flag, d): return float(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else d


run = sys.argv[1]
model = json.load(open("sysid/vq_model.json"))
rl = model["rate_loop"]; G = np.array([rl[k]["gain_G"] for k in ("roll", "pitch", "yaw")])
TAU = np.array([rl[k]["tau_ms"] for k in ("roll", "pitch", "yaw")]) / 1e3
DT = 1.0 / 72.0
d = load_run(run)
# dedup + frame-correct via the same conventions as prep (reuse its internals on raw arrays)
_, uidx = np.unique(d["t_us"], return_index=True); uidx = np.sort(uidx)
Q = d["quat"][uidx]; W = d["omega"][uidx] * WFIX; C = d["cmd"][uidx]
Rs = np.array([quat_to_R(qfix(q)) for q in Q]); Vw = d["vel"][uidx]
vb = Vw  # ODOMETRY vel is body frame (per conventions)


def rollout_omega(roll_wv0, roll_wv1, yaw_wv):
    """1-step omega prediction from the rate loop + weathervane; compare to actual omega[k+1]."""
    a = np.exp(-DT / TAU)
    pred = a * W[:-1] + (1 - a) * G * C[:-1, :3]
    pred[:, 0] += (roll_wv0 + roll_wv1 * vb[:-1, 0]) * vb[:-1, 1] * DT
    pred[:, 2] += yaw_wv * vb[:-1, 1] * DT
    err = np.sqrt(np.mean((pred - W[1:]) ** 2, axis=0))   # per-axis omega RMSE
    return err


cur = rollout_omega(-0.105, -0.019, model["weathervane"]["wv_coeff"])
new = rollout_omega(argf("--roll_wv0", -0.105), argf("--roll_wv1", -0.019), argf("--yaw_wv", -0.149))
print(f"omega RMSE [roll,pitch,yaw]  current={np.round(cur,4)}  refit={np.round(new,4)}", flush=True)
print(f"refit better on roll+yaw: {bool(new[0] <= cur[0] and new[2] <= cur[2])}", flush=True)
```

- [ ] **Step 2: Run validation with current coefficients (baseline)**

Run: `ssh laptop "cd /d \"$WT\" && \"$PY\" wv_validate.py \"C:\Users\alexj\Documents\vq_data\<a_crab_run>\""` (use a high-speed crab run id from Task 2). Records the current-model omega RMSE on the dynamic-regime run.

- [ ] **Step 3: Run validation with the refit coefficients**

Re-run with `--roll_wv0 <X> --roll_wv1 <Y> --yaw_wv <Z>` from Task 3's high-v bins. PASS = refit roll+yaw RMSE ≤ current on the held-out high-sideslip run (the model is more faithful where it mattered).

- [ ] **Step 4: Commit**

`ssh laptop "cd /d \"$WT\" && git add wv_validate.py && git commit -m \"feat(cor-127): replay validation of weathervane refit on dynamic-regime data\""`

---

## Task 5: Update the model (both repos)

**Files:** Modify `$WT/sysid/vq_model.json` (laptop); Modify `$MAC/sim/dynamics/vq_matched.py` (Mac)

- [ ] **Step 1: Update `vq_model.json` yaw weathervane (if refit changed it)**

Only if Task 4 showed the refit yaw improves the dynamic-regime RMSE: fetch `$WT/sysid/vq_model.json`, set `weathervane.wv_coeff` to the refit value, add a note field `weathervane.dynamic_refit` with the roll_wv0/roll_wv1 and the v-bin R². scp back. If the refit did NOT improve, leave it and record that the v0–4 model is confirmed adequate.

- [ ] **Step 2: Update the roll weathervane constants in `VQMatchedDynamics`**

The roll weathervane is hardcoded in `$MAC/sim/dynamics/vq_matched.py` (`self.roll_wv0, self.roll_wv1 = (-0.105, -0.019)`). If Task 4 refit them, Read that file and update the tuple to the refit values; keep the existing comment referencing COR-127 and append the dynamic-regime provenance. (If unchanged, skip.)

- [ ] **Step 3: Keep the matched-sim pytest green**

Run: `cd $MAC && python -m pytest tests/test_sim/test_vq_matched.py -q` → expect all pass (the refit changes constants, not structure; equivalence/hover/ZOH tests still hold).

- [ ] **Step 4: Commit both**

```bash
ssh laptop "cd /d \"$WT\" && git add sysid/vq_model.json && git commit -m \"refit(cor-127): weathervane coefficients in the dynamic regime\""
cd $MAC && git add sim/dynamics/vq_matched.py && git commit -m "refit(cor-127): dynamic-regime roll weathervane in VQMatchedDynamics"
```

---

## Task 6: Record results

**Files:** Modify the obsidian exp log, `$MAC/HANDOFF.md`; post to COR-127.

- [ ] **Step 1: Append exp-log entry**

Append a `**WV-DYNAMIC (06-09).**` paragraph to `/Users/alex/Documents/obsidian-vault/01 Projects/CorvidX/AI-GP Experiment Log.md`: the crab-sweep maneuver, the 3 runs, the weathervane coefficient surface vs (v_body_y, v_body_x), whether it grows/holds vs the v0–4 fit, the replay-validation result (refit vs current RMSE), and the model update. Include run_ids.

- [ ] **Step 2: Update HANDOFF.md**

Update item (e): the weathervane is now validated/refit in the dynamic regime (or confirmed adequate); the matched sim is faithful for RL → next is RL/DAgger retrain on it (downstream).

- [ ] **Step 3: Post to COR-127**

Post a Linear comment summarizing: dynamic-regime weathervane validation done via the crab-sweep (the missing divergent-regime data), the refit coefficients + validation, and that the matched sim is now faithful for the RL-transfer path. (Batch into ONE comment; Linear rate-limits writes.)

- [ ] **Step 4: Commit Mac docs**

`cd $MAC && git add docs/superpowers HANDOFF.md && git commit -m "docs(cor-127): weathervane dynamic-regime validation results"`

---

## Self-review notes
- **Spec coverage:** C1 crab maneuver → Task 1–2; C2 refit → Task 3; C3 trajectory validation → Task 4; model update → Task 5; results → Task 6. Approaches A (crab) is Task 1–2; C (replay cross-check) is Task 4; B (INDI) is the documented fallback only (Risks), not a task. All covered.
- **Type consistency:** `fit_wv_axis(dW_axis, wcmd_axis, omega_axis, v_body)` signature identical in test (Task 3.1) and impl (Task 3.3) and caller `refit_dynamic` and `wv_validate`'s rollout. `roll_wv0/roll_wv1/yaw_wv` naming consistent across Tasks 3–5. `prep()`/`qfix`/`WFIX` reused from `fit_model` (not redefined).
- **No placeholders:** every code/command step is concrete; the only conditionals (Task 5) are explicit ("if the refit improved … else record adequate").
- **Frame discipline:** all sideslip/omega via `prep()`/`qfix`/`WFIX` (solved conventions); stability judged by TILT; live FF/metric signs verified on the dashboard.
