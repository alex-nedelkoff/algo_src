# AI-GP Vision Gate Fly-Through Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect the next gate from the camera feed alone and fly the drone through it, with Rerun 3D/mask telemetry and a live web HSV tuner.

**Architecture:** Two pure, unit-tested modules (`gate_detect`, `visual_servo`) feed a live loop (`fly_gate`) that reuses the proven `race_cruise` body-rate inner loop, driving heading/altitude from the gate bearing. `viz` logs to a Rerun viewer on the Mac; `hsv_tuner` is a Flask web app for live threshold tuning. Success = the sim's `active_gate_index` increments.

**Tech Stack:** Python 3.13, OpenCV, NumPy, pymavlink, Flask, rerun-sdk; `aigp` conda env on the Windows sim host.

---

## Working environment (read first)
- Code + tests run in the **`aigp` conda env on the Windows host** (`ssh alexj@100.120.233.90`).
- Edit on Mac → `scp` to `C:/Users/alexj/Documents/algo_src/.claude/worktrees/aigp-client/<path>` → run over ssh. Do NOT pipe `| tail`/`findstr` *inside* the ssh command (runs in Windows cmd); pipe on the Mac side.
  - Run pattern: `ssh alexj@100.120.233.90 "cd C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client && conda run -n aigp <cmd> 2>&1" | tail -30`
- Flight scripts require a **live qualifier flight** (only the user drives the GUI); `fresh_start()` gates actuation on race-live (acting before live DQs the sim).
- Mac Tailscale IP (Rerun viewer): **100.101.13.126**. Windows/sim host: **100.120.233.90**.
- Reference: `aigp/race_cruise.py` (working rate controller — reuse its inner loop), `aigp/control_math.py`, `aigp/geometry.py` (intrinsics `K`), `aigp/io_layer.py` (`VisionIO` frames, `Store`).

## File structure
- Create `aigp/gate_detect.py` — red-gate detector (pure). Responsibility: frame → nearest gate bbox/center.
- Create `aigp/visual_servo.py` — bearing → setpoints (pure). Responsibility: detection → (fwd_speed, yaw_sp, z_sp).
- Create `aigp/viz.py` — Rerun logging helpers. Responsibility: log 3D/images/scalars, never crash the loop.
- Create `fly_gate.py` (worktree root, like `race_cruise.py`) — live loop tying it together.
- Create `servo_sign_probe.py` (worktree root) — bring-up: pin pixel→control signs.
- Create `hsv_tuner.py` (worktree root) — Flask web HSV tuner → `detect_params.json`.
- Create `detect_params.json` (worktree root) — shared thresholds.
- Create `tests/test_gate_detect.py`, `tests/test_visual_servo.py`, `tests/fixtures/gate_frame_*.jpg`.

---

## Task 0: Setup — deps, fixtures, params

**Files:**
- Create: `detect_params.json`, `tests/fixtures/gate_frame_0.jpg` (and `_1`, `_2`)

- [ ] **Step 1: Install deps into the aigp env**

Run: `ssh alexj@100.120.233.90 "conda run -n aigp pip install flask rerun-sdk 2>&1" | tail -5`
Expected: both install successfully.

- [ ] **Step 2: Capture/copy test fixtures**

The 3 frames were already captured this session at the worktree root (`gate_frame_0.jpg`..`_2.jpg`). Move them into fixtures:
Run: `ssh alexj@100.120.233.90 "cd C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client && mkdir tests\fixtures 2>nul & copy gate_frame_*.jpg tests\fixtures\ 2>&1"`
If missing, re-grab with `grab_frame.py` (already on the worktree) while the sim is live, then copy.
Expected: `gate_frame_0.jpg gate_frame_1.jpg gate_frame_2.jpg` in `tests/fixtures/`.

- [ ] **Step 3: Write the default params file**

Create `detect_params.json`:
```json
{"h_lo1":0,"h_hi1":12,"h_lo2":165,"h_hi2":180,"s_min":110,"v_min":70,"min_area_px":80,"square_tol":0.5,"max_v_frac":0.72}
```

- [ ] **Step 4: Confirm the gate's pixel location in fixture 0** (for the Task 1 test)

Open `tests/fixtures/gate_frame_0.jpg` and read off the nearest (largest) red gate's center. Expected ≈ **(u=315, v=180)** ± ~70 px. Record the actual value; use it in Task 1's assertion.

- [ ] **Step 5: Commit**
```bash
git add detect_params.json tests/fixtures/gate_frame_*.jpg
git commit -m "chore: gate-detect fixtures + default HSV params"
```

---

## Task 1: `gate_detect.py` — red-gate detector (TDD)

**Files:**
- Create: `aigp/gate_detect.py`
- Test: `tests/test_gate_detect.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gate_detect.py
import os
import cv2
import numpy as np
from aigp.gate_detect import detect_gate, red_mask, load_params, GateDetection

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
P = load_params(os.path.join(os.path.dirname(__file__), "..", "detect_params.json"))


def _frame(n):
    return cv2.imread(os.path.join(FIX, f"gate_frame_{n}.jpg"))


def test_mask_has_gate_pixels():
    m = red_mask(_frame(0), P)
    assert m.dtype == np.uint8 and m.shape == (360, 640)
    assert m.sum() > 0  # gate is bright red -> mask non-empty


def test_detect_nearest_gate_center():
    det = detect_gate(_frame(0), P)
    assert isinstance(det, GateDetection)
    # nearest gate center ~ (315,180) in this fixture (confirm in Task 0 step 4)
    assert abs(det.u - 315) < 80 and abs(det.v - 180) < 80
    assert det.area >= P["min_area_px"]


def test_detect_none_on_desaturated():
    gray = np.full((360, 640, 3), 60, np.uint8)  # flat grey, no saturation
    assert detect_gate(gray, P) is None


def test_rejects_low_reflection():
    # a red square low in the frame (a 'reflection') must not be selected
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(img, (300, 330), (340, 358), (0, 0, 255), -1)  # BGR red, v>0.72*360
    assert detect_gate(img, P) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ssh ... "cd ... && conda run -n aigp pytest tests/test_gate_detect.py -q 2>&1" | tail -20`
Expected: FAIL (`ModuleNotFoundError: aigp.gate_detect`).

- [ ] **Step 3: Implement `aigp/gate_detect.py`**

```python
"""Classical gate detector: find the nearest saturated-red gate in a camera frame.
Pure (no sim deps). Gates are the only saturated objects in the desaturated env."""
from __future__ import annotations
import json, os
from dataclasses import dataclass
import cv2
import numpy as np

DEFAULT_PARAMS = {"h_lo1": 0, "h_hi1": 12, "h_lo2": 165, "h_hi2": 180,
                  "s_min": 110, "v_min": 70, "min_area_px": 80,
                  "square_tol": 0.5, "max_v_frac": 0.72}


def load_params(path: str = "detect_params.json") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return {**DEFAULT_PARAMS, **json.load(f)}
    return dict(DEFAULT_PARAMS)


@dataclass
class GateDetection:
    u: float; v: float; w_px: float; h_px: float; area: float
    bbox: tuple  # (x, y, w, h)


def red_mask(bgr, p: dict) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo1 = np.array([p["h_lo1"], p["s_min"], p["v_min"]]); hi1 = np.array([p["h_hi1"], 255, 255])
    lo2 = np.array([p["h_lo2"], p["s_min"], p["v_min"]]); hi2 = np.array([p["h_hi2"], 255, 255])
    m = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def detect_gate(bgr, params: dict | None = None):
    p = params or load_params()
    H, W = bgr.shape[:2]
    mask = red_mask(bgr, p)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < p["min_area_px"]:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        if abs(w / float(h) - 1.0) > p["square_tol"]:        # roughly square
            continue
        if (y + h / 2.0) > p["max_v_frac"] * H:              # reject low track reflections
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)
    if best is None:
        return None
    area, x, y, w, h = best
    return GateDetection(x + w / 2.0, y + h / 2.0, float(w), float(h), area, (x, y, w, h))


def draw_overlay(bgr, det) -> np.ndarray:
    out = bgr.copy()
    if det is not None:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(det.u), int(det.v)), 3, (0, 255, 0), -1)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ssh ... "cd ... && conda run -n aigp pytest tests/test_gate_detect.py -q 2>&1" | tail -20`
Expected: PASS (4 passed). If `test_detect_nearest_gate_center` fails on coords, fix the expected center from Task 0 step 4 (the fixture is ground truth). If the mask misses the gate, tune `s_min/v_min/h_*` via the tuner (Task 3) and update `detect_params.json`.

- [ ] **Step 5: Commit**
```bash
git add aigp/gate_detect.py tests/test_gate_detect.py
git commit -m "feat: classical red-gate detector with reflection rejection"
```

---

## Task 2: `visual_servo.py` — bearing → setpoints (TDD)

**Files:**
- Create: `aigp/visual_servo.py`
- Test: `tests/test_visual_servo.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_visual_servo.py
from aigp.gate_detect import GateDetection
from aigp.visual_servo import servo, ServoCfg, ServoCmd

CFG = ServoCfg()  # defaults; signs +1 (pinned later in bring-up)


def _det(u, v):
    return GateDetection(u, v, 40, 40, 1600, (int(u) - 20, int(v) - 20, 40, 40))


def test_centered_no_correction():
    cmd = servo(_det(320, 180), yaw_cur=1.0, z_cur=-5.0, cfg=CFG)
    assert cmd.have_gate
    assert abs(cmd.yaw_sp - 1.0) < 1e-6
    assert abs(cmd.z_sp - (-5.0)) < 1e-6
    assert cmd.fwd_speed == CFG.fwd_speed


def test_gate_right_yaws_positive():
    cmd = servo(_det(480, 180), yaw_cur=0.0, z_cur=-5.0, cfg=CFG)
    ex = (480 - 320) / 320.0
    assert abs(cmd.yaw_sp - CFG.k_yaw * CFG.sign_x * ex) < 1e-6


def test_gate_low_descends():
    cmd = servo(_det(320, 300), yaw_cur=0.0, z_cur=-5.0, cfg=CFG)
    ey = (300 - 180) / 320.0
    assert cmd.z_sp > -5.0  # NED z down +, gate below center -> descend
    assert abs((cmd.z_sp + 5.0) - CFG.k_alt * CFG.sign_y * ey) < 1e-6


def test_no_gate_hovers_without_last():
    cmd = servo(None, yaw_cur=0.5, z_cur=-5.0, cfg=CFG)
    assert not cmd.have_gate and cmd.fwd_speed == 0.0 and cmd.yaw_sp == 0.5


def test_no_gate_coasts_with_last():
    last = ServoCmd(CFG.fwd_speed, 0.7, -4.0, True)
    cmd = servo(None, yaw_cur=0.0, z_cur=-5.0, cfg=CFG, last=last)
    assert cmd.yaw_sp == 0.7 and cmd.z_sp == -4.0
    assert 0.0 < cmd.fwd_speed < CFG.fwd_speed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ssh ... "cd ... && conda run -n aigp pytest tests/test_visual_servo.py -q 2>&1" | tail -20`
Expected: FAIL (`ModuleNotFoundError: aigp.visual_servo`).

- [ ] **Step 3: Implement `aigp/visual_servo.py`**

```python
"""Visual servo: gate bearing -> heading / altitude / forward setpoints. Pure.
Camera intrinsics fx=fy=320, cx=320, cy=180 (aigp.geometry.K). Pixel->control signs
(sign_x, sign_y) are pinned empirically in bring-up (servo_sign_probe.py)."""
from __future__ import annotations
from dataclasses import dataclass

FX = FY = 320.0
CX = 320.0
CY = 180.0


@dataclass
class ServoCfg:
    k_yaw: float = 0.8       # rad per unit ex (horizontal bearing)
    k_alt: float = 6.0       # m per unit ey (z-setpoint nudge)
    fwd_speed: float = 1.8   # m/s along camera heading (capped, weathervane-safe)
    sign_x: float = 1.0      # pinned in bring-up
    sign_y: float = 1.0
    max_dz: float = 8.0      # clamp altitude nudge
    coast_frac: float = 0.5  # forward-speed fraction when gate momentarily lost


@dataclass
class ServoCmd:
    fwd_speed: float
    yaw_sp: float
    z_sp: float
    have_gate: bool


def servo(det, yaw_cur: float, z_cur: float, cfg: ServoCfg, last: ServoCmd | None = None) -> ServoCmd:
    if det is None:
        if last is not None:
            return ServoCmd(cfg.fwd_speed * cfg.coast_frac, last.yaw_sp, last.z_sp, False)
        return ServoCmd(0.0, yaw_cur, z_cur, False)
    ex = (det.u - CX) / FX
    ey = (det.v - CY) / FY
    yaw_sp = yaw_cur + cfg.k_yaw * cfg.sign_x * ex
    dz = cfg.k_alt * cfg.sign_y * ey
    dz = max(-cfg.max_dz, min(cfg.max_dz, dz))
    return ServoCmd(cfg.fwd_speed, yaw_sp, z_cur + dz, True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ssh ... "cd ... && conda run -n aigp pytest tests/test_visual_servo.py -q 2>&1" | tail -20`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**
```bash
git add aigp/visual_servo.py tests/test_visual_servo.py
git commit -m "feat: visual-servo bearing-to-setpoint mapping"
```

---

## Task 3: `hsv_tuner.py` — live web HSV tuner

**Files:**
- Create: `hsv_tuner.py`

- [ ] **Step 1: Implement the tuner**

```python
"""Live HSV gate tuner. Serves frame+mask (MJPEG) + sliders at http://<win-tailscale-ip>:8088.
'Save' writes detect_params.json (read by aigp.gate_detect). Run while the sim feeds frames."""
import json, threading, time
import cv2
import numpy as np
from flask import Flask, Response, request, redirect
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.gate_detect import red_mask, detect_gate, draw_overlay, load_params

PARAMS = load_params()
app = Flask(__name__)
s = Store(); m = MavlinkIO(s); m.wait_heartbeat(10); m.start(); VisionIO(s).start()


def _mjpeg():
    while True:
        fr, _ = s.get_frame()
        if fr is None:
            time.sleep(0.05); continue
        bgr = fr[0]
        mask = red_mask(bgr, PARAMS)
        det = detect_gate(bgr, PARAMS)
        overlay = draw_overlay(bgr, det)
        maskc = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combo = np.hstack([overlay, maskc])
        ok, jpg = cv2.imencode(".jpg", combo)
        if ok:
            yield (b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        time.sleep(0.04)


PAGE = """<html><body style='background:#111;color:#ddd;font-family:sans-serif'>
<h3>AI-GP HSV gate tuner — left: overlay, right: mask</h3>
<img src='/stream' width='1280'><br>
<form method='post' action='/save'>
{rows}
<button>Save detect_params.json</button></form></body></html>"""
SLIDERS = ["h_lo1","h_hi1","h_lo2","h_hi2","s_min","v_min","min_area_px"]


@app.route("/")
def index():
    rows = "".join(
        f"{k}: <input name='{k}' type='range' min='0' max='{255 if k!='min_area_px' else 4000}' "
        f"value='{int(PARAMS[k])}' oninput=\"this.nextElementSibling.value=this.value\">"
        f"<output>{int(PARAMS[k])}</output><br>" for k in SLIDERS)
    return PAGE.format(rows=rows)


@app.route("/stream")
def stream():
    return Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/save", methods=["POST"])
def save():
    for k in SLIDERS:
        if k in request.form:
            PARAMS[k] = int(request.form[k])
    with open("detect_params.json", "w") as f:
        json.dump(PARAMS, f, indent=0)
    return redirect("/")


# live slider preview without saving: update PARAMS on every change via query
@app.route("/set")
def setp():
    for k in SLIDERS:
        if k in request.args:
            PARAMS[k] = int(request.args[k])
    return ("", 204)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088, threaded=True)
```

(For live preview while dragging, wire each slider's `oninput` to also `fetch('/set?'+name+'='+value)`; optional — the Save round-trip already updates the mask.)

- [ ] **Step 2: Run it (sim must be live/streaming frames)**

Run: `ssh alexj@100.120.233.90 "cd ... && conda run -n aigp python hsv_tuner.py" &` (leave running)
Open `http://100.120.233.90:8088` in the Mac browser.
Expected: overlay (left) + mask (right) update live; the gate appears as a clean white square in the mask; dragging sliders + Save changes the mask. Tune until only gates show, reflections gone, then Save.

- [ ] **Step 3: Verify the saved params still pass detector tests**

Run: `ssh ... "cd ... && conda run -n aigp pytest tests/test_gate_detect.py -q 2>&1" | tail -10`
Expected: PASS (the tuned `detect_params.json` still boxes the fixture gate). If a test breaks, the tuning is too tight — loosen and re-Save.

- [ ] **Step 4: Commit**
```bash
git add hsv_tuner.py detect_params.json
git commit -m "feat: web HSV gate tuner (frame+mask MJPEG, writes detect_params.json)"
```

---

## Task 4: `viz.py` — Rerun telemetry helpers

**Files:**
- Create: `aigp/viz.py`

- [ ] **Step 1: Pin the rerun API for the installed version**

Run: `ssh ... "cd ... && conda run -n aigp python -c \"import rerun as rr; print(rr.__version__)\" 2>&1" | tail -2`
Note the version. The code below targets rerun-sdk 0.2x (`rr.init`, `rr.connect_grpc`/`rr.connect`, `rr.Points3D`, `rr.Image`, `rr.Scalars`/`rr.Scalar`, `rr.Transform3D`). If an API name differs, adjust to the installed version (check `rr.<name>` exists).

- [ ] **Step 2: Implement `aigp/viz.py`**

```python
"""Rerun telemetry for gate fly-through. Best-effort: never raise into the control loop.
Connects to a Rerun viewer on the Mac (run `rerun` there first)."""
from __future__ import annotations
import numpy as np

MAC_VIEWER = "100.101.13.126:9876"
_ok = False


def init():
    """Connect to the Mac viewer; on any failure, fall back to a local .rrd recording."""
    global _ok
    try:
        import rerun as rr
        rr.init("aigp-gate", spawn=False)
        try:
            rr.connect_grpc(f"rerun+http://{MAC_VIEWER}/proxy")  # 0.2x API
        except Exception:
            try:
                rr.connect(MAC_VIEWER)                            # older API
            except Exception:
                rr.save("aigp_gate.rrd")                          # offline fallback
        try:
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # NED-ish
        except Exception:
            pass
        _ok = True
    except Exception as e:
        print(f"[viz] disabled: {e}", flush=True)
        _ok = False


def _scalar(rr, path, val):
    for fn in ("Scalars", "Scalar"):
        if hasattr(rr, fn):
            rr.log(path, getattr(rr, fn)(float(val))); return


def log_step(t_s, drone_pos, drone_vel, frame_bgr, mask, overlay, det, cmd):
    """Log one control step. All args optional-safe."""
    if not _ok:
        return
    try:
        import rerun as rr
        rr.set_time_seconds("t", float(t_s))
        if drone_pos is not None:
            rr.log("world/drone", rr.Points3D([drone_pos], radii=0.3))
        if frame_bgr is not None:
            rr.log("cam/frame", rr.Image(frame_bgr[:, :, ::-1]))   # BGR->RGB
        if mask is not None:
            rr.log("cam/mask", rr.Image(mask))
        if overlay is not None:
            rr.log("cam/overlay", rr.Image(overlay[:, :, ::-1]))
        if det is not None:
            _scalar(rr, "det/u", det.u); _scalar(rr, "det/v", det.v)
            _scalar(rr, "det/size", det.w_px)
        if cmd is not None:
            _scalar(rr, "ctrl/yaw_sp", cmd.yaw_sp)
            _scalar(rr, "ctrl/fwd", cmd.fwd_speed)
            _scalar(rr, "ctrl/have_gate", 1.0 if cmd.have_gate else 0.0)
        if drone_vel is not None:
            _scalar(rr, "ctrl/speed", float(np.linalg.norm(drone_vel[:2])))
    except Exception:
        pass
```

- [ ] **Step 3: Smoke-test the link**

On the Mac: `rerun` (opens the viewer, listens on 9876). Then:
Run: `ssh ... "cd ... && conda run -n aigp python -c \"import numpy as np; from aigp import viz; viz.init(); [viz.log_step(i*0.1, np.array([i*0.1,0,-2.0]), np.array([1,0,0]), np.zeros((360,640,3),np.uint8), np.zeros((360,640),np.uint8), np.zeros((360,640,3),np.uint8), None, None) for i in range(30)]\" 2>&1" | tail -5`
Expected: the Mac Rerun viewer shows a `world/drone` point moving and `cam/*` image panels. If nothing arrives, check Tailscale reachability to 100.101.13.126:9876; otherwise the `.rrd` fallback file is written (scp + open).

- [ ] **Step 4: Commit**
```bash
git add aigp/viz.py
git commit -m "feat: rerun telemetry helpers (3D, masks, scalars) with offline fallback"
```

---

## Task 5: Live bring-up — `servo_sign_probe.py` then `fly_gate.py`

**Files:**
- Create: `servo_sign_probe.py`, `fly_gate.py`

- [ ] **Step 1: Implement `servo_sign_probe.py` (pin pixel→control signs)**

```python
"""Bring-up: hover (position hold), command a small +yaw step, and report how the detected
gate's u (and a small climb -> v) move in-image. Sets ServoCfg.sign_x / sign_y.
Camera looks -body-x (empirical); geometry.py's +body-x assumption is NOT trusted."""
import json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_accel, desired_attitude, mat_to_quat,
                               attitude_error_quat, collective_accel, accel_to_thrust_norm)
from aigp.gate_detect import detect_gate, load_params

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = np.array([0.7, 1.6, 1.0]); WMAX = 4.0; LOOP_DT = 0.004
KP = np.array([0.0, 0.0, 1.8]); KD = np.array([1.0, 1.0, 3.0]); TILT = np.tan(np.radians(10)) * 9.81
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
P = load_params()

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time() * 1000); c = Commander(m.conn, boot)


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    t = time.time()
    while time.time()-t < 1.0: idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time()-t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)): break
        time.sleep(0.02)
    while time.time()-t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0: return True
        time.sleep(0.02)
    return False


def gate_u():
    fr, _ = s.get_frame()
    if fr is None: return None
    det = detect_gate(fr[0], P)
    return det.u if det else None


assert fresh_start(), "not live"
ds0 = s.get_drone(); spawn = ds0.pos_ned.copy()
yaw0 = float(np.arctan2(quat_to_R(ds0.quat_wxyz)[1, 0], quat_to_R(ds0.quat_wxyz)[0, 0]))
c.arm()


def hold(yaw_target, secs):
    """Position-hold at spawn with a commanded heading; return mean detected gate u."""
    us = []; t0 = time.time()
    while time.time()-t0 < secs:
        ds = s.get_drone()
        if ds is not None:
            a = desired_accel(ds.pos_ned, ds.vel_ned, spawn, np.zeros(3), KP, KD)
            n = float(np.linalg.norm(a[:2]))
            if n > TILT: a[:2] = a[:2]/n*TILT
            Rc = quat_to_R(ds.quat_wxyz); yc = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            q = mat_to_quat(desired_attitude(a, yc))
            w = KP_ATT * attitude_error_quat(ds.quat_wxyz, q)
            w[2] = 2.0 * ((yaw_target - yc + np.pi) % (2*np.pi) - np.pi)
            thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
            c.send_attitude_target(np.clip(w/RG, -WMAX, WMAX), thr)
            if time.time()-t0 > secs*0.5:
                u = gate_u()
                if u is not None: us.append(u)
        time.sleep(LOOP_DT)
    return float(np.mean(us)) if us else None


u0 = hold(yaw0, 4.0)
u1 = hold(yaw0 + np.radians(8), 4.0)   # small +yaw step
print(f"gate u at yaw0={u0}, at yaw0+8deg={u1}", flush=True)
if u0 is not None and u1 is not None:
    # sign_x maps ex (=(u-cx)/fx) to the yaw correction that re-centers the gate.
    # If +yaw moved u in +direction, then to null +ex we need -yaw -> sign_x = -sign(du/dyaw).
    sign_x = -1.0 if (u1 - u0) > 0 else 1.0
    print(f"du/d(+yaw) = {u1-u0:+.1f}px  -> ServoCfg.sign_x = {sign_x:+.0f}", flush=True)
else:
    print("gate not detected during hold — tune detector first (Task 3).", flush=True)
```

- [ ] **Step 2: Run the sign probe (sim live)**

Run: `ssh ... "cd ... && conda run -n aigp python servo_sign_probe.py 2>&1" | tail -8`
Expected: prints `du/d(+yaw)` and a recommended `sign_x`. Record it. (For `sign_y`: gate above center should require climb; default `sign_y=+1` descends when gate is low (v large) — confirm direction during the first closed-loop run and flip if it climbs away.)

- [ ] **Step 3: Set the pinned sign in `ServoCfg` defaults**

Edit `aigp/visual_servo.py`: set `sign_x` (and `sign_y` if step 2/first-run shows it inverted) to the measured value. Re-run `pytest tests/test_visual_servo.py -q` (the sign only flips the expected direction; update the two directional asserts if you changed defaults).

- [ ] **Step 4: Implement `fly_gate.py` (closed loop)**

```python
"""Fly through ONE gate using the camera only. Visual servo (gate bearing -> yaw/altitude)
on top of the race_cruise body-rate inner loop. Success = active_gate_index increments."""
import json, time
import numpy as np
from pymavlink import mavutil
from aigp.io_layer import MavlinkIO, VisionIO
from aigp.state import Store
from aigp.commander import Commander
from aigp.geometry import quat_to_R
from aigp.control_math import (desired_attitude, mat_to_quat, attitude_error_quat,
                               collective_accel, accel_to_thrust_norm)
from aigp.gate_detect import detect_gate, red_mask, draw_overlay, load_params
from aigp.visual_servo import servo, ServoCfg, ServoCmd
from aigp import viz

r = json.load(open("sysid/sim_response.json"))
HOVER = r["hover_thrust"]; KA = r["k_a"]
RG = np.array([r["rate_gain_axes"]["roll"], r["rate_gain_axes"]["pitch"], r["rate_gain_axes"]["yaw"]])
KP_ATT = np.array([0.7, 1.6, 1.0]); KP_YAW = 3.0; KD_YAW = 0.3
KD_H = 1.2; KP_Z = 1.8; KD_Z = 3.0; AL_MAX = 0.5
WMAX = 4.0; LOOP_DT = 0.004; DURATION = 40.0
TILT = np.tan(np.radians(15)) * 9.81
IDLE = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
P = load_params(); CFG = ServoCfg()

s = Store(); m = MavlinkIO(s); assert m.wait_heartbeat(10); m.start(); VisionIO(s).start()
boot = int(time.time()*1000); c = Commander(m.conn, boot); viz.init()


def idle():
    m.conn.mav.set_attitude_target_send(int(time.time()*1000)-boot, m.conn.target_system,
                                        m.conn.target_component, IDLE, [1, 0, 0, 0], 0, 0, 0, 0)


def fresh_start():
    t = time.time()
    while time.time()-t < 1.0: idle(); time.sleep(0.02)
    prev = s.get_race(); pb = prev["boot_ms"] if prev else None
    c.sim_reset(); t = time.time()
    while time.time()-t < 30:
        idle(); r2 = s.get_race()
        if r2 and (not r2["race_live"] or (pb and r2["boot_ms"] < pb)): break
        time.sleep(0.02)
    while time.time()-t < 30:
        idle(); d = s.get_drone()
        if s.get_race_live() and d is not None and np.linalg.norm(d.vel_ned) < 3.0: return True
        time.sleep(0.02)
    return False


assert fresh_start(), "not live"
ds0 = s.get_drone(); z_sp0 = ds0.pos_ned[2]
gi0 = s.get_gate_idx()
c.arm()
last = None; t0 = time.time(); passed = False; logn = 0
while time.time()-t0 < DURATION:
    ds = s.get_drone()
    if ds is not None:
        Rc = quat_to_R(ds.quat_wxyz); yaw_cur = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
        fr, _ = s.get_frame(); bgr = fr[0] if fr else None
        det = detect_gate(bgr, P) if bgr is not None else None
        cmd = servo(det, yaw_cur, ds.pos_ned[2], CFG, last); last = cmd
        # camera heading = -body-x; fly forward along it, steered by cmd.yaw_sp
        fwd = -np.array([np.cos(cmd.yaw_sp), np.sin(cmd.yaw_sp)])
        a = np.zeros(3)
        a[:2] = np.clip(KD_H * (cmd.fwd_speed - ds.vel_ned[:2] @ fwd), -4.0, AL_MAX) * fwd
        a[2] = KP_Z * (cmd.z_sp - ds.pos_ned[2]) + KD_Z * (0.0 - ds.vel_ned[2])
        n = float(np.linalg.norm(a[:2]))
        if n > TILT: a[:2] = a[:2]/n*TILT
        q = mat_to_quat(desired_attitude(a, yaw_cur))
        w = KP_ATT * attitude_error_quat(ds.quat_wxyz, q)
        w[2] = KP_YAW * ((cmd.yaw_sp - yaw_cur + np.pi) % (2*np.pi) - np.pi) - KD_YAW * float(ds.omega[2])
        thr = accel_to_thrust_norm(collective_accel(a, ds.quat_wxyz), HOVER, KA)
        c.send_attitude_target(np.clip(w/RG, -WMAX, WMAX), thr)
        if s.get_gate_idx() > gi0:
            passed = True; print(f"GATE PASSED at t={time.time()-t0:.1f}s (idx {gi0}->{s.get_gate_idx()})", flush=True); break
        logn += 1
        if logn % 12 == 0 and bgr is not None:
            viz.log_step(time.time()-t0, ds.pos_ned, ds.vel_ned, bgr, red_mask(bgr, P),
                         draw_overlay(bgr, det), det, cmd)
    time.sleep(LOOP_DT)
print(f"RESULT: {'PASSED gate' if passed else 'did NOT pass'} in {time.time()-t0:.0f}s", flush=True)
```

- [ ] **Step 5: Run the closed-loop fly-through (sim live; Rerun viewer open on Mac)**

Run: `ssh ... "cd ... && conda run -n aigp python fly_gate.py 2>&1" | tail -20`
Expected: `GATE PASSED ...` (active_gate_index increments). Watch Rerun for the 3D path + detection overlay.
Tuning loop (if it misses): adjust `ServoCfg.k_yaw` (steering authority), `fwd_speed` (slower = easier), `sign_y`/`k_alt` (vertical). If it loses the gate near the end (gate fills/leaves frame), the `coast` keeps it going straight — usually enough to pass.

- [ ] **Step 6: Confirm repeatability**

Re-run twice more. Expected: passes gate 0 at least 2 of 3.

- [ ] **Step 7: Commit**
```bash
git add servo_sign_probe.py fly_gate.py aigp/visual_servo.py
git commit -m "feat: closed-loop single-gate fly-through via visual servoing"
```

---

## Self-Review notes
- **Spec coverage:** detector (T1), reflection rejection (T1 test + `max_v_frac`), servo (T2), success=`active_gate_index` (T5), Rerun 3D+masks+scalars (T4), web HSV tuner→`detect_params.json` (T3), camera-sign empirical (T5 probe), bring-up order (T0→T5), deps (T0). All present.
- **Type consistency:** `GateDetection(u,v,w_px,h_px,area,bbox)`, `ServoCfg`/`ServoCmd(fwd_speed,yaw_sp,z_sp,have_gate)`, `servo(det,yaw_cur,z_cur,cfg,last)`, `detect_gate(bgr,params)->GateDetection|None`, `red_mask`/`draw_overlay` used consistently across tuner, probe, fly_gate, viz.
- **Live caveats (not pure-testable):** T3/T4/T5 are validated by running against the live sim, not unit tests — expected observations are stated for each.
