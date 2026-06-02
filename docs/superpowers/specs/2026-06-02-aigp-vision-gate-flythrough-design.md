# AI-GP vision gate fly-through — design (2026-06-02)

Linear: **COR-125**. Repo: `algo_src` branch `aigp-gate-data-collection`, package `aigp/`,
run on the Windows sim laptop in the `aigp` conda env. Companion to the control work in
`race_cruise.py` (see vault `01 Projects/CorvidX/AI-GP flight-control handoff 2026-06-02`).

## Goal & scope
Fly the drone **through one gate** using **only the camera feed** to find the gate — no
ground-truth gate poses (the track payload is NOT available as a control input). Closed loop:
camera → detect gate → bearing → visual servo → body-rate controller → pass through.

**In scope:** detect the next gate, steer through it, confirm passage. Plus diagnostic tooling
(3D flight view + CV mask layers + live HSV tuning).
**Out of scope (later):** full multi-gate lap, metric 3D gate estimation, learned detector.

## Why classical CV (not a learned detector)
The qualifier environment is "high-contrast, desaturated." Captured frames confirm: **gates are
the only saturated objects** — bright red/pink square frames against a desaturated grey/blue/black
world (buildings, blue track tube, black sky). So color+shape thresholding isolates gates with no
training data — which also sidesteps the "no gate poses to auto-label with" problem entirely.

## Architecture

```
camera frame ─▶ GateDetector ─▶ target (u,v,size) ─▶ VisualServo ─▶ setpoints ─▶ rate ctrl ─▶ motion
                    │ params.json        │ None if no gate                          ▲
              HSV tuner (web)      Rerun logging (frame, masks, 3D, plots)    drone state (race_cruise inner loop)
```

Five small units, each with one job, the first two pure (no sim deps, unit-testable):

### 1. `aigp/gate_detect.py` — detector (pure)
`detect_gate(bgr, params) -> GateDetection | None`
- BGR→HSV; threshold **two red hue bands** (red wraps HSV 0/180) ∧ `S ≥ s_min` ∧ `V ≥ v_min`.
- Morphological close; `findContours`.
- Keep contours that are **large enough** (`area ≥ min_area_px`) and **roughly square**
  (bbox aspect within `square_tol`); reject **track reflections** (appear low in the frame on
  the blue track surface, and are dimmer/distorted) by an upper-bound on `v` of the bbox and a
  solidity check.
- **Target = largest qualifying gate** (nearest = the one to fly through).
- `GateDetection`: `u, v` (center px), `w_px, h_px`, `area`, `bbox`, plus the binary `mask` and
  the annotated overlay for logging. Returns `None` if nothing qualifies.
- Reads thresholds from `detect_params.json` (shared with the tuner) — no recompile to retune.

`detect_params.json` schema:
```json
{"h_lo1":0,"h_hi1":10,"h_lo2":165,"h_hi2":180,"s_min":120,"v_min":80,
 "min_area_px":80,"square_tol":0.5,"max_v_frac":0.72}
```

### 2. `aigp/visual_servo.py` — bearing → setpoints (pure)
`servo(det, state, cfg) -> ServoCmd(fwd_speed, yaw_sp, z_sp)`
- Intrinsics `fx=fy=320, cx=320, cy=180`. Bearing errors:
  `ex = (u-cx)/fx` (horizontal), `ey = (v-cy)/fy` (vertical).
- **Yaw:** `yaw_sp = yaw_cur + K_yaw * SIGN_X * ex` — rotate so the camera points at the gate.
- **Altitude:** `z_sp = z_cur + K_alt * SIGN_Y * ey` (gate below center → descend; NED z down +).
- **Forward:** steady modest `fwd_speed` along the **current camera heading** (−body-x), capped
  (`AL_MAX`) to stay under the weathervane threshold (per race_cruise findings).
- **Gate lost** (`det is None`): hold last `yaw_sp`/`z_sp`, coast at reduced speed for `T_coast`,
  then slow toward hover. `SIGN_X/SIGN_Y` are pinned empirically in bring-up (see Risks).

### 3. `aigp/fly_gate.py` — integration loop
- `fresh_start()` (clears throttle → `sim_reset` → wait live, as in race_cruise) → `arm()`.
- Each step (~250 Hz control; uses the **latest** camera frame, which arrives slower):
  `frame → detect_gate → servo → ` feed `(fwd_speed, yaw_sp, z_sp)` into the **proven
  race_cruise inner loop** (desired accel → desired attitude → attitude error → body rates;
  altitude hold; thrust from vertical accel). Send `set_attitude_target` (rates+thrust).
- **Success = `active_gate_index` increments** (race progress from race-status — not gate-pose
  data, fair to use). Log success + `last_gate_time`; end the run (or continue to next gate later).
- Reuses `race_cruise`'s gains and caps; the only change is heading/altitude come from the gate
  bearing instead of a fixed line.

### 4. `aigp/viz.py` — Rerun telemetry helpers
- `rr.init("aigp-gate")` → connect to the **Rerun viewer on the Mac** at
  **`100.101.13.126:9876`** (gRPC; exact API/port confirmed against installed `rerun-sdk` at
  impl). Fallback: save an `.rrd` file, scp to Mac, open offline.
- World coords NED (set `ViewCoordinates` so down renders down).
- Logged per step:
  - **3D:** drone position trail (line strip) + attitude (`Transform3D`) + velocity arrow;
    a ray from the drone along the detected gate bearing.
  - **Images:** `cam/frame` (raw), `cam/mask` (binary), `cam/overlay` (box on chosen gate).
  - **Scalars:** `ex`, `ey`, gate `u/v`, apparent size, `fwd_speed`, thrust, tilt, yaw error,
    `active_gate_index`.
- Logging is best-effort and must never block/crash the control loop (wrap in try/except).

### 5. `aigp/hsv_tuner.py` — web HSV tuner (served from Windows, used in Mac browser)
- Flask app on the Windows machine, `0.0.0.0:8088`; open `http://<windows-tailscale-ip>:8088`
  on the Mac (Windows Tailscale IP = the sim host `100.120.233.90`).
- Runs its own `VisionIO`; streams **frame + live mask side-by-side** via MJPEG.
- Sliders: `h_lo1/h_hi1`, `h_lo2/h_hi2` (two red bands), `s_min`, `v_min`, `min_area_px`.
  Drag → mask updates live.
- "Save" writes `detect_params.json` (the same file `gate_detect.py` reads).

## Camera & coordinate notes
- Intrinsics: `fx=fy=320, cx=320, cy=180`, image 640×360 (`aigp/geometry.py::K`).
- **Camera looks along −body-x** (empirical: "camera-forward / racing" = −body-x, which
  race_cruise flies controllably). NOTE: `geometry.py::R_BODY_TO_CAM` assumes **+body-x** — treat
  it as unverified; the servo pins pixel→control **signs empirically** (bring-up step 2), not from
  that matrix. (Flag: the auto-label projection in geometry.py may have a sign bug, but COR-125
  no longer depends on it.)
- Forward-through-gate = travel along −body-x = race_cruise's `fwd` direction.

## Success criteria
1. Detector boxes the correct (nearest) gate on saved frames and live feed; rejects reflections.
2. Closed-loop run: **`active_gate_index` increments** (drone passes gate 0). Repeatable ≥2×.
3. Rerun shows the 3D trajectory + mask layers + plots for a run; HSV tuner adjusts the mask live
   and persists thresholds.

## Bring-up order (de-risks the unknowns)
1. **Detector offline** on captured frames (`/tmp/gate_frame_*.jpg`) — verify the right gate is
   boxed, reflections rejected. Tune via the web tuner on the live feed.
2. **Servo-sign check** — hover (race_cruise position hold), command a small yaw step, observe
   which way the gate's `u` moves in-image → lock `SIGN_X`; pitch/altitude nudge → `SIGN_Y`.
3. **Closed-loop fly-through** — start at spawn (gate 0 visible ahead), servo forward, confirm
   `active_gate_index` ticks. Tune `K_yaw, K_alt, fwd_speed`.

## Risks / mitigations
- **Track reflections** misdetected as gates → position/solidity filter + pick-largest; validate
  offline first.
- **Camera sign convention** wrong → pinned empirically in bring-up step 2 (don't trust geometry.py).
- **Weathervane at speed** (from control work) → keep `fwd_speed` modest, capped.
- **Frame rate/latency** lower than control rate → control loop always uses latest frame; servo
  gains tuned for the effective update rate; `gate lost` coast handles dropouts.
- **Rerun network** (Windows→Mac gRPC over Tailscale, port 9876 reachable) → `.rrd` file fallback.
- **No actuation before race-live** (DQ) → all flight via `fresh_start` gating, as in race_cruise.

## Dependencies
- `pip install flask rerun-sdk` into the `aigp` env (cv2, numpy already present).
- Mac Tailscale IP (Rerun viewer): **100.101.13.126**. Windows/sim host: **100.120.233.90**.

## Testing
- `gate_detect`: unit tests on saved frames (assert a box near the known gate center; assert
  reflection rejected). Param-driven so tests pin thresholds.
- `visual_servo`: unit tests on synthetic detections (centered → ~zero correction; off-center →
  correct sign/magnitude; `None` → hold).
- `fly_gate`: live integration — the gate-pass run (success = `active_gate_index` increments).
