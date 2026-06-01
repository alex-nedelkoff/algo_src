# AI-GP Sim Response Characterization + Body-Rate Controller — Design

**Date:** 2026-06-01
**Status:** Approved (brainstorming), pending spec review
**Branch:** `aigp-gate-data-collection` (continues COR-125 work; relates to COR-96 sysID)

## Goal

The AI-GP sim's outer (position/velocity) control loop is unusable as-is — commanding `SET_POSITION_TARGET_LOCAL_NED` velocity setpoints produced in-place oscillation with the camera panning wildly (yaw instability). Replace it with our own control: characterize the sim's response to thrust + body-rate commands (gently, near hover), then build a stable **body-rate cascade controller** that keeps the sim's inner rate loop and runs our own position/velocity/attitude loops on top. Success = stable hover hold → stable orbit around a gate → resumed auto-labeled data capture (COR-125 Task 13).

## Context (confirmed this session)

- Sim accepts `SET_ATTITUDE_TARGET` with **either** a target quaternion **or** body rates (type-mask selects). We use **body-rate mode** (`ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE` set), thrust normalized 0..1.
- Telemetry confirmed: `ODOMETRY` (pos/vel/quat/omega, 73 Hz), `ATTITUDE` (114 Hz), `HIGHRES_IMU` (114 Hz).
- `SIM_RESET` (cmd 31000) restarts the race and re-spawns the drone at the start — used to recover between probe segments.
- Gate acquisition works (6 gates; gate 0 at NED ≈ (-23.3, -0.4, -0.05), 2.72 m square).
- Our grey-box model (`sim/dynamics/`) is CrazyFlie-scale (mass 0.027 kg) — wrong for the racing drone, so the controller is built **mass-agnostic** from the empirical thrust→accel response rather than from model params. (Full grey-box parameter fit = COR-96 follow-up, out of scope here; the probe logs are reusable for it.)
- Command-side reference: `control/base_policies/pd_waypoint_tracker.py` outputs `[thrust_N, ωx, ωy, ωz]` — the same cascade we adapt here.

## Architecture

All in the existing `aigp/` package (numpy/pymavlink/opencv; reuses `geometry.py`, `io_layer.py`, `state.py`). Components, each independently testable:

### 1. Commander extension (`commander.py`)
- Add `send_attitude_target(body_rates, thrust_norm)` — `set_attitude_target` in **rate mode**: `type_mask = ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE`, dummy quaternion `[1,0,0,0]`, `body_roll/pitch/yaw_rate` = `body_rates`, `thrust` = `clip(thrust_norm, 0, 1)`.

### 2. Excitation probe (`probe_response.py`) — open-loop, no controller needed
From the spawned/at-rest start state, inject scripted **gentle** open-loop excitations and log everything; recover with `SIM_RESET` between segments.
- **Thrust sweep:** hold body-rates = 0, ramp `thrust_norm` slowly across a band (e.g. 0.3–0.7); the hover point is where vertical acceleration crosses zero.
- **Body-rate doublets:** hold `thrust_norm ≈ hover`, apply a short (~0.4 s) ±rate doublet on one axis at a time (roll, pitch, yaw), small amplitude (≤ ~1.5 rad/s).
- **Safety bounds (gentle, near hover):** tilt magnitude < 30° abort, |pos drift| limit, altitude floor; each excitation burst short; auto `SIM_RESET` between segments and on any bound breach.
- Logs `(t, cmd_thrust, cmd_rates, ODOMETRY pos/vel/quat/omega, HIGHRES_IMU acc)` to `sysid/<run>/log.jsonl`.

### 3. Analysis (`analyze_response.py`) — offline, pure
Fit from the logs and emit `sysid/<run>/sim_response.json`:
- **`hover_thrust`** — normalized thrust where vertical accel ≈ 0 (linear fit of accel_z vs thrust_norm, solve for zero).
- **`thrust_accel_gain` (k_a)** — slope d(accel)/d(thrust_norm) near hover [m/s² per unit].
- **Per-axis rate tracking** — commanded vs measured body rate during doublets: steady-state **gain** and first-order **time constant**; flags whether the inner rate loop is clean enough to build on.

### 4. Body-rate controller (`attitude_control.py`) — mass-agnostic cascade
Given drone state (pos, vel, quat, omega), a position/velocity setpoint, and yaw setpoint:
1. Position error → desired velocity → **desired acceleration** `a_des` (PD), add gravity comp.
2. Desired thrust direction = unit(`a_des` + g·ẑ); **collective thrust magnitude** along body-z → `thrust_norm = hover_thrust + (a_along_z) / k_a` (mass-agnostic, uses probe constants).
3. Desired attitude from desired thrust direction + commanded yaw → rotation error vs current attitude → **body-rate command** `ω_cmd = kp_att · att_err_body − kd · omega` (the PD-tracker form).
4. Yaw handled as a **rate** term (rate-limited) — kills the wild panning seen with yaw-angle setpoints.
5. Output `(ω_cmd, thrust_norm)` → `send_attitude_target`.

### 5. Verification harness (`fly_check.py`)
Three staged live checks (each prints pass/fail metrics):
- **Stage 1 — hover hold:** setpoint = capture the spawn position; confirm it holds within a small box (e.g. < 1 m) for 15 s with no oscillation/divergence and bounded yaw rate.
- **Stage 2 — orbit:** feed `OrbitPattern` geometry (existing) into the new controller; confirm a smooth circle around gate 0, no panning.
- **Stage 3 — resume capture:** re-run COR-125 Task 13 capture using the new controller; confirm a labeled dataset is produced.

## Frame handling

Reuse `geometry.py` (NED). New transform: desired thrust direction → desired rotation matrix (thrust along body-z, heading from yaw setpoint), then rotation error → body-frame angular error. Pure, unit-tested. All rates are body-frame; thrust collective along body-z.

## Testing

**Pure/offline TDD (no sim):**
- desired-accel → `thrust_norm` mapping (uses hover_thrust + k_a; hover input → hover thrust; +accel → more thrust).
- desired thrust-direction + yaw → desired attitude → body-rate command (zero error → zero rate; known tilt error → correct-sign rate).
- analysis fits on **synthetic logs** with known constants (recover hover_thrust, k_a, rate gain/τ within tolerance).

**Live validation:** the 3-stage `fly_check.py`. Probe is validated by producing a sane `sim_response.json` (hover_thrust in (0,1), k_a > 0, rate gain ≈ O(1)).

## Error handling

- Probe: bound breaches → log + `SIM_RESET` + continue to next segment; never let the drone run away.
- Controller: clamp `thrust_norm` to [0,1] and `ω_cmd` to a max; if no fresh ODOMETRY, hold last safe command.
- No heartbeat → clear error/exit.

## Out of scope

- Full grey-box parameter fit (mass/inertia/coeffs) — COR-96 follow-up; probe logs are reusable.
- Attitude-setpoint (quaternion) control mode — rejected in favor of body-rate.
- MPC / RL sim-to-sim transfer; motor-level control; aggressive-envelope characterization (gentle near-hover only).

## Open items (resolved during execution)

1. Whether the sim's inner rate loop tracks cleanly enough for body-rate control — answered by the probe's rate-tracking fit; if poor, revisit.
2. Exact gentle bounds (thrust band, rate amplitude, tilt/drift limits) — start conservative, widen only if the fit is under-excited.
