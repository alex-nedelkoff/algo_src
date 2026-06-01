# AI-GP Drone Dynamics sysID (bare-physics, lumped-ratio) — Design

**Date:** 2026-06-01
**Status:** Approved (brainstorming), pending spec review
**Branch:** `aigp-gate-data-collection` (continues COR-125; realizes COR-96 against the real sim)

## Goal

Identify the AI-GP drone's open-loop motion model directly from the simulator, in
**lumped-ratio** form, by commanding at the **motor level**
(`set_actuator_control_target`) — bypassing the sim's inner attitude/rate
controllers — and fitting the rigid-body + rotor equations of motion to the
measured response. Deliverable: a validated `sim_dynamics.json` of identified
coefficients usable for model-based control (MPC), trajectory optimization, and
sim-to-sim policy transfer.

This is the **bare-physics** sysID (the real "drone physics"), distinct from the
control-oriented attitude-setpoint model. It realizes COR-96's grey-box intent
against the *actual* sim (COR-96's MVP was sim-to-sim).

## Why lumped ratios

From flight input/output, absolute `mass`, `inertia`, and `k_thrust` are coupled
(scaling k_thrust and mass together is unobservable). The **identifiable**
quantities fully determine the motion and suffice for control:
- `c_T` — specific thrust (thrust/mass) per motor input
- `c_L, c_M, c_N` — roll/pitch/yaw torque-over-inertia coefficients
- `k_torque/k_thrust` ratio; motor **mixing map** (channel → axis/sign); arm geometry
- `tau_motor` — first-order motor lag
- translational **drag** coefficients

Absolute physical units are out of scope (would need an external anchor, e.g. a
spec mass); can be back-computed later if a reference value appears.

## Key facts that shape the approach (from prior sessions)

- **Actuation only after race-live** (`boot_ms >= start_ms`), else DQ + sim close.
  Drone is **held until armed**, released on arm, runs away without control.
- **Collective excitation is attitude-stable** (all motors equal → pure vertical
  motion, no tumble → long windows). **Differential excitation tumbles** → short
  bursts + reset between.
- **`HIGHRES_IMU` measures specific force** (thrust/mass) + body angular rates at
  ~114 Hz — read thrust/mass straight off the accelerometer, no velocity
  differentiation needed. This is the primary sensor.
- **ODOMETRY quaternion is `xyzw`** (already reordered to wxyz in `io_layer`);
  NED position/velocity are trustworthy. IMU accel/gyro sign+axis convention is
  **unknown** → calibrated empirically in the motor-calibration step.
- **Motor interface is untested**: `set_actuator_control_target` channel
  count/range, and motor→axis mapping, are unknown (the template's motor
  constants were placeholders). This is the gating prerequisite.

## Approach: static least-squares from short bursts

Open-loop bare dynamics are unstable, so we fit **instantaneously** (specific
force / angular acceleration vs motor input) over short windows rather than long
rollouts. Most fits are **linear** in the parameters → `numpy.linalg.lstsq` (no
scipy/torch dependency). `tau_motor` (nonlinear) via a small grid search wrapping
the linear fit.

## Architecture (in the `aigp/` package)

### 1. `commander.send_motor_command(u)`
Wrapper for `set_actuator_control_target` (group 0). `u` is the per-channel input
vector (length determined in calibration; template used 8). Sent only after
race-live. Clipped to the valid range.

### 2. `dyn_probe.py` — live excitation (no unit test; open-loop)
- **Motor calibration** (`calibrate_motors`): from held→armed→released, pulse one
  channel above baseline at a time, briefly, logging IMU specific-force + gyro.
  Determines: number of active channels, input→thrust direction, and the
  **mixing map** (which channel produces +roll/+pitch/+yaw, with signs) — plus the
  IMU accel/gyro sign+axis convention (cross-checked against world-frame motion).
- **Excitation campaign** (`run_campaign`): from fresh race-live + arm,
  - *thrust/translational*: all-equal collective at several levels (steps),
    attitude-stable, ~1 s each.
  - *torque*: per-axis differential doublets (small), short (~0.3 s) bursts,
    reset between.
  - *lag*: a sharp collective step, log the specific-force rise.
  Logs `(t, u[], HIGHRES_IMU acc+gyro, ODOMETRY pose+vel+omega)` to
  `sysid_dyn/<run>/log.jsonl`, tagged by segment. Safety: short windows, auto
  `SIM_RESET` between segments / on divergence.

### 3. `dyn_fit.py` — pure fitters (unit-tested)
- `fit_thrust(u_sum_or_sq, imu_specific_force)` → `c_T`, and selects input vs
  input² by comparing fit residuals.
- `fit_axis_torque(motor_mix, ang_accel)` → `c_L`/`c_M`/`c_N` per axis (linear LS).
- `fit_motor_lag(t, u_step, specific_force)` → `tau_motor` (grid search).
- `fit_drag(vel, residual_force)` → drag coeffs (linear LS on faster-motion data).
- `angular_accel(t, gyro)` helper (finite-diff with light smoothing).

### 4. `dyn_model.py` — lumped forward model (pure, unit-tested)
`step(state, u, dt)` applying the identified lumped EOM (specific thrust along
body-up, per-axis torque/inertia, first-order motor lag, drag, gravity). Mirrors
`sim/dynamics/numpy_quad.py` structure but parameterized by the lumped coeffs.
Used for validation rollouts.

### 5. `dyn_validate.py` — held-out validation + report
Splits logs into fit/validation; predicts specific-force + angular-accel from
`u` and reports RMSE per channel; checks coefficient signs are consistent and
positive where expected; writes `sysid_dyn/<run>/sim_dynamics.json` (the lumped
coeffs + mixing map + input-power choice + RMSEs).

## Data flow

held→armed→released → `dyn_probe` commands motor patterns → logs IMU/ODOMETRY →
`dyn_fit` (LS) → lumped coeffs → `dyn_model` rollout vs held-out → `dyn_validate`
→ `sim_dynamics.json`.

## Testing

**TDD, pure (synthetic data, no sim):**
- `dyn_model.step` conserves expected behavior (e.g. equal thrust → vertical accel;
  differential → angular accel of correct sign).
- `dyn_fit` recovers known coeffs from data generated by `dyn_model` (c_T, c_L/M/N,
  tau_motor, drag within tolerance).
- `angular_accel` finite-diff on a known signal.

**Live validation:** `dyn_probe` runs against the sim; `dyn_validate` produces a
`sim_dynamics.json` with physically-consistent, sign-consistent coefficients and
low held-out RMSE.

## Error handling

- Motor interface not honored / no response in calibration → stop with a clear
  message (the prerequisite failed); fall back to documenting findings.
- Divergence during a burst → log + `SIM_RESET` + next segment.
- Stale/missing telemetry → skip sample.
- No race-live within timeout → abort (countdown/DQ guard).

## Out of scope

- Absolute physical params (mass/inertia/k_thrust in SI) — needs an external anchor.
- Aerodynamic effects beyond simple linear/quadratic drag (blade flapping, ground
  effect).
- The control-oriented attitude-setpoint model (separate, already partially have it).
- Using the identified model in a live controller (downstream follow-up).

## Open items (resolved in implementation)

1. Exact `set_actuator_control_target` semantics (channels, range, mapping) —
   determined by `calibrate_motors` (Task 0); gates the rest.
2. Thrust ∝ input vs input² — selected by residual comparison in `fit_thrust`.
3. `tau_motor` observability at ~114 Hz (~2 samples for 0.02 s) — may only bound
   it; report a confidence/range.
4. IMU accel/gyro sign+axis convention — calibrated against world-frame motion in
   `calibrate_motors`.
