# AI-GP Closed-Loop Lateral-Aero & Weathervane SysID — Design Spec

**Date:** 2026-06-02 · **Status:** Design (Phase 2) · **Extends:** `2026-06-02-aigp-aero-sysid-design.md`
**Repo:** algo_src, branch `aigp-gate-data-collection`, env `aigp` (Windows host).

## 1. Motivation
Phase 1 identified and validated **horizontal rotor drag `D_x≈0.52/s`** (matches published 0.49–0.54) but **lateral drag `D_y`, the weathervane moment, and rotational damping did not converge**. Three open-loop attempts (diagonal coast, powered weave, steady circle) failed for distinct reasons that share one root cause: **lateral-aero ID is gated on lateral-flight control, which is blocked by the very weathervane we're measuring** — the platform is too weathervane-unstable to fly the steady sideslip trajectories open-loop ID requires (commanded circles overshoot to 7 m/s, tilt spikes ~58°, drift 30 m). Literature (NeuroBEM, Faessler, Crazyflow, MRFT) resolves this with **closed-loop ID**: keep a stabiliser running and treat the **control effort as the measurement**.

## 2. Enabling telemetry finding
A live message scan shows the sim streams **`ACTUATOR_OUTPUT_STATUS` at ~94 Hz** (per-motor outputs), alongside HIGHRES_IMU (specific force + gyro), ODOMETRY / LOCAL_POSITION_NED (privileged world vel/attitude), ATTITUDE. **This is the unlock:** with the per-motor outputs we can reconstruct the **actual control torque `τ_motor` and total thrust `T`** from a motor+mixer model — no need to model the rate-PID or run a full closed-loop rollout to recover the control effort. The moment ID becomes a direct grey-box regression on `τ_aero = I·ω̇ + ω×(I·ω) − τ_motor`.

## 3. Goal / Deliverable
Identify, with held-out validation:
- **Lateral drag `D_y`** (and the **thrust-coupling** of drag: powered `D_x≈0.34` ≠ coast `0.52` ⇒ `D = D₀ + D_T·T`).
- **Weathervane moment** — the CP-migration term: sideslip body velocity → yaw/pitch moment (refine the `wv_*` functional form + magnitude; sign `wv_z>0` already confirmed).
- **Rotational damping `d`** (now identifiable from in-flight excitation, unlike the free tumble).
Output: extend `sim_aero.json` (+ a motor model block) and the report. Reuse the validated `D_x`.

## 4. Architecture — measured-control-allocation closed-loop ID

### 4.1 Principle
A basic stabiliser (the existing `race_cruise` ACRO rate loop) keeps the vehicle flying inside a protected envelope. We inject small, safe excitation, **measure the control effort directly from `ACTUATOR_OUTPUT_STATUS`**, and fit the aero terms as the residual between observed dynamics and the modeled rigid-body + measured-control-allocation prediction. The weathervane moment shows up as the moment the controller must counter to hold heading during sideslip.

### 4.2 Reference governor (envelope protection) — REQUIRED
A layer between excitation/setpoint and the controller that clamps **velocity-setpoint slew rate, max commanded tilt, max speed, and max body-rate command** to the controller's known stable envelope. The open-loop attempts blew straight through these (overshoot, 58° tilt, divergence); the governor makes excitation safe. Start conservative (e.g. tilt ≤ 20°, speed ≤ 4 m/s, bounded slew), widen as stability allows.

### 4.3 Motor + mixer model (new sub-model, calibrated)
Per rotor `i` from `ACTUATOR_OUTPUT_STATUS` output `uᵢ`: thrust `fᵢ = k_f·g(uᵢ)`, reaction torque `qᵢ = k_q·g(uᵢ)` (`g` linear or quadratic — pick by fit). With the quad geometry (chassis 280×280×160 mm → arm `L`):
- Total thrust `T = Σ fᵢ` (body −z); body control torque `τ_motor = mixer(f₁..f₄, q₁..q₄, L)` (roll/pitch from thrust differential × `L`, yaw from reaction-torque differential).
- **Calibration:** hover gives `Σfᵢ = m·g ⇒ k_f`; a known yaw input gives `k_q`; `L` from geometry. Calibrate on a quiet hover + small single-axis inputs before the main fit.

### 4.4 Signals (extend the logger)
Add to the per-sample row: the 4 `ACTUATOR_OUTPUT_STATUS` outputs, the commanded body-rate (`wcmd`) and commanded thrust (control effort). Keep IMU specific force + gyro, ODOMETRY vel/attitude. (Phase-1 maneuvers already log the rest.)

### 4.5 Targets (extend `aero_dataset`)
- **Aero moment** `τ_aero = I·ω̇ + ω×(I·ω) − τ_motor` (motor torque now MEASURED, not omitted) — clean in powered flight, no coast needed.
- **Aero force (horizontal)** `a_aero_{x,y} = f_body_{x,y}` (thrust is body −z ⇒ thrust-independent horizontally). Vertical still excluded (VRS).
- Apply **IMU lever-arm correction** `a_CG = a_IMU − ω×(ω×r) − ω̇×r` and **velocity↔IMU latency `τ_lag`** alignment (the negative-`D_y` artifact was a phase error).
- **Derivatives via cubic splines** (NeuroBEM practice) — less noisy `ω̇` than finite-diff + boxcar.

### 4.6 Excitation
- **2-1-1 doublets** in the lateral-velocity and yaw channels near a modest forward (fwd, toward-course) trim — excites sideslip transients without leaving the stable region. Preferred over large sideslip excursions for unstable platforms.
- Collect to coverage over (sideslip angle × speed × thrust); convergence rule from Phase 1. Crazyflow fits comparable models in **<4 min** of flight, so keep it lean.

### 4.7 Estimator
- **Primary (direct grey-box):** with `τ_motor` measured and the corrected targets, fit `D_y`, `D_T`, the weathervane `wv_*`, and damping `d` by least-squares / robust regression (reuse `aero_fit`), extended with the thrust-coupled drag column and the CP-weathervane terms.
- **Refinement / validation (rollout):** **Nelder-Mead** minimising orientation + body-rate + horizontal-specific-force error over short closed-loop rollouts (reuse `dyn_model`), to (a) jointly refine `r`, `τ_lag`, motor-model params against degeneracy and (b) validate that the identified model **reproduces the measured control effort** on held-out runs. Multi-start to avoid local minima.

### 4.8 Model terms (extend `aero_model`)
- Drag: `D = D₀ + D_T·T` (thrust-coupled rotor drag) + quadratic parasitic `C` (unchanged).
- Weathervane: CP-migration form for the yaw/pitch moment vs sideslip velocity (and possibly `T`); generalise the `wv_*` block.
- Damping `d` (per axis), now in-flight-identifiable.

## 5. Validation
- Held-out **runs** (70/20/10), full-envelope coverage.
- `D_y` vs literature (0.24–0.39); `D_x` / `D_T` consistency with Phase 1 (coast 0.52, powered 0.34).
- Weathervane **magnitude** (sign already confirmed), validated by the model **predicting the held-out control effort** (commanded moments).
- Short-horizon rollout: predicted vs measured orientation + velocity.
- **Physical cross-check:** the identified weathervane should reproduce the **speed-dependent tail-first instability** observed in COR-127 / the camera-forward work.

## 6. Risks / open questions
- `ACTUATOR_OUTPUT_STATUS` units (normalised vs PWM vs rad/s) — confirm in T0; the motor-model calibration absorbs the scale but the `g(u)` form (linear/quadratic) must be fit.
- Joint identifiability of `r`, `τ_lag`, motor-model, and aero — may need to fix some via separate calibration (T0/T4) to avoid degeneracy; the rollout multi-start guards the rest.
- Reference-governor tuning: too tight → no excitation; too loose → divergence. Start conservative.
- Moment-ID accuracy is bounded by `dyn_model`/mixer fidelity and the inertia estimate (`I_ratio≈3.7`); consider refining `I` here too.
- Sideslip bluff-body force may be non-diagonal/nonlinear at large β — keep excitation near trim (small β) where the linear model holds; let the residual MLP (temporal/TCN) mop up the rest.

## 7. Task outline (for the plan)
- **T0** Telemetry + units: confirm `ACTUATOR_OUTPUT_STATUS` layout/units; log the 4 outputs + commanded effort.
- **T1** Reference governor + 2-1-1 excitation driver (extend `aero_capture`).
- **T2** Motor + mixer model + calibration (hover + single-axis inputs).
- **T3** Extend `aero_dataset`: measured `τ_motor`, lever-arm + latency correction, spline derivatives.
- **T4** Extend `aero_model` + `aero_fit`: thrust-coupled drag, CP-weathervane, damping (direct fit).
- **T5** Nelder-Mead rollout refiner/validator on `dyn_model` (joint `r`/`τ_lag`/motor/aero; held-out control-effort prediction).
- **T6** Live collect-to-coverage (governor-protected 2-1-1 battery).
- **T7** Fit + validate + extend `sim_aero.json` + report.

> This campaign is dual-purpose: it identifies `D_y` + the weathervane, **and** it builds the measured-control-allocation + envelope-protection tooling the team needs to actually tame the tail-first instability.
