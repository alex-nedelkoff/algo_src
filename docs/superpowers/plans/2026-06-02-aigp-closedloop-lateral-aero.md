# AI-GP Closed-Loop Lateral-Aero & Weathervane SysID — Implementation Plan

> **For agentic workers:** use **superpowers:subagent-driven-development** task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Identify lateral drag `D_y`, the thrust-coupling of drag (`D=D₀+D_T·T`), the weathervane moment (CP-migration), and rotational damping `d` — via **closed-loop ID** using the measured per-motor control allocation — validated on held-out flight. Extend `sim_aero.json` + report. Reuse the validated Phase-1 `D_x≈0.52`.

**Spec:** `docs/superpowers/specs/2026-06-02-aigp-closedloop-lateral-aero-design.md`.
**Builds on:** Phase-1 pipeline (`aigp/aero_model|dataset|fit|validate.py`, 15 tests), `aigp/dyn_model.py`, `aigp/io_layer.py`, `aigp/state.py`, `race_cruise.py`, `aero_capture.py`, `aero_run.py`.

## Working environment (read first)
- Code + tests run in the **`aigp` conda env on the Windows host** (`ssh alexj@100.120.233.90`), worktree `C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client`, branch `aigp-gate-data-collection`.
- Edit on Mac → `scp` to the worktree → run over ssh. Do **NOT** pipe (`| tail`) *inside* the ssh command (Windows cmd breaks); pipe on the Mac side. Files are **CRLF** — read exact text before Edit, or rewrite via scp.
- Run: `ssh alexj@100.120.233.90 "cd <worktree> && conda run -n aigp <cmd> 2>&1" | tail -30`.
- Tests in `tests/test_aigp/`; commit on the branch; end messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Live runs need the user-driven sim + `fresh_start()`; gate on race-live.

## Conventions (additions to Phase 1)
- Body **FRD**; thrust along body **−z**; IMU = specific force (gravity excluded) + gyro.
- **`ACTUATOR_OUTPUT_STATUS`** (~94 Hz): per-motor outputs `u₁..u₄`. Motor model → per-rotor thrust `fᵢ` and reaction torque `qᵢ`; mixer → total thrust `T=Σfᵢ` and control torque `τ_motor`.
- **Aero moment target** `τ_aero = I·ω̇ + ω×(I·ω) − τ_motor` (motor torque MEASURED — powered flight is clean).
- **Aero horizontal force** `a_aero_{x,y} = f_body_{x,y}` (thrust-independent). Vertical excluded (VRS).
- **Lever-arm** `a_CG = a_IMU − ω×(ω×r) − ω̇×r`; **latency** `τ_lag` aligns velocity↔IMU; **spline** derivatives for `ω̇`.

## New / extended files
- `aigp/io_layer.py`, `aigp/state.py` — parse + store `ACTUATOR_OUTPUT_STATUS` (T0).
- `aero_capture.py` — reference governor + 2-1-1 excitation + log actuators/effort (T1).
- `aigp/motor_model.py` (new) + `tests/test_aigp/test_motor_model.py` — motor+mixer + calibration (T2).
- `aigp/aero_dataset.py` — `build_targets_cl(...)` with measured `τ_motor` + corrections (T3).
- `aigp/aero_model.py`, `aigp/aero_fit.py` — thrust-coupled drag col + CP-weathervane + damping fit (T4).
- `aigp/aero_rollout.py` (new) + `tests/test_aigp/test_aero_rollout.py` — Nelder-Mead rollout refiner/validator (T5).
- `aero_run.py` — `closedloop` fit/validate mode + report (T7).

---

## Task 0: Telemetry — `ACTUATOR_OUTPUT_STATUS` units + logging (LIVE)

**Files:** `aigp/io_layer.py`, `aigp/state.py`

- [ ] **Step 1 (probe, sim live):** Receive `ACTUATOR_OUTPUT_STATUS` during a quiet hover and print `active`, `actuator[0:8]`, and their range over a few seconds. Decide: how many motors (4?), and units — normalized `[0,1]`, PWM `~1000–2000`, or rad/s. Record the finding inline as a comment.
- [ ] **Step 2:** In `io_layer.py`, add a handler: on `ACTUATOR_OUTPUT_STATUS`, `store.set_actuators(np.array(m.actuator[:4]), m.time_usec)`. In `state.py`, add `_act`, `set_actuators`, `get_actuators()` (mirror `set_imu`/`get_imu`).
- [ ] **Step 3:** Verify: a short script reads `s.get_actuators()` live and prints the 4 outputs at hover vs a small roll input (they should differ across motors). Confirms wiring + that the differential is observable.
- [ ] **Step 4: Commit** `feat(aero-cl): parse + store ACTUATOR_OUTPUT_STATUS per-motor outputs`.

---

## Task 1: Reference governor + 2-1-1 excitation driver (extend `aero_capture.py`) (code + LIVE smoke)

**Files:** `aero_capture.py`

- [ ] **Step 1:** Add a `ReferenceGovernor` that clamps the commanded setpoint to the stable envelope: velocity-setpoint **slew-rate limit**, **max speed**, **max commanded tilt** (cap the horizontal accel like the existing `TILT_MAX_ACC` but tighter), **max body-rate command**. Pure function `govern(vsp_prev, vsp_raw, dt) -> vsp` + a tilt/rate clamp inside `control()`.
- [ ] **Step 2:** Add maneuver `doublet211 <axis lat|yaw> <trim_speed> <amp> [dwell]`: hold a modest **fwd** (toward-course) trim at `trim_speed`, then inject a **2-1-1** pattern (`+2τ, −1τ, +1τ` … standard 2-1-1) on the lateral-velocity (or yaw-rate) channel, governed. Log every sample including the 4 actuators + `wcmd` + `thr` + `coast=False`. Abort on tilt.
- [ ] **Step 3:** Extend `Logger.COLS` with `u0..u3` (actuators), `wcx,wcy,wcz` (commanded body-rate), `thr_cmd`. Update `Logger.log(...)` to take + store them.
- [ ] **Step 4 (sim live):** Smoke `doublet211 lat 2.5 1.0`: confirm it stays bounded (governor holds tilt/speed in envelope — contrast the Phase-1 circle that hit 58°), a parquet appears with the new columns, and the actuators show a clear lateral-doublet signature.
- [ ] **Step 5: Commit** `feat(aero-cl): reference governor + 2-1-1 doublet excitation + actuator/effort logging`.

---

## Task 2: Motor + mixer model + calibration (TDD + LIVE calib)

**Files:** `aigp/motor_model.py`; `tests/test_aigp/test_motor_model.py`

- [ ] **Step 1 (failing tests):** `motor_outputs_to_wrench(u, params) -> (T, tau_motor)` where `u=(4,)`, `params={k_f,k_q,L,g_form}`. Tests on a synthetic symmetric quad: equal outputs → `T=4·k_f·g(u)`, `tau_motor≈0`; a roll-differential pattern → nonzero `tau_x`, others ≈0; yaw-reaction pattern → nonzero `tau_z`. `g_form ∈ {linear, quadratic}`.
- [ ] **Step 2:** Implement `motor_model.py`: per-rotor `fᵢ=k_f·g(uᵢ)`, `qᵢ=k_q·g(uᵢ)`; quad-X mixer geometry (signs from rotor layout, arm `L`) → `T=Σfᵢ`, `tau=[L·Σ(±fᵢ), L·Σ(±fᵢ), Σ(±qᵢ)]`. Add `calibrate(df_hover, df_inputs, mass, I_ratio)` that solves `k_f` from hover (`Σf=m·g`), `k_q` from a known yaw input, `L` from geometry (chassis 280×280 → motor arm). Pure numpy. Run → green.
- [ ] **Step 3 (sim live, calibration):** capture a quiet hover + a small single-axis roll and yaw input (reuse T1's driver at tiny amp), run `calibrate`, print `k_f,k_q,L` and the residual (predicted `T` vs `m·g` at hover). Save to `aero_data/motor_model.json`.
- [ ] **Step 4: Commit** `feat(aero-cl): motor+mixer model (outputs->thrust/torque) + calibration`.

---

## Task 3: Dataset extension — measured `τ_motor`, lever-arm/latency, spline derivatives (TDD)

**Files:** `aigp/aero_dataset.py`; `tests/test_aigp/test_aero_dataset.py` (add cases)

- [ ] **Step 1 (failing tests):** `spline_deriv(t, w) -> ẇ` (cubic-spline differentiation; test on a known sinusoid vs analytic derivative, tighter than the boxcar). `leverarm_correct(a_imu, omega, omega_dot, r) -> a_cg` (test: pure spin about z with offset r gives the expected centripetal removal). `build_targets_cl(df, I_ratio, motor_model, r, lag) -> dict` adding `tau_aero = I·ω̇ + ω×(I·ω) − τ_motor` (τ_motor from `motor_outputs_to_wrench` on the logged `u`), `a_aero` lever-arm-corrected + latency-shifted, plus `v_body`, `omega`, `T`. Test on a synthetic powered hover: `tau_aero≈0`, `T≈m·g`.
- [ ] **Step 2:** Implement. Reuse Phase-1 `world_to_body_vel`/`quat_to_R`; replace finite-diff with `spline_deriv`; apply `leverarm_correct`; shift velocity vs IMU by `lag` samples. Keep Phase-1 `build_targets` intact (don't break the 15 tests).
- [ ] **Step 3:** Run the full `test_aero_dataset.py` → all green (old + new).
- [ ] **Step 4: Commit** `feat(aero-cl): closed-loop targets (measured tau_motor, lever-arm, latency, spline deriv)`.

---

## Task 4: Model + fit extension — thrust-coupled drag, CP-weathervane, damping (TDD)

**Files:** `aigp/aero_model.py`, `aigp/aero_fit.py`; tests in `test_aero_model.py` / `test_aero_fit.py`

- [ ] **Step 1 (failing tests):** extend `force_features` with a **thrust-coupled** drag column so `a_drag = −(D₀+D_T·T)·v` is linear in `[D₀, D_T]` (pass `T` in). Extend `moment_features` with the **CP-weathervane** form (yaw/pitch moment vs sideslip body velocity, optionally ×`T`) + keep damping `d`. Synthetic round-trip tests recover known `D₀,D_T, wv*, d` to 1e-6 (extend `_make_synth`).
- [ ] **Step 2:** Implement the new columns (keep `FORCE_COLS`/`MOMENT_COLS` ordering stable; append new names). `fit_parametric` already stacks features generically — verify it fits the extended θ; add a coast/effort weighting hook if needed.
- [ ] **Step 3:** Run `test_aero_model.py` + `test_aero_fit.py` → green.
- [ ] **Step 4: Commit** `feat(aero-cl): thrust-coupled drag + CP-weathervane features + extended fit`.

---

## Task 5: Nelder-Mead rollout refiner / validator (TDD)

**Files:** `aigp/aero_rollout.py`; `tests/test_aigp/test_aero_rollout.py`

- [ ] **Step 1 (failing test):** `rollout_cost(params, run, motor_model, dyn) -> float` simulates the run forward (reuse `dyn_model.py` rigid-body + the candidate aero from `aero_model.predict` + measured `τ_motor`/`T` as the control input) from the recorded initial state, and returns a weighted trajectory error (orientation + body-rate + horizontal specific-force + velocity). `refine(params0, runs, ...) -> params` wraps `scipy.optimize.minimize(method="Nelder-Mead")`, multi-start. Test: on synthetic data generated from known params + a known lever-arm/lag, `refine` recovers them within tolerance and drives the cost down.
- [ ] **Step 2:** Implement. Parameter vector = `{D₀,D_T, wv*, d, r(3), lag}` (motor-model fixed from T2, or included if identifiable). Short-horizon rollouts (windowed) to bound integration error. scipy is installed.
- [ ] **Step 3:** Run → green.
- [ ] **Step 4: Commit** `feat(aero-cl): Nelder-Mead rollout refiner/validator on dyn_model`.

---

## Task 6: Live collect-to-coverage (LIVE)

**Files:** none (data → `aero_data/`)

- [ ] **Step 1 (sim live):** run the governed **2-1-1 battery**: lateral + yaw doublets at a few trim speeds (within envelope), toward the course. Track coverage over (sideslip angle × speed × thrust). Keep collecting until the Phase-1 convergence rule trips. Target <4 min total flight (Crazyflow benchmark). Hold out ≥1 run per excitation type.
- [ ] **Step 2:** Quick sanity per run: load → `build_targets_cl` → print `τ_aero` and horizontal `a_aero` ranges + that the actuators show the doublet. Confirms the capture→targets path.

---

## Task 7: Fit + validate + extend `sim_aero.json` + report (LIVE + offline)

**Files:** `aero_run.py`, `docs/aero_sysid_report.md`, `docs/sim_aero.json`

- [ ] **Step 1:** add `aero_run.py closedloop` mode: load the 2-1-1 runs (minus held-out) → `build_targets_cl` → `fit_parametric` (extended) for `D₀,D_T, wv*, d` → `aero_rollout.refine` to clean `r,lag` (+ validate) → optional temporal residual.
- [ ] **Step 2 (validate):** held-out single-step (force + moment RMSE/R²) + a **control-effort prediction** check (does the identified model reproduce the held-out commanded moments?) + short-horizon rollout. Compare `D_y` to literature (0.24–0.39), `D_x/D_T` to Phase-1 (coast 0.52 / powered 0.34).
- [ ] **Step 3:** **Physical cross-check:** confirm the identified weathervane reproduces the speed-dependent tail-first instability (COR-127).
- [ ] **Step 4:** extend `sim_aero.json` (new θ + motor model + `r`/`lag`); regenerate `docs/aero_sysid_report.md` (add the lateral/moment section with the held-out + control-effort metrics); copy `sim_aero.json` → `docs/`.
- [ ] **Step 5: Commit** `feat(aero-cl): closed-loop fit/validate + identified D_y/weathervane/damping + report`.

---

## Self-review notes
- **Offline TDD:** T2 (motor model), T3 (targets), T4 (features/fit), T5 (rollout) are pure + unit-tested with synthetic data; **live:** T0, T1 smoke, T2 calibration, T6, T7.
- **Reuse:** `quat_to_R`, `dyn_model`, `aero_fit.fit_parametric`, `aero_validate`, `race_cruise.fresh_start/control` — don't reinvent. Keep Phase-1 `build_targets` + the 15 tests intact.
- **Identifiability guard:** if joint `{r, lag, motor-model, aero}` is degenerate, fix `r`/`lag` from dedicated calibration (T0/T2) and only refine aero in T5; multi-start Nelder-Mead.
- **Dual-purpose:** the governor + measured-control-allocation tooling here is the same machinery needed to tame the tail-first instability.

---
### T0 — DONE (2026-06-02), commit `6d98e35`
- `ACTUATOR_OUTPUT_STATUS`: `active=15` (4 motors), `actuator[]` len 32, **motors on channels 0–3, normalized [0,1]** (idle 0.05).
- Live verify (hover/roll/yaw): hover `u≈[0.264,0.264,0.269,0.269]` (~hover_thrust); roll input → `[0.08,0.31,0.08,0.11]`; yaw input → `[0.44,0.60,0.46,0.46]`. **Differential clearly observable** ⇒ `τ_motor` reconstruction feasible.
- Motor-layout hint for T2 mixer: a +roll command raised **motor 1** and lowered **0 & 2** (3 mid); yaw raised **motor 1** most → use these signatures to pin the quad-X mixer sign pattern during T2 calibration.
- Wired `io_layer.py` (`ACTUATOR_OUTPUT_STATUS` handler) + `state.py` (`set_actuators`/`get_actuators`). Next: **T1** (reference governor + 2-1-1 excitation + actuator/effort logging).

---
### T1 — DONE (2026-06-02), commit `c8c3f8d`
- `ReferenceGovernor` (velocity-setpoint speed cap + slew-rate, ramp from rest) + a **forward-accel cap** added to `control(al_dir,al_max)` — the latter is the race_cruise anti-runaway fix and was **essential**: without it the fwd (tail-first) trim ran away to 6.6 m/s and tumbled (tilt 99°). With it, the smoke held **speed ~2–3 m/s, tilt ≤27°, no divergence**.
- `doublet211 <lat|yaw> <trim_speed> <amp> [dwell]`: governed fwd trim, 2-1-1 multistep (signs +,−,+, durations 2d,d,d) on the lateral-velocity or yaw-rate channel; doublet added ON TOP of the governed trim (sharp, not slewed).
- `Logger` now logs **24 cols**: + `u0..u3` (motor outputs from `get_actuators()`), `wcx,wcy,wcz` (commanded body-rate effort), `thr_cmd`. All existing maneuvers keep working (new cols default to actuators+zeros).
- Verified: actuators vary during the doublet; yaw effort `wcz` ±0.28 (controller countering sideslip = the weathervane signal). **But lateral excitation was modest (`v_body_y`~1.2 m/s)** with dwell=0.5 — **T6 must tune dwell/amp/trim up** for stronger sideslip + weathervane signal.
- Next: **T2** motor+mixer model + calibration (use the T0 motor-layout hint: +roll raised motor 1, lowered 0&2).

---
### T2 — DONE (2026-06-02), commits `…`(model) + calibration
- `aigp/motor_model.py`: `motor_outputs_to_wrench(u, params) -> (T, tau_motor)`; per-rotor `f=k_f*g(u)`, `q=k_q*g(u)` (g quadratic); mixer `T=Σf`, `tau=[L·sx@f, L·sy@f, sz@q]`. 5 unit tests pass.
- **Calibration (live):** de-meaned single-axis motor differentials (subtract the collective/altitude-loop component) → `sx=[-1,1,1,-1]` (roll), `sy=[-1,-1,1,1]` (pitch), `sz=[-1,1,-1,1]` (yaw) = **−(sx·sy) diagonal** — a consistent quad-X. **Validated: hover torque τ≈[0,0.02,0]** (alternating signs ⇒ pure-thrust hover). `k_f` from hover (`T=9.81`); `k_q` seeded (refine in T5). Saved `docs/motor_model.json`; baked into `DEFAULT_GEOM`.
- Gotcha: a sustained roll/pitch/yaw rate command also shifts the collective (drone tilts → altitude loop raises all motors); must **de-mean** the per-motor delta to recover the differential sign pattern.
- Next: **T3** (offline TDD) — `build_targets_cl`: aero moment `τ_aero = I·ω̇ + ω×(Iω) − τ_motor` (τ_motor from the calibrated model), lever-arm + latency correction, spline derivatives.

---
### T3 / T4 / T5 — DONE (2026-06-02), offline TDD, full aero suite 122 passed
- **T3** `074609f` — `build_targets_cl`: `tau_aero = kappa*(I_ratio*ω̇ + ω×(I_ratio*ω)) − tau_motor` (motor torque now MEASURED from the calibrated model, not omitted); `spline_deriv` (cubic-spline ω̇, cleaner than finite-diff); `leverarm_correct` (`a_cg = a_imu − ω×(ω×r) − ω̇×r`); latency `lag`. Phase-1 `build_targets` + tests untouched.
- **T4** `44ded0d` — `force_features_cl(v,T)`: thrust-coupled drag `−(D0 + D_T·T)·v − C·v|v|` (FORCE_COLS_CL, 9). `moment_features_cl(v,ω)`: damping + weathervane, **yaw from lateral v_y** (`phi[2,5]=v_y`, the sideslip→yaw the 2-1-1 excites). `fit_parametric_cl`.
- **T5** `d3d4380` + review fix `b11f61c` — `aero_rollout.py`: `refine_nuisance` (Nelder-Mead over lever-arm r + latency lag, inner loop = build_targets_cl→fit_parametric_cl, objective = fit-residual; precompute cache for ~60× speed). Recovers r≈[0.05,−0.03] and lag in tests.
- ⚠️ **Lag-convention bug caught + fixed in review:** the optimizer originally returned lag with the OPPOSITE sign to `build_targets_cl` → would have double-shifted in T7. Now UNIFIED: `lag>0` = velocity is `lag` samples later than force; `refine_nuisance`'s returned `lag` feeds directly into `build_targets_cl`. (Sign caveat noted in the rollout docstring.)
- Review also spot-checked T3/T4 formulas (tau_aero elementwise I_ratio, cross order, lever-arm signs, feature columns) — all correct.
- Next: **T6** (live, sim up) — governed 2-1-1 battery (lat+yaw, a few trim speeds; boost dwell/amp for stronger sideslip per the T1 note). Then **T7** fit/validate → extend `sim_aero.json` + report.
