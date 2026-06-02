# AI-GP aerodynamic sysID — design (2026-06-02)

Linear: follow-up to **COR-96** (grey-box dynamics sysID) and the **COR-26** fidelity roadmap
(COR-28 v² drag, COR-30 advance-ratio thrust, COR-32 residual-learning). Repo: `algo_src`
branch `aigp-gate-data-collection`, package `aigp/`, run on the Windows sim laptop in the `aigp`
conda env. Literature: NotebookLM notebook **"AI-GP Drone Aero SysID — Modeling"**; primer in the
vault `01 Projects/CorvidX/Drone flight physics & aero sysID primer.md`. Companion control work:
[[AI-GP drone dynamics sysID]] (COR-96 lumped params), `race_cruise.py` (current controller).

## Goal & scope
Identify the AI-GP sim drone's **aerodynamics** — the velocity/orientation-dependent forces and
**moments** the current lumped model ignores — as a **parametric model + a learned residual**,
**validated against held-out flight data**. This is the keystone for a faithful replica sim and
for transferring/retraining the TRPY policy at racing speed (course peaks ~120 km/h ≈ 33 m/s,
vs. our current ~3 m/s controlled flight).

**Motivation (why now):** the dominant unmodeled forces — quadratic body drag and a
velocity-dependent yaw/pitch moment (the "weathervane") — scale with **v²**, so going 3 → 33 m/s
multiplies them ~**100×**. They are exactly what destabilized camera-forward flight and what a
racing controller/policy must account for.

**In scope:** identify + fit the aero **force and moment** terms (+ residual), validate on held-out
AI-GP flight (low single-step accel error, low short-horizon rollout RMSE, residual variance share
reported). Output = the coefficients (`sim_aero.json`) + a validation report.
**Out of scope (separate follow-ons):** folding the model into a runnable replica sim, setting
domain-randomization ranges, RL/control training, online/adaptive ID during deployment.

## What we already have (reuse, don't rebuild)
- **Lumped rigid-body params** (COR-96): thrust/mass `c_T≈9.95`, per-axis torque/inertia mixing
  `c_L/c_M/c_N`, inertia ratio `Ixx/Iyy≈3.7`, hover ≈0.20 motor / 0.23 normalized.
- **Differentiable EOM** `torch_quad` (COR-96) + the half-wired **drag fitter** in `aigp/dyn_fit.py`
  + excitation scaffolding in `aigp/dyn_probe.py`.
- **IO** (`aigp/io_layer.py`): MAVLink + HIGHRES_IMU (specific force + gyro, ~114 Hz), ODOMETRY /
  LOCAL_POSITION_NED (privileged world velocity/attitude). NED world, FRD body.
- **Controller** `race_cruise.py` for stable powered flight; the **motor interface**
  (`set_actuator_control_target`) for open-loop bursts.

## Model (parametric terms + residual)
Body frame; fit coefficients per axis. Literature-grounded (NeuroBEM, Faessler rotor-drag, the
racing-quad sysID + wind-tunnel sources).

**Forces**
- Linear rotor/induced drag: `F = −D·v` (D per-axis). First-order drag effect (Faessler); helps tracking.
- Quadratic parasitic body drag: `F = −C·v·|v|` (C per-axis). **Dominant force at speed** (NeuroBEM).
- Thrust-vs-speed correction (**required at our speeds**): start with the racing shortcut
  `T_corr = T_cmd + k_h·v_h²` (linear in `k_h`); escalate to BEMT advance-ratio/inflow only if the
  thrust residual stays large.

**Moments**
- Weathervane / CP-migration + hub moment: multinomial in advance ratio
  `C_M = k₁₁·μ + k₁₂·λ_c·μ` (+ hub term `∝ e`). **The instability term.**
- Aerodynamic rotational damping: `τ = −d·ω` (d per-axis).

**Residual:** small differentiable MLP `r(v_body, ω, R, u) → (ΔF, Δτ)`; its variance share is the
"how fully known" metric. Differentiable so it drops into a torch/JAX replica + RL later.

## Data-collection protocol
**Per-sample row (logged at IMU rate ~114 Hz):** `t`; `v_world`,`quat` (→ `v_body`); `ω` (gyro);
`f_body` (specific force = (thrust+aero)/m); `u` (motor/TRPY command); `coast_flag`.
**Targets (preprocess):** `aero_force = f_body − thrust_model(u)` (clean when coasting, thrust≈0);
`aero_moment = α − motor_torque_model(u) + ω×(I_ratio·ω)`, `α` = low-pass-filtered `dω/dt`.
No wind in sim → **airspeed = ground velocity**; air density folds into the drag coefficients.

**Maneuver battery (observability-driven):**
| # | Maneuver | Mode | Identifies |
|---|---|---|---|
| 1 | hover / near-hover | powered | baseline (aero≈0), hover thrust |
| 2 | per-axis speed sweeps (±x,±y,±z) | powered | linear + quadratic drag, per axis |
| 3 | circles + lemniscates | powered | drag coeffs + cross-coupling (max-excite body x/y velocity) |
| 4 | roll/pitch/yaw rate doublets at several speeds | powered | aero moments (damping + velocity term) + torque mixing |
| 5 | collective thrust ramps/chirps at several speeds | powered | thrust-vs-speed (advance-ratio/inflow droop) |
| 6 | zero-thrust coast bursts (incl. tail-first) | coast | clean high-speed aero force + the weathervane moment |

**Coast-probe (the key technique):** (1) build speed in a chosen direction/orientation (controller
or short open-loop motor burst); (2) cut thrust to ~0 → `f_body = aero directly` (no thrust-model
error); (3) capture the ~200–500 ms window — the aero force decelerates it and any aero moment
shows as a growing `ω` before tumble; (4) repeat across orientation (nose / sideways / **tail-first**)
and entry speed. **Tail-first coast = cleanest weathervane-moment measurement** — the unstable
regime the controller can't hold.

**Budget = collect to coverage + convergence (no fixed cap):**
- Track a **coverage map** over body-velocity-direction × speed-bin × orientation × rate-excitation;
  weight collection toward the **sparse/hard corners** (high-speed, off-axis, tail-first).
- **Convergence stop (per term):** fitted coeffs stable (Δ within noise) ∧ held-out single-step RMSE
  plateaus ∧ residual variance stops shrinking.
- Lopsided run count: powered sweeps/circles/lemniscates give long continuous streams (few runs cover
  a lot); **coasts are ~200–500 ms each and target the hardest regime → expect many dozens**, varying
  orientation × entry speed. Over-sample coasts; let the convergence rule call "done."
- **Validation split by run** (hold out whole runs: a lemniscate at an unseen speed + a set of coasts)
  → no within-run leakage.

**Envelope caveat:** we likely reach only ~15–25 m/s (coasts/nose-first), not 33 m/s. Fit the
physically-motivated parametric terms in-envelope and **extrapolate** the v² drag / μ-moment to racing
speed; **trust the learned residual only inside the tested envelope** and flag extrapolation (matches
NeuroBEM's >65 km/h caveat).

**Pitfalls handled:** thrust-model error → lean on coast samples for force/moment targets, powered for
coverage; gyro-derivative noise → low-pass `α` or fit integrated form; stream time-alignment →
interpolate IMU/odometry/commands to a common grid; Huber loss + normalized features for robustness.

## Fitting pipeline
**Files:** `aigp/aero_model.py` (terms, differentiable), `aigp/aero_fit.py` (fitter).
- **Stage 1 — parametric backbone via least-squares.** Drag, the weathervane multinomial, damping,
  and the `k_h` thrust shortcut are **linear in their unknowns** → one convex `lstsq` over stacked
  regressors. Fast, interpretable, gives each term's contribution. (Nonlinear BEMT inflow, if needed,
  → small Adam fit.)
- **Stage 2 — residual.** Subtract the parametric prediction; fit the MLP `r(features)` on the leftover.
- **(Optional Stage 3)** joint fine-tune via differentiable multi-step rollout MSE (COR-96 Adam-on-
  rollout) to reduce integration drift — only if rollout error demands it.
- Weight coast samples higher; Huber loss; normalized features.

## Validation & deliverable
**File:** `aigp/aero_validate.py`. On held-out runs:
1. **Single-step prediction** — aero force & moment RMSE / R², per axis, physical units.
2. **Short-horizon rollout** — integrate the full identified model over ~0.5–2 s windows from recorded
   initial states + commands vs the sim's actual trajectory (the real replica test).
3. **Residual share** — fraction of aero variance explained by parametric terms vs residual, with a
   per-term breakdown. "Fully known" ⇔ residual small + bounded.
4. **Envelope/coverage report** — where validated vs extrapolated; explicit speed-extrapolation flag.

**Deliverable:** `sim_aero.json` (coefficients + residual weights) + validation report (RMSE tables,
residual share, measured-vs-fit force/moment-vs-velocity plots, coverage map). This = the "validated
aero model" the scope defines.

## Architecture / files
- `aigp/aero_model.py` — parametric terms + residual (differentiable, pure). Builds on `dyn_model.py`.
- `aigp/aero_fit.py` — lstsq backbone + residual fit (reuses `torch_quad`, the `dyn_fit` drag fitter).
- `aigp/aero_dataset.py` — raw log → (features, aero targets); coast/powered split; time-alignment.
- `aigp/aero_capture.py` — data-collection driver (maneuver battery + coast probes) → parquet + manifest.
- `aigp/aero_validate.py` — held-out single-step + rollout + residual-share report.
- Maneuver scripts (sweeps / circle-lemniscate / doublets / thrust-chirp / coast-probe), reuse
  `fresh_start` gating from `race_cruise.py`.
- Tests in `tests/test_aigp/` (synthetic round-trip; pure components).

## Testing
- **Synthetic round-trip:** generate aero from known `θ` (+ a known residual), confirm the fitter
  recovers them within tolerance (COR-96 equivalence-test pattern).
- `aero_model` / `aero_fit` / `aero_dataset` are pure → unit-tested offline. Data-capture + final
  validation need the live sim.

## Risks / open questions
- **Envelope / extrapolation:** can't reach 33 m/s controlled; rely on parametric extrapolation +
  residual-in-envelope-only. Biggest risk to racing-speed fidelity.
- **Inertia 3.7× oddity:** `Ixx/Iyy≈3.7` is suspicious for a square 280×280 mm frame — re-check during
  the moment fit (could be an axis-convention artifact); affects the moment EOM.
- **Thrust-model contamination** of powered force estimates → mitigated by coast-sample weighting.
- **Identifiability:** some terms may be correlated (e.g. linear vs quadratic drag at low speed) →
  rely on the wide speed coverage + convergence stopping rule; report parameter correlations.

## Dependencies
- numpy / scipy (lstsq), torch (residual MLP + `torch_quad`), pandas/pyarrow (parquet), matplotlib
  (report plots) in the `aigp` env. Chassis geometry 280×280×160 mm available as an inertia prior.
- Mac Tailscale IP (if Rerun used for live capture viz): 100.101.13.126. Sim host: 100.120.233.90.
