# SysID Multi-Drone Capability Demo — Implementation Plan

**Date:** 2026-05-05
**Goal:** Two-phase deliverable that demonstrates our ability to learn drone dynamics across (a) different drone scales and (b) sources where the data generator has physics outside our 8-param grey-box model.

**Why two phases:**
- **Phase 1** validates the existing pipeline scales across drones (cheap, ~half-day, low risk). Self-fit (`numpy_quad` → `numpy_quad`) — same model, same parameters, so a reviewer will recognize it's mostly a "does the optimizer converge" demo. But it's the right starting point: if it doesn't work cleanly across drone scales, Phase 2 is irrelevant.
- **Phase 2** is the actual headline result — fitting our model against `gym-pybullet-drones`, which has propeller wake / ground effect / blade flapping that our 8 params can't represent. Shows predictive accuracy *despite* model mismatch, which is the realistic deployment scenario.

Final deliverable: single HTML report (matches the `/ship-it` template) with both phases side-by-side.

---

## Phase 1 — Multi-drone numpy_quad benchmark (~half-day)

### Task 1.1: Define 4 drone presets

Create `configs/sysid/drones/`:

| File | Class | Mass | Arm | T:W ratio | Notes |
|---|---|---|---|---|---|
| `cf21.yaml` | CrazyFlie 2.1 micro | 0.027 kg | 0.040 m | ~2.0 | existing baseline |
| `fpv_racer.yaml` | 5" FPV race quad | 0.600 kg | 0.120 m | ~6.0 | aggressive |
| `f450.yaml` | DJI F450 photography | 1.500 kg | 0.225 m | ~2.5 | mainstream |
| `heavy.yaml` | Industrial payload | 5.000 kg | 0.400 m | ~2.0 | upper-end |

For each, derive consistent params:
- Inertia ≈ `0.5 · mass · arm² · diag(1, 1, 1.8)` (point-mass approximation, Z scaled higher)
- `k_thrust` chosen so `4·k_thrust·max_omega² / (mass·g) = T:W ratio`
- `k_torque ≈ 0.02 · k_thrust · arm` (typical for similar-size props)
- `tau_motor` scales roughly with `√mass` (~0.02 s for CF, ~0.08 s for heavy)
- `max_rpm` decreases with prop size (15 000 → 6 000)

Sanity check: each preset must produce sane hover trajectories before being added.

### Task 1.2: Per-drone benchmark runner script

`scripts/sysid/benchmark_drones.py`:

1. Load each drone preset
2. Generate dataset (`generate_dataset.py`-equivalent, 200 trajectories × 200 steps × dt=0.01)
3. Run fit twice:
   - Unconstrained (init_perturbation=1.5, no `--fix`)
   - Constrained (`--fix mass --fix arm_length` to ground truth)
4. Run held-out validation (separate seeds)
5. Save results (per-drone) to `outputs/sysid/multi_drone/{drone_name}/`:
   - `dataset.npz` (training) / `validation.npz`
   - `fit_unconstrained.pt` / `fit_constrained.pt`
   - `metrics.json`: per-param error, rollout RMSE per state component, convergence loss curve

### Task 1.3: Aggregate report

`scripts/sysid/_make_multi_drone_report.py` (one-shot, follows `_gen_ship_report.py` pattern):

- Per-drone parameter recovery bar chart (8 params × 2 conditions per drone)
- Convergence curves overlay (4 drones × 2 conditions = 8 lines)
- Held-out predictive RMSE table (rows: drone, columns: pos/vel/quat/omega/motor)
- Inline tables embedded in HTML, base64-encoded plots

**Success criteria for Phase 1:**
- All 4 drones converge (loss → 1e-5 or below)
- Constrained fit recovers mass + arm + tau_motor exactly (pinned), other 5 params within 10 %
- Held-out RMSE: position < 1 mm/s rollout, velocity < 1 cm/s, quat < 1e-3 over 0.5 s

---

## Phase 2 — PyBullet model-mismatch demo (~1.5 days)

### Task 2.1: PyBullet data generator

`scripts/sysid/generate_dataset_pybullet.py`:

- Use `gym-pybullet-drones` (already a dep — verify, install if not)
- Mirror the `numpy_quad` action styles (random_walk + sinusoidal + step) so trajectories are comparable
- Match dt=0.01, traj length=200 steps
- Use the gym-pybullet-drones CrazyFlie 2.1 URDF as the "ground truth" — same nominal params as our `cf21.yaml`, so a clean self-fit comparison is possible

Output schema same as numpy_quad dataset (`states (N, T+1, 17), actions (N, T, 4)`) so the existing `fit_params.py` works unchanged. Caveat: PyBullet's state may include forces/torques we don't track — discard, keep only the 17-dim subset.

### Task 2.2: Run cross-fit

Same `fit_params.py` and `validate.py` against PyBullet-generated data:
- Init params from CrazyFlie 2.1 nominal values (same as `cf21.yaml` Phase 1)
- Constrained fit (`--fix mass --fix arm_length`)
- Compare:
  - **Self-fit (Phase 1, cf21):** numpy_quad data → numpy_quad model
  - **Cross-fit (Phase 2):** pybullet data → numpy_quad model

Expected outcome: parameter recovery is degraded (PyBullet has effects our model can't represent, so the fitter compensates by drifting params), but **predictive accuracy on held-out PyBullet trajectories** is still useful (e.g., position RMSE < 5 cm over 0.5 s, vs ~70 µm in self-fit). That delta is the headline.

### Task 2.3: Failure mode analysis

Generate residual plots — `(pybullet_state − fitted_model_prediction)` over time, broken down by component. Look for:
- Velocity-dependent bias → unmodeled drag (we know about this)
- Near-ground bias → ground effect
- Aggressive-maneuver bias → blade flapping or motor model nonlinearity

This is what a reviewer wants: "they can characterize what their model can and can't capture."

### Task 2.4: Extend the report

Add Phase 2 sections:
- Predictive accuracy table: self-fit vs cross-fit, per-component RMSE
- Residual plots (4 components × 2-3 example trajectories)
- Discussion of model-mismatch sources

**Success criteria for Phase 2:**
- PyBullet data generator runs end-to-end without divergence
- Cross-fit converges (loss is higher than self-fit, but stable)
- Held-out predictive RMSE on PyBullet trajectories: position < 5 cm, velocity < 0.1 m/s over 0.5 s rollout
- Residual plots show identifiable systematic biases (not just white noise) — evidence the model-mismatch is structural, characterizable

---

## Deferred / out of scope

- **Drag identification.** Currently `torch_quad` doesn't model drag; `numpy_quad` does (with default `[0.01, 0.01, 0.005]`). Phase 1 self-fit is on drag-bearing data with a drag-less model — small but nonzero misfit. We'd want to add drag to `torch_quad` and make `drag_coeff` learnable as a follow-up. **Skipping for this deliverable** to keep scope tight; the existing 8-param model is what we want to demonstrate.
- **Hybrid grey-box + MLP residual.** The natural Phase 3. Defer until Phase 2 surfaces the residual structure.
- **Real flight log fitting.** Defer until we have either the AI Grand Prix sim telemetry or a physical drone.
- **Online / adaptive sysID.** Defer.

---

## Estimated total: 2 days

| Task | Effort |
|---|---|
| 1.1 Drone presets | 2 h |
| 1.2 Benchmark runner | 2 h |
| 1.3 Phase 1 report | 1.5 h |
| 2.1 PyBullet data gen | 4 h |
| 2.2 Cross-fit run | 1 h |
| 2.3 Residual analysis | 2 h |
| 2.4 Phase 2 report | 1.5 h |
| **Total** | **~14 h** |

## Linear issue

Recommend filing as new COR-XXX with this plan as the description, milestoned `M1` Phase 1 + `M2` Phase 2. Self-contained, can be picked up by anyone after the team sync.

## Cross-references

- `docs/superpowers/specs/2026-05-04-moe-deployment-next-steps.md` — broader sysID follow-ups context
- `scripts/sysid/{dataset,fit_params,validate}.py` — existing pipeline (unchanged for Phase 1, extended for Phase 2)
- `sim/dynamics/{numpy_quad,torch_params,torch_quad}.py` — the 8-param grey-box model under test
- COR-96 — original sysID MVP issue (parent context)
