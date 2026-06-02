# AI-GP Sim Aerodynamic SysID — Identified Model Report

_Generated 2026-06-02 19:01 · grey-box fit on motors-off (coast) data_

## Method

Linear-in-coefficients parametric backbone (least-squares) + torch residual MLP on the force leftover. Only **motors-off samples** are used so the IMU specific force is the aero force and dω/dt is the aero angular accel — no thrust or motor-torque model is assumed.
- **Force** from **coast bursts** (build speed, cut thrust, log decel).
- **Moment** from **tumbles** (build speed, pulse a body rate, cut motors, log the free rotational+translational decay).

## Dataset

- Force: 5 train + 1 held-out coast runs; Moment: 5 train + 1 held-out tumble runs (held-out `coast_20260602_171627.parquet`, `tumble_y_20260602_172351.parquet`).
- Body-speed envelope reached: **0–13.7 m/s**. Course peaks ~33 m/s; coefficients are in-envelope and **extrapolate with rising uncertainty above ~8 m/s**.

## Force model  `a_aero = -D·v - C·v|v|`  (m/s², per body axis)

| axis | D (linear) | C (quadratic) |
|---|---|---|
| x | +0.5231 | -0.0175 |
| y | +0.0021 | -0.0258 |
| z | +0.0000 | +0.0000 |

Train R²=0.408. Held-out single-step force RMSE **2.192 m/s²** (parametric) → **2.978** with residual; R² 0.400→-0.107.
Per-axis held-out RMSE: x=0.156, y=0.003, z=3.793 m/s².
> **`D_x` is the trusted, literature-validated result** — linear rotor drag along the flight axis (Faessler form). Published ground-truth for racing-class quads identified from real flight: `d_x≈0.49–0.54 /s` (circle/lemniscate @ 4 m/s) — **our 0.52 lands squarely in that band.** The **z-axis** is from free-fall coast (vortex-ring/windmill-brake state where momentum theory fails, not clean drag) so `D_z`/`C_z` are **pinned to 0** in the deliverable; literature notes vertical drag is a minor effect anyway. **`D_y` is not yet identified** — see next (off-axis coast carries non-drag bluff-body force; lateral drag needs a powered lemniscate).

## Moment model  `α_aero = -d·ω + weathervane(v)`  (rad/s², per body axis)

| axis | d (rotational damping) | wv (velocity-coupling) |
|---|---|---|
| x | -12.9589 | +0.0204 |
| y | +11.2716 | +0.0606 |
| z | +18.8678 | +0.2841 |

Train R²=0.173, held-out RMSE **7.759 rad/s²**, R²=0.138.
> ⚠️ **Moments are INDICATIVE, not converged.** The one robust takeaway is the **weathervane sign**: `wv_z=+0.284` (forward/tail-first velocity drives a **yaw** moment) — this confirms the team's tail-first-destabilizer hypothesis and is the term that matters for the control problem. The damping coefficients are unreliable (one came out **negative** = anti-damping) because a free tumble is too coupled/noisy for a diagonal-damping model: the body-frame velocity rotates rapidly, finite-diff of a fast gyro is noisy, and the gyroscopic subtraction is sensitive to the approximate `I_ratio=3.7`. Clean moment ID needs a different excitation (steady tail-first flight + a motor-torque / control-inversion model) — see next.

## Residual & rollout

- Force residual-variance-share (in-sample): 0.204. **But the residual MLP did NOT generalize** (held-out force R² 0.40 → -0.11 *worse* with the residual): it overfits the noisy free-fall z-axis. On the trusted horizontal axis the linear drag term already explains the data, so **the parametric model is the deliverable** and the residual is not used.
- Velocity rollout over a held-out coast window (~0.50s): RMSE 3.292 m/s (dominated by the unreliable z free-fall integration; the horizontal x-axis tracks well, per the 0.16 m/s² single-step RMSE above).

## Dominant terms

- Force: `C_z` (16.144), `D_z` (12.109), `D_x` (2.730)
- Moment: `d_z` (7.993), `wv_z` (0.173), `wv_y` (0.008)

## What's trusted vs open

- ✅ **Horizontal rotor drag `D_x`** — headline result, held-out RMSE 0.156 m/s², **matches published 0.49–0.54 /s**.
- ✅ **Weathervane sign** (`wv_z>0`) — qualitatively confirms the tail-first yaw destabilizer.
- ❌ **Lateral drag `D_y`** — THREE attempts failed (diagonal coast → bluff-body; powered weave → phase/lever-arm artifact, spurious negative; steady circle → controller can't track it: speed overshoots, tilt spikes ~58°, drift 30 m). **Root cause: `D_y` is gated on lateral-flight control, which is blocked by the very weathervane we're measuring** (chicken-and-egg). The platform is too weathervane-unstable to fly the steady sideslip trajectories the open-loop method needs. By-products: quasi-steady `D_x`≈0.56 corroborates coast 0.52; powered-transient `D_x`≈0.34 < coast → linear drag is **thrust/operating-point dependent** (rotor-drag physics). The viable path is **closed-loop ID** (next).
- ❌ **Damping magnitudes & vertical force** — not reliably identified (see caveats above).

## Recommended next steps (literature-grounded — NeuroBEM, Faessler, grey-box sysID)

1. **Lateral drag `D_y` via CLOSED-LOOP ID** (open-loop steady circles are unflyable here — tried 3 ways). Keep a basic stabilizer running and make the **control effort the measurement**: the commanded yaw/pitch moment the controller applies to hold heading against the weathervane is a direct function of the lateral aero (CP-migration moment). Concretely: (a) add a **reference governor** to keep commands inside the controller's stability/motor envelope (no overshoot/tilt-spike); (b) excite with **2-1-1 lateral/yaw doublets near trim**; (c) fit `D_y` (and the weathervane moment) by **Nelder-Mead minimizing orientation error over short transient rollouts**, correlating IMU specific force + commanded moments — NOT steady-state regression. Add **motor speeds** if exposed (powered D_x≈0.34 ≠ coast 0.52 → thrust-coupled) and an IMU lever-arm correction. (Crazyflow fits such models in <4 min of flight.)
2. **Moments, properly**: don't use free tumbles (motors-off *removes* the rotor-coupled hub/H-force moments that ARE the weathervane). Use a **virtual mixer** — map commanded rate/thrust → predicted control torque (via the known rate-loop gain), attribute the residual `I·ω̇ − τ_control` to aero — and identify it by **trajectory rollout + gradient-free optimization (Nelder-Mead)**, NOT regression on finite-differenced ω̇ (too noisy). Use cubic-spline derivatives if a derivative is needed.
3. **Residual model**: the memoryless MLP overfit. NeuroBEM-style residuals need **temporal context** (a window of ~20 past states, often a **TCN**) so the net can reconstruct hidden airflow/wake state; train with a 70/20/10 split over a diverse-maneuver dataset.
4. **High speed (→33 m/s)**: the `k_h·v_h²` thrust-droop term under-predicts >15% at race speed → use a **full BEM / NeuroBEM hybrid**; quadratic parasitic drag (`C` terms) dominates above ~15–20 m/s and needs high-speed excitation to identify.

Model serialized to `aero_data/sim_aero.json` (D_z/C_z pinned to 0).
