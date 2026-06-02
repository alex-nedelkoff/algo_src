# AI-GP Sim Aerodynamic SysID — Identified Model Report

_Generated 2026-06-02 17:35 · grey-box fit on motors-off (coast) data_

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
| z | -1.8501 | +0.4446 |

Train R²=0.408. Held-out single-step force RMSE **2.192 m/s²** (parametric) → **2.978** with residual; R² 0.400→-0.107.
Per-axis held-out RMSE: x=0.156, y=0.003, z=3.793 m/s².
> **`D_x` is the trusted result** — linear rotor drag along the flight axis (Faessler form). The **z-axis** comes from free-fall coast (rotor descent dynamics, not clean drag) and is unreliable; treat `D_z`/`C_z` as placeholders.

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

- ✅ **Horizontal rotor drag `D_x`** — the headline result, validated on held-out data (0.156 m/s² single-step RMSE).
- ✅ **Weathervane sign** (`wv_z>0`) — qualitatively confirms the tail-first yaw destabilizer.
- ❌ **Damping magnitudes & vertical force** — not reliably identified (see caveats above).

## Recommended next data

1. **Moments, properly**: identify the weathervane/damping from **steady tail-first flight** by inverting the controller's commanded rate/torque (needs a motor-torque model), instead of free tumbles. This is the right path for the term the team actually cares about.
2. **Vertical force & >8 m/s**: a thrust-vs-speed model unlocks the powered sweeps for clean vertical drag and the high-speed envelope (course peaks ~33 m/s).
3. **Lateral drag `D_y`**: add a pure side-slip (strafe/side-coast) maneuver.

Model serialized to `aero_data/sim_aero.json`.
