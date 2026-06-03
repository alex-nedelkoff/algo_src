# AI-GP Virtual Qualifier drone — parameter pack

Single source-of-truth for the identified VQ-sim drone characteristics, for instantiating a **parallel RL
training surrogate** (PyBullet / `torch_quad`). Machine-readable: [`vq_drone_params.json`](vq_drone_params.json).
Regenerate: `python build_param_pack.py`; validate: `python validate_param_pack.py`.

- **Frame:** body FRD (x-fwd, y-right, z-down); world NED.
- **Normalization:** mass-normalized (forces in m/s², `T(hover)=9.81`). Absolute mass is unobservable from
  flight — the surrogate is mass-normalized; anchor with the published chassis (280×280×160 mm) if an
  absolute scale is needed.
- **Identified speed envelope: 0–8 m/s.** The race peaks at ~33 m/s, so high-speed terms are **DR-only**.

## Parameters

| Group | Param | Value | Conf | Source | DR range | Note |
|---|---|---|---|---|---|---|
| rigid | mass | 0.65 kg | low | chassis-bbox prior | 0.45–0.85 | absolute scale degenerate |
| rigid | I_ratio (Ixx:Iyy:Izz) | 3.7 : 1 : **4.7** | med | aero + dyn3 reconcile | Izz/Iyy [1,22] | Ixx/Iyy=3.7 agreed; Izz under-identified |
| rigid | I_y (pitch, mass-norm) | 0.0085 | med | kappa.json pitch-pulse | 0.005–0.013 | the pulse axis; Ix=3.7·Iy |
| rigid | I_z (yaw, mass-norm) | 0.040 | low | ⊥-axis (Ix+Iy); dyn3 said 22× | 0.0085–0.187 | **under-identified** (weak yaw); weathervane ∝ wv_z/Izz |
| rigid | arm L | 0.14 m | med | motor_model | — | |
| prop | k_f | 34.26 | high | hover calib | — | `T=k_f·Σg(u)`, hover→9.81 |
| prop | k_q | 0.685 | **low** | seeded k_q/k_f=0.02 | 0.27–1.7 | weak yaw; not pinnable |
| prop | thrust form | quadratic g(u)=u² | high | motor_model + collective sweep | — | ✓ confirmed (T~u^1.64, quad R²0.935≫linear 0.851) |
| prop | c_T | 9.95 | med | dyn | — | |
| prop | hover (throttle / motor-u) | 0.23 / 0.27 | high | sim_response / live | — | |
| mixer | sx,sy,sz | [-1,1,1,-1],[-1,-1,1,1],[-1,1,-1,1] | high | motor_model | — | hover-torque≈0 validated |
| force | **D_x** | **0.52 /s** | **high** | sim_aero coast | — | lit-validated 0.49–0.54 |
| force | D_y | 0.36 /s | med | sim_aero_cl | — | closed-loop |
| force | D_z | 0 | low | pinned (free-fall) | 0–0.3 | unidentified |
| force | D_thrust-coupling | [-0.023,-0.024] | med | sim_aero_cl | — | drag drops with thrust |
| force | C (quadratic) | ~0.02 | low | low-speed only | 0–0.05 | dominates >15 m/s, **unmeasured at race speed** |
| moment | weathervane (axial v_x→yaw) | +0.28 | med | sim_aero | — | tail-first destabilizer; sign trusted |
| moment | weathervane (sideslip v_y→yaw) | −0.0033 | low | sim_aero_cl yaw-fit | — | **k_q-conditional magnitude** |
| moment | angular damping | 0 | low | unidentified (collinear) | 0–0.5 | DR-only |
| acro | rate_gain (per axis) | ≈ −1.9 (sysID) / −2.5 (flight) | med | sim_response / race_cruise | — | ⚠ use flight value |
| acro | k_a | 62 | high | sim_response | — | thrust→accel |
| acro | motor lag τ | — | **missing** | fitter unrun | 0.010–0.040 s | DR-only |

## Validation (against real flight)

**Layer 1 — consistency cross-check** (`vq_drone_params_consistency.json`):

| Check | Status | Finding |
|---|---|---|
| thrust_form | **OK (resolved)** | Collective sweep (378 samples, u 0.14–0.43) confirms **quadratic**: T~u^1.64, `T=a·u²` R²=0.935 vs linear 0.851 (affine-quad best, R²=0.958). `sim_dynamics` power=1 (rough 3-level fit) superseded. `aero_capture.py collective` + `thrust_form_fit.py`. |
| hover_torque_zero | OK | reconstructed |τ|=0.000 at hover → mixer signs correct |
| D_x_thrust_dependence | OK | coast 0.52 vs powered 0.33 → operating-point dependent (expected) |
| acro_rate_gain | WARN | sysID −1.98 vs flight −2.5 (~1.3×); use the flight-measured value |
| inertia_anchor | OK | κ(pitch)=0.0085 used; κ(roll) noise-limited (3.7× inertia) |
| k_q_pinned | WARN | not pinnable from flight (weak yaw authority) → weathervane magnitude k_q-conditional |

**Layer 2 — held-out single-step** (`coast_20260602_171627`, 126 coast samples): drag-axis **RMSE = 0.156 m/s²**
(reproduces the published headline). R² is degenerate (−50) here because a steady coast has near-constant
aero force → ~zero target variance; RMSE + rollout are the meaningful metrics at a single operating point.

**Layer 3 — velocity rollout** (6 held-out coast bursts): forward-integrating the drag model predicts the
**final body-x velocity to 0.08 m/s mean error** (per-burst 0.03–0.18 m/s) over ~1.5 m/s of deceleration.
The drag model reproduces real motion.

## Surrogate-readiness verdict

**Ready for a first-cut parallel RL surrogate**, with two required extensions and domain randomization:

1. ✅ **DONE — aero terms wired into `numpy_quad`**: `VehicleParams.linear_drag_coeff`
   (`−D·v`) and `weathervane_coeff` (`v→moment`, the defining instability), DR-registered, TDD'd.
   For the VQ drone set `linear_drag_coeff=[0.52, 0.36, 0]`; `weathervane_coeff` per the pack (sign/
   magnitude k_q-conditional → DR + validate vs live flight). *Follow-ups:* `torch_quad` parity
   (fitting twin, currently drag-less) + a VQ-drone config wiring the pack values.
2. **Domain-randomize the unidentified params** (motor lag, angular damping, D_z, high-speed C,
   weathervane & k_q magnitude) over the DR ranges above — the Swift recipe: train in a randomized
   approximate sim, close the residual gap on real VQ flight.

**Biggest fidelity gap: the >8 m/s race regime is unmeasured** (quadratic drag + BEM thrust droop). Expect a
sim-to-real gap at speed; close it with residual learning on real VQ flight and/or a high-speed sysID campaign.

**Cheap next wins:** ~~resolve the thrust-form conflict~~ ✓ done (quadratic confirmed); run the existing
motor-lag fitter; (later) a high-speed sysID campaign for the >8 m/s regime.

*Out of scope here:* the end-to-end "does a surrogate built from this transfer" check — that needs the
surrogate built first. This pack is validated *against real flight*, the right pre-surrogate bar.
