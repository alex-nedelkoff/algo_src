# Camera-tilt reconciliation: COR-142 T2a "18.6°" vs G0 adjudication "≈0°"

**Date:** 2026-07-23. **Verdict: no contradiction — the two numbers live in
different reference frames.** Camera-in-body tilt is ~+1–3° up, in BOTH
epochs, once the body-pitch term is made explicit. The 20° `CAM_TILT` nominal
does not describe the rendered camera in either epoch.

## Explicit frame definitions

- `pitch_body` — body-FRD x-axis elevation above the gravity horizon
  (positive nose-up). Measured from the at-rest accelerometer (specific
  force = −gravity in body frame).
- `tilt_cam_gravity` — camera optical-axis elevation above the gravity
  horizon. Measured from the vanishing point of receding horizontal world
  lines: `v_vp = cy + fy·tan(tilt_cam_gravity)`.
- `tilt_cam_body` — camera optical axis above the body x-axis. This is what
  `CAM_TILT` / `dcl_constants.CAMERA_PITCH_UP_DEG` claim to be (nominal 20°).
- Identity: `tilt_cam_body = tilt_cam_gravity − pitch_body`.

T2a's VP method measures `tilt_cam_gravity`; converting it to a camera mount
angle requires `pitch_body`, which T2a assumed to be zero ("cannot
definitively separate camera tilt from residual body pitch" — their own
caveat, calibration-report §D).

## Measurements

| quantity | COR-142 T2a (2026-07-02, run_04/C2b) | this work (2026-07-22/23 corpora) |
|---|---|---|
| `tilt_cam_gravity` (VP of receding lines) | 18.6° ± 0.4 (horizon row 288.0 ± 2.7) | **+2.25°** (VP row 192.1 ± 0.5, x 319.0 ≈ cx; 12 frames of `g0_calibration_20260722_0205`, same Canny→HoughLinesP→length-weighted robust VP method) |
| `pitch_body` at rest (accel) | **+17.8°** — reported in §G and *dismissed as a "canned sim constant"* (per `sim_kalibr_imucam_chain.yaml` comment) | **−0.88°** (3,402 low-actuator HIGHRES_IMU samples, yaw-scan corpus; \|f\| = 9.797) |
| ⇒ `tilt_cam_body` | 18.6 − 17.8 = **+0.8°** | 2.25 − (−0.88) = **+3.1°** |
| G0 gate projection cross-check | — | gate centre ~9.5 px above cy at ~11 m with gate ~0.3 m above camera ⇒ consistent with `tilt_cam_gravity` ≈ 0–2° |

## Reconciliation

T2a's 18.6° was a **gravity-referenced** measurement with the body-pitch term
silently set to zero. Their own §G rest-gravity measurement (17.8° pitched)
was the body-pitch term — treating it as a sim artifact instead of a real
attitude moved 17.8° of body pitch into the camera mount. Subtracting it,
both epochs agree: **camera-in-body ≈ +1–3° up**. Whether the 07-02 build
really rested nose-up 17.8° (spawn/catapult attitude, since changed) or the
IMU gravity vector was then miscanned cannot be settled from here — but under
either reading the camera mount is near-level, and on the CURRENT build all
three independent observables (rest accel, VP row, G0 gate projection) agree
with each other to ≤3°.

## Flight-run addendum (2026-07-24, vq2_mig1–3)

The canned tilted gravity vector is CONFIRMED in the current build in the
post-reset regime: all three migration flights printed `post-reset pitch
-17.8` (the ESKF initializes attitude from the accel it sees at reset). This
does NOT overturn the reconciliation — it completes it: the flight stack's
attitude(−17.8, canned) + `CAM_TILT`(+20°) **cancel to ≈ +2.2°**, which
matches the true rendered camera elevation (+2.25° VP). The stack has been
self-consistently wrong-in-pairs; **flipping CAM_TILT alone would break the
cancellation and mis-elevate every camera ray by ~18°.**

Open regime question: the yaw-scan static-window median gave pitch −0.88°
(level) while the post-reset accel reads −17.8° (canned). The two rest
measurements disagree between regimes — when the stream switches (arming?
race start?) is unresolved and belongs to the tilt/attitude experiment. A
second measured input for that experiment: the G1 identity check carries a
~20° systematic reprojection bias at the pad (uniform ~123/159px at fx=320
across all 8 corners; banked in vq2_mig1's `obs_ident_fail` rows).

## Consequences

1. The G0 focal adjudication's "20° tilt REJECTED, true tilt ≈ 0°" stands,
   now WITH the T2a reconciliation it previously lacked (refined to ≈ +3°
   camera-in-body on current imagery).
2. `vq2/camera.py CAM_TILT = 20°` and `dcl_constants.CAMERA_PITCH_UP_DEG = 20`
   describe the config nominal, not the rendered camera. Every consumer of
   `M_BODY_CAM` / `pixel_rays_body` carries a ~17° elevation error on camera
   rays. Empirically-calibrated paths (RANGE_AFF, servoing on bearings) have
   partly absorbed it; geometric consumers (map building, ray casting into
   reset-NED) must not.
3. The CAM_TILT flip remains a SEPARATE flight-stack decision requiring tick
   re-validation (unchanged from the migration plan) — but it is no longer
   blocked on an unexplained measurement conflict.
4. COR-148 Phase B should use tilt_cam_body ≈ +3° (or 0° with ±3°
   uncertainty), per the Phase B note that map math applies the corrected
   extrinsic locally.

## Repro

- VP method: `scratch vp_tilt.py` (session scratchpad; method per T2a report
  §D — Canny→HoughLinesP→oblique-line filter 6–80°→Huber-IRLS length-weighted
  VP), 12 frames sampled at 30–80% of the stationary corpus.
- Rest pitch: median specific force over HIGHRES_IMU rows gated to
  `max(actuator[0..3]) < 0.2`, `pillar_g0_yawscan_20260722_0145/mavlink.jsonl`;
  `pitch = atan2(f_x, sqrt(f_y² + f_z²))`, `roll = atan2(−f_y, −f_z)`.
- Kalibr chain provenance: `algo_src_main/state_estimation/configs/
  sim_kalibr_imucam_chain.yaml` (records both the 18.6° VP number and the
  dismissed 17.8° §G gravity pitch).
