# CAM_TILT pair experiment — CORRECTED VERDICT (2026-07-24)

> **This file's first committed version (0df6eb43) concluded the −17.8° rest
> attitude was canned and recommended a CAM_TILT migration. That verdict was
> WRONG and is retracted here** — its decisive test (accel pitch stepping
> −17.8→0 at arming with ~0 gyro) is attitude-blind: once thrust rises,
> specific force aligns with body-z and accel-implied "pitch" reads ~0
> regardless of true attitude. Alex's challenge ("is the drone starting at an
> angle on the pad?") prompted the re-examination. `cam_tilt_reconciliation.md`
> is likewise superseded on its tilt conclusions (its focal work stands).

## Verdict: the spawn tilt is REAL and the stack is CORRECT as built

1. **The drone physically sits ≈ −18° nose-down on the pad**, and physically
   levels off at takeoff. Decisive chain (vq2_mig2):
   - Raw ∫ygyro over takeoff = −19.5° = **+19.5° nose-up in the flight
     stack's validated wfix convention** (`state['pitch'] += (-gyr[1])*dt`,
     eskf wfix "~exact on the sim IMU"), unfolding over ~1.5 s at 0.2–0.5
     rad/s — matching the campaign's own 07-12 note ("the spawn sits at
     −17.8° pitch and the takeoff level-off rotates ~0.5 rad/s") exactly.
   - RELEVEL print: `pitch +1.4 -> -0.2`. ESKF init −17.8 (rest accel) +
     gyro-integrated +19.2 = +1.4 at hover. Under a level-spawn/canned story
     the noiseless gyro would integrate ~0 and RELEVEL would have printed
     −17.8 → −0.2. It did not. The rotation was physical.

2. **Camera-in-body tilt is REAL ≈ +20°** (mount angle, matching the 20°
   nominal). Direct hover-VP measurements (level body ⇒ VP row reads the
   mount): **mig2 known-level hover frames (kf_pose-selected): row
   304.7 ± 1.0 px → +21.4°**; COR-142 T2a static hover: 288.0 ± 2.7 →
   18.6° ± 0.4. Pad VP (+2.25°) decomposes as spawn(−19..−16) + tilt
   (+18.6..+21.4) — consistent, sum-only as always.

3. **The rest accel is honest.** Bit-frozen presentation (|f| = 9.81 exactly,
   gyro exactly 0 — COR-142's noiseless-IMU finding), truthful value. The
   regime timeline (canned-looking −17.8 only disarmed-post-reset; level
   post-landing) reflects the drone's real attitude in each state: on the
   tilted spawn vs on flat floor after landing.

4. **Consequently the current flight stack needs NO tilt migration:**
   attitude init from the rest accel is correct; `CAM_TILT = 20°` is
   correct; RELEVEL is correct; in-flight camera rays are correctly
   elevated. The earlier "wrong-in-pairs cancellation" framing is retracted
   — both terms are individually right. The `OBSZ=0` "z corrupt" mystery
   loses the tilt explanation and remains open.

5. **The G0 focal adjudication's tilt rejection is RETRACTED** ("a level
   dead-ahead gate would project 116 px below centre" assumed a LEVEL body;
   with the real −18° spawn pitch, the 20° tilt projects the gate near
   centre, as observed). Its **fx = 320 conclusion STANDS** on
   tilt-independent evidence: aperture pixel WIDTH at the certified datum
   depth (40–46 px ⇒ fx≈320; width is unaffected by pitch), Janahan's
   Kalibr chain, and T2a's 0.74 px PnP reprojection.

6. **The wrong-constellation bug stands, unaffected by this reversal**: the
   G1 branch of vq2wp's identity check reprojects `G2RIB_C` — the GATE-2
   constellation (centroid [11.68, 5.17, −1.06], bearing 23.9° from the
   pad) — against G1 observations (measured du +122 px horizontal, dv +6).
   It survives as a coarse junk-rejector under the (now correctly rescaled)
   angular tolerance. A real G1 constellation is the proper fix.

## What changes downstream

- COR-148 / mapping extrinsics: use camera-in-body ≈ **+20° (18.6–21.4
  measured)** — NOT the "≈0–3°" my superseded docs claimed. The built
  Station-22 candidate is unaffected (its builder chains orientation
  through the anchor PnP and never consumed CAM_TILT).
- `vq2/camera.py` CAM_TILT stays 20°. The vq2 in-repo comment claiming the
  adjudication found ~0 true tilt should be cleaned up when next touched.
- Attitude-regime tooling (`accel_regime.py`) remains valid as a timeline
  tool; interpret disarmed-post-reset readings as REAL spawn attitude.

## Repro

- wfix sign: vq2wp.py:489-490 (`roll += -gyr[0]`, `pitch += -gyr[1]`).
- Gyro level-off: `accel_regime.py` + fine-grained integral around t_arm
  (vq2_mig2); RELEVEL print in `fly_mig2.log`.
- Hover VP: `vq2/tools/vp_tilt.py` on `work/mig2_hover_sample/`
  (kf_pose-selected level-hover frames; row 304.7 ± 1.0).
- Wrong-constellation decomposition: `vq2/tools/reproj_decompose.py` +
  `work/mig1_pad_detections.jsonl`.
