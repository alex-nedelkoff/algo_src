# CAM_TILT pair experiment — verdict (2026-07-24, all offline, no sim used)

Question: `CAM_TILT=20°` and the −17.8° rest attitude are wrong-in-pairs —
what is actually true, when, and what must change together?

## Verdict (each item measured, banked corpora only)

1. **The −17.8° disarmed gravity vector is CANNED (fake).** Decisive test:
   at arming (vq2_mig2, t≈12.4 s) the accel pitch STEPS −17.80° → −0.01°
   within ~2 samples while the cumulative body-y gyro across the step is
   −0.4° (a real 17.8° pitch-up in 0.16 s needs ~2 rad/s; observed |gyro|
   peak 1.05, integrating only the ordinary climb transient). Pre-arm the
   IMU is bit-frozen (|f| = 9.81 exactly, gyro exactly 0, pitch −17.80
   exactly). COR-142 T2a §G was right; the RELEVEL comment's "the spawn
   sits at −17.8° pitch" is a misreading — the spawn *reads* −17.8, the
   drone *sits* level.

2. **Regime law** (accel-implied pitch vs time, mig1/mig2/yaw-scan):
   canned −17.8 exists ONLY disarmed-on-the-pad-post-reset; the accel goes
   live the instant thrust rises; post-landing rest reads level. (The
   earlier "−0.88, body level" rest measurement in
   `cam_tilt_reconciliation.md` was a regime-mixing artifact of a
   low-actuator filter — superseded by this timeline.)

3. **True constants:** body at spawn ≈ LEVEL (0°); camera-in-body tilt ≈
   **+2.2° up** (VP row 192.1 ± 0.5 on stationary G0 frames; corroborated
   by the gate projection and by this decomposition's dv residuals).

4. **The cancellation, exactly:** modeled elevation = attitude + CAM_TILT.
   - At the pad: (−17.8 canned) + (+20 fake) = +2.2 = (0 true) + (+2.2
     true). Correct SUM, both terms wrong → pad-phase geometry works.
   - **In flight, after RELEVEL** re-zeros attitude from the live accel:
     (0 true) + (+20 fake) = +20 vs true +2.2 → **+17.8° elevation error
     on every camera ray for the rest of the flight.** This is the likely
     root cause of the "z channel corrupt" verdict that forced `OBSZ=0`
     (obs z disabled) — the error signature and magnitude match.

5. **The ~21° pad reprojection bias is NOT a tilt/attitude effect — it is a
   wrong-constellation bug.** Vector decomposition of the identity-check
   error on mig1 pad frames (of-record corners vs the exact vq2wp chain):
   du median **+121.6 px**, dv **+6.2 px** — horizontal, not vertical. The
   G1 branch of the constellation check reprojects `G2RIB_C`
   (`g2rib_corners_world.npy`, centroid **[11.68, 5.17, −1.06] = gate 2**,
   bearing 23.9° from the pad) against G1 observations. It has functioned
   as a coarse junk-rejector only because the wrong-gate offset sat under
   the loose 226-scaled tolerance while true junk reprojects far off.
   dv confirms item 4: as-flown (−17.8+20) and true (0+2.25) both leave
   dv ≈ +6 px; single-term variants blow dv to ±110 px.

## Safe migration recipe (the terms that must move TOGETHER)

1. `CAM_TILT` 20° → **+2.2°** (or 0° with a documented +2.2 residual).
2. Attitude init at spawn: **ignore the canned disarmed accel** — init
   pitch/roll to 0 post-reset (the accel is trustworthy for leveling only
   once armed; RELEVEL already handles post-takeoff).
3. Done together, modeled elevation is correct in ALL regimes (today it is
   correct only at the pad). Do NOT change either term alone: CAM_TILT
   alone breaks the pad phase (−17.8+2.2 = −15.6 vs true +2.2); attitude
   init alone breaks it the other way (0+20 = +20).
4. Follow-ons unlocked by the fix: `OBSZ` may be re-enableable (z channel
   was likely never "corrupt", just mis-elevated); the G1 identity check
   should get a real G1 constellation (item 5 — independent bug, fix
   separately); `pixel_rays_body`/mapping consumers stop needing the
   quarantine note in COR-148.
5. Flight re-validation required (fresh sim + gidx2) — one variable: this
   pair. NOT flown in this experiment (sim intermittently in use by
   another agent); all evidence above is from banked corpora.

## Repro

- Regime timeline: `accel_regime.py` (scratchpad) over
  vq2_mig1 / vq2_mig2 / pillar_g0_yawscan corpora.
- Arming step: fine-grained accel-pitch + cumulative gyro-y around t_arm
  (vq2_mig2 mavlink.jsonl). Caveat: naive integrals over this stream
  double-count its 38% duplicate re-sends; the step conclusion needs only
  the ~0 integral across 2 samples.
- Reprojection decomposition: `reproj_decompose.py` (scratchpad) — exact
  vq2wp chain (Rz·Ry·Rx·M_BODY_CAM, fx=320) on
  `work/mig1_pad_detections.jsonl` (of-record corners, 5 mig1 pad
  frames) vs `g2rib_corners_world.npy`, sweeping (pitch, tilt, cam-z).
