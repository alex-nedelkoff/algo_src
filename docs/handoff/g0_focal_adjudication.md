# G0 focal-length adjudication (evidence only — NOT approved)

Branch `feat/g0-calib-adjudication`. Offline. `vq2/camera.py` untouched; no
`human_approved` flag set. Reproduce: `python -m vq2.tools.g0_focal_adjudication
<frames_dir> --n 30`.

## Question
Which sim-camera intrinsic explains BOTH the known 1.5 m G0 aperture geometry
AND the certified G0 depth (~10.6 m)?
- H1 mapping/GateNet "226": fx=fy=226, cx=319.5, cy=179.5 @640x360, +20 deg tilt.
- H2 Janahan/Kalibr "320": fx=fy=width/2=320, cx=320, cy=180, no distortion, no tilt.
- H3 free fit from corners.

## Data
- 111 fresh-reset stationary G0 frames, `g0_calibration_20260722_0205/frames`.
- **Resolution: 640x360** (verified). So H2's fx=width/2=320 applies directly —
  no resolution/crop rescale needed.

## Corner localization (non-circular)
Aperture corners from red-frame vs blue-interior segmentation (redness =
R - max(G,B)), median over 30 frames; cross-checked with per-column/row
zero-crossing edge fits on 3 frames. Two brackets on the true 1.5 m opening:
- **interior (mid-rim):** ~39w x 40.5h px, centre (320.5, 172.8).
- **front-rim:** ~42w x 47h px, centre (320.4, 168.3).
Per-corner uncertainty ~2-4 px (bottom corners std 0.5 px; top corners std ~4 px
because of a bright inner-thickness bloom band). Monte-Carlo (sigma=3px/corner)
depth spread ~0.4-0.55 m.

Robust facts independent of the bracket:
- Aperture is **~40-46 px wide** (NOT ~30 px). A 1.5 m square at 10.6 m needs
  fx*1.5/10.6 px: 32 px @fx=226 vs 45 px @fx=320. Measured ~40-45 px => fx~320.
- Gate centre x ~= cx (dead ahead). Gate centre y ~= 169-173, i.e. only ~7-11 px
  ABOVE the principal point.

## Per-hypothesis table (implied camera-to-G0 depth vs certified 10.595 m)

interior (mid-rim) corners:
| model        | fx  | reprojRMS | orth   | eqscale | depth Zopt | dVsCert |
|--------------|-----|-----------|--------|---------|------------|---------|
| fitted (H3)  | 118 | 0.00      | 0.000  | -0.000  | 4.59       | -6.00   |
| mapping-226  | 226 | 1.67      | +0.073 | -0.447  | 8.27       | -2.33   |
| legacy-320   | 320 | 1.79      | +0.136 | -0.670  | 11.71      | +1.11   |
focal reproducing 10.595 m: **fx=289**

front-rim corners:
| model        | fx  | reprojRMS | orth   | eqscale | depth Zopt | dVsCert |
|--------------|-----|-----------|--------|---------|------------|---------|
| fitted (H3)  | 122 | 0.00      | 0.000  | -0.000  | 4.33       | -6.27   |
| mapping-226  | 226 | 1.87      | +0.105 | -0.412  | 7.12       | -3.48   |
| legacy-320   | 320 | 1.96      | +0.211 | -0.597  | 10.04      | -0.56   |
focal reproducing 10.595 m: **fx=338**

- **H3 free fit is ill-conditioned** (near-fronto-parallel square): returns
  unphysical fx≈118, fy anisotropic, depth ~4.5 m. Unusable — as the tool's
  design warns. Do not use a free fit from this view.

## Adjudication — root cause is FOCAL, not tilt/resolution/datum
- **Focal (root cause):** direct aperture geometry reproduces the certified
  10.6 m only at **fx = 289-338 (mid ~310-320)**. fx=226 forces the gate to
  **7.1-8.3 m — 2.3-3.5 m short** of the datum in every bracket. 226 cannot fit
  the real aperture pixels and the certified depth simultaneously.
- **20 deg tilt: REJECTED.** With a 20 deg upward camera tilt, a level dead-ahead
  gate at 10.6 m would project ~fx*tan20 ≈ 116 px BELOW image centre (y≈296).
  Observed gate centre is y≈170 (ABOVE centre). The effective pitch is ~2 deg
  (explained by the gate sitting ~0.25 m above the camera). No 20 deg tilt exists
  in the rendered imagery.
- **Resolution/crop: not the cause.** Frames are 640x360; fx=320=width/2 is the
  native value, no rescale.
- **Datum misread: not the cause — datum is CORROBORATED.** Gate is dead-ahead
  (centre x≈cx), so the datum X=10.595 m ≈ true camera-to-gate range, and direct
  geometry at fx~310-320 reproduces it. The "226 -> 11.18 m" figure is GateNet's
  own (circular) pipeline output, not the direct-geometry depth (which is ~7-8 m
  under 226 — an internal inconsistency that further indicts 226).

## RECOMMENDATION (for human sign-off — NOT approved)
Consolidate on the H2 / Janahan Kalibr-backed model at 640x360:
- **fx = fy = 320**   (direct geometry brackets 289-338; H2=320 and AirSim
  fx=320.54 sit inside it; three upstream sources agree)
- **cx = 320.0, cy = 180.0**  (measured gate centre 320.5,172.8 is consistent;
  319.5/179.5 is within noise)
- **tilt = 0 deg**  (20 deg tilt refuted by gate image position)
- zero distortion
Drop the local "226 + 20 deg tilt" model.

## VO scale correction (close the loop)
VO used vq2.camera FX=226 and derives metric scale from gate-PnP range, which is
linear in focal. Under the recommended fx=320:
- **correction factor = 320/226 = 1.416**
- **0.30 m/unit -> 0.30 x 1.416 = 0.425 m/unit** (bracket over fx 289-338:
  1.28-1.50x => 0.38-0.45 m/unit; central ~0.42).
