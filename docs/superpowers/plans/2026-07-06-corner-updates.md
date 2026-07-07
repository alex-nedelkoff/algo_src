# Tight per-corner updates + corner identity (playbook 0.3 / ADR-VINS / QuAdGate)

Goal: replace solved-pose gate fixes with per-corner pixel-reprojection
updates + content-level obs identity, so hallucinated truss/fixture solves
can't feed the filter and partial views (2-3 corners inside the <6 m blind
zone) still update. Offline-first on the bench; fly only after it wins on
vq2_accept5 AND vq2_accept (creep).

## Why (evidence chain)
- VQ2-VETO-01: no single geometric cue (row, pitch) separates hallucinated
  solves from real gates; some miss>8 "junk" may be real G1 seen by a
  drifted filter. Only content-level identity separates.
- ADR-VINS: corner reprojection innovations valid from 2 corners, no PnP.
- QuAdGate (MonoRace): corner candidates matched to projected priors via
  local descriptors + RANSAC affine; tolerant of 150 px prior error.
- Bonus: corners stay in frame closer than the full gate -> shrinks the
  blind leg the campaign has fought since day 1.

## Measurement model (fits current PosVelKF, no filter restructure)
Attitude trusted (wfix chain). For map gate corner c_w (known 3D point):
  uv_pred = project(K, M_BODY_CAM^T @ R_wb(roll,pitch,yaw)^T @ (c_w - p))
  innovation = uv_detected - uv_pred          (2-vector per corner)
  H = d(uv)/d(p) = 2x3 Jacobian (analytic; only position block nonzero)
KF update per corner (or stacked 2N x 3), R from detections' sigma_diag
(per-corner sigma already in detections.jsonl!) x huber r_scale on pixel
residual. Chi2 gate per corner at 5.99 (2 dof, 95%).

## Map prerequisite: gate corner positions
- Gate aperture ~1.5 m square (VQ1-size, verified). Need gate YAW to place
  corners. G1 faces course (-x normal, aperture in y-z plane at x=11):
  corners = [11, ±0.75, -1.3±0.75] (check aperture half-size 0.75 vs VQ1
  gate spec in perception configs). G2 normal ~N2=[0.95,0.2,0] (vq2wp:76).
- Validate corner map against pad-rest frames: project corners with rest
  attitude from origin -> overlay on frame -> must land on the red gate
  corners (visual check via rrd emitter or single-frame plot).

## Identity (QuAdGate-lite)
1. Project map-gate corners through current estimate -> priors.
2. Match detected corners (corner_xy, 4 per inst, usable flags) to priors
   by nearest-neighbor in pixels with generous radius (150 px).
3. Require >=2 matched corners AND RANSAC-free consistency check: the 2D
   affine translation implied by matches must agree within ~40 px spread
   (poor man's RANSAC for 4 points).
4. Unmatched instances = no identity = never update (kills truss solves
   regardless of geometry: their corner constellation won't fit a
   projected gate square).

## Implementation steps (TDD, offline only until bench win)
1. `vq2/gates3d.py`: GATE_CORNERS map + `project_corners(p, att, gate)` +
   analytic H. Unit tests: pad-rest projection lands on known pixel box
   (validate against a real pad frame's detected corner_xy!); Jacobian vs
   numeric diff.
2. eskf: `update_pixel(uv_innov, H, R)` generic 2x3 update (or reuse via
   small refactor). Test: synthetic corner updates recover position.
3. fusion: `corner_updates=True` path — detections' corner_xy + sigma_diag
   through identity match -> per-corner updates (replaces update_position
   when on). obs_events stages: corner_match / corner_ident_reject.
4. Bench: corners vs huber_area on vq2_accept5 + vq2_accept close-range
   metrics + junk-immunity (accepted obs with miss>8 should ~vanish, real
   ones survive via corner identity).
5. Port to vq2wp det_loop (corner_xy available from dec objects same as
   vq2_detect.py builds them); POLICY=corners env. One recorded flight.

## Facts needed (verify, don't assume)
- Aperture half-size: check perception config gate_physical_size=1.5 (full
  size? then half=0.75) in vq2/gatenet/postprocess.py + VQ1 gate spec.
- Corner ORDER/convention in corner_xy (which index = which physical
  corner; visibility flags) — read perception/decode/associate.py.
- Pixel frame of corner_xy: padded image coords (pad_bottom to pad_to_h)
  vs raw 640x360 — vq2_detect uses padded input; check whether corner_xy
  is stride-corrected to input pixels (postprocess: (sub+0.5)*stride).
- G1 corner yaw sign vs course frame; stacked high gate corners too (it's
  a REAL gate — give it corners so identity can positively match it and
  keep it out of G1 updates).

## Session state (2026-07-06 end)
- Branch vq2-estimation @ be8e987; suite 35 passed + 1 xfail.
- Corpora: vq2_data/vq2_accept (creep), vq2_accept2_wp (attempt-2),
  vq2_accept5 (run-6 huber_area flight, has cmds + vq2wp_log.jsonl).
- Live stack: huber_area active (POLICY=radius rollback), RATE_MAX=0.6,
  phase z-guard, RECORD=<dir> full recording incl. cmds.
- Tools: vq2/bench.py, vq2/flight_report.py, vq2/rrd_emit.py (rerun venv).
- Laptop: vq2wp.py + eskf.py deployed at C:\Users\alexj\; detect pass =
  vq2_detect.py <frames_dir>; flights via RECORD/VMAX/NOVIZ/POLICY envs.
