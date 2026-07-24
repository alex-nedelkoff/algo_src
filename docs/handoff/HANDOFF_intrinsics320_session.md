# HANDOFF — intrinsics320 session (2026-07-23/24, Vagon)

Branch `feat/intrinsics320-cor148` (worktree `algo_src-mig`), all committed +
pushed to origin. Suite 211 passed / 11 skipped. Linear COR-147/COR-148 carry
the full comment trail; this file is the resume map.

## What this session settled (chronological, each flight/measurement banked)

1. **Intrinsics migration VALIDATED.** fx=320 flip + suite green; the ×1.416
   logged-range correction was a double-scale bug (removed, `340d73a1` —
   logged obs t_cam was always of-record fx=320 + RANGE_AFF affine); of-record
   GateNet G0 depth 11.18 m is 320-based and unchanged (+0.58 m vs datum is a
   rim/origin convention question, gauge-fixed in the COR-148 candidate).
2. **Flight re-validation: 4/4 gate-1 REAL TICKS** (vq2_mig2/3/4/5, gidx2,
   fresh-sim protocol) vs 4/5 baseline. mig1 abort caught the 226-scaled
   identity tolerance (rescaled to 85px@11m, `0ab910d3`).
3. **kf_pose @ frame cadence** (`55dd6210`): every recorded frame logs KF p/v
   keyed by frame sim_ns. Deployed + in-branch. This unlocked most of what
   followed.
4. **g1 DR-bridged scale anchor** (`31d1f5eb`): blind gate-1 leg has its
   metric anchor back; mig2 g1 0.373 vs gate-2 B 0.376 (agreement 0.992).
   VO scale @320 ≈ 0.37 m/unit on the fly_pos recipe.
5. **Tilt saga SETTLED after one full reversal** (`413118cb`): the spawn REALLY
   sits ≈ −18° nose-down (honest rest accel; gyro records the level-off;
   RELEVEL print consistent), cam-in-body REALLY ≈ +20° (hover-VP 18.6–21.4).
   Stack correct as built — NO tilt migration. G0 adjudication tilt-rejection
   retracted; fx=320 stands. LESSON: accel-implied pitch is only valid
   thrust-free; hover-VP + gyro integrals are the attitude oracles.
6. **G1 identity-check wrong-constellation bug FIXED** (`503c65c2`): the G1
   branch reprojected G2RIB_C (gate-2 corners). Real G1 = G2RIB_C − [5.14,
   5.26, 0] (`g1rib_corners_world.npy`, derivation fallback at load). Err
   122→40 px, 19/20 in tol; mig4 flew clean (21 obs / 0 ident_fail).
7. **Janahan pulls audited**: his frames are y-up VP-Manhattan/z-fwd/origin@
   gate-0, psi about UP (sign flips vs our NED yaw), likely different course —
   NEVER import values without a frame fixture. No Station-22 cross-validation
   possible (his stations ~22 m downcourse); duplicate station numbers
   confirmed real. His banner size [0.788×3.254 m] consistent with our
   text-block assumption. His `live_estimator.py` (GTSAM fixed-lag, runs
   live) = convergence candidate for our Phase 3-4.

## THE OPEN PROBLEM — gate-2 mid-leg reversal (top priority, judge-relevant)

Measured on mig2 (kf_pose + optical flow + frames):
- Yaw convention CORRECT (drone physically yaws right; estimator +65°
  matches; est yaw = −∫raw gyro-z, same wfix family as pitch).
- Post-tick the drone starts CORRECTLY toward g2 (+y, y +3.2 @ tick+1 s) then
  REVERSES to −y; by tick+5 s it closes on the Station-22 pillar with the
  gate sliding off-axis; ends in the fgpursuit collision every staged run
  shows. fg62 (pure FGPURSUIT, no staging) ticked g2 — never ran this path.
- NOT a mirror, NOT the map normal (tried [0.866,0.5,0] in mig5 — reverted
  untested; gate-2 true yaw unknown; see course_map_plumbing.json
  normal_note).

**Resume probe:** replay mig2/mig5 livelogs through the staged-chain state
transitions (`mapfollow` / `fgp` / `punch` / `stage` rows) and find the exact
row where the commanded carrot crosses back over the course line. Suspects in
order: STAGE-PRI handoff, MF carrot sequencing, staged-entry waypoint
(g2 − MF_STAGE_BACK·N2) under the pillar-occluded gate view. kf_pose gives
frame-cadence ground truth for the divergence instant.

## Deploy state (flight kit C:\Users\Administrator\)

Trunk-campaign vq2wp lineage + surgical fixes: camera.py fx=320, identity
check 320+85px tol + G1RIB_C, kf_pose logging. Backup of pre-session files:
`deploy_backup_20260723\`. course_map_plumbing.json G2 normal back at
[1,0,0] with the audit note. fly_mig1..5.bat = fly_pos recipe clones.

## Other open threads

- OBSZ=0 root cause (tilt explanation retracted — open again).
- COR-148 Phase B promotion path: axis work continues on the Codex agent's
  `feat/pillar-axis-adjudication-cor148` (worktree `algo_src-axis`, based at
  `8ff8edc2` — 6+ commits behind this branch, unpushed; no tilt assumptions,
  safe, but should sync before its next round). Extrinsics for any
  constants-based Phase B math: cam-in-body ≈ +20°, spawn ≈ −18°.
- COR-147 Phase 3-4 (OdomFactor → smoother, VO_SM=1) vs converging on
  Janahan's live_estimator — discuss with Janahan first.
- Local `feat/vo-loop` is 12 commits ahead of its origin (unpushed merges);
  vo-loop/vo-loop-pest/g0-calib-adjudication are fully contained in this
  branch (archaeology only). Merging this branch → trunk `vq2-estimation` is
  the eventual integration step (Alex's call; trunk worktree has uncommitted
  campaign work — never touch it).
