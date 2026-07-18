# HANDOFF — LINE_TRANSIT gate-2 campaign (2026-07-19, GATE 2 TICKED)

**Branch `vq2-estimation`, commits `1cf9408` + `128bd52`** (vq2wp.py;
deployed copy at `C:\Users\Administrator\vq2wp.py` = repo file with
alexj→Administrator path rewrite — redeploy after any repo edit).
33 flights this session (fly_test15..47.bat, corpora vq2_test15..47).
G1 recipe: 17/19 post-tick-capable flights ticked G1.

## ✅ GATE 2 TICKED — fly_test46, corpus vq2_test46

**TICKS=2, LAND: mission complete.** G1 @5.9 s, G2 @16.2 s race clock.
The G2 crossing was flown THROUGH under continuous pursuit steering
(hole converged 0.19→−0.04 rad while w grew 40→173; no punch fired, no
impact). vq2_test46 is the first-ever 2-tick corpus — bank it for the
estimator work (its transit has tick-truth at BOTH ends).

**Repeatability: 1/2** on the final config (test47 missed G2). That is
the next campaign — the architecture is right, the variance is in the
sweep/settle timing.

## THE decisive discovery (test45 frames — look at them)

**The fastgate hole detector reads LETTERFORMS on the orange sponsor
banner as apertures.** test45 punched a dead-center "w=113 hole" that
was the negative space of a letter (see vq2_test45 frames ~740-802: the
camera fills with giant white glyphs, then solid orange). The banner
stands adjacent to G2, so map-bearing and vertical checks CANNOT
separate them. Half the campaign's "inexplicable slides" were terminal
locks onto typography.

**Fix = LINE-MATCH:** the cyan racing line passes through the SCORING
aperture only. A terminal candidate must sit where the line's far head
points (`|line_head_off * LT_HEAD2RAD − b0| < LT_LINE_MATCH`, defaults
1.75/0.28). This gate turned the very next flight into the tick.

## The LINE_TRANSIT machinery (all env-gated, in vq2wp.py)

Phase logic per control cycle (post-tick, MAPFOLLOW):
1. **Line fresh + course-aligned → line-follow**: yaw at the LINE HEAD
   (mean x of the topmost 30% of detected line pixels — v1's centroid
   yaw parked the drone ON the line), roll centers the centroid
   (LT_KYAW 0.9, LT_KLAT 0.25, LT_PITCH 0.06). Alignment = gyro yaw
   within LT_DIR_GATE (1.75 rad) of LT_COURSE_YAW (0.77 = judge-anchor
   G1→G2 heading): the line has no arrow; after a scan the drone WILL
   lock it backwards (test18 followed it west). Do NOT use DR position
   for this gate — v5 did and it misfired mid-curve (test19).
2. **Line stale 1–10 s (or fresh-but-misaligned) → yaw-scan** toward the
   course heading, pb 0.03, LINE_LOST_DESCEND sinking. Reacquired in
   test18 after 6 s.
3. **Line stale >10 s → the PROVEN teardrop staging** (fg62 family) owns.
4. **Vision release while line fresh**: w ≥ LT_VIS_W AND |b0| < LT_VIS_B
   AND |b1| < LT_VIS_BV (vertical plausibility — REJECTS the non-gate
   arch the line passes under ~4 s post-G1: b1 → −0.72 at close range;
   both test16 AND test17 chased it and died before this gate existed).
   Line stale → original staging discipline (stage_done + 45/0.6).
5. **PN punch gate** (E1 re-applied, `PN_PUNCH=1`): punch additionally
   requires |LOS-rate| < PN_LOSRATE (0.12) on both axes.
6. **Terminal climb** (`LT_CLIMB` 0.035): candidate hole moderately above
   axis (−0.45 < b1 < −0.05, centered, w≥60) → climb bias (approach from
   below keeps the hole in the 20°-up camera; the arch's b1 −0.72 is
   excluded).

Config du jour (fly_test27.bat): fg62 base + E3 (LINE_ROW_REF **0.80**
— 0.72 flew into floor clutter, test22; 0.80 never crashed in
line-follow) + LINE_MIN_PX 80 (150 flickered at the thin-strip count) +
LT_VIS_W 60 / LT_VIS_B 0.25 / LT_VIS_BV 0.2 + FGP_PUNCH_CTR 0.18 +
PN_PUNCH 1.

## Hard-won facts (do not re-learn)

- **The blind est/DR carrot cannot fly the transit**: test16 walked a
  70 s phantom orbit around its imagined gate; DR under-rotates real
  turns. The line was the fix, as the 07-17 postmortem predicted.
- **Line extinction near the gate is STRUCTURAL**: at aperture height
  the floor line ahead hides below the 20°-up camera (row→0.95+, count
  →threshold). Vision must own the terminal; the scan + staging fallback
  covers the handoff gap.
- **A fast-growing centered hole ≠ the gate**: the line passes UNDER an
  arch ~4 s post-G1. Vertical bearing plausibility (|b1| gate) is the
  discriminator that finally let flights survive past it (test18+).
- Restart protocol works headless: scratchpad `sim_restart.py`
  (EnumWindows 'AI-GP', one key per call, screenshot each step). Spawn
  verified −17.8° via simprobe before the campaign.
- The judge tick lags the physical crossing ~0.75 s (measured on the
  estimator side, test10) — G2 tick timing analysis must account.

## The terminal stack that ticked (tests 28–46, all measured; env in
fly_test46.bat)

1. **Ownership-gated push integral** — the `_att_ri` trim adapter used
   to integrate the bearing of ANY fresh hole; during line-follow it ate
   the +0.6–0.8 rad bearings of holes the line passes by, wound to
   +0.18 FULL SCALE in 1.5 s, and dragged every hover sideways (test37/38
   ib/ri traces). Now integrates only while the pursuit owns control,
   and resets per leg (the "global trim" assumption is wrong across
   legs). NOTE: a duplicate integral briefly existed keyed on the same
   ATT_KI env — removed; `_att_ri` is the single push integral.
2. **Terminal latch + 2-detection debounce** (LT_TERM_S 6 s): once a
   candidate passes the strict release TWICE within 1.2 s, the pursuit
   owns terminal — no line-steal, no threshold flapping. Single noisy
   blips must not release: test42 beelined off the line's safe corridor
   mid-transit and hit scenery.
3. **Terminal brake at the release width** (LT_BRAKE_W 58, just under
   LT_VIS_W): line pitch → 0 when the candidate appears; braking earlier
   (45) stalls OUTSIDE release and the drone hovers uselessly (test39).
4. **Map-consistency** (LT_MAP_B 0.5): candidate bearing must agree with
   the DR+gyro bearing to the gate's map position.
5. **LINE-MATCH** (see above) — the one that ticked it.
6. **Below-height approach preserved**: asymmetric punch vertical cone
   (FGP_PUNCH_BVUP 0.32 above), per-leg punch width (FGP_PUNCH_W2 85)
   and pitch (ATT_PUNCH_PITCH2 −0.12). G1 keeps FGP_PUNCH_W 95 /
   ATT_PUNCH_PITCH −0.18 — softening the shared knobs broke G1 twice
   (test34 double-punch, test44 velocity runaway).
7. PN collision-course gate on the LATERAL axis (PN_LOSRATE 0.12);
   FGP_PUNCH_KD 0.30 D-term in punch steering.

## Next session, in order

1. **Repeatability campaign**: fly the test46 config 5×. Variance lives
   in the sweep/settle timing (test47 missed). Candidate lever: extend
   LT_TERM_S, or a second latch window after a missed first pass (the
   drone survives misses now — timeouts, not crashes).
2. Then generalize to legs 3+ (LT_COURSE_YAW per-leg from the map;
   TICKS=3+; the line machinery is leg-agnostic).
3. Estimator track (HANDOFF_estimator_pillars.md): test12 outlier,
   Phase D — and vq2_test46 is the first corpus with tick-truth at BOTH
   transit ends: re-run the smoother validation on it.
4. Housekeeping: 33 corpora now on disk (vq2_test15..47, ~several GB);
   prune the failed-run frame dirs if disk pressure returns.
