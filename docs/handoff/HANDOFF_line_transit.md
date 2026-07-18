# HANDOFF — LINE_TRANSIT gate-2 campaign (2026-07-19 evening)

**Branch `vq2-estimation`, commit `1cf9408`** (vq2wp.py; deployed copy at
`C:\Users\Administrator\vq2wp.py` = repo file with alexj→Administrator
path rewrite — redeploy after any repo edit). 13 flights this session
(fly_test15..27.bat in `C:\Users\Administrator`, corpora vq2_test15..27).
G1 recipe INTACT: 12/13 ticks (one w=126 punch lottery miss, test21).

## Where it stands

**The transit problem is structurally solved; the tick is not yet.**
Post-G1, line-guided flight now reliably delivers a centered G2 terminal
shot: test23 fired a real punch AT the aperture (w=100, b=[0.08,−0.09])
and near-missed; tests 24–27 isolated the remaining wall:

**THE LAST WALL = the known ~0.3 m/s rightward push in the final 3–4 m**
(same wall as the 17-run fgpursuit campaign, now with clean telemetry on
the G2 leg): test27 released vision at w=60 → pursuit converged the hole
0.67→−0.03 rad over 3 s → then the bearing slid negative and w SHRANK
64→42 = drone slipping right past the aperture before punch width.
Punch-cone knobs don't fix it (0.10 never fires / 0.18 fires unconverged
into the frame / PN correctly refuses a non-collision course). Next
session: closed-loop push compensation in the ATT pursuit — e.g. a vy/roll
feedforward learned from the LOS-rate residual during the tracked
approach (the PN machinery already computes `_att_bd`), or the punch
steering gain on b0 raised with the PN gate as safety.

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

## Next session, in order

1. **Push compensation in the ATT pursuit terminal** (the one remaining
   wall): feedforward from LOS-rate residual, or bearing-P gain up with
   PN as the safety. Evidence base: test23 (punch near-miss),
   test24/27 (rightward slide 64→42 px), test25 (unconverged punch → imp
   6.1). Also consider velocity-mode punch for G2 (PUNCH_VYBIAS exists
   and is already −0.3 for G1).
2. If 2–3 more attempts don't tick: frame-dump review of a terminal
   approach (vq2_test23/27 frames around the release) to measure the
   actual lateral miss distance and direction.
3. After the G2 tick: generalize — the same line machinery should carry
   legs 3+ (LT_COURSE_YAW becomes per-leg; the map has the headings).
4. Estimator track (separate handoff HANDOFF_estimator_pillars.md):
   test12 outlier, Phase D productionization.
