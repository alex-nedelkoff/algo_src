---
name: vq2-fastgate
description: "★★★ TICKS=2 MISSION COMPLETE 07-15 (fg62, gidx2: 0→1 @17.55s, 1→2 @22.06s). Recipe fly_servo_fg62.bat: vision-only ATTMODE pursuit + punch-while-locked per gate, kinematic DR (cmd-speed × gyro-yaw, MF_DR=1) + staging waypoint between gates, tick-as-anchor, est zero-trust. Roll-mirror fix underneath it all. Next: consistency ×3, then R2-SUBMISSION."
metadata: 
  node_type: memory
  type: project
  originSessionId: ac528723-aa01-44f3-ad7b-c8f642be8516
---

**FASTGATE built & validated 2026-07-14** (C:\Users\alexj\fastgate.py + fastgate_loop
/ FGPURSUIT in vq2wp.py, env FASTGATE=1 / FGPURSUIT=1, runs fg1-fg8):
- Classical CV: orange-frame + dark-hole contour -> tracks the ACTUAL aperture hole
  center (no PnP-origin offset). **100% detection, 1.8 ms median (250x faster than
  GateNet's 450 ms), 30 Hz.** Color mask handles the washed pink-white close-range
  core (cyan trace glow contaminates blue: key on r-g>25, allow b to r+30).
- FGPURSUIT: pure pursuit on the live hole bearing, replaces route/approach/punch.
  When tracking, it PINS the bearing to ~0.01 rad through the whole approach.
- Association lessons (each cost a run): acquisition MUST be by size (hole >= 40 px
  = the chute gate at ~6 m; bay gates <= 20 px — 'most-centered' grabbed a 9 px far
  gate and tracked it perfectly to nowhere); tracking MUST be temporal continuity
  (<= 0.15 rad/frame, biggest-hole re-picks switch targets when the hole clips);
  NEVER steer on clipped holes (remnant center biased toward visible side); DR
  range lies (over-integrates ~2 m) — use hole SIZE as range cue (w>=120 px ~ <4 m).

Terminal FOV: NOSE-DOWN TILT implemented (FGP_PITCH env, ramps to ~-9 deg inside
4.5 m; rotates the 20-up cam down -> hole in frame to ~1.5 m; pitch-forward IS the
punch). fg9/fg10 detection fixes: aspect band 0.15-3.5 (oblique square reads ~0.3),
9x9 morph close (white text/checker bands break the orange ring -> hole un-enclosed).

Pole occlusion FIXED (fg12/13): frame-centroid fallback (big orange blob with no
enclosed hole -> bbox center) + horizontal 1x25 morph close bridges the thin pole.
Detection now survives text bands, oblique views, glow, and the pole.

**★★ 07-15 10:45 — GATE-1 TICK CONSISTENT: 3/3 (fg45/46/47), ROLL MIRROR
FIXED.** The 25-day root cause: est roll was −(true roll). Roll was the only
axis integrating the RAW gyro (`+gyr[0]`; pitch/yaw use the negated "wfix")
— unobservable at rest (spawn roll=0), and SIGN_R=-1 had been tuned to make
the MIRRORED loop converge, locking it in. Proof: fg44 frames show the
horizon banked opposite est roll (-0.10/-0.18). It caused: the "IMU-
invisible rightward push" family, the est-y truth gap, and attmode's
positive-feedback divergence (correcting left banked right). FIX:
`state['roll'] += (-gyr[0])*dt` + `SIGN_R=+1` (both together or the rate
loop destabilizes). Recipe fly_servo_fg45.bat = fg36 config + fix; ticks at
clock 20.0-20.5 s, 3 consecutive fresh-window runs.
**NEW JUDGE LAW: the sim AUTO-RESTARTS the race seconds after a wreck** →
corpora can hold multiple resets; gidx2 upgraded to per-segment scoring
(was keying on the LAST reset and silently dropping valid first segments).
NEXT: gate 2 — post-tick FGPURSUIT handoff works; the leg dies on the
FGP_PLANE_X=25 backstop / terminal collisions; iterate like gate 1.

**07-15 late — MAPFOLLOW built (fg48-50), gate-1 now 6/6 consecutive:**
st2 probe: post-roll-fix DR lateral drift ~0.06 m/s (was 0.2-0.3) — transit-
navigable; ~1 m static est-y takeoff offset remains. MAPFOLLOW=1 (vq2wp):
pose carrot toward next_gate_w in ATTMODE blind stretches, pursuit auto-
takeover on acquisition; distance runaway bound replaces the x-plane overrun
(dogleg exits the chute already past gate-2 x); tick-as-FULL-fix (position:=
ticked gate, velocity:=2 m/s along heading — est goes to fantasy within
seconds of a punch otherwise). fg50: re-anchor + carrot + re-acquisition all
worked; the OBLIQUE chase (ribbon approached 40-60 deg off-normal: w stuck
40-67, bearing walks right, vy rails, fast maneuvers re-bank est error)
remains the gate-2 blocker. **TICKS=2 status (fg52-57): gate-1 13/13; the
gate-2 staging leg is blocked by POST-PUNCH EST DIVERGENCE** — tried tick
anchor, post-dash re-anchor, brake+accel-relevel, quiet-gated relevel: the
accel never settles post-punch (no quiet hover -> no attitude re-zero -> no
trustworthy velocity -> no closed-loop brake; chicken-and-egg). **★★★ 07-15 14:41 — TICKS=2 MISSION COMPLETE (fg62): gidx2 REAL TICKS: 2
(0->1 @ 17.55 s chute, 1->2 @ 22.06 s ribbon), fresh window, controlled
mission-complete landing.** Full chain (fly_servo_fg62.bat): gate-1
vision punch -> 0.4 s dash -> KINEMATIC DR seed (cmd-speed x gyro-yaw,
MF_DR=1, no KF/accel) -> direct corridor to the staging point on the
ribbon normal (geometry-aware: direct if west of staging plane, teardrop
otherwise) -> vision released head-on -> fastgate punch -> tick 2.
Final unlocks: est-velocity guard disabled under MF_DR (est zero-trust);
waypoints geometry-aware (stale teardrop hit the pillar in fg60).
lietorch: clean 14.38 rebuild still AVs (exit 5) — parked, not needed.
NEXT: consistency x3 on fg62.bat; scout R2-SUBMISSION protocol; commit
review. (Superseded plan below kept for history.)
**NEXT SESSION: finish MF_DR (3 edits listed in the 07-15 13:55 outbox
entry: _mf_lastpb setter, tick-handler _mf_p seed, overrun check on _mf_p)
and fly fg59; separately re-do the lietorch rebuild AFTER deleting
DPVO\build (the 14.38 rebuild reused stale 14.44 objects — SE3 CUDA still
dies, silent exit 5).** Original idea kept for reference:
yaw+time choreographed teardrop — NO est position.
Gyro yaw is short-term exact (YAWPROBE 0.2 deg/415 deg); fly: pitch -0.06,
yr +0.3 until yaw~200 deg, straighten to yaw~20 deg -> head-on ribbon
corridor -> fastgate acquires -> the proven punch. Est stays guards-only.
STAGING WAYPOINT BUILT (fg52/53): carrot routes
via [gate−3m along +x], vision/punch suppressed until staged; fg52 STAGED
successfully (dogleg cleared the pillar). Gate-1: 9 consecutive ticks
(fg45-53). LAST KNOB: the post-tick punch+dash exits the chute at x≈12.5,
PAST staging x 8.4 -> approach from the wrong side; brake-turn-in-place
(fg53) destabilizes the est (rotation across IMU gaps). FIX QUEUED: shorten
the post-tick clearance dash (1.2 s @ 1.5 -> ~0.3 s) so staging stays
AHEAD; then TICKS=2 = transit -> stage -> head-on punch. Config:
fly_servo_fg53.bat; GN_UNLOAD=1 frees the GPU (DPVO-ready); MF_PITCH/
MF_YRMAX/MF_STAGE_BACK knobs.

**★ 07-15 09:46 — FIRST JUDGE TICK (fg36): gidx2 TRANSITION idx 0->1 @ clock
22.56 s, fresh window.** The CHUTE ticked. Recipe: ATTMODE=1 pure-attitude
pursuit (bearing PID P0.6/D0.25/I0.5 -> roll_bias + throttle-delta directly in
level_cmd(att=True); fixed ATT_PITCH -0.08 for speed; vision range-rate
governor ATT_VMAX 1.6; NO est-velocity feedback) + punch-while-locked at w>=95
with centering tightened to FGP_PUNCH_CTR=0.10 + RELEVEL=1, RECENTER=0,
YAWCAL=0 (integrator self-trims), CLIMB_S=0.9, RECENTER_Z=-0.4. Config:
fly_servo_fg36.bat. fg35 (CTR 0.18) crossed IN-aperture without ticking ->
the judge wants a genuinely CENTERED crossing; every historical "well-placed"
chute crossing actually missed (frame forensics). RIBBON HYPOTHESIS DEAD.
REMAINING FOR TICKS=2: the post-tick gate-2 leg now re-enters FGPURSUIT
(implemented, untested by a ticking run).
**CONSISTENCY (fg37-43): 1 tick in 8 attmode runs — NOT yet repeatable.**
Instrumented failure signature (att jlog: rb/ri/pb/rrate/r_est/p_est):
approach starts clean, bear walks LEFT while roll ramps to the -0.22 clamp,
and fg43 PROVED the commanded roll is achieved (r_est tracks rb to -0.196) --
an 11-deg real left tilt failed to stop the walk. Prime suspect:
UNCOMMANDED RIGHTWARD YAW (the 07-12 "swivel") -- body-azimuth bearing can't
distinguish right-yaw from right-slide; explains hover-clean, maneuver-only,
tilt-immune. NEXT: add yaw+gyro-z to the att jlog, one run; if yaw walks
right under a left yr command, control yaw from vision (hold hole azimuth by
yaw; drop turn-to-target in final approach). Fixes landed on the way:
YAWCAL cal-loop pitch bias (fg38 n=0), est-vel guard 20 m/s in ATTMODE
pursuit/punch (false kills fg38/39), governor heavy-EMA + w>=55 (fg40 noise
braking), blind coast levels out (held tilt = held acceleration, fg40).

**07-15 SESSION (fg15-fg19): pursuit-approach failure chain isolated, 3 fixes in.**
Ribbon hypothesis STILL untested — every run died before a clean crossing:
- fg15: crossed 0.3 m OVER the chute top (est-z lottery), re-acquired the ribbon
  gate correctly (0.75 rad right, converging) but FGP_PLANE_X=13 backstop killed
  the live pursuit (est overran truth ~5 m). -> FGP_PLANE_X=25 since fg16.
- fg16: approached BELOW the aperture, punched at w=128 with hole 0.6 rad above
  center -> hit the frame. -> PUNCH-CENTERING GATE in vq2wp (FGP_PUNCH_CTR=0.18:
  _fg_close only arms when |bear| < 0.18 both axes).
- fg17: THE BIG ONE (acquisition frame): post-RECENTER the drone sits 2.5-3.5 m
  up (est-z under-reads the climb ~1-1.5 m: est -1.33 when truth ~3) and from
  there the 20-up camera CANNOT see near gates at all (~40 deg below axis) while
  FAR bay gates (w exactly 40-48 px) stay visible -> far-gate acquisition,
  wasted run. -> DESCEND-UNTIL-ACQUIRE in vq2wp (pre-first-track: vx 0.3,
  vz -0.35 FGP_SEEK_VZ, floor guard est z > -0.5).
- fg18: chute tracked continuously from pad, turn-to-target converging, then hit
  the RIGHT START-LIGHT POLE at light-box height at the recenter->pursuit
  handoff: RECENTER_Y=-1.5 parks the drone in line with the right pole (fg15
  only survived it by flying higher). ANY contact instantly garbages the NOFIX
  est (est x +27 m in 2 s -> bogus 'deep DR overrun' brake). -> RECENTER_Y=0
  (fg19+): the pad-gate centerline is pole-clear.
- Sim-side: transient 0xC0000409 during GateNet load happens (~1/5 launches);
  just relaunch, no scene poison if the drone never armed. bear[1] convention
  verified: (lat/fwd, dwn/fwd), positive vz = UP in level_cmd — vertical loop
  sign is CORRECT; the failures were geometry/est, not sign.

**07-15 LATE (fg20-fg29): approach SOLVED, terminal push is the last wall.**
The pursuit now reliably delivers a centered, collision-free approach to punch
range: hole-width range scale (290/w, fixes vy saturation from the deep
FGP_PLANE_X), RECENTER_Z=-0.4 (aperture-height start; RECENTER=0 even better —
skips the pole-lottery blind leg), through-hole alias block (w<0.55x last
rejected), bearing INTEGRATOR on vy (nulls constant-bearing drift, replaces the
wrong-signed APPROACH_VYBIAS feed-forward in this branch), tilt + punch keyed
on hole width (punch-while-locked at w>=95+centered — the hole clips at
w 95-108, below the old 120 vanish trigger).
- REMAINING WALL, measured 5x (fg23/26/27/28/29 frames): in the final ~3 m /
  1.6-2.5 s the drone displaces 0.5-0.9 m RIGHT and exits beside the right
  post — with straight punch, coast, AND a bearing-steered punch (last
  bearings go stale once the hole clips). This is the vq2-terminal-push
  (~0.3-0.5 m/s) owning whatever segment flies without fresh vision.
- NEXT EXPERIMENT (below-height approach, the 07-14 memory idea, now
  quantified): approach at truth ~0.5-1 m (RECENTER_Z ~ -0.2 est or
  descend-until-acquire deeper) so the 20-up camera keeps the hole visible to
  ~1 m; steer laterally through the crossing; pop up only at the plane.
- Judge note: 4 more well-placed near-chute passes (2 arguably grazing the
  aperture edge region) = 0 ticks on gidx2-verified fresh windows.

**TWO ITEMS FOR NEXT SESSION:**
1. **RIBBON-GATE HYPOTHESIS (potentially the whole answer):** fg12 frames show the
   cyan racing line diving through the STACKED/ribbon gate at ~x 11.5 y 5.3 -- and
   the G2FULL note records arch17/23/27 all ticked THERE, never at the chute. ~30
   perfect-ish chute crossings today = 0 ticks. The judge's gate 1 may be the ribbon.
   fg13 tried to cross it (FGP_PLANE_X=13) but couldn't reach it laterally (below).
2. **TURN-TO-TARGET + GATE-TO-GATE BUILT (fg14), needs one debug:** yaw-unlock
   (yr=1.4*bear_y), small strafe assist, punch-through-then-drop-track (next gate =
   next biggest close hole -> the course flies itself in proximity order), and
   Alex's FG_MIN_W=40 close-gates-only filter -- all in vq2wp fgpursuit. fg14
   PROVED turn-to-target works (corrected bearing 0.49->0.04 in 3 samples) then
   died to velocity runaway under sustained yaw. **YAW ISSUE RESOLVED (YAWPROBE,
   07-14 late): the sim's gyro reports EULER-ANGLE RATES, not FRD body rates.**
   415-deg hover spin: simple integrator err 0.2 deg (exact); EULERFIX coupling
   terms err 0.6 deg at hover and ~5 deg/s phantom roll at fg14's 9-deg tilt +
   0.56 rad/s yaw (tan(pitch)-scaled) -> the runaway. **FIX: EULERFIX=0 always**
   (flag kept, default off; it was my 07-14 addition -- the campaign's simple
   integrator was right all along). YAWPROBE=1 mode added to vq2wp for future
   attitude-model checks. fly_servo_fg15.bat staged = fg14 config minus EULERFIX
   -> NEXT SESSION: fly fg15 = gate-to-gate pursuit + ribbon-hypothesis tick test.
   (Residual: under NOFIX the est velocity still slow-drifts from the ~4% accel
   under-read -> the 8 m/s guard can fire on long flights; pursuit steering is
   vision-relative and immune, only the DR backstop/tilt-ramp touch the est.)
Related: [[vq2-terminal-push]], [[vq2-perception-latency]], [[vq2-pad-acquisition]].
