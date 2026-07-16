# HANDOFF — VQ2 gate-1 tick campaign, fastgate/pursuit state (2026-07-14, ~30 runs)

Read `C:\Users\alexj\algo_src\CLAUDE.md` first (mission, judge laws, sim protocol).
This file is the delta: everything established on 07-13/14. Trust it; don't re-derive.
Auto-memory (`~/.claude/projects/C--Users-alexj-algo-src/memory/`) has the same facts
in note form with evidence pointers.

## THE FIRST ACTION
Fly `C:\Users\alexj\fly_servo_fg15.bat` (staged, ready) on a fresh sim window.
It is the **ribbon-hypothesis tick test**: gate-to-gate pursuit that crosses the
chute gate AND the stacked/ribbon gate. Score ONLY with
`python gidx2.py vq2_servo_fg15`. If it ticks, the campaign's gate-identity
question is answered and TICKS=2 is within reach the same day.

## ESTABLISHED FACTS (all measured this session — do not relitigate)
1. **Roll/pitch TRIM was the "10 m drift"**: the windowed force-balance trim
   misreads maneuver accel as attitude error and injects ±2° false roll.
   `NOTRIM=1` (always on now): approach drift 10.06 m → 0.46 m. (vq2_blind1 vs _nt)
2. **Sim gyro reports EULER-ANGLE RATES, not body rates.** YAWPROBE (415° hover
   spin): simple integrator exact (0.2° err); coupling terms (my EULERFIX) inject
   tan(pitch)-scaled phantom roll → fg14 velocity runaway. **EULERFIX=0 always.**
   `YAWPROBE=1` mode exists in vq2wp for attitude-model checks.
3. **A REAL, IMU-invisible rightward push (~0.3 m/s, variable 0.8–2.2 m effect)
   acts in the course/gate area** (07-10 "inner assist" was right). Plus a
   takeoff est-truth gap ~+2 m right (accel under-reads under thrust). The
   historical `AIMBIAS_Y=-2` compensated these, not a judge quirk. No feed-forward
   beats the variance → only closed-loop vision through the crossing works.
4. **Pad acquisition = spawn-pitch lottery, not GateNet.** Fresh spawn is −17.8°
   nose-down (gate visible); a poisoned restart gives 0.0° (20°-up camera misses
   the below-pad gate). ALWAYS verify with `python spawn_att_probe.py`
   (expect "NOSE-DOWN") before a scored launch.
5. **GateNet is ~450 ms / 2 Hz** (VRAM-starved) and drops the gate <4 m. It's fine
   for the at-rest pad lock; useless for terminal control. DPVO: unusable for
   gate 1 (freezes on hover, diverges on the dive; scale ~metric but tracking
   fails; point cloud outlier-dominated) — reserve for continuous-motion legs
   only, or not at all.
6. **Judge scoring:** gidx2.py ONLY, fresh sim only, frames beat telemetry.
   ESC→pause menu→**keyboard DOWN+ENTER** (clicks don't register) → RESTART gives
   a valid fresh window. App relaunch → login (ENTER) → event menu → R2-TRAINING
   (keyboard DOWN×2+ENTER) if the session is fully stale/logged out.

## FASTGATE (the detector — done, hardened)
`C:\Users\alexj\fastgate.py` — classical CV: orange frame + dark hole → the ACTUAL
aperture-hole center. **100% detection, ~1.8 ms, 30 Hz** (vs GateNet 450 ms).
Hardening (each lesson cost a flight):
- Color mask keys on `r−g>25, r−b>−30, r>120` — the close-range core is washed
  pink-white with BLUE contamination from the cyan trace glow.
- 9×9 morph close: white text/checker bands on the gate face break the orange
  ring → hole un-enclosed. 1×25 horizontal close bridges the start-light pole.
- Frame-centroid fallback: big orange blob with no enclosed hole → bbox center.
- Aspect band 0.15–3.5 (oblique square reads ~0.3).
- `clipped` flag when hole touches frame border (bearing biased — don't steer on it).
- Range from hole width is glow-corrupted (over-reads) — use bearings + hole SIZE
  as proximity cue, never fastgate range as metric truth.

## FGPURSUIT (the controller — in vq2wp.py, env FASTGATE=1 + FGPURSUIT=1)
Pure pursuit on the live hole bearing. No spline, no DR steering, immune to the
est-truth gap. Association (in `fastgate_loop`):
- **Acquisition by size**: only holes ≥ `FG_MIN_W` (40 px = close gates) can start
  a track (most-centered acquired a 9 px far bay gate once — never again).
- **Temporal continuity** ≤0.15 rad/frame keeps the SAME hole (biggest-hole
  re-picks switched targets when the hole clipped).
- Clipped detections rejected for steering; fallback holds last correction decayed.
- **Gate-to-gate**: hole grows ≥120 px then vanishes → punch straight 1.6 s → drop
  track → re-acquire next biggest close hole → the course flies itself in
  proximity order. Turn-to-target: `yr = 1.4·bear_y` (proven: 0.49→0.04 rad in 3
  samples) + small strafe assist + `FGP_PITCH` nose-down tilt (−0.15 inside 4.5 m;
  rotates the 20°-up cam down → hole in view to ~1.5 m; pitching forward IS the punch).

## THE RIBBON HYPOTHESIS (the open question that likely explains everything)
The cyan racing-line trace dives through the **stacked/ribbon gate (~x 11.5,
y 5.3)**, and the campaign's only historical ticks (arch17/23/27) happened THERE —
never at the chute. ~30 well-placed chute crossings this session = 0 ticks.
**The judge's gate 1 may be the ribbon, not the chute.** fg13 tried to reach it
but heading-locked strafing can't fly the 40°-off-axis dogleg; fg14's turn-to-
target can, and died only to the (now-fixed) EULERFIX runaway. fg15 = the test.

## KNOWN RESIDUALS
- Under NOFIX the est velocity slow-drifts (~4% accel under-read) → the |v|>8
  guard can fire on long flights (>20-30 s). Pursuit steering is vision-relative
  and immune; only `_fwd_dr` backstop (`FGP_PLANE_X=13`) and the tilt ramp touch
  the est. If the guard kills healthy pursuits, consider hole-size-based tilt
  ramp and dropping the DR backstop.
- ~22 un-killable zombie python.exe processes (access denied from both shells)
  linger from 07-14; they haven't blocked flights but a reboot clears them.
- Rerun streaming to the Mac (100.101.13.126) DEADLOCKS the process when the
  tailnet is down — use `NOVIZ=1` or `RRD=<path>` (local .rrd file, safe), never
  bare default. `.rrd` recordings of runs: vq2_servo_fg11–14.rrd.
- Debug printfs still in `C:\Users\alexj\DPVO\dpvo\lietorch\src\lietorch_gpu.cu`.

## ENV KNOBS (vq2wp.py, all added 07-13/14)
NOTRIM, EULERFIX(=0!), YAWPROBE, FASTGATE, FGPURSUIT, FG_MIN_W(40), FG_ACQ_W(40),
FGP_PITCH(-0.15), FGP_PLANE_X(13), RECENTER_Y, STRAIGHTCHUTE, G1_IDENT_R(2.5),
PUNCH_TRIG/PUNCH_WIN/PUNCH_LAT/PUNCH_STRAIGHT/PUNCH_VX/PUNCH_VYBIAS,
APPROACH_VYBIAS, AIMBIAS_Y/Z, GNSCALE(dead end), DPVO_BRIDGE(dead end for gate 1),
RRD(path→local .rrd), NOVIZ.

## FILES
Flight: `vq2wp.py`, `fastgate.py`, `fly_servo_fg15.bat` (NEXT), fly_servo_fg1-14.bat,
fly_blind_nt*.bat (blind-mode configs). Probes: `spawn_att_probe.py` (pitch),
`settle_probe.py`, YAWPROBE mode. Analysis: `gidx2.py` (scoring), `crash.py`,
`det_latency.py`, `imu_gap_analysis.py`, `scale_drift.py`, `recon_vs_dr.py`,
`gate_in_cloud.py`, `cloud_viz.py`, `detect_pillars.py` (station-number landmark
prototype — the COURSE-completion path: pillar map + EKF, see memory
vq2-fastgate/dpvo-tracking notes). Sim control: `simctl.ps1` (focus/keys/click/
shot via simctl_cmd.txt), `simlaunch.ps1`. Corpora: vq2_servo_fg*/, vq2_blind_nt*/.

## PER-ATTEMPT PROTOCOL (unchanged + additions)
1. ESC → DOWN → ENTER restart (verify pad screenshot if unsure).
2. `python spawn_att_probe.py` → must read NOSE-DOWN −17.8°.
3. Launch the .bat via `cmd /c "...bat > log 2>&1"` (background), watch the log.
4. Score with gidx2 ONLY. Pull crossing FRAMES for any surprising result —
   frames overturned the telemetry story four separate times this session.
