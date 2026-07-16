# Handoff — VQ2 qualifier racing: fly gates via spline + vision (2026-07-05, 43 flights)

## Mission
Score judge ticks through gate 1 and gate 2 of the VQ2 qualifier with `vq2wp.py`
(spline route + GateNet vision servo). Flight stack works end-to-end; the ONE
remaining blocker is state estimation on the blind final ~6 m (see "The blocker").
**Next step decided with Alex: build VIO offline against the recorded corpora —
do NOT resume live patch-per-flight iteration.**

## READ FIRST
- Obsidian session log: `02 Areas/Claude Code/2026-07-05 VQ2 spline waypoint racer.md`
  (full flight-by-flight forensics; this file is the distilled version).
- Auto-memory: `project-vq2-judge-tick-blocker.md` (same conclusions, short).
- Predecessor doc: `HANDOFF_interface_latency.md` (VQ1-era; env/ssh facts still valid).

## Environments (deltas vs the old handoff)
- Laptop (`ssh laptop`, Tailscale): sim = **AI-GP Simulator v1.0.3379**,
  `C:\Users\alexj\Documents\AI-GP Simulator v1.0.3379\` — contains the OFFICIAL
  `README.md` + `PyAIPilotExample-v2` (protocol reference; read them).
  Launch: `FlightSim.exe` (GUI login required for the qualifier!).
- Python for racing: `C:\Users\alexj\miniconda3\envs\monorace\python.exe`
  (torch 2.5.1+cu121, transformers **5.10.2** — required by the 5.x-layout
  `gatenet_b2_cov.pt`; vq2wp.py shims the missing `torch.float8_*` dtypes).
  rerun-sdk 0.33.0 installed (matches Mac viewer 0.33).
- Deployed script: `C:\Users\alexj\vq2wp.py` (repo copy: `vq2/live/vq2wp.py`).
  Run: `ssh laptop "C:\...\monorace\python.exe C:\Users\alexj\vq2wp.py"`.
  Logs: `C:\Users\alexj\vq2_wp\{log.jsonl, frames/}`.
- Rerun dashboard: start `~/.rerun33-venv/bin/rerun` on the Mac; vq2wp streams to
  `rerun+http://100.101.13.126:9876/proxy` (3D actual-vs-ref path, pose, gate
  boxes, carrot, FPV, alt/gate_idx plots). `NOVIZ=1` disables.

## Protocol facts (hard-won, all verified)
- **RACE_STATUS** = ENCAPSULATED_DATA type 1, `<BQqqIq>` =
  (dtype, sim_boot_ms, race_start_boot_ms, race_finish_ns, active_gate_index,
  last_gate_race_time). Judge counts gates IN ORDER via active_gate_index.
- **Hard race reset = `command_long 31000 param1=1`** → sim_boot_ms resets,
  3-2-1 countdown, race re-arms. `param1=0` = soft reset (respawn only, race
  NOT re-armed — this cost ~25 flights before it was found).
- Qualifier requires the sim GUI **login**; without it the judge is inert.
- VQ2 sends NO track broadcast, NO ODOMETRY/ATTITUDE/LOCAL_POSITION_NED
  (vision-only by design; the type-2 track protocol in `aigp/protocol.py` is
  VQ1 legacy). 144 Hz HIGHRES_IMU + 30 fps UDP JPEG (port 5600) is all you get.
- Collision ids: **1001 = gate structure, 1002 = environment** (official).
- Sending commands during the countdown can DQ (official example comment).

## Course facts (VQ2 qualifier, spawn frame: x downcourse, y right, z down)
- Red gate 1 at ~[11.0, 0, −1.3] (banner center; VQ1-size aperture ~1.5 m).
  GateNet PnP ranges are TRUE (gate dims match VQ1 — do not rescale).
- G2 ~[30.5, 8.5, −1.5] (from the June tick-flight reacquire).
- A second gate is stacked HIGH above gate 1 (~[10.4, 0, −4.0], repeatable
  GateNet detections); course ribbon drops vertically from ceiling near it.
- Start-light poles flank the line at x≈4-5, y≈±1.5.
- **No obstacle between pad and gate 1** (Alex free-cam verified).

## The blocker (why gates still don't tick)
1. **KF under-integrates: distances compress ~0.6×** (drone travels ~1.6× farther
   than KF believes; same signature as the June survey's "bled odometry").
   Vision position fixes re-anchor it while locked, and vision-velocity fixes
   (static gate ⇒ v = −d(g_w)/dt, implemented) hold v true while locked, BUT:
2. **GateNet loses the gate ~6 m out** (gate slides below the 20°-up camera;
   worse above ~1.5 m/s). The final ~6 m are always blind, flown on the
   compressed KF → every pass misses/clips the aperture. 43 flights confirm.
- Root-cause candidates for the compression: accel_level attitude leak,
  IMU dt handling, thrust-lag — un-diagnosed. VIO sidesteps it.

## vq2wp.py — what's in it (all flight-proven)
Hard-reset race arming; pad GateNet lock; spline route (GateTrajectory carrot,
tangent yaw, per-gate s-stop); map-anchored obs acceptance (xy-only KF fixes,
match hysteresis, ACCEPT_R 3.0); vision-velocity KF fixes (consistency-gated);
obs-z fully quarantined (per-run −0.6…−4.9 bias — never use it for estimation);
vision-primacy approach (DR fallback only >2.5 s stale), align-then-shoot,
misalignment speed governor; punch (fresh-obs <2.6 m AND obs-DR fallback
<2.8 m est) with lateral hold; retreat-and-retry with aperture-height sweep;
tick-as-position-fix; Rerun dashboard. Flight-measured signs SX/SY/SZ=+1
(probes opt-in via `PROBES=1`).

## NEXT SESSION: VIO (the plan)
1. Offline first: corpora in `~/Documents/drone-ai-grand-prix/vq2_data/`
   (`vq2_rec`, `vq2_motion` + tars) through `vq2/replay.py`.
2. Optical-flow ground-plane velocity (floor texture is rich) fused into
   `PosVelKF.update_velocity` at frame rate — likely sufficient. Full odometry
   fallback: cor-106 MASt3R-SLAM worktree on the laptop.
3. Acceptance test offline: reintegrated trajectory must hold <0.5 m error over
   a 12 m run (vs the gate-position ground truth in the recordings).
4. Then wire into vq2wp.py as a velocity source and fly. Everything downstream
   is ready; a single accurate blind-leg should tick gate 1 → route → G2.

## Watch-outs
- transformers in monorace must STAY 5.10.2 (ckpt layout); the fp8 shim lives in
  vq2wp.py. Don't "fix" it by downgrading.
- Kill stray flights: `ssh laptop "taskkill /f /im python.exe"`.
- Sim state: if mavlink heartbeat dies, relaunch FlightSim.exe via the GUI and
  RE-LOGIN to the qualifier (a scheduled-task headless start lands in a
  non-race scene).
- Trust frames + Alex's screen over derived telemetry — every major root cause
  this session was found visually (see memory note
  `feedback-visual-evidence-over-derived-telemetry`).
