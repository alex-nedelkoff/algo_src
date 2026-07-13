# VQ2 Campaign — laptop-local session bootstrap (2026-07-12)

You are running LOCALLY on Alex's Windows laptop (the flight machine).
Everything below was established over 07-05..07-12; trust it, don't re-derive.
Mission: judge ticks through gates 1 and 2 of the AI-GP VQ2 qualifier
(TICKS=2). Sim: AI-GP v1.0.3385, `C:\Users\alexj\Documents\AI-GP Simulator
v1.0.3385\FlightSim.exe`. 18 days remain on the qualifier clock.

## Being local: what changed
- No ssh/schtasks dance: you run IN the desktop session. SendKeys/screenshots
  to the sim work directly from PowerShell. Flight scripts run as plain
  processes (still prefer detached with logs — flights outlive tool timeouts).
- Flight kit is at `C:\Users\alexj\` (vq2wp.py, dpvo_odom.py, eskf.py,
  gate_traj.py, gidx2.py, crash.py, g2fresh.bat + friends, per-run corpora
  vq2_*). The git repo clone is the SOURCE; deploy = copy to C:\Users\alexj.
- **Obsidian logging (Alex's standing rule):** session logs + experiment rows
  live in the Mac vault `/Users/alex/Documents/obsidian-vault/` (session note:
  `02 Areas/Claude Code/2026-07-05 VQ2 VIO implementation plan.md`, handoff:
  `01 Projects/CorvidX/VQ2/HANDOFF — VQ2 state (2026-07-11).md`). The Mac is
  on the tailnet at **100.101.13.126** — when reachable, write/append logs
  there (ssh/scp as user alex). If unreachable, accumulate log entries in
  `C:\Users\alexj\obsidian_outbox.md` and flush when the tailnet returns.

## Judge laws (all measured, decoder-verified)
- RACE_STATUS = ENCAPSULATED_DATA type 1, `<BQqqIq`: [1]=race clock ms
  (zeroes on hard reset 31000/1), [4]=active_gate_index (0→1 on gate-1 tick,
  flip LAGS crossing ~2 s), [5]=tick timestamp (−1 until tick).
- **Judge scores ONLY on a fresh sim session** (restart via ESC menu or app
  relaunch). Hard resets restart the race but never revive a stale judge.
  Recordings start pre-reset: score ONLY with `gidx2.py` (post-reset
  transitions); `gidx.py` counts stale rows (phantom ticks — burned us).
- Judge gate 1 = the CHUTE gate visible from the pad (pad lock ~[6.2-6.4,
  0.0], aperture z −1.35, inner 1.5 m sq). Gate 2 ≈ [11.7, 5.2, −1.35]
  spawn frame (right turn after gate 1, cross along +x).
- Sim scene state machine: ESC TOGGLES the pause menu (blind sequences
  desync — verify with screenshots); app relaunch boots to the pause menu
  with RESTART pre-highlighted; the sim can silently LOG OUT (login screen,
  credentials pre-filled → click SUBMIT). Wrecked drones survive hard
  resets — a crashed run poisons the next one; restart the sim after crashes.

## Error budget (07-12, all measured — the day's big result)
- Sim IMU is NOISELESS (at-rest gyro/accel sigma + bias exactly 0). Stream
  quirks: 38% duplicate re-sends (deduped in vq2wp), bursty 7/14/28 ms
  cadence; rotation across gaps at high rate banks attitude error → RATE_MAX.
- In-flight GateNet detection lateral noise σ 0.3-1.4 m/obs — position fixes
  were injecting the entire ±1.5 m crossing spread (NOFIX=1 = pure-DR chute).
- Takeoff physically displaces the drone ~+1.0 y/−0.55 z (real motion, est
  tracks it) → RECENTER=1 flies it out post-climb.
- The route scan-sweep yawed the whole blind chute (obs always stale under
  NOFIX) and yaw-across-IMU-gaps rotates the DR frame → heading lock
  (yr=0 pre-tick, in code).
- Residual after ALL fixes: est≈truth diverges ~1 m via an un-modelled push
  (07-10 "inner assist" signature) INVISIBLE to inertial sensing by
  construction → blind gate-1 capped at ~12-33%. Detection-center coasts
  don't tick either (detection center ≠ judge aperture at close range).
  ⇒ DPVO is the only path (task #24). Zero magic constants remain: one
  verified tick at TRUE CENTER aim with the full fix stack.

## DPVO (task #24 — the active thread)
- Windows port BUILT AND WORKING on the RTX 3050 (4 GB): 56 fps offline
  (night42/672f in 11.9 s). Repo `C:\Users\alexj\DPVO` (+ weights dpvo.pth).
  Port recipe (if rebuild needed): DISTUTILS_USE_SDK=1; nvcc
  -allow-unsupported-compiler; CL=/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;
  `<long>`→`<int64_t>` (10 files); ba_cuda.cu compound literals → local
  arrays. Import torch FIRST; import dpvo from the REPO (pip wheel lacks
  dpvo.loop_closure) — dpvo_odom.py handles both.
- Live integration (vq2/live/dpvo_odom.py + vq2wp DPVO=1): thread feeds
  state['frame'] → DPVO → spawn-frame position → KF.update_position @10 Hz.
  Anchor = KF pose at first fed frame; monocular scale calibrated vs KF
  displacement during climb (independent until first update), frozen after.
- **CURRENT BLOCKER (where you pick up):** 4 GB VRAM. sim (~1.8G) + GateNet
  (~1G) + DPVO don't fit → silent process death mid-flight. Fix in flight
  (pair 7, log `C:\Users\alexj\g2test.log`): GateNet unloads at GO
  (state['go_passed'], pad lock already done), 1 s settle, DPVO_PATCHES=32,
  GateNet reloads post-tick for the gate-2 servo leg. If still dying:
  DPVO_PATCHES=24, half-res input, or defer GateNet reload.
- Scale-lock telemetry: jlog dpvo_cal/dpvo_scale/dpvo_upd in livelog.jsonl.
  "scale locked" print = updates flowing. No print = flying blind.
- Alex's target architecture (after gate-1 ticks): judge-blessed offline
  reference line (DPVO over a ticked run's frames) + online DPVO pose;
  GateNet out of the control loop entirely (optional recovery only).

## Flight config (current best, g2fresh.bat)
DIRTEST=center, NOFIX=1, RECENTER=1, DPVO=1, DPVO_PATCHES=32, G2TEST=1,
TICKS=2, ARCHTEST=1, VMAX=0.6, POLICY=huber, OBSZ=0, per-run
RECORD=C:/Users/alexj/vq2_freshN. Judge scoring: `gidx2.py <corpus>`.
Crossing forensics: crash.py (route trace), frames in <corpus>/frames
(timestamp blocks split runs — dirs ACCUMULATE, always take the last block).

## Protocol per attempt (freshness discipline)
1. Restart sim (menu RESTART, verify by screenshot — pad scene, green
   lights) or app relaunch → ENTER on RESTART. Watch for the login screen.
2. Launch the run(s) within the fresh window; ~2-3 runs max per restart.
3. Score with gidx2 ONLY; bank per-run corpora; frames = truth.

## Repo pointers
Branch `vq2-estimation`. Key files: vq2/live/vq2wp.py (flight),
vq2/live/dpvo_odom.py, vq2/live/eskf.py, gate_traj.py (nearest_s_window),
vq2/truth_crossings.py, vq2/local_map.py, vq2/build_map.py. Banked offline
DPVO trajectories: vq2_data/dpvo_today/ (Mac). RunPod recipe in the handoff
(pod use: offline DPVO; A4000 $0.17/hr; PUBLIC_KEY env mandatory).

## Discipline (Alex-enforced, hard-won)
- ONE variable per experiment. Frames/visual evidence beat derived telemetry.
- Check COLLISION contact + judge clock before trusting any probe data.
- No unverified success prints; exit codes checked; per-run recording dirs.
- gidx2 only. Fresh sim only. When a theory fails twice, measure instead.
