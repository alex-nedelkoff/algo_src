# AI-GP HANDOFF — next agent starts here (updated 2026-06-10 ~01:00)

> **#1 lesson (still true):** frames/conventions + sysID are SOLVED. Do NOT re-derive. Read the sources of truth below FIRST.

## ★ NEXT = close the IMITATION gap on the honest plant (then PPO beyond teacher)
**TL;DR of tonight:** closed-loop sanity PASSED → pod RL launched → it exposed a **chirality bug in the MIMO fit** (one-handed corner data ⇒ left turns impossible in sim) → **fixed by mirror-symmetrizing A,B; re-validated (omega RMSE unchanged); baked into `sysid/vq_model.json` everywhere** → reran RL: **no more policy erosion, but PPO is flat at warm-start (best 2.62/6 on v6 curves; teacher = 3.58/6)**. Standalone `dagger_v2` (per-dim standardized BC targets) reached **teacher parity 5.9/6** at v4 — so the gap is **BC quality in `rl_finetune`, not the plant or PPO stability.** Next agent, in order:
1. **Port `dagger_v2`'s action standardization into `rl_finetune`'s BC/DAgger warm-start** (or BC the dagger_v2 student straight into the SB3 policy). Its docstring explains why: rate channels ~0.02, abs-MSE under-weights them.
2. **Curriculum v4→v6 curves** (teacher is 48/48 on v4 curves = perfect demos; v6 = 14/48 frontier; v8 = 0/48).
3. PPO beyond teacher only AFTER student ≈ teacher. Keep best-by-eval saving (`ft_<tag>_best.zip`, already wired).
4. Transfer fidelity backlog: `--thrust_lag 0.085` (measured τ~85 ms), **quadratic low-speed drag** in `vq_matched` (linear Dx overestimates at low v — REFIT-02 says c=0.057 quadratic; this is why the live-faithful entry stalls <2 m/s in the closed-loop script), broaden MIMO fit data (esp. **collect a live RIGHT-hand corner** — we have ZERO; if the real plant is chiral, sym under-models it).

## What changed tonight (06-09 night session, chronological)
1. **Closed-loop matched-sim sanity built + passed** — `scripts/sysid/closed_loop_corner.py` (`--v0` spawns at speed): corner_speed law vs `VQMatchedDynamics`. Original-MIMO plant reproduced the live pump/over-tilt/tumble; the DIAGONAL plant shows NOTHING (placid 60 s) — closed-loop proof pre-MIMO RL trained on fantasy physics. Caveats: entry accel ~30% slow (linear drag), pump magnitude ~10 vs live 13–26.
2. **Frame relations data-VERIFIED** (canonical doc updated): qfix/wfix frame IS physical (quat-rate vs wfix·omega slopes ±1.00); live io_layer frame = `q_true[[3,0,1,2]]` shuffle (NOT a rotation), `om_live=[w0,−w1,w2]`, vel=body. Live "tilt" readings are frame-warped in turns (60–80° reading ↔ 24–46° true). Spawn true-attitude ≈ yaw-180.
3. **Teacher for the RL env = corner_speed equivalents** (in `dagger_v2.py` + `rl_finetune.py`): **anti-velocity yaw** (nose-at-gate = camera-BACKWARD = wrong/stable wv branch!), **YR_CAP 1.5** (uncapped π-error yaw cmd = ~12 rad/s = instant resonance tumble), **weathervane-FF** (v4: 45/48 vs 18/48 without — live β-bounding reproduced in-env), **dt=1/72 enforced** (MIMO A,B dt-specific; `vq_matched` raises on mismatch).
4. **DAgger diagnostic (dagger_v2, v4 sinusoids): student 5.9/6, 15/16 — teacher parity.** BC alone 0.2/6 (covariate shift) → relabel rounds close it.
5. **MIMO-CHIRALITY found + fixed**: teacher 0/6 on ALL-LEFT curves (dead at gate 0) vs 6/6 ALL-RIGHT. Root = one-handed live corner data ⇒ mirror-violating cross terms (roll↔pitch, pitch↔yaw) unconstrained → diverge on the un-data'd direction (live slaloms refute: both directions stable). **Fix: A←(A+MAM)/2, M=diag(−1,1,−1)** (keeps the legit roll↔yaw coupling + diagonals). matched_resonance re-validated: turn-band RMSE [0.057,0.037,0.051] ≈ unchanged. Teacher after: LEFT 32/32 + RIGHT 32/32 @v4. **`sysid/vq_model.json` is now the SYM model** (laptop canonical, backup `vq_model_premirror.json`; Mac + pod synced; `fit_rate_mimo.py` symmetrizes future fits).
6. **RL results**: pre-fix runs eroded (half of tracks impossible — explains every plateau-then-decay). Post-fix v6curve: warm 2.5 → PPO 1.6–2.6 flat, **best 2.62/6** saved. Artifacts: pod `/workspace/algo_src_cor127/ft_mimo_sym_v6curve{,_best}.zip`, Mac `sysid/ft_*.zip`. Exp-log rows: **MATCHED-CLOSEDLOOP, MIMO-CHIRALITY, RL-MIMO-SYM-01**.

## READ FIRST (sources of truth, in order)
1. **`01 Projects/CorvidX/AI-GP Frames & Conventions (CANONICAL).md`** — now includes the 06-09 verified frame relations + live-tilt-warp warning.
2. `01 Projects/CorvidX/AI-GP Experiment Log.md` — newest rows: MATCHED-RESONANCE, MATCHED-MIMO-FIX, **MATCHED-CLOSEDLOOP, MIMO-CHIRALITY, RL-MIMO-SYM-01** (06-09/10).
3. This file. Linear **COR-127** (comments 06-09 incl. the closed-loop/teacher/DAgger comment; post the chirality+RL results comment if not yet there).

## HARD RULES (anti-loop)
- Reuse proven blocks (`race_cruise`/`coord_turn`/`corner_speed` laws; `dagger_v2.ff_batch`+`aim` = the env teacher). Don't re-derive frames. Judge stability by TRUE tilt sim-side.
- **Env teacher MUST fly camera-forward** (anti-velocity yaw). Nose-at-gate = stable wv branch = doesn't transfer to racing.
- **dt=1/72 everywhere the MIMO plant runs.**
- Before live runs: kill stray MAVLink :14550 python, log+poll, stream Rerun dashboard (Mac 100.101.13.126:9876).
- Linear writes: batch them (Cloudflare WAF blocks bursts; reads OK).

## KEY LOCATIONS
- **Mac repo** (branch `alexnedelkoff/cor-127-…`, pushed): `sim/dynamics/vq_matched.py` (MIMO + dt guard + ENU), `scripts/rl/{dagger_v2,rl_finetune,ff_corridor}.py` (camfwd teacher, dt=1/72, best-save), `scripts/sysid/closed_loop_corner.py`, `sysid/vq_model.json` (SYM; gitignored), `sysid/ft_mimo_sym_v6curve*.zip`.
- **Pod** (A4500, `runpod-cor106` ssh alias — host/port rotates, update `~/.ssh/config`): `/workspace/algo_src_cor127` = this branch checked out + synced; container layer ephemeral → `pip install stable_baselines3 gymnasium scipy pytest` on restart; vq_model.json (sym) present. Old `/workspace/algo_src` = stale non-git copy, ignore. **Pod left RUNNING at session end — STOP it if not continuing** (`runpodctl pod stop <id>`... it was started manually by Alex; `runpodctl pod list`).
- **Laptop** (VQ sim host, `ssh laptop`): aigp-client worktree; `sysid/vq_model.json` = SYM canonical (+`vq_model_premirror.json`, `vq_model_sym.json`); `fit_rate_mimo.py` patched (symmetrizes); flight scripts unchanged.
- Data: `~/Documents/vq_data` (laptop). Corner runs incl. `135925`/`140352` (validation pair).
