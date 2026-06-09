# AI-GP HANDOFF — next agent starts here (updated 2026-06-09 eve)

> **#1 lesson (still true):** the frame/conventions + most sysID are SOLVED. Do NOT re-derive them — it has wasted hours every prior session. Read the sources of truth below FIRST.

## ★ IMMEDIATE NEXT TASK — implement the cornering plan
**Build the cornering-at-speed controller.** Spec + step-by-step plan are written, committed, and self-reviewed:
- Plan (execute this): `docs/superpowers/plans/2026-06-09-cornering-at-speed.md`
- Spec (the why): `docs/superpowers/specs/2026-06-09-cornering-at-speed-design.md`

**What it is:** `corner_speed.py` = the proven `coord_turn.py` (coordinated turn: nose-tracks-velocity so β≈0, bank `atan(v²/Rg)`, yaw-rate FF `v/R`, pitch-comp) + an **explicit weathervane feedforward** (cancel `wv·v_body_y` in the rate command — the new piece, enabled by this session's WV-DYNAMIC validation, that every prior coordinated-turn attempt lacked) + a tilt-budget speed cap. Validate on ONE corner at ~6 m/s with an FF on/off A/B (Task 3) that PROVES the FF and verifies its sign live.
**How to run it:** subagent-driven (Tasks 1–2 = TDD helpers + the script, pure code) then drive Task 3 live yourself (needs the VQ sim up + dashboard). The plan has exact ssh/scp/pytest commands + the env quirks.
**Critical:** the FF needs live `v_body_y` (sideslip) — the exact frame trap that bit this session. **Verify the FF sign empirically** (β must SHRINK with FF on, via `--no-wvff` / `--wvff_sign`); never assume the body/world velocity frame. Judge by TILT + β on the dashboard.

## READ FIRST (sources of truth, in order)
1. **`01 Projects/CorvidX/AI-GP Frames & Conventions (CANONICAL).md`** (obsidian) — single source of truth: 3 frames, qfix/wfix/body-vel, B=diag(1,−1,−1), deploy bridge, β meaning, measured params, which controllers work. (Updated 06-09: the "wall" is drag/tilt-saturation, not uncollectable.)
2. `01 Projects/CorvidX/AI-GP Experiment Log.md` (+ `… 2026-06-07 supplement.md`) — full blow-by-blow. Newest rows: RAMP-WALL, REFIT-01/02, LATERAL-WALL, TRACK-01, **WV-DYNAMIC** (06-09).
3. The spec/plan chain in `docs/superpowers/{specs,plans}/2026-06-09-*` (3 specs, 3 plans — tilt-budget controller, weathervane validation, cornering).
4. This file. Linear: **COR-127** (2 comments posted 06-09 with the wall + weathervane findings).

## HARD RULES (anti-loop — we keep violating these)
- **Reuse the proven building blocks; don't rewrite them.** Camera-forward straight = `race_cruise.py`; coordinated turn = `coord_turn.py`; the tracker = `traj_track.py`. New controllers (`traj_track`, `corner_speed`) are built ON these — reuse their frame handling verbatim.
- **Do NOT re-derive the frame / re-characterize the weathervane / re-collect solved sysID.** It's in `vq_model.json` + the canonical note. **Judge stability by TILT** (tumble = tilt>~70° growing), NEVER a hand-rolled β/heading metric — those are FRAME-BUGGY (body-vel vs world-camera → sign-inverted; a proven cam-forward flight reads β≈180 / align≈−0.94 = NORMAL, not divergence).
- **Before live runs:** kill stray python on MAVLink udp 14550 (`powershell Get-NetUDPEndpoint -LocalPort 14550 | %{Stop-Process -Id $_.OwningProcess -Force}`), run `python -u > log`, poll the log (ssh `| tail` block-buffers till exit). Stream the Rerun dashboard (Mac 100.101.13.126:9876). `fresh_start()` recycle can be slow after a tumble — be patient.

## STATE OF PLAY (06-09) — what this session solved
- **The cam-forward "wall" = DRAG / TILT-BUDGET SATURATION, not an aero weathervane.** `v_max ≈ √(g·tan(tilt_cap)/c)`, c≈0.057. Tunable ~8–10 m/s (18°→8, 25°→9, 30°→10.5, confirmed live). The old 2.8 was just `AL_MAX=0.6`.
- **Lateral wall = the SAME tilt budget.** Jink capability shrinks with speed (~5.5m@v3 → ~1.2m@v7 → ~0@v8). Both walls are one constraint: forward-drag tilt + maneuver tilt ≤ cap.
- **Tilt-budget speed controller BUILT (TRACK-01).** `gate_traj.py` speed law `v=√(g·tan(φ)·margin/(c+κ))` (unit-tested) + budget-aware cross-track + startup ramp in `traj_track.py`. **STRAIGHT flies clean live at 8.5 m/s (3× the old governor).** Translated/curve still TUMBLE in the startup/lateral-acquisition transient — **nose-follows-tangent yaw is the trigger** (fixed-heading crab flew translated to 82%); the cornering plan's weathervane-FF + velocity-tracking yaw is the fix.
- **Weathervane VALIDATED in the dynamic regime (WV-DYNAMIC).** First controlled high-sideslip data via `collect_vq_crab.py`. The weathervane does NOT grow with speed (weakens/reverses); the matched sim's directional model already matches (+0.028 at v7 vs +0.029 fit) → **no coeff change; the matched sim is faithful.** Coeffs: `roll_wv0=−0.105, roll_wv1=−0.019, yaw_wv=−0.149`.
- **Open after cornering (downstream, each its own spec):** (1) gate-to-gate chaining of the coordinated turn into the `gate_traj` tracker; (2) DAgger/RL retrain on the validated matched sim — train to BOUND sideslip (avoid the divergent regime the sim captures), NOT model a steady high-sideslip weathervane that doesn't exist. RL-beyond-imitation has failed all session WITHOUT a teacher → the cornering controller becomes the teacher.

## KEY LOCATIONS
- Repo (this branch `alexnedelkoff/cor-127-…`): matched dyn `sim/dynamics/vq_matched.py`, env `sim/envs/gate_race_env.py` (action_mode `vq_rate`), RL `scripts/rl/{ff_corridor,dagger_v2,rl_finetune}.py`. Specs/plans `docs/superpowers/`. (HANDOFF is committed here.)
- Laptop (VQ sim host): `ssh laptop`, aigp env `C:\Users\alexj\miniconda3\envs\aigp\python.exe`, worktree `C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src\.claude\worktrees\aigp-client`. Flight scripts: `race_cruise`, `coord_turn`, `traj_track`, `gate_traj`, `collect_vq*` (+`_ramp/_alramp/_lateral/_crab`), `wv_speed_fit`/`wv_validate`/`drag_speed_fit`, `vq_deploy*`/`vq_rate_verify`, `sysid/vq_model.json`, `sysid/sim_response.json`, `ft_dagger*.zip`. `scp -O` required.
- Data: `~/Documents/vq_data` on the laptop (recorder runs; `index.jsonl`). Dynamic-regime crab runs: `20260609T11{3419,3626,3740}_collect_vq_crab`.
- Pod (GPU, RL only): runpodctl; STOPPED. New SSH each start; alias `runpod-cor106`.
