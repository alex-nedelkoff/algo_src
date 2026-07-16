# Handoff — AI-GP: match the RL sim's control interface + latency to the VQ sim

## Mission
Make the RL physics sim (`numpy_quad`) match the **VQ (DCL FlightSim) drone's control interface + timing**, then prove it with a **trajectory-level holdout**. Only then are the sysID params confidently mapped and an RL policy transferable. Right now they are NOT — params were fit through the VQ's control loop at VQ timing, and porting them into our different interface fails (cmd-replay diverged: |vel| 2.4 VQ vs 35.9 sim).

## READ FIRST (don't re-derive — a prior session wasted effort doing so)
- **Linear COR-127** "TRPY rate-loop + mixer calibration" — READ THE 3 COMMENTS. Has: motor model, rate loop, the cam-fwd roll-divergence blocker. Use `linear-grandprix` MCP (team COR). Also **COR-96** (dynamics sysID), **COR-106** (vision policy).
- **Obsidian** `01 Projects/CorvidX/AI-GP Experiment Log.md` (main, may be iCloud-locked) + two `… 2026-06-07 supplement*.md` (residual-RL + this session). Session log `02 Areas/Claude Code/2026-06-07 AI-GP directional weathervane + interface fidelity.md`. Older handoff `AI-GP flight-control handoff 2026-06-02.md`.

## Environments
- **Windows laptop** (the VQ sim host): `ssh laptop` (Tailscale). `scp -O` required (capital O). Avoid nested PowerShell quoting — write scripts to /tmp, `scp -O`, run by path.
  - VQ python env: `C:\Users\alexj\miniconda3\envs\aigp\python.exe`. Base numpy: `C:\Users\alexj\miniconda3\python.exe`.
  - VQ client worktree (branch `aigp-gate-data-collection`): `C:\Users\alexj\Documents\drone-ai-grand-prix\algo_src\.claude\worktrees\aigp-client`.
  - RL repo (branch vq-course1, has `numpy_quad` + VQ params + corridor): `C:\Users\alexj\Documents\algo_src`.
  - Recorded VQ data: `C:\Users\alexj\Documents\vq_data\<timestamp>_<run>\data.npz` (keys: t_wall,t_us,cmd,pos,vel(BODY),quat,omega,acc,gyro,act,coll_seq,gate_idx,live).
- **RunPod** (GPU, RL training only — NOT needed for interface/latency work, which is CPU numpy): `ssh root@213.173.107.105 -p 15053 -i ~/.ssh/id_ed25519` (migrated; **only `/workspace` persists**, container deps wiped on migration → `pip install numpy` or `scripts/cloud/bootstrap.sh`). Currently idle + billing — **stop it** (`~/.local/bin/runpodctl stop pod <id>`) unless training. GPU-host-full on restart is common (retry / it migrates).
- **Mac** (this machine): repo `~/Documents/drone-ai-grand-prix/algo_src` (branch cor-106…; **lacks VQ params/corridor** — can't run the RL sim locally; use the laptop's `C:\Users\alexj\Documents\algo_src` or the pod).
- **Rerun** viewer (Mac): `~/.rerun33-venv/bin/rerun <file.rrd>` (0.33; system py caps at 0.26). The VQ client streams a live flight dashboard via `aigp/flight_telemetry.py` (gRPC to Mac over Tailscale; ON by default in race_cruise/goto). HARD RULE: every sim run streams telemetry to the Rerun dashboard.
- **wandb**: project `janahanr-corvidx/corvidx-drone-racing`.

## What's already validated (USE these, don't refit)
- `docs/motor_model.json` (laptop aigp-client): sign map `sx=[-1,1,1,-1]`(roll) `sy=[-1,-1,1,1]`(pitch) `sz=[-1,1,-1,1]`(yaw), L=0.14, k_f=34.26, k_q=0.685, **quadratic**, **hover≈0.27**. NOTE: numpy_quad yaw torque sign is FLIPPED vs `sz` — reconcile.
- `sysid/sim_response.json`: **rate_gain −1.93** (roll −1.98/pitch −1.89/yaw −2.29), hover_thrust 0.23, k_a 62.
- `sysid/vq_model.json` (grey-box, COR-96): drag Dx=Dy=0.229(mass-norm), thrust hover 0.2675, **rate_loop per axis {α, τ_ms (roll26/pitch24/yaw44), gain_G (−2.54/+2.53/−2.29), r²~0.99}**, weathervane yaw −0.1488, **delays_measured {comms 19ms, thrust_lag 40-90ms, odom 72Hz/imu 144Hz/actuator 73Hz}**, holdout body-accel RMSE x1.79/**y3.95**/z1.78.
- **Directional ROLL weathervane (this session, validated 25k pts, posted COR-127):** `roll_wv(vx)=−0.105−0.019·v_body_x` (fwd ~2× bwd). Yaw ~constant −0.15. numpy_quad already patched (pod copy). The y=3.95 holdout residual is the unmodeled roll wv → should drop on refit.
- Interface facts: VQ is **ACRO rate mode** (`SET_ATTITUDE_TARGET`, body-rate+thrust, `ATTITUDE_IGNORE`) via `commander.send_attitude_target(w_cmd, thr)` with `w_cmd = w_des/rate_gain`; OR direct motors `commander.send_motor_command(u[0,1]×4)` (`SET_ACTUATOR_CONTROL_TARGET`, bypasses inner loop). numpy_quad RL uses **TRPY feedforward mixer** (`τ=J·rate_gain·cmd`, rate_gain=1.0) — NO rate loop. THIS is the core mismatch.

## The task — identify control interface + latency, build it in, validate jointly
1. **Confirm the rate-loop model.** vq_model `rate_loop` (α/τ/gain_G per axis, first-order) already fits the cmd→ω response (r²~0.99). Validate it: on the VQ, step a body-rate command (`send_attitude_target`), record `omega` response, confirm the first-order gain+lag matches. (Use `aigp/flight_telemetry.py` / a probe script; `fresh_start()` before each run.)
2. **Identify latency precisely.** Where does the 19ms comms + 40-90ms thrust lag enter (cmd→actuation vs sensor→telemetry)? Step-input timing on the VQ. Distinguish actuation delay from sensor delay (matters for the sim's delay buffer placement).
3. **Build the matched interface into numpy_quad** (replace the feedforward `TRPYMixer`): an ACRO rate-tracking loop (gain −1.93, first-order α/τ per axis from vq_model) + a latency buffer (delay cmd by actuation lag, delay obs by sensor lag) + fix the yaw torque sign. Keep the validated aero (drag, thrust, yaw wv −0.149, directional roll wv).
4. **Trajectory-level holdout (the confidence test).** Replay held-out VQ command sequences through the matched sim, compare FULL trajectories (pos/vel/omega/quat over time) to VQ telemetry — not 1-step accel. Infra exists: `replay_compare.py` (laptop, `C:\Users\alexj\replay_compare.py`) — it FAILED (|vel| 35.9) precisely because the interface was unmatched; with the matched interface it should reproduce VQ (target: trajectory RMSE comparable to vq_model's ~1.8 m/s² accel, and bounded position drift). 
5. **Only params that survive the holdout are confidently mapped.** Then RL trained on the matched sim is transfer-credible. Update COR-127 (#4 rate loop) + COR-96 (refit weathervane).

## Watch-outs
- Frames: numpy_quad is ENU/z-up; VQ is NED. VQ telemetry needs qfix (`wxyz=stored[1,2,3,0]`), wfix `[1,-1,1]`, vel-is-BODY (`world=R@v_body`) — all in vq_model `frame_conventions`.
- Param confidence is CONDITIONAL on the interface+timing model — do NOT trust the aero params for transfer until the matched-interface trajectory holdout passes.
- Camera-forward = the racing direction = the unstable one (vx>0, roll wv ~2×). Backward is stable but camera can't see gates — don't "solve" it by flying backward.
- Check Linear/vault BEFORE deriving anything — most of the motor/rate-loop/interface work already exists in COR-127/96.
