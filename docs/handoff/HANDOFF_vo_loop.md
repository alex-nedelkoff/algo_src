# HANDOFF — feat/vo-loop: VO into the smoother (2026-07-21)

═══════════════════════════════════════════════════════════════════
## ⭐⭐ START HERE — SESSION BANKED 2026-07-22 (COR-147)
═══════════════════════════════════════════════════════════════════

Branch `feat/vo-loop` (worktree `C:\Users\Administrator\algo_src-vo`, pushes
to origin = algo_src-alex). Linear: **COR-147**. Env: conda `monorace`
(`C:\Users\Administrator\miniconda3\envs\monorace\python.exe`). All work below
is committed + pushed; 14 tests pass (`python -m pytest vq2/tests -q`).

### THE ARCHITECTURE (decided this session, with Alex)
Estimator = **map-based absolute localization** (gate PnP + pillar-azimuth
against a KNOWN offline map) as the drift-free backbone, with **monocular VO
as the relative bridge** between/through detections. **No IMU** — with a good
map + continuous landmark visibility, per-frame resection observes full pose,
so IMU is insurance, not a requirement (and the un-modelled "push" is only
visible to vision anyway). VO is unit-scale; gate crossings give metric scale.
Both sources lower to `SlidingSmoother` factors via the `vq2/relpose.py`
contract — the pillar-PnP branch (Alex) feeds the SAME contract.

### WHAT'S BUILT (all committed, flight-validated where noted)
1. **Measurement contract** `vq2/relpose.py` — `LandmarkRelPose`, `OdomDelta`
   (+`to_factor`). The shared seam for both branches. (trunk `9ec4a9a3`)
2. **Monocular VO** `vq2/live/vo_cv.py` — `MonoVO`: KLT + essential-matrix,
   **keyframe-by-parallax** (median flow >= kf_flow, NOT adjacency), emits
   `VoStep`/`OdomDelta` (body-frame unit dir + dyaw, scale_locked=False). Uses
   the real `vq2.camera` model (FX=226, cam->body M_BODY_CAM). Front-end/back-
   end split: `track()` (real-time) + `solve_job()` (threadable). `gpu=True`
   swaps in CUDA-graph optical flow. Solve caps: `max_solve_pts=200`,
   `ransac_prob=0.99`.
3. **GPU optical flow** `vq2/live/gpu_klt.py` — torch pyramidal LK. `GpuKLTGraph`
   captures the fixed-shape LK as a CUDA graph, one launch: 6.7ms wall, 0.00ms
   CPU-busy (100% freed), 0.02px vs cv2.
4. **Live threaded consumer** `vq2/live/vo_live.py` (`LiveVO`) + hook
   `vq2/live/vo_live_hook.py` (`start_vo_live`, gated `VO_LIVE=1`). Evict-oldest
   queue under backpressure.
5. **Association** `vq2/live/vo_association.py` — `calibrate_scale` (metric
   scale from a known gate leg), inlier-weighted sigma, `feed_smoother` ->
   `SlidingSmoother.push_odom`.
6. **Offline tools** `vq2/tools/vo_replay.py` (VO health + estimator-agreement
   with livelog carry-forward-ns join), `vo_fusion_demo.py`, `vo_live_replay.py`
   (real-time cadence validator). Scratch probes in `scratch/vo_*.py`.
7. **Deployed hook**: 4 gated lines in `C:\Users\Administrator\vq2wp.py` after
   cam_loop starts (VO_LIVE off by default — safe for other flights).

### KEY MEASURED FACTS (trust these; each earned)
- **VO is sound** (two truth-anchored checks): 86% forward-dominant on a clean
  transit; reproduces the map's surveyed gate-1 right turn within **14 deg**
  (residual = judge-tick ~2s lag + VO drift, NOT VO error).
- **Map-as-reference works**, and beats the estimator (p_est is DR, not truth).
  Datum/scale sidestepped via RELATIVE geometry (turn angle, length ratios).
- **Vagon box = 2 physical cores**; the sim (DCGame/Unreal) alone holds ~one
  (86% load). This is THE constraint.
- **CUDA-graph GPU tracking holds real-time** where CPU can't: front-end
  submit p50 6.8 / p90 15ms, 0% over the 33ms budget (CPU: 17/37, 12% over).
- **Solve optimization validated in flight** (votest2): subsample+prob+throttle
  +evict-oldest took solve p50 300ms-1s -> **36-70ms**, p90 2.1s -> ~420ms,
  drops -> 0. VO correct live: 76% fwd-dominant, +x 0.90, inliers 85.

### OPEN ISSUES / LIMITATIONS (honest)
- **GPU CUDA-graph capture is NOT flight-safe yet.** Default `global` capture
  mode crashed the co-resident GateNet detector thread mid-flight ("operation
  not permitted when stream is capturing") -> pad-lock fail -> abort.
  `thread_local` (now set) helps but capture is fundamentally unsafe against
  concurrent CUDA that ALLOCATES (GateNet startup autotune). votest1 worked by
  timing luck. **FIX NEEDED: capture the graph in a GateNet-idle window** (or
  eager-capture at known 640x360 before autotune). Until then fly `VO_GPU=0`.
- With CPU tracking the **front-end is the bottleneck** in flight (track p50
  32-36ms, over budget), worsened by concurrent load. GPU tracking (12-15ms)
  is the fix once flight-safe.
- **Datum reconciliation** (Janahan map reset-NED vs flight pad-lock) is the
  load-bearing prerequisite for Step B quantitative use. Relative geometry is
  datum-free; absolute anchoring is not.
- Only 2 ticked gates on tick flights => the 2D VO->map alignment is fully
  determined, so gate-offset is not an independent check (turn-angle is).

### NEXT (pick one)
- **(a) Flight-safe GPU capture** — capture in a GateNet-idle window; then GPU
  front-end (12-15ms) + solve opts (36-70ms) combine = fully real-time live VO.
- **(b) Step B: map-localization backbone** — per-frame gate PnP + pillar
  azimuth against the map + online scale from gate crossings, CPU tracking as
  the working baseline. This is the actual product estimator.

### RUNBOOK
- Fly VO smoke: `fly_votest2.bat` (VO_GPU=0 reliable; STRAIGHTTEST fwd 8s). VO
  log -> `VO_LOG` (vo_votest*.jsonl). **Straight test lands the drone ~6m
  downrange -> RESTART the sim (menu/relaunch) to re-home before each flight**
  (a race hard-reset does NOT reposition it).
- Shared machine: a Codex agent has also been active here. Screenshot the sim
  state before flying; do not grab the sim window from active human/agent use.

═══════════════════════════════════════════════════════════════════
## UPDATE 2026-07-22 — DPVO IS DEAD IN-VM; OpenCV VO IS THE PATH
═══════════════════════════════════════════════════════════════════

**DECISION (Alex): swap the VO source from DPVO to an in-VM OpenCV
monocular VO. DPVO stays OFFLINE-only, on a box where it runs.** The
`OdomDelta` contract makes the source swappable — the smoother side is
unchanged.

**Why DPVO is dead here (measured, root-caused — do NOT retry online
DPVO on Vagon):**
- CUDA works (A10G passthrough, torch 2.3.1+cu121, `cuda_corr` imports
  AND executes — frame 0 ran, 2.4 GB allocated, 35% util).
- But DPVO's **lietorch CUDA extension access-violates on first kernel
  execution** (SE3 `inv()` in `lietorch/group_ops.py:14`, hit via
  `motion_probe → reproject`). `smoke_lietorch.py` (DPVO's own test)
  imports OK then SEGFAULTs the instant a lietorch op runs. Exit 139.
- Signature = ABI mismatch (extension built against a different
  torch/CUDA than dpvo231's) OR a build-provenance issue from the
  laptop port. Not worth grinding: a rebuild MIGHT fix it but the
  decision is to move on.
- The `dpvo231` conda env now has a FULL working DPVO dep set
  (torch_scatter, numba, pypose, kornia, evo, plyfile, matplotlib,
  numpy<2, opencv 4.11) — usable for OFFLINE DPVO on a working box.

**Why OpenCV VO is viable (measured on real corpora):**
- KLT (`goodFeaturesToTrack` + `calcOpticalFlowPyrLK`) + essential-
  matrix (`findEssentialMat` RANSAC + `recoverPose`) runs clean in-VM,
  no segfault, ~500 features tracked/frame, 0 NaN, 0 E-failures.
- CRITICAL LESSON (cost me several probes): **corpus frames 0..~360
  are the PAD-STATIC PRELUDE** (drone waits for GO; median flow 0.0px).
  Sampling there makes VO look degenerate (recoverPose ~1 inlier).
  ALWAYS find the motion window first (scan median flow; test46 flight
  = frames ~380..1100). `scratch/vo_motionscan.py` does this.
- On the MOTION window, recoverPose is healthy and scales with
  baseline (test46, inliers p50/p90 of ~500 feats):
    stride 1 (1.7px flow): 13/115  | stride 5 (8.6px): 164/313
    stride 3 (5.2px):      80/241  | stride 8 (12.6px):177/328
- ⇒ **The VO front-end MUST keyframe by parallax**, not adjacency:
  accumulate frames until median flow ≈ 8-13px (cf DPVO
  KEYFRAME_THRESH=15px), then solve the relative pose. Emit ONE
  `OdomDelta` per keyframe pair, `scale_locked=False` (monocular =
  direction only; reuse the DPVO scale machinery — climb-cal /
  GNSCALE — to scale later).

**Env note:** the vq2/VO runtime env is `monorace` (cv2 5.0.0, numpy 2).
cv2 5.0 vs 4.11 gave IDENTICAL VO results (A/B'd) — version is not a
factor. Scratch probes: `scratch/vo_cv_smoke.py`, `vo_baseline.py`,
`vo_motionscan.py`, `vo_probe.py`.

**NEXT (pick up here):** build `vq2/live/vo_cv.py` — a keyframe-by-flow
monocular VO emitting `OdomDelta` — then the offline replay harness
(Phase 1 below), validating VO-only smoother drift vs the `p_est`
reference in `livelog.jsonl`. Frame↔time link: jpg filename IS the
`sim_ns`; order via `frames_dedup.jsonl`.

The DPVO-specific plan below is SUPERSEDED for the online path; its
scale-strategy and replay-harness structure still apply to the OpenCV
VO. Everything else (contract, discipline, phase gates) stands.

═══════════════════════════════════════════════════════════════════
## MISSION
═══════════════════════════════════════════════════════════════════

Get VO ego-motion into the estimation loop as first-class smoother
factors. Concretely: the DPVO bridge emits `OdomDelta` records
(`vq2/relpose.py`, the shared measurement contract), an association/
scheduling layer lowers them to `OdomFactor`s in the `SlidingSmoother`
(`vq2/window_smoother.py`). The existing direct-to-KF path
(`dpvo_odom.py` → `KF.update_position`) stays intact behind its flag —
you are ADDING a parallel, cleaner path, not replacing the old one
until the new one measurably wins.

Why this matters (07-12 error budget, CLAUDE.md): the residual ~1 m
est-vs-truth divergence is an un-modelled push INVISIBLE to inertial
sensing by construction. VO is the only sensor that sees it. Alex's
target architecture: judge-blessed offline reference line + online
DPVO pose, GateNet out of the control loop entirely.

**Division of labor: Alex is on the digit recognizer (feat/pillar-pnp
side) and managing you. Janahan (upstream) owns mapping; his map is
trusted. Your lane is VO + smoother integration ONLY.**

═══════════════════════════════════════════════════════════════════
## THIS MACHINE (Vagon — verified 2026-07-21, don't re-derive)
═══════════════════════════════════════════════════════════════════

- GPU: **NVIDIA A10G, 23 GB** — the laptop's 4 GB VRAM wall (sim +
  GateNet + DPVO co-residency, silent process death) is GONE here.
  The GateNet-unload-at-GO dance in vq2wp may be unnecessary on this
  box; do not delete it (laptop still flies), just don't fight it.
- DPVO repo: `C:\Users\Administrator\DPVO` with weights
  `C:\Users\Administrator\DPVO\dpvo.pth` (build logs present — the
  Windows port was built here or copied built).
- Python: conda env `monorace` at
  `C:\Users\Administrator\miniconda3\envs\monorace\python.exe`.
  vq2 tests pass in it. WHETHER DPVO IMPORTS IN IT IS UNVERIFIED —
  Janahan documented three env silos on upstream/main
  (`build(cor144)` commit); DPVO may need its own. Verify in Phase 0.
- Flight corpora: `C:\Users\Administrator\vq2_test1..144+` (per-run
  recording dirs; frames in `<corpus>/frames` — timestamp blocks
  split runs, dirs ACCUMULATE, always take the LAST block).
- Repo/worktree: you work in `C:\Users\Administrator\algo_src-vo`,
  branch `feat/vo-loop`, pushes to `origin` (algo_src-alex fork).
  Trunk `vq2-estimation` lives in `..\algo_src` and has an ACTIVE
  line-transit campaign with uncommitted work — NEVER touch that
  worktree. `upstream` remote = team repo (Janahan's main).

⚠️ `vq2/live/dpvo_odom.py` has HARDCODED laptop paths
(`C:\Users\alexj\DPVO\dpvo.pth`, `...\config\default.yaml`,
lines ~28-29). First code change: env-var override with the laptop
values as fallback (`DPVO_HOME` or similar). Same audit for
`dpvo_odom_bridge.py` / `bridge_dpvo.py` / `dpvo_route.py`.

═══════════════════════════════════════════════════════════════════
## HARD-WON FACTS (measured on the laptop campaign — trust these)
═══════════════════════════════════════════════════════════════════

- DPVO offline: 56 fps (night42/672f in 11.9 s) on an RTX 3050.
  Fast live config: PATCHES 24, input 320x180, REMOVAL/OPT window
  12/6 → 171 ms sim-closed vs 300 ms stock.
- Import order: `import torch` FIRST; import dpvo from the REPO, not
  the pip wheel (wheel lacks `dpvo.loop_closure`) — dpvo_odom.py
  already handles both.
- Convention: DPVO `pg.poses_` stores world-to-camera; `.inv()` gives
  camera-to-world (translation = camera position in the cam0 frame).
  `M_BODY_CAM` columns are camera axes in body frame (cam→body).
- Anchoring: DPVO's world = its first fed camera frame. Feed starts
  at 'airborne'; anchor = KF pose at that moment (at-rest attitude
  exact, position ~0.1 m).
- **Monocular scale is PER-SESSION NON-DETERMINISTIC** (full debate:
  `docs/handoff/dpvo_scale_review.md`). Current strategy: calibrate
  vs KF displacement during climb (independent — no DPVO updates
  applied yet), FREEZE at first KF update to avoid circularity.
  A second mechanism exists: GNSCALE / gate-scale apply
  (`dpvo_gate_scale.py`, commit 955f577) — gate-PnP range as a scale
  reference. These are A/B candidates, not settled.
- Scale-lock telemetry: jlog keys `dpvo_cal` / `dpvo_scale` /
  `dpvo_upd` in `livelog.jsonl`. "scale locked" print = updates
  flowing; no print = flying blind. vq2wp already gates GateNet
  handoff on `state['dpvo_scale_locked']` (~line 675).
- Sim IMU is NOISELESS (zero sigma, zero bias) but the stream has 38%
  duplicate re-sends and bursty 7/14/28 ms cadence (deduped in vq2wp).

═══════════════════════════════════════════════════════════════════
## THE CONTRACT (already on your branch — build against it)
═══════════════════════════════════════════════════════════════════

`vq2/relpose.py` (tests: `vq2/tests/test_relpose.py`, 5 pass):

- `OdomDelta(t0, t1, dp_local, dyaw, source='dpvo', sigma_p, sigma_yaw,
  scale_locked)` — ego-motion between capture times. Time-keyed; the
  producer NEVER sees state indices.
- `OdomDelta.to_factor(i, j)` → `window_smoother.OdomFactor`. The
  association layer owns the time→index mapping
  (capture-time-nearest, same convention as PosUnary).
- `scale_locked=False` deltas are direction-only information: inflate
  sigma_p hard or drop. Decide by measurement, not taste.
- dp_local frame: yaw-relative frame of the earlier state (OdomFactor
  convention). DPVO gives you cam0-frame motion — you own the
  rotation into the state-i yaw frame (use the smoother's current
  yaw estimate at t0; document the circularity implications).
- If the contract needs a field, propose it to Alex — contract edits
  land on trunk `vq2-estimation` first, then rebase. Do NOT fork the
  contract on this branch.

═══════════════════════════════════════════════════════════════════
## PHASED PLAN — offline first, gates before phases
═══════════════════════════════════════════════════════════════════

**Phase 0 — bring-up (gate: DPVO runs offline on this machine).**
1. Fix hardcoded paths (env-var override).
2. Verify dpvo imports in `monorace`; if not, build/locate its env
   (check upstream/main env recipes, `docs/vagon-bootstrap.md`).
3. Run DPVO offline over ONE banked corpus's frames; sanity-check
   fps (expect >> 56 on the A10G) and a visually sane trajectory.

**Phase 1 — offline replay harness (gate: drift number vs truth).**
Build `vq2/tools/vo_replay.py` (or similar): corpus frames → DPVO →
`OdomDelta` stream → `SlidingSmoother` replay (IMU factors + OdomDelta
only, no GateNet) → compare vs recorded truth (`vq2/truth_crossings.py`
knows the truth channels). Metrics: drift %/m per leg, endpoint error
at gate-1/gate-2 crossing times. Run over ≥5 corpora spanning good and
bad flights (include test142's corpus — the wedge flight — and some of
100-107). THIS HARNESS IS THE MAIN DELIVERABLE — every later decision
cites it.

**Phase 2 — scale strategy A/B (gate: one strategy wins on ≥5 corpora).**
A: climb-calibration (current). B: GNSCALE gate-scale apply.
C (cheap to add): A then B refinement. Same harness, same corpora,
report per-corpus scale estimates and drift. Per-session scale
non-determinism means N matters — do not conclude from n=1-2.

**Phase 3 — live wiring behind a NEW flag (gate: latency + no
regression).** `VO_SM=1` (leave `DPVO=1` legacy path untouched):
bridge thread emits OdomDelta into a queue; smoother side drains and
lowers to factors. Measure end-to-end latency (frame→factor) and
smoother solve time at flight rate on this box. No flight yet.

**Phase 4 — flight validation (gate: TICKS, nothing else).**
Sim lives on this machine per vagon-bootstrap. Fly the standard
protocol (fresh sim, gidx2-only scoring, per-run RECORD dirs,
2-3 runs per restart). Compare tick RATE vs the non-VO stack over
MULTIPLE flights. Read the line-transit handoff's methodology warning
first: `docs/handoff/HANDOFF_line_transit.md` — proxy metrics lied
all session; only ticks count; n=1 is an anecdote, not a finding.

═══════════════════════════════════════════════════════════════════
## DISCIPLINE (Alex-enforced, hard-won — violating these burned us)
═══════════════════════════════════════════════════════════════════

- ONE variable per experiment. Frames/visual evidence beat derived
  telemetry. Check COLLISION contact + judge clock before trusting
  any probe data.
- No unverified success prints; exit codes checked; per-run recording
  dirs. gidx2 only. Fresh sim only.
- When a theory fails twice, MEASURE instead of theorizing.
- Never declare "root cause found / solved" off a proxy metric.
- Commit checkpoints frequently on feat/vo-loop, push to origin
  (backup). Update THIS FILE at end of session: what was measured
  (with n), what's still assumed, exact pickup point.

## NON-GOALS (out of your lane)
- fastgate.py / vq2wp terminal & line-transit logic (trunk campaign,
  Alex's). Digit recognizer / placards (Alex). Map building (Janahan).
- No GateNet retraining, no contract edits in-branch, no touching
  `..\algo_src` (trunk worktree with uncommitted work).

## POINTERS
- Contract: `vq2/relpose.py`, `vq2/window_smoother.py` (factor defs)
- Bridge code: `vq2/live/{dpvo_odom,dpvo_odom_bridge,bridge_dpvo,
  dpvo_gate_scale,dpvo_route}.py`
- Design record: `docs/handoff/dpvo_scale_review.md` (scale debate,
  tick=aperture-not-center, external Sim3 wrapper rationale)
- Machine setup: `docs/vagon-bootstrap.md`; laws + judge protocol:
  `CLAUDE.md` (root)
- Truth: `vq2/truth_crossings.py`; corpora `C:\Users\Administrator\vq2_test*`
