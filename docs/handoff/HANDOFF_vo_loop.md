# HANDOFF — feat/vo-loop: VO (DPVO) into the smoother (2026-07-21)

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
