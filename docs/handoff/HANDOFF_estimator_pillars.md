# HANDOFF — window-smoother estimator + pillar landmarks (2026-07-19)

**Branch `vq2-estimation`, commits this campaign:** `ca67d9d` VP module →
`e512f07` window smoother + pillar machinery → `d75e25b` SlidingSmoother →
`72c57cd` corpus segmentation fix → `7993105` gate-PnP pose/yaw (Phase C) →
`038a1ac` pillar-map contract + loader → `6896394` map v2.
All tests green (150+). Plan of record:
`docs/superpowers/plans/2026-07-18-vq2-window-smoother-flight-integration.md`
(Phases A–C done, D pending). Working scripts in the session scratchpad are
listed at the bottom — they are NOT in the repo.

## ⚠ START HERE — Alex's correction (2026-07-19)

> "The same pillar has multiple #22 markings, it's not three different
> pillars."

One physical pillar carries the station number at MULTIPLE heights/faces
(lit top panel ≈ z −7.16, face panels, vertical "Station NN" text at
mid-height). The survey pipeline assumed ALL readings sit at z = −7.16;
a lower marking therefore computes a longer elevation-range and plants a
**phantom pillar displaced along the view ray**. Map v2's `22b`
(23.13, 4.0) and `22c` (20.24, 3.35) are believed to be ONE pillar seen
via different markings. The measured "2.6–4.9 m approach floor" and the
"twin under-determination" verdict are downstream of this modeling error.

**The structural fix (next session's first job):**
1. **Azimuth-primary pillar factors** — the bearing to a *vertical* pillar
   is height-independent (any marking lies on the axis). Rework
   `PillarFactor` (vq2/window_smoother.py) so azimuth is always applied
   and elevation-range only when the marking is identifiably the TOP panel
   (e.g. highest read of that pillar in-frame, or panel-shape signature).
2. **Map v3: one entry per PHYSICAL pillar** — collapse same-number
   entries that lie along a common view-ray family (22b+22c), keep
   markings as sub-features with their own z. Contract already supports
   this direction (extend `landmarks[]` with a `markings[]` list).
3. Re-run the 9-corpus survey azimuth-only to re-derive pillar axes
   (script: `survey_v2.py`, gate/self-exclusion logic already built).

## Current measured state (tick-truth harness, janahan datum)

- **Approach (G1@tick):** best configs 1.2–1.7 m on day-3 corpora; on six
  fresh ticks 2.6–4.9 m (flight-dependent; partly the phantom-pillar
  factor corruption above).
- **Transit (pre-crash):** 0.2–0.9 m median on clean corpora.
- **6/6 G1 ticks** on 2026-07-18 flights (fly_test9..14, crossings
  5.1–6.1 s); ALL crash on the G2 fgpursuit leg (the known wall).
- 12 corpora banked (`vq2_test1..14`, no 7's tick, 5+8 est-reversed at
  tick — auto-excluded by the survey's kinematic sanity gate).

## Key subsystems + their one-line truths

- `vq2/window_smoother.py` — batch + SlidingSmoother; capture-time factor
  folding (absorbs 0.5–1 s staleness); Huber IRLS; **use scipy Cholesky
  (np.linalg.solve is 225× slower on this box — measured)**.
- Gate-PnP pose+yaw (Phase C harness): gate world frame fitted from 1262
  rest solves; **junk-solve gate = solve-implied roll/pitch must match the
  trusted chain ±4°** (cuts 70 % of events, decisive for quality).
- Pillar detector/OCR: 9–10 ms/frame CPU; identity reads ~10–15 % of obs;
  per-obs OCR floor at 9×7 px (far panels unreadable — by design now:
  runtime identity = close reads + map association).
- DPVO: **observe-only verdict stands.** Offline replay automated
  (867/532/833 poses on test9/10/11). A/B: robustness floor only —
  rescues the worst corpus, neutral elsewhere; ~half the stream discarded
  as keyframe-removal graph shifts (fix path: graph-shift-aware deltas
  via the keyframe counters in pose replies). **The bridge protocol
  returns POSITION ONLY — DPVO contributes zero bearing as wired.**
- Judge ticks: decode ENCAPSULATED_DATA hex, struct `<BQqqIq`; gidx2.py
  errors on these corpora. Crash-aware transit metric ends at the first
  post-tick COLLISION row.

## Datum map (three coordinate worlds — do not mix)

1. **Janahan/offline datum** (`janahan_map_v1.json`, all offline work,
   pillar maps): G1 (10.595, 0.051), G2 (26.845, 9.139). Offline GateNet
   range scale matches THIS datum (G1 at 11.06 m from pad).
2. **Flight datum** (`course_map_plumbing.json` + pad-lock): G1 ≈ (6.28,
   0.02). Phase D must reconcile (range-scale calibration via pad-lock).
3. DPVO's per-session arbitrary frame (Sim2 alignment per corpus).

## Ops runbook

- **Fresh-sim restart** (judge law): focus window 'AI-GP' (EnumWindows —
  the FlightSim PID has no window), ESC → screenshot-verify menu → DOWN →
  **screenshot-verify RESTART highlighted** → ENTER → verify race clock
  < 15 s. Blind key chains desync (measured twice).
- **Flights:** `fly_testN.bat` (clone of test6 config; RECORD dir per
  run). Watch via `until grep -q exit fly_testN.log`.
- **Pipeline per corpus:** GateNet pass (`scratchpad/vq2_detect_vagon.py`,
  ~200 ms/frame GPU) → pillar scan+reads (`scan_batch.py`) → tick decode
  (`decode_ticks.py`).
- **DPVO bridge (laptop over tailnet):** ssh key auth ready
  (`~/.ssh/id_ed25519` → alexj@100.120.233.90, Windows OpenSSH). Restart:
  `ssh alexj@100.120.233.90 "wsl -d Ubuntu -- bash /mnt/c/Users/alexj/launch_bridge.sh"`.
  Aborted client sessions WEDGE the single-connection service → bounce it.
  Replay: `scratchpad/replay_dpvo.py N…` (120 s cold-start timeout).
- Wrecked drone persists after crashes → always fresh-restart before a
  scored run. Vagon bills per-minute — shut down when idle.

## Scratchpad inventory (session-local, NOT in repo — port what earns it)

`validate_harness.py` (Phase C + tick scoring, reads pillar_map_v2.json),
`survey_v2.py` (9-corpus survey, anchor self-gate), `validate_dpvo.py`
(DPVO A/B), `replay_dpvo.py`, `scan_batch.py`, `decode_ticks.py`,
`vq2_detect_vagon.py` (Vagon-pathed GateNet batch), `pillar_harvest/`
`pillar_cluster/pillar_ocr.py` (detector+OCR, prototypes reference local
corpora), `fix_trace.py`, `tick_score.py`, `map_v2.py`.

## Next session, in order

1. Azimuth-primary pillar factors + map v3 (physical pillars, markings as
   features) — re-survey, re-validate on the 9 ticked corpora.
2. If approach ≤ 2 m holds across fresh ticks → Phase D: productionize
   detector/OCR into `vq2/pillars.py`, datum reconciliation via pad-lock,
   SlidingSmoother beside the KF in vq2wp, observe-only flights (health
   triad jlogged).
3. Race track in parallel: the G2 fgpursuit crash (6/6) is the qualifier
   blocker — line-follow/LINE_TRANSIT work is independent of all of this.
