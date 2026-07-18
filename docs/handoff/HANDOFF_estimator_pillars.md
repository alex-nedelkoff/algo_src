# HANDOFF — window-smoother estimator + pillar landmarks (2026-07-19)

**Branch `vq2-estimation`, commits this campaign:** `ca67d9d` VP module →
`e512f07` window smoother + pillar machinery → `d75e25b` SlidingSmoother →
`72c57cd` corpus segmentation fix → `7993105` gate-PnP pose/yaw (Phase C) →
`038a1ac` pillar-map contract + loader → `6896394` map v2 →
`95f3085` **azimuth-primary factors + map v3** (Alex's multi-marking
correction, DONE). All tests green (152). Plan of record:
`docs/superpowers/plans/2026-07-18-vq2-window-smoother-flight-integration.md`
(Phases A–C done, D pending). Working scripts in the session scratchpad
are listed at the bottom — they are NOT in the repo.

## ✅ Alex's correction is implemented and validated (2026-07-19)

> "The same pillar has multiple #22 markings, it's not three different
> pillars."

- `PillarFactor` (vq2/window_smoother.py) is now **azimuth-primary**: the
  height-independent bearing residual (tangential, range-scaled, m) is
  always applied; elevation-range (radial, sigma_r 1.5) ONLY when
  `top=True`. Regression test proves a lower-marking read with top=False
  helps instead of corrupting.
- **Top-read classification** (harness rule, port to production in Phase
  D): a read is identifiably-TOP if it is the highest of stacked
  same-number reads in a frame (|Δu|<30 px, min y, ≥2 in column) OR small
  (h ≤ 10 px → far → only the lit top panel is readable). Singleton close
  reads contribute azimuth only.
- **Map v3** (`vq2/pillar_map_v3.json`): one entry per PHYSICAL pillar
  with `markings[]` sub-features (contract §markings, loader validates).
  22c collapsed into 22b at the far-read position (23.13, 4.0) — same
  bearing from the course line (Δ1.5°) = one pillar; far reads see the
  lit top panel so their z assumption (hence xy) is the valid one.

**Result (tick-truth, six fresh-tick corpora 9–14):**
G1@tick **1.12 / 1.12 / 1.30 / 2.86 / 1.89 / 0.06 m** (median 1.21;
was 2.6–4.9). Transit medians 0.49–1.08 except test9 4.34 (15-s
pre-crash span). The ≤2 m bar holds on 5/6 — test12 (2.86) is the
outlier. 2×2 A/B (map v2/v3 × gated/always-top): the factor rework
carries the win; the map collapse is neutral-positive; top-gating ≈
always-top on these corpora (protection matters when close lower reads
dominate, which tick brackets dilute).

## ⚠ New measured facts (2026-07-19, don't re-derive)

- **The judge tick LAGS the physical G1 crossing ~0.75 s** (test10:
  GateNet range-to-G1 hits 2.5 m and flips to G2 at t=18.2; tick at
  18.96). G1@tick min-over-window scoring absorbs this; anything that
  treats the tick as the crossing instant must not.
- **Fusion-est translation is unusable as a survey baseline**: on test10
  the est moves ~1.5 m during a ~13 m approach (DR under-responsiveness —
  the same reason pillar anchors exist). Anything needing real positions
  must use gate-PnP pose events (Phase C junk-gate recipe).
- **Pure azimuth triangulation is degenerate on the approach leg**:
  perpendicular baseline 0.1–0.7 m; LSQ line-intersection collapses onto
  the flight line (bearings from ~one point intersect AT the drone).
  Usable geometry is the ALONG-track elevation profile with marking-z as
  a free unknown (survey_v3.py solves (x, y, z_marking) per bbox-u
  track). Sanity: high-travel tracks land within 1–1.5 m of the
  truth-surveyed 22a.
- **23b twin candidate at (24.03, 7.02)** (quarantined in map v3): from
  the two best-travel tracks of the whole survey (12.1/13.5 m, tests
  9/10, spread 0.85, marking z ≈ −3.1). Mapping it REGRESSED validation
  (test10 1.12→3.28, median 1.21→1.99): helps 11/12, hurts 9/10/13/14.
  Needs an independent fix before promotion. Same for the degenerate
  5-corpus '23' cluster at (8.2, 7.5) — consistent-but-biased, do not map.

## Key subsystems + their one-line truths

- `vq2/window_smoother.py` — batch + SlidingSmoother; capture-time factor
  folding; Huber IRLS; azimuth-primary pillar factors (top flag threads
  through push_pillar); **use scipy Cholesky (np.linalg.solve is 225×
  slower on this box — measured)**.
- Gate-PnP pose+yaw (Phase C): junk-solve gate = solve-implied roll/pitch
  must match the trusted chain ±4° (cuts 70 % of events, decisive).
- Pillar detector/OCR: 9–10 ms/frame CPU; identity reads ~10–15 % of obs;
  OCR floor 9×7 px (far panels unreadable by design; far reads = lit top
  panel — that asymmetry is what makes the far-read survey positions
  valid).
- DPVO: observe-only verdict stands; bridge protocol returns POSITION
  ONLY.
- Judge ticks: decode ENCAPSULATED_DATA hex, struct `<BQqqIq`;
  crash-aware transit metric ends at the first post-tick COLLISION row.
  Remember the 0.75 s tick lag above.

## Datum map (three coordinate worlds — do not mix)

1. **Janahan/offline datum** (`janahan_map_v1.json`, all offline work,
   pillar maps): G1 (10.595, 0.051), G2 (26.845, 9.139).
2. **Flight datum** (`course_map_plumbing.json` + pad-lock): G1 ≈ (6.28,
   0.02). Phase D must reconcile (range-scale calibration via pad-lock).
3. DPVO's per-session arbitrary frame (Sim2 alignment per corpus).

## Ops runbook

- **Fresh-sim restart** (judge law): focus window 'AI-GP' (EnumWindows),
  ESC → screenshot-verify menu → DOWN → screenshot-verify RESTART
  highlighted → ENTER → verify race clock < 15 s. Blind key chains
  desync (measured twice).
- **Flights:** `fly_testN.bat`; watch via `until grep -q exit
  fly_testN.log`. Wrecked drone persists after crashes → always
  fresh-restart before a scored run. Vagon bills per-minute — shut down
  when idle. GPU can CHANGE between sessions (T4→A10G) — re-check
  nvidia-smi before any CUDA build.
- **Pipeline per corpus:** GateNet pass (`vq2_detect_vagon.py`, ~200
  ms/frame GPU) → pillar scan+reads (`scan_batch.py`) → tick decode
  (`decode_ticks.py`).
- **DPVO bridge (laptop over tailnet):** ssh alexj@100.120.233.90;
  restart `wsl -d Ubuntu -- bash /mnt/c/Users/alexj/launch_bridge.sh`;
  aborted client sessions WEDGE the single-connection service.

## Scratchpad inventory (session-local, NOT in repo — port what earns it)

Session 915338c7 (2026-07-19) has everything from the previous session
(f4a76a39) plus: `survey_v3.py` (gate-PnP unknown-z survey),
`validate_v3.py` (map-v3 + top-gated harness — the canonical scorer now),
`validate_ab_v2.py`/`validate_ab_v3top.py`/`validate_ab_v2gated.py`/
`validate_v3b.py`/`validate_v3c.py` (the 2×2+ablation grid),
`map_v3b/v3c.json` (ablation maps). Carried over: `validate_harness.py`,
`survey_v2.py`, `validate_dpvo.py`, `replay_dpvo.py`, `scan_batch.py`,
`decode_ticks.py`, `vq2_detect_vagon.py`, `pillar_ocr.py`, per-corpus
`pillar_reads_*.jsonl` / `pillar_obs_*.jsonl` for tests 1–14.

## Next session, in order

1. **test12 outlier** (G1@tick 2.86): the one fresh tick over the bar —
   diagnose (pillar association? corridor? gate-PnP dropouts around its
   tick?).
2. **Phase D**: productionize detector/OCR into `vq2/pillars.py` with the
   top-read rule, datum reconciliation via pad-lock, SlidingSmoother
   beside the KF in vq2wp, observe-only flights (health triad jlogged).
3. **23b resolution** (optional, cheap on next flights): one deliberate
   slow pass by the 23-field would give the survey real parallax and
   settle 23a/23b in one corpus.
4. Race track in parallel: the G2 fgpursuit crash (6/6) is the qualifier
   blocker — line-follow/LINE_TRANSIT work is independent of all of this.
