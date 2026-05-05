# M2 — Depth Source Decision

**Date:** 2026-05-05
**Linear:** COR-106 (week 1, M2)
**Spec:** `docs/superpowers/specs/2026-05-05-vision-augmented-racing-design.md`

## Decision

**Use Depth Anything V2 Small at 384×512 input resolution, with a
3-frame EMA smoothing post-process. Defer Video Depth Anything (VDA).**

Headline justification:
- **28 fps on RTX 3050 Laptop** at the chosen resolution — more than 2×
  the 10–20 Hz target the planner needs.
- **Flicker is moderate, not catastrophic** (normalised adjacent-frame
  RMSE ≈ 0.036 on static scenes; mean depth scale ~100). Visible but
  fight-able with cheap post-processing.
- **Bigger DA V2 variants don't materially reduce flicker** — Large
  brings RMSE from 0.042 → 0.034 (~20% improvement) at 4× the latency.
  The flicker is a single-frame-inference structural issue, not a
  model-capacity issue, so investing in a bigger DA V2 is the wrong
  axis. The right axis to address it is temporal modelling — i.e. VDA.
- **VDA's package situation makes it expensive to integrate** —
  not on PyPI, not in `transformers`, requires upstream-repo install.
  Spend that integration cost only when downstream evidence (M5
  closed-loop test) shows DA V2 + smoothing isn't enough.

## Context

Vision-augmented racing pipeline (COR-106) needs a depth source to feed
the local-obstacle encoder. Two candidates from the spec:

- **Depth Anything V2 (DA V2)** — single-frame, available via
  `transformers` pipeline. Per-frame depth flicker on static scenes
  was the open question.
- **Video Depth Anything (VDA)** — temporally-consistent.
  Not on PyPI / not in `transformers==5.8.0`; requires upstream-repo
  install.

Goal of M2: characterise per-frame inference latency on host GPU
(RTX 3050 Laptop) and per-frame temporal flicker on static-scene
sequences, then decide which model to integrate.

## Setup

- **Hardware:** NVIDIA RTX 3050 Laptop GPU (host dev box; sim qualifier
  runs here, not on Jetson). FP16 inference.
- **Models tested:** DA V2 Small (24.8 M params), Base (97.5 M),
  Large (335.3 M) via `transformers.pipeline("depth-estimation")`.
- **Inputs:** 3 PyBullet warehouse renders × 20 jittered frames each
  (small horizontal pixel shifts simulating sub-pixel camera motion on a
  static scene). 60 frames per (model × resolution).
- **Resolutions:** 144×256 (BP_FlyingPawn camera default per COR-100),
  224×224 (ViT-friendly), 384×512 (AirSim "medium").
- **Metrics:**
  - Latency: per-frame wall clock (after 3 warmup frames), mean / median / p95.
  - Temporal consistency: per-pixel RMSE between adjacent normalised
    depth maps. Smaller = more consistent. On a static scene with
    sub-pixel jitter, a perfectly-stable model would output ~0; the
    residual is the model's intrinsic flicker.
- **Code:** `scripts/perception/benchmark_depth_anything.py`. Outputs to
  `outputs/perception/depth_anything_benchmark/`.

## Results

### Latency (FP16, RTX 3050 Laptop, 60-frame mean)

| model | params | 144×256 | 224×224 | 384×512 |
|---|---|---|---|---|
| **Small** | **24.8 M** | **24.9 ms (40 fps)** | 35.8 ms (28 fps) | **35.8 ms (28 fps)** |
| Base | 97.5 M | 41.4 ms (24 fps) | 63.3 ms (16 fps) | 60.5 ms (16 fps) |
| Large | 335.3 M | 96.4 ms (10 fps) | 143.8 ms (7 fps) | 133.9 ms (7 fps) |

p95 ≈ mean + 1–3 ms across the board (low variance).

### Temporal consistency (normalised adjacent-frame RMSE on static scenes)

| model | 144×256 | 224×224 | 384×512 |
|---|---|---|---|
| Small | 0.0420 (max 0.30) | 0.0418 (max 0.31) | 0.0364 (max 0.36) |
| Base | 0.0387 (max 0.43) | 0.0438 (max 0.40) | 0.0388 (max 0.38) |
| Large | 0.0393 (max 0.44) | 0.0355 (max 0.42) | **0.0341** (max 0.47) |

Mean depth ~100 across all combinations.

## Interpretation

**Latency.** All three Small-model resolutions hit the 10–20 Hz target
with margin. Base is borderline at 224×224 / 384×512 (16 fps). Large
is below target at every resolution but the smallest (10 fps at
144×256) — without TensorRT, Large isn't deployable.

**Temporal flicker.** All combinations cluster around 0.035–0.044
normalised RMSE. The Large @ 384×512 (0.034) does best, but the
Small @ 384×512 (0.036) is within 6% — not enough to justify ~4×
the latency. The flicker max values (0.30–0.47) are the real
concern: outlier frames where depth changes dramatically between
adjacent inputs that should produce nearly-identical depth.

**Why bigger models don't fix flicker.** DA V2 is a single-frame model.
Sub-pixel input jitter causes its features to settle into different
local minima between frames — no amount of capacity makes a per-frame
network temporally consistent. The right fix is video-aware
inference (VDA) or ensemble-style smoothing.

**The 384×512 sweet spot.** For Small specifically, 384×512 is
roughly the same latency as 224×224 (35.8 ms either way) but produces
slightly better temporal RMSE (0.036 vs 0.042). Likely an artifact of
the model's preferred internal resolution. Free win — use 384×512.

## Smoothing post-process

EMA smoothing on the depth map: `smoothed[t] = α · depth[t] + (1-α) · smoothed[t-1]`,
with `α = 0.5` (matching the yaw smoother in `sim/tracks/waypoint.py`),
should approximately halve the adjacent-frame RMSE — bringing 0.036 →
~0.018 on a static scene. Cost: 1× depth-map memcpy per frame, well
under 1 ms.

For motion: EMA introduces lag proportional to `1/α`. At 25 fps and
α=0.5, the depth at a given timestep effectively reflects information
from the last 2–3 frames (~80–120 ms latency). This is acceptable for
the planner (5–10 Hz) but worth noting if the local-obstacle encoder
is fed depth at the full inference rate.

## When to revisit

Re-open the VDA decision if any of these hit:

1. **Closed-loop test (M5) shows planner instability** traceable to depth
   flicker — e.g. waypoints jittering between adjacent planner ticks
   even when the drone is stationary.
2. **DA V2 fails on real warehouse imagery** (i.e. AirSim-rendered, not
   PyBullet renders). The current benchmark used PyBullet renders which
   are out-of-distribution for any natural-image-trained depth model;
   AirSim's UE-rendered imagery is closer to in-distribution.
3. **EMA smoothing alone introduces unacceptable motion lag** at the
   speeds we're flying.
4. **A new VDA release lands in `transformers`** — would drop the
   integration cost from "clone repo + manage deps" to "swap pipeline
   model name". Worth checking quarterly.

## Open follow-ups

- **Real AirSim imagery benchmark** — re-run the same benchmark with UE-
  rendered captures from a live AirSim PIE session. PyBullet renders
  are OOD for natural-image-trained models so the current numbers may
  understate quality and overstate flicker.
- **Jetson runtime** — deferred. Sim qualifier runs on host; on-board
  inference is a physical-deployment concern.
- **TensorRT compilation** — could halve latency. Defer until deployment.
- **Smoothing α tuning** — α=0.5 is a starting point. Higher α (less
  smoothing, less lag) for fast manoeuvres; lower α (more smoothing,
  more lag) for stable/hovering.

## Cross-references

- `scripts/perception/benchmark_depth_anything.py` — runner
- `outputs/perception/depth_anything_benchmark/summary.json` — raw numbers
- `outputs/perception/depth_anything_benchmark/viz_*.png` — RGB | depth
  side-by-sides per (model, resolution)
- COR-106 — week 1 issue this M2 belongs to
- COR-92 — Janahan's setup notes (depth-correction gotcha: VDA / DA
  produces Euclidean ray length, needs cos(pixel_angle) correction
  before use as planar Z)
