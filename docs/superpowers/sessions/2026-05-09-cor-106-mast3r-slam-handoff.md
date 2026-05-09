# Session handoff — COR-106 MASt3R-SLAM Phase 1, paused 2026-05-09

This is a one-page brief for a fresh Claude Code session to pick up where the previous one left off. Read top-to-bottom; everything else is referenced by path.

## Current status

- **Branch**: `cor-106-mast3r-slam-cloud-phase1` on `alex-nedelkoff/algo_src` (personal fork)
- **Linear**: COR-106 (vision-augmented racing policy) — `get_issue` + `list_comments` for full context
- **Cloud**: RunPod RTX 3090 instance with `/workspace/algo_src` checked out, all bootstrap deps installed, MASt3R ViT-Large checkpoint (2.75 GB) on volume disk. Pod may be stopped (volume disk persists across stop) — needs Start before iteration.
- **Phase**: 1 (validate MASt3R-SLAM tracks PyBullet-rendered RGB before closed-loop integration). NOT yet green.

## What works

- Build + install end-to-end on Linux RunPod (and on Windows local with workarounds, see `reference_mast3r_slam_install` memory)
- Procedurally textured 14×10×3 m room with 4 CC0 Polyhaven materials at `sim/assets/textured_room_v1/`. `room.obj` is triangulated (open3d quirk).
- Mast3rLocalizer constructs, model loads (688.6M params on cuda), env←slam alignment math correct
- Bootstrap script `scripts/cloud/bootstrap.sh` is idempotent and survives a fresh pod
- Smoke test infra: scripted oval flight, dual ICP+MASt3R-SLAM logging, rerun .rrd output, web viewer on port 8888

## What we tried, what failed

**Racing-speed smoke test (7.37 m/s mean speed) failed**: both ICP and MASt3R-SLAM diverged from GT after frame 0. Per the rerun output:
- ICP error spiked to ~1000 cm at t=1s, dropped back as the GT oval looped past the stuck estimate
- MASt3R error spiked to ~700 cm with similar pattern
- ICP fitness stayed 0.7-0.9 — it converged but to a wrong local minimum (likely a different wall of the symmetric room geometry)
- Camera RGB was dim and feature-poor; PyBullet's EGL plugin failed to load on the pod, falling back to TINY_RENDERER

Hypothesis: **both** failures share an upstream cause (lighting / speed × correspondence radius), not localizer-specific bugs.

## Where you should pick up

A diagnostic variant exists at `scripts/perception/mast3r_slam_smoketest_easy.py` (commit `95808fd`). Three modes ranked easiest → hardest:

1. **`--mode static`**: drone parked, looking at metal_plate wall. Validates that ICP returns ~zero error when fed GT. If this fails, frame-convention bug in `R_BODY_TO_CAM` or `pose_to_matrix` shared by both localizers.
2. **`--mode line`**: slow forward translation along +x, no rotation. Validates motion handling.
3. **`--mode oval`**: same as racing test but at ~1 m/s instead of 7.37.

Defaults for the diagnostic: ICP correspondence radius 5 cm (vs 20 in racing test), GT-prior-every-frame (no carry-forward), brighter lighting (`lightAmbientCoeff=1.0`).

### Immediate next commands on the pod

```bash
cd /workspace/algo_src
git pull origin cor-106-mast3r-slam-cloud-phase1

# Static — should print near-zero error if localizers are correct
python -m scripts.perception.mast3r_slam_smoketest_easy --mode static --n-frames 30

# Then escalate
python -m scripts.perception.mast3r_slam_smoketest_easy --mode line --n-frames 50
python -m scripts.perception.mast3r_slam_smoketest_easy --mode oval --n-frames 100
```

Per-frame stdout shows: `t=N fit=X.XXX rmse=X.X cm corr_n=N pos_err=X.X cm GT=[…] est=[…]`. Watch for the first frame where `pos_err` blows up.

To view the resulting `.rrd` (Mac side): scp it from the pod (RunPod's *Connect → SSH over exposed TCP* gives the host/port), then `rerun mast3r_smoketest_easy_<mode>.rrd` locally.

## Relevant files

| Path | Purpose |
|---|---|
| `perception/localization/mast3r_localizer.py` | The localizer — env←slam alignment, lietorch.Sim3 → 4×4, opt-in 4 GB workarounds |
| `perception/localization/icp_localizer.py` | The ICP baseline — depth + map cloud, point-to-plane |
| `scripts/perception/mast3r_slam_smoketest.py` | Original racing-speed smoke test (failed) |
| `scripts/perception/mast3r_slam_smoketest_easy.py` | **Diagnostic variant — start here** |
| `sim/assets/textured_room_v1/build_room.py` | Procedural room generator (regen if needed) |
| `scripts/cloud/{bootstrap,run_smoke}.sh` | RunPod setup + smoke test runner |
| `scripts/cloud/README.md` | End-to-end cloud workflow |
| `docs/superpowers/artifacts/2026-05-08-cor-106-mast3r-slam-phase1-report.html` | Mid-progress report (status: in_progress) |

## Open questions / decisions parked

- **EGL plugin failure on RunPod**: PyBullet falls back to TINY_RENDERER which works for textures but lighting is dim. May need `apt install libnvidia-gl-<driver-major>` or similar. Could also bake brighter values into the textures themselves.
- **ICP correspondence radius**: 20 cm let it latch onto wrong walls. 5 cm in diag may be too tight; tune.
- **MASt3R feature contrast**: untested whether brighter lighting alone is enough or if we need genuinely photoreal input (Phase 2 splat reconstruction).

## Memory files relevant to this work

These persist across Claude sessions on the same machine. On the Mac they won't be present — the new session will rebuild them as topics come up.

- `reference_mast3r_slam_install` — Windows install patches (CUDA_HOME, long→int64_t, SingleProcessManager shim)
- `reference_rerun_drone_fpv` — body→cam rotation matrix (`R_BODY_TO_CAM` here)
- `project_perception_complexity` — context for why we need on-board localization
- `project_competition_platform` — DCL/Anduril sim is Windows-native

## How to invoke the next session

In your Mac terminal, in the algo_src project root:

```bash
git pull
git checkout cor-106-mast3r-slam-cloud-phase1
claude  # or however you invoke Claude Code
```

First prompt: *"Read `docs/superpowers/sessions/2026-05-09-cor-106-mast3r-slam-handoff.md` and `get_issue COR-106` to load context. We left off about to run the easy-mode diagnostic smoke test on the RunPod pod — start there."*
