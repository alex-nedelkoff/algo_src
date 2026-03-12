# In-Training Artifact Pipeline Design

**Date**: 2026-03-11
**Status**: Draft
**Related**: COR-53 (experiment tracking & artifact pipeline), COR-51 (viz & enhanced metrics)
**Builds on**: `docs/superpowers/specs/2026-03-11-experiment-tracking-pipeline-design.md`

## Context

COR-53 established the post-hoc artifact pipeline: training produces `.npz` trajectories and model checkpoints locally, then `sync_artifacts.py` uploads everything to R2 and registers with W&B after training completes. COR-51 added trajectory recording (`TrajectoryRecorderCallback`) and Rerun visualization (`rerun_generator.py`).

The post-hoc approach has two problems:
1. **No visibility during training** — Rerun visualizations and trajectory data aren't available until someone runs `make sync-artifacts` after training ends.
2. **Artifact loss on ephemeral instances** — if a cloud instance terminates before the post-hoc sync, trajectory and Rerun artifacts are lost.

Metrics (reward curves, gate success rates, lap times) already stream to W&B in real-time via `sync_tensorboard=True`. This design adds real-time visibility for **visual artifacts** (Rerun 3D visualizations and raw trajectory data).

## Goal

Rerun viewer links appear in W&B within seconds of each eval checkpoint during training, so teammates can inspect 3D flight visualizations while training is still running.

## Design

### 1. Scope Split: In-Training vs. Post-Hoc

| Artifact type | Upload path | Rationale |
|---------------|------------|-----------|
| Trajectories (`.npz`, ~100KB) | In-training (`ArtifactUploader`) | Small, fast to upload, needed for visibility |
| Rerun archives (`.rrd`, ~2-5MB) | In-training (`ArtifactUploader`) | What users want to see ASAP |
| Model checkpoints (`ppo_*_steps.zip`, 50-200MB) | Post-hoc (`sync_artifacts.py`) | Large files, network issues shouldn't affect training |
| Best model (`best_model.zip`) | Post-hoc (`sync_artifacts.py`) | Same as checkpoints |
| Final model (`final_model.zip`) | Post-hoc (`sync_artifacts.py`) | Same as checkpoints |
| Config (`config.yaml`) | Post-hoc (`sync_artifacts.py`) | Tiny, no urgency |

### 2. Component Architecture

```
Training Loop
  │
  ├─ GateMetricsCallback → TensorBoard → W&B (unchanged)
  ├─ EvalCallback → TensorBoard → W&B (unchanged)
  ├─ CheckpointCallback → local checkpoints/ (unchanged)
  │
  ├─ TrajectoryRecorderCallback
  │    saves .npz → generate_rrd() → uploader.submit(npz, rrd)
  │
  └─ ArtifactUploader (daemon thread + queue)
       upload .npz + .rrd to R2 → register W&B artifacts

Post-training
  │
  └─ sync_artifacts.py (unchanged)
       uploads checkpoints, best_model, final_model, config
       skips .npz/.rrd already uploaded (idempotency)
```

Three modules involved:

| Module | Responsibility | New/Changed |
|--------|---------------|-------------|
| `artifacts/r2.py` | Shared R2 upload + idempotency logic | New (extracted from `sync_artifacts.py`) |
| `artifacts/uploader.py` | `ArtifactUploader` class — background thread, queue, W&B registration | New |
| `control/trajectory_recorder.py` | After saving `.npz`, calls `generate_rrd()` then `uploader.submit()` | Changed (~10 lines) |
| `scripts/sync_artifacts.py` | Post-hoc checkpoint upload, imports from `artifacts/r2.py` | Refactored (uses shared module) |
| `control/__main__.py` | Creates `ArtifactUploader`, passes to callback, calls `close()` at end | Changed (~10 lines) |

### 3. Data Flow

**During training** (every `viz_freq` steps):

```
TrajectoryRecorderCallback._on_step()
  │
  ├─ 1. Run eval episodes, save .npz to checkpoints/trajectories/step_{ts}/
  ├─ 2. generate_rrd(npz_path) → .rrd saved alongside .npz (CPU-only, ~seconds)
  └─ 3. uploader.submit(npz_path, rrd_path, step)
       │
       └─ ArtifactUploader (background thread)
            ├─ Upload .npz to R2: runs/{run_id}/trajectories/step_{ts}/eval_ep_{i}.npz
            ├─ Upload .rrd to R2: runs/{run_id}/rerun/step_{ts}/eval_ep_{i}.rrd
            └─ Log to W&B run:
                 ├─ Reference artifact with R2 URLs
                 └─ Rerun viewer URL (https://app.rerun.io/version/.../?url=...)
```

### 4. ArtifactUploader Lifecycle

```python
__init__(run_id, r2_config, wandb_run)
  # Creates R2 client (boto3)
  # Starts daemon thread

submit(npz_path, rrd_path, step)
  # Puts work item on queue (non-blocking, instant return)

_worker()  # daemon thread loop
  # Pulls from queue
  # Uploads via artifacts/r2.py (shared logic)
  # Registers W&B artifacts
  # Logs errors, never raises

close(timeout=60)
  # Sends sentinel to queue
  # Joins thread with timeout
  # Logs any items that didn't make it
```

**Wiring in `control/__main__.py`:**
```python
uploader = ArtifactUploader(run_id=wandb_run.id, r2_config=cfg.r2, wandb_run=wandb_run)
trajectory_callback = TrajectoryRecorderCallback(..., uploader=uploader)
# ... training ...
uploader.close(timeout=60)  # best-effort drain
model.save("final_model")
# User runs: make sync-artifacts RUN_DIR=outputs/...
```

### 5. Shared Upload Module (`artifacts/r2.py`)

Extracted from `sync_artifacts.py` so both `ArtifactUploader` and `sync_artifacts.py` share identical upload logic:

| Function | Purpose |
|----------|---------|
| `make_r2_client(account_id, access_key, secret_key)` | Creates boto3 S3 client pointed at R2 |
| `upload_file(client, local_path, bucket, r2_key)` | Upload with idempotency (size + etag check) |
| `register_wandb_artifact(wandb_run, r2_key, artifact_type)` | Log R2 URL as W&B reference artifact |
| `rerun_viewer_url(r2_public_base, r2_key, rerun_version)` | Build `app.rerun.io` viewer URL |

`scripts/sync_artifacts.py` becomes a thinner orchestrator — it still handles discovery, priority ordering, and the CLI interface, but delegates upload/check to `artifacts/r2.py`.

### 6. Failure Handling

- **Upload failure**: Logged as warning, never raised. Training continues unaffected. `sync_artifacts.py` catches missed uploads post-hoc via idempotency checks.
- **R2 credentials missing**: `ArtifactUploader.__init__` logs a warning and enters a no-op mode. `.rrd` files are still generated locally (available for post-hoc upload).
- **W&B not initialized**: Uploader skips W&B registration, still uploads to R2.
- **Training crash**: Daemon thread dies with the process. Any `.npz`/`.rrd` files already on disk are picked up by `sync_artifacts.py` later.
- **`close()` timeout**: Items remaining in queue are logged. `sync_artifacts.py` handles them.

### 7. What Doesn't Change

- Metrics pipeline (GateMetricsCallback → TensorBoard → W&B)
- Checkpoint saving (CheckpointCallback, EvalCallback)
- Post-hoc checkpoint upload flow (`make sync-artifacts`)
- Rerun generator code (`sim/viz/rerun_generator.py`)
- Config files
- R2 bucket structure (same paths as COR-53 spec)

### 8. Non-Goals

- In-training checkpoint upload — stays post-hoc via `sync_artifacts.py`
- Artifact versioning / DVC
- Self-hosted experiment tracker
- Priority queue — all in-training artifacts are small (~5MB total per eval), no contention

### 9. Dependencies

No new dependencies. `boto3` (R2 uploads) and `rerun-sdk` (.rrd generation) are already in `pyproject.toml` optional groups `[artifacts]` and `[viz]` respectively.

### 10. File Changes Summary

| File | Change | Lines (est.) |
|------|--------|-------------|
| `artifacts/__init__.py` | New — package init | ~0 |
| `artifacts/r2.py` | New — extracted R2 upload core | ~120 (moved from sync_artifacts.py) |
| `artifacts/uploader.py` | New — `ArtifactUploader` class | ~80 |
| `control/trajectory_recorder.py` | Changed — add `generate_rrd()` + `uploader.submit()` after .npz save | ~10 lines added |
| `control/__main__.py` | Changed — create uploader, pass to callback, `close()` at end | ~10 lines added |
| `scripts/sync_artifacts.py` | Refactored — import from `artifacts/r2.py` instead of inline logic | Net negative (smaller) |

### 11. End-to-End User Experience

1. Start training → metrics appear in W&B immediately (existing)
2. Every `viz_freq` steps → Rerun viewer links appear in W&B within seconds (new)
3. Teammate opens W&B → sees metrics + clicks Rerun link → 3D viz opens in browser (new)
4. Training ends → `make sync-artifacts` uploads checkpoints and models (existing)
5. `sync_artifacts.py` skips trajectories/Rerun already uploaded during training (existing idempotency)
