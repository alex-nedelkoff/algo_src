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

**Note**: The COR-53 spec listed "automated upload during training" as a non-goal to avoid network issues affecting training stability. This design supersedes that non-goal for small artifacts only (trajectories + Rerun). Experience showed that deferring all uploads to post-hoc leaves the ephemeral-instance data loss problem unsolved. Model checkpoints (the large, network-sensitive files) remain post-hoc.

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
  │    saves .npz → uploader.submit(npz_path, step)
  │
  └─ ArtifactUploader (daemon thread + queue)
       generate_rrd() → upload .npz + .rrd to R2 → register W&B artifacts

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
| `control/trajectory_recorder.py` | After saving `.npz`, calls `uploader.submit()` | Changed (~5 lines) |
| `scripts/sync_artifacts.py` | Post-hoc checkpoint upload, imports from `artifacts/r2.py` | Refactored (uses shared module) |
| `control/__main__.py` | Creates `ArtifactUploader`, passes to callback, calls `close()` at end | Changed (~10 lines) |

### 3. Data Flow

**During training** (every `viz_freq` steps):

```
TrajectoryRecorderCallback._on_step()
  │
  ├─ 1. Run eval episodes, save .npz to checkpoints/trajectories/step_{ts}/
  └─ 2. uploader.submit(npz_path, step)  # non-blocking, instant return
       │
       └─ ArtifactUploader (background thread)
            ├─ generate_rrd(npz_path) → .rrd saved alongside .npz
            ├─ Upload .npz to R2: runs/{run_id}/trajectories/step_{ts}/eval_ep_{i}.npz
            ├─ Upload .rrd to R2: runs/{run_id}/rerun/step_{ts}/eval_ep_{i}.rrd
            └─ Register W&B artifacts via wandb.Api() (thread-safe public API):
                 ├─ Reference artifact with R2 URLs
                 └─ Rerun viewer URL (https://app.rerun.io/version/.../?url=...)
```

`.rrd` generation runs on the background thread, not the training thread. `generate_rrd()` iterates every timestep and renders wireframe cameras — this can take more than a few seconds for long episodes. Keeping it off the main thread avoids extending the already-blocking eval rollout window.

### 4. ArtifactUploader Lifecycle

```python
__init__(run_id, wandb_entity, wandb_project)
  # run_id: W&B run ID, also used as R2 path prefix (runs/{run_id}/...)
  # wandb_entity/wandb_project: needed to construct run_path for wandb.Api()
  # Creates R2 client (boto3) from env vars:
  #   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
  # Creates wandb.Api() for thread-safe artifact registration
  # Starts daemon thread

submit(npz_path, step)
  # Puts work item on queue (non-blocking, instant return)

_worker()  # daemon thread loop
  # Pulls from queue
  # Calls generate_rrd(npz_path) → .rrd file
  # Uploads .npz + .rrd via artifacts/r2.py (shared logic)
  # Registers W&B artifacts via wandb.Api() (not the live Run object)
  # Logs errors, never raises

close(timeout=60)
  # Sends sentinel to queue
  # Joins thread with timeout
  # Logs any items that didn't make it
```

**R2 credentials**: Read from environment variables (same as `sync_artifacts.py`), not Hydra config. Secrets stay out of config files.

**W&B thread safety**: The background thread uses `wandb.Api()` (the public REST API client) to register artifacts, not the live `wandb.Run` object. The `Run` object is not thread-safe and is already used by `sync_tensorboard` on the main thread. `wandb.Api()` is independent and safe to use from any thread.

**Wiring in `control/__main__.py`:**
```python
uploader = ArtifactUploader(
    run_id=wandb_run.id,
    wandb_entity=wandb_run.entity,
    wandb_project=wandb_run.project,
)
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
| `register_wandb_artifact(api, run_path, r2_key, artifact_type)` | Log R2 URL as W&B reference artifact via `wandb.Api()` |
| `rerun_viewer_url(r2_public_base, r2_key, rerun_version)` | Build `app.rerun.io` viewer URL |

`scripts/sync_artifacts.py` becomes a thinner orchestrator — it still handles discovery, priority ordering, and the CLI interface, but delegates upload/check to `artifacts/r2.py`. As part of this refactor, `sync_artifacts.py` will also switch from `wandb.init(resume="allow")` to `wandb.Api()` for artifact registration, matching the `ArtifactUploader`. This eliminates the risk of two processes writing to the same W&B run if `sync_artifacts.py` is triggered (e.g., via `trap EXIT`) while training is still active.

### 6. Failure Handling

- **Upload failure**: Logged as warning, never raised. Training continues unaffected. `sync_artifacts.py` catches missed uploads post-hoc via idempotency checks.
- **R2 credentials missing**: `ArtifactUploader.__init__` logs a warning and enters a no-op mode. `.npz` files are still saved locally (available for post-hoc upload).
- **`rerun-sdk` not installed**: `_worker` catches `ImportError` from `generate_rrd()`, logs a warning, and uploads the `.npz` only (no `.rrd`). Training is unaffected.
- **W&B not initialized**: Uploader skips W&B registration, still uploads to R2.
- **Training crash**: Daemon thread dies with the process. Any `.npz`/`.rrd` files already on disk are picked up by `sync_artifacts.py` later.
- **`close()` timeout**: Items remaining in queue are logged. `sync_artifacts.py` handles them.
- **Queue backpressure**: Queue has `maxsize=10`. If uploads fall behind (slow network + low `viz_freq`), `submit()` drops the oldest item and logs a warning. Dropped items still have their `.npz` on disk — to generate `.rrd` for dropped items, run `python -m sim.viz <trajectory_dir>` and re-sync. In practice this won't happen — each work item is ~5MB and `viz_freq` defaults to 1M timesteps.

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
| `artifacts/uploader.py` | New — `ArtifactUploader` class (thread, queue, .rrd generation, upload, W&B registration) | ~100 |
| `control/trajectory_recorder.py` | Changed — call `uploader.submit()` after .npz save | ~5 lines added |
| `control/__main__.py` | Changed — create uploader, pass to callback, `close()` at end | ~10 lines added |
| `scripts/sync_artifacts.py` | Refactored — import from `artifacts/r2.py` instead of inline logic | Net negative (smaller) |

### 11. End-to-End User Experience

1. Start training → metrics appear in W&B immediately (existing)
2. Every `viz_freq` steps → Rerun viewer links appear in W&B within seconds (new)
3. Teammate opens W&B → sees metrics + clicks Rerun link → 3D viz opens in browser (new)
4. Training ends → `make sync-artifacts` uploads checkpoints and models (existing)
5. `sync_artifacts.py` skips trajectories/Rerun already uploaded during training (existing idempotency)
