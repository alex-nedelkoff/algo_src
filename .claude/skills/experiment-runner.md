---
name: experiment-runner
description: Guide team members through setting up and running training experiments with W&B tracking and R2 artifact storage. Use when someone asks how to run training, launch an experiment, set up W&B, configure R2 uploads, check training results, monitor metrics, or troubleshoot experiment infrastructure. Also triggers on "how do I train", "run an experiment", "set up logging", "where are my W&B metrics", "launch monorace", "launch playground".
---

# Experiment Runner

Guides you through setting up and running training experiments with full W&B metric tracking and R2 artifact storage.

## First-Time Setup

### 1. Get credentials

Create a `.env` file at the repo root (`algo_src/.env`). It's gitignored — never commit it.

**W&B API key** — get your own from [wandb.ai/authorize](https://wandb.ai/authorize). Create a free account if you don't have one, then join the `corvidx-drone-racing` project (ask the team lead for an invite).

**R2 credentials** — ask the team lead for the R2 keys (shared via email). These are shared across the team.

Your `.env` should look like:

```
# Your personal W&B key
WANDB_API_KEY=wandb_v1_your_key_here

# Shared R2 credentials (from team lead)
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=corvidx-artifacts
```

The training entrypoint auto-loads `.env` via `_load_dotenv()`, so R2 credentials are picked up automatically. `WANDB_API_KEY` must also be exported to the shell for Docker compose environment variable substitution.

### 2. Create docker-compose.override.yml

```bash
cp docker-compose.override.yml.example docker-compose.override.yml
```

Edit the `control` service to pass the W&B key and set your experiment:

```yaml
services:
  control:
    volumes:
      - ./outputs:/app/outputs
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
    command: ["python", "-m", "training", "+experiment=monorace_baseline"]
```

Change `+experiment=monorace_baseline` to whichever experiment you want to run.

### 3. Verify GPU access

Training requires an NVIDIA GPU with Docker GPU support:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Available Experiments

| Config | Environment | Description |
|--------|------------|-------------|
| `monorace_baseline` | GateRaceEnv | MonoRace M23 baseline, 100M steps |
| `playground_phase1` | RateCtrlEnv | Playground SymPy dynamics, 100M steps |
| `playground_phase2` | RateCtrlEnv | Phase 2 (curriculum) |
| `playground_phase3` | RateCtrlEnv | Phase 3 (perception noise) |
| `playground_phase4` | RateCtrlEnv | Phase 4 (sim-to-real) |
| `asymmetric_critic` | GateRaceEnv | Asymmetric actor-critic |
| `perception_noise` | GateRaceEnv | With perception noise injection |
| `ablation_no_dr` | GateRaceEnv | No domain randomization (ablation) |
| `smoke_test` | GateRaceEnv | Quick 50K step sanity check (4 envs) |

To see the full config for any experiment:

```bash
python -m training +experiment=monorace_baseline --cfg job
```

## Launching Training

### Quick start

```bash
source .env && export WANDB_API_KEY && make train
```

This builds the GPU Docker image and launches the control service with your override config.

### Direct docker run (alternative)

```bash
source .env && export WANDB_API_KEY
docker run --rm --gpus all \
  -e WANDB_API_KEY \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/outputs:/app/outputs \
  algo-src-control \
  python -m training +experiment=monorace_baseline
```

### CLI overrides

Override any config value from the command line:

```yaml
# In docker-compose.override.yml
command: ["python", "-m", "training", "+experiment=monorace_baseline",
          "sim.n_envs=200", "total_timesteps=50_000_000", "seed=123"]
```

### Smoke test first

Before a long run, verify everything works with the smoke test config (50K steps, ~1 min):

```yaml
command: ["python", "-m", "training", "+experiment=smoke_test"]
```

## W&B Metrics

All experiments log to the `corvidx-drone-racing` W&B project. The metrics contract (`metrics/contract.py`) ensures consistent panels across all environments.

### Racing metrics (`racing/` prefix)

| Panel | What it shows |
|-------|--------------|
| `gates_per_ep` | Mean gates passed per episode |
| `laps_per_ep` | Mean laps completed per episode |
| `success_rate` | Fraction of episodes ending successfully |
| `gate_passage_rate` | Fraction of episodes passing at least 1 gate |
| `lap_completion_rate` | Fraction of episodes completing at least 1 lap |
| `lap_time_mean` / `lap_time_best` | Rolling window lap times (seconds) |
| `avg_speed` | Mean velocity magnitude (m/s) |
| `steps_to_first_gate` | Mean steps before first gate passage |
| `best_lap_time_ever` | All-time best lap time |
| `best_gates_per_ep_ever` | All-time max gates in one episode |
| `best_laps_per_ep_ever` | All-time max laps in one episode |
| `best_ep_reward_ever` | All-time best episode reward |
| `reward_<component>` | Per-component reward breakdown (env-specific names) |

### Termination breakdown (`termination/` prefix)

Shows the fraction of episodes ending for each reason (dynamic, env-specific):
- GateRaceEnv: `ground`, `ceiling`, `quat`, `nan`, `arena_oob`, `body_rate`, `gate_collision`, `timeout`
- RateCtrlEnv: `crash`, `timeout`

### SB3 default metrics

PPO also logs standard SB3 metrics under `train/` and `rollout/` prefixes (loss, entropy, value estimates, episode reward/length).

## Artifact Storage (R2)

Training artifacts (checkpoints, trajectory `.npz` files, ONNX exports) are uploaded to Cloudflare R2 (`corvidx-artifacts` bucket) automatically when W&B logging is enabled.

The `ArtifactUploader` is initialized with the W&B run ID and uploads in a background thread. Artifacts appear at:

```
s3://corvidx-artifacts/<wandb-run-id>/trajectories/step_<N>/eval_ep_<i>.npz
s3://corvidx-artifacts/<wandb-run-id>/checkpoints/...
```

To manually sync artifacts after a run:

```bash
make sync-artifacts RUN_DIR=outputs/<run-dir>
```

## Troubleshooting

### "WANDB_API_KEY not set"
Make sure you've both sourced and exported:
```bash
source .env && export WANDB_API_KEY
```

### W&B panels empty
- Check the run appeared in the `corvidx-drone-racing` project
- Metrics log every `log_freq` steps (default 100) — wait for a few hundred steps
- Racing metrics only appear after the first episode completes

### "No NVIDIA GPU detected"
The control Docker image requires `--gpus all`. If using `make train`, docker compose handles this via the `deploy.resources.reservations.devices` section. If using `docker run`, pass `--gpus all` explicitly.

### Training runs but no R2 uploads
- Verify `.env` has all 4 `R2_*` variables
- Check the training log for "ArtifactUploader" messages
- Trajectory uploads happen at `viz_freq` intervals (default varies by experiment)

## Quick Reference

```bash
# First time setup
cp docker-compose.override.yml.example docker-compose.override.yml
# Edit the command to your experiment

# Run smoke test
source .env && export WANDB_API_KEY && make train

# Check W&B
# → https://wandb.ai/corvidx-drone-racing
```
