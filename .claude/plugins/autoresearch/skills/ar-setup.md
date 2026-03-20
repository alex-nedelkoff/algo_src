---
name: ar-setup
description: Setup instructions for the auto-research system. Use when a new team member needs to get started or when troubleshooting setup issues.
---

# Auto-Research Setup

One-time setup to get the auto-research system running on your machine.

## Prerequisites

- Git clone of `algo_src` with the `autoresearch-mvp` branch
- Python 3.10+
- uv (recommended) or pip

## Steps

### 1. Install dependencies

```bash
# Create venv and install all deps including training stack
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[control]"
```

### 2. Install and login to W&B

```bash
uv pip install wandb
wandb login
```

When prompted, enter your W&B API key. You can find it at https://wandb.ai/authorize.

The W&B project is `corvidx-drone-racing`.

### 3. Create `.env` file

Create a `.env` file in the repo root with R2 credentials for artifact storage:

```
R2_ACCOUNT_ID=<ask team for credentials>
R2_ACCESS_KEY_ID=<ask team for credentials>
R2_SECRET_ACCESS_KEY=<ask team for credentials>
R2_BUCKET=corvidx-artifacts
```

This file is gitignored. Ask a teammate for the credential values.

### 4. Verify setup

```bash
# Activate venv
source .venv/bin/activate

# Verify autoresearch package
python -m pytest tests/test_autoresearch/ -v

# Verify training pipeline
python -m training +experiment=smoke_test total_timesteps=50000

# Verify W&B connection
python -c "import wandb; api = wandb.Api(); print(f'Logged in as: {api.viewer.entity}')"

# Verify autoresearch state
python -c "
from autoresearch.archive.serialization import load_archive
from autoresearch.tree.serialization import load_tree
from autoresearch.coordination.claims import load_claims
a = load_archive('autoresearch/state/archive.json')
t = load_tree('autoresearch/state/tree.json')
c = load_claims('autoresearch/state/claims.json')
print(f'Archive: {a.n_occupied}/{a.n_cells} cells')
print(f'Tree: {t.total_experiments} experiments')
print(f'Claims: {len(c)} active')
"
```

All 4 checks should pass. If the smoke test completes and shows a W&B run URL, you're ready.

### 5. Start researching

```bash
# In Claude Code, with venv activated:
/auto-research              # interactive mode (default)
/auto-research autonomous   # auto-approve hyperparameter/algorithm changes
/auto-research yolo         # full autonomy (hard constraints still enforced)
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'autoresearch'`
Make sure you installed with `-e` (editable mode): `uv pip install -e ".[control]"`

### `ModuleNotFoundError: No module named 'wandb'`
Install wandb in the same venv: `uv pip install wandb`

### W&B login fails
Check `~/.netrc` for wandb credentials, or re-run `wandb login`

### Training crashes with CUDA errors
Your machine may not have a GPU. Training runs on CPU automatically (`device: auto`). It's slower but works.

### Docker DNS issues
This machine requires `--network=host` for Docker builds/runs. See CLAUDE.md for details.

## GPU Notes

- **With NVIDIA GPU**: training uses CUDA automatically
- **CPU only (laptops)**: works but slower. Use `smoke_test` experiment (50K steps, ~7s) for validation. Full experiments (5M+ steps) should run on a GPU machine or via Docker.
- **Docker GPU**: `docker run --network=host --gpus all algo-src-control python -m training ...`
