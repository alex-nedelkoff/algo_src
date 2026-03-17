#!/usr/bin/env bash
# Build and run the VPR data pipeline container.
# Usage: ./run.sh [command]
#   ./run.sh                          # interactive shell
#   ./run.sh download replica         # download Replica dataset
#   ./run.sh process replica room0    # process one scene
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$TRAINING_DIR")")"
DATA_DIR="${DATA_DIR:-$HOME/corvidx/data}"
IMAGE_NAME="corvidx/vpr-pipeline"

mkdir -p "$DATA_DIR"

# Build from perception/training/ as context
docker build --network=host -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile" "$TRAINING_DIR"

# Run with GPU, host networking (for DNS), mount data dir + source for live editing
exec docker run --rm -it \
    --gpus all \
    --network=host \
    -v "$DATA_DIR:/data" \
    -v "$TRAINING_DIR:/app/perception/training" \
    -v "$REPO_ROOT/artifacts:/app/artifacts" \
    -e PYTHONPATH=/app \
    "$IMAGE_NAME" \
    "$@"
