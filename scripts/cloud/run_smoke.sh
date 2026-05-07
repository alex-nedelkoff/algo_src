#!/usr/bin/env bash
# Run the Phase 1 MASt3R-SLAM smoke test and serve the rerun web viewer.
#
#     bash scripts/cloud/run_smoke.sh
#
# Open the viewer URL printed at the end (RunPod's "Connect to Pod"
# panel exposes any port you bind to 0.0.0.0). Default port 9876.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON=${PYTHON:-python}
OUT="${OUT:-outputs/perception/mast3r_smoketest.rrd}"
PORT="${RERUN_PORT:-9876}"
N_FRAMES="${N_FRAMES:-300}"

echo "[smoke] running smoke test ($N_FRAMES frames)..."
$PYTHON -m scripts.perception.mast3r_slam_smoketest \
    --out "$OUT" --n-frames "$N_FRAMES"

echo ""
echo "[smoke] starting rerun web viewer on 0.0.0.0:$PORT"
echo "[smoke] open http://<runpod-public-host>:$PORT/ in your browser"
echo "[smoke] (RunPod auto-exposes the port if you set 'Expose HTTP Ports')"
exec rerun "$OUT" --web-viewer --bind 0.0.0.0 --web-viewer-port "$PORT"
