#!/usr/bin/env bash
# Bootstrap for RunPod RTX 3090 (or similar Linux + CUDA 12.1 + PyTorch 2.5).
#
# Run from the repo root after cloning:
#     cd /workspace/algo_src
#     bash scripts/cloud/bootstrap.sh
#
# What this script does:
#   1. Apt installs system libs (EGL for headless OpenGL, build tools)
#   2. Inits MASt3R-SLAM submodules
#   3. Installs Python deps + MASt3R-SLAM editable (with curope CUDA RoPE)
#   4. Downloads the MASt3R ViT-Large checkpoint (2.75 GB, one-time)
#   5. Builds the textured room asset (procedural OBJ + MTL + URDF)
#   6. Runs an import smoke test
#
# After this, you can run the actual smoke test with:
#     bash scripts/cloud/run_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[bootstrap] repo root = $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. System libraries
# ---------------------------------------------------------------------------
echo "[bootstrap] step 1/6 — apt-get system deps"
if [[ $EUID -ne 0 ]]; then
    SUDO=sudo
else
    SUDO=""
fi
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
    git build-essential ninja-build cmake \
    libegl1 libgl1-mesa-glx libgles2-mesa libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    curl wget

# ---------------------------------------------------------------------------
# 2. Clone MASt3R-SLAM (not vendored in this repo to keep it small).
#    Upstream is github.com/rmurai0610/MASt3R-SLAM with nested submodules
#    for mast3r, dust3r, croco, asmk, pyimgui.
# ---------------------------------------------------------------------------
echo "[bootstrap] step 2/6 — clone MASt3R-SLAM (recursive submodules)"
MAST3R_DIR=external_packages/MASt3R-SLAM
mkdir -p external_packages
if [[ ! -d "$MAST3R_DIR" ]]; then
    git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git "$MAST3R_DIR"
else
    echo "[bootstrap]   $MAST3R_DIR already present, updating submodules"
    (cd "$MAST3R_DIR" && git submodule update --init --recursive)
fi

# ---------------------------------------------------------------------------
# 3. Python deps + MASt3R-SLAM editable install
# ---------------------------------------------------------------------------
echo "[bootstrap] step 3/6 — pip installs"
PYTHON=${PYTHON:-python}
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r scripts/cloud/requirements-cloud.txt

# torchvision must match whatever torch the host image has. Detect torch
# version + CUDA tag and install a compatible torchvision wheel from
# the PyTorch index. Skipped if torchvision is already there.
echo "[bootstrap]   matching torchvision to host torch"
$PYTHON - <<'PY'
import subprocess, sys
import torch
torch_ver = torch.__version__.split('+')[0]
cuda_tag = ('cu' + torch.version.cuda.replace('.', '')
            if torch.version.cuda else 'cpu')
mm = '.'.join(torch_ver.split('.')[:2])
# torchvision pairs by minor: torch 2.X.Y → torchvision 0.(X+15).Y, e.g.
# 2.4 → 0.19, 2.5 → 0.20, 2.6 → 0.21, 2.7 → 0.22.
table = {
    '2.1': '0.16', '2.2': '0.17', '2.3': '0.18',
    '2.4': '0.19', '2.5': '0.20', '2.6': '0.21',
    '2.7': '0.22', '2.8': '0.23',
}
tv = table.get(mm)
if tv is None:
    print(f"  WARN: unknown torch {torch_ver}, will pip install bare torchvision")
    pkg = "torchvision"
    idx = []
else:
    pkg = f"torchvision=={tv}.0+{cuda_tag}" if cuda_tag != 'cpu' else f"torchvision=={tv}.0"
    idx = ["--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"]
print(f"  torch {torch_ver} ({cuda_tag}) → {pkg}")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", pkg, *idx])
PY

# Install MASt3R-SLAM and its thirdparty packages.
#
# Three real Python packages here, in dependency order:
#   - thirdparty/mast3r  — auto-installs asmk + curope (CUDA RoPE2D)
#                          via its install_requires path-deps
#   - thirdparty/in3d    — separate package
#   - root MASt3R-SLAM   — builds mast3r_slam_backends.so
#
# dust3r is NOT a pip package — it has no setup.py. mast3r's path-helper
# `mast3r.utils.path_to_dust3r` adds dust3r/ to sys.path at import time;
# that's the supported pattern (see mast3r_slam/mast3r_utils.py). The
# import smoke test below mirrors that pattern.
#
# in3d's setup.py bundles the pyimgui C++ source as a path dep. That
# build is fragile (Windows MSVC + Linux compiler quirks). Patch in3d
# to use the PyPI imgui wheel instead — the GUI features in3d wraps
# aren't used by our headless smoke test or by mast3r_slam's tracker.
IN3D_SETUP=external_packages/MASt3R-SLAM/thirdparty/in3d/setup.py
if grep -q 'f"imgui @ {pyimgui_path}"' "$IN3D_SETUP"; then
    echo "[bootstrap]   patching in3d to use PyPI imgui instead of pyimgui submodule"
    sed -i 's|f"imgui @ {pyimgui_path}"|"imgui"|' "$IN3D_SETUP"
fi

# --no-build-isolation: use the host torch we already have.
echo "[bootstrap]   building MASt3R-SLAM stack (compiles CUDA extensions, ~5 min)"
PIP_FLAGS="--no-build-isolation"
$PYTHON -m pip install $PIP_FLAGS -e external_packages/MASt3R-SLAM/thirdparty/mast3r
$PYTHON -m pip install $PIP_FLAGS -e external_packages/MASt3R-SLAM/thirdparty/in3d
$PYTHON -m pip install $PIP_FLAGS -e external_packages/MASt3R-SLAM

# ---------------------------------------------------------------------------
# 4. Download MASt3R checkpoint (2.75 GB)
# ---------------------------------------------------------------------------
echo "[bootstrap] step 4/6 — fetching MASt3R checkpoint"
CKPT_DIR=external_packages/MASt3R-SLAM/checkpoints
CKPT="$CKPT_DIR/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
mkdir -p "$CKPT_DIR"
if [[ ! -s "$CKPT" || $(stat -c%s "$CKPT") -lt 2700000000 ]]; then
    rm -f "$CKPT"
    wget --tries=3 -O "$CKPT" \
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
else
    echo "[bootstrap]   checkpoint already present, skipping download"
fi

# ---------------------------------------------------------------------------
# 5. Build textured room asset
# ---------------------------------------------------------------------------
echo "[bootstrap] step 5/6 — building textured_room_v1 OBJ/MTL/URDF"
$PYTHON sim/assets/textured_room_v1/build_room.py

# ---------------------------------------------------------------------------
# 6. Smoke import test
# ---------------------------------------------------------------------------
echo "[bootstrap] step 6/6 — import smoke test"
$PYTHON -c "
import torch
print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} '
      f'device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
import lietorch, lietorch_extras
import mast3r_slam, mast3r_slam_backends
print('  lietorch + mast3r_slam_backends OK')
import mast3r.utils.path_to_dust3r  # noqa: F401  — adds dust3r/ to sys.path
import mast3r, dust3r
print('  mast3r + dust3r OK')
try:
    from dust3r.croco.models.curope import cuRoPE2D
    print('  curope CUDA extension OK')
except ImportError as e:
    print(f'  WARNING: curope not available, will use slow Python RoPE ({e})')
import perception.localization.mast3r_localizer  # noqa: F401
print('  Mast3rLocalizer import OK')
"

echo ""
echo "[bootstrap] All done. Run the smoke test with:"
echo "    bash scripts/cloud/run_smoke.sh"
