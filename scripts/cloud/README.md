# Cloud GPU workflow — RunPod RTX 3090 for MASt3R-SLAM smoke test

End-to-end: spin up a pod, clone repo, bootstrap, run the smoke test, view results in your browser.

## 1. Spin up the pod

On [runpod.io](https://runpod.io):

- **Template**: any *RunPod PyTorch 2.x* image with **CUDA 12.x** is fine. As of writing
  the menu offers PyTorch 2.4, 2.6, 2.7, 2.8 — the bootstrap detects whichever you pick
  and installs a matching torchvision wheel. If you have a choice, prefer 2.5 or 2.6
  (closest to the version we exercised locally). Avoid CUDA 11.x images — curope's
  build expects 12.x headers.
- **GPU**: RTX 3090 24GB (about $0.40/hr secure cloud, $0.20–0.30/hr community)
- **Container disk**: 30 GB minimum
- **Volume disk**: 30 GB persistent (so the MASt3R checkpoint survives stops)
- **Expose HTTP ports**: `9876` (rerun web viewer)
- **Expose TCP ports**: `22` (SSH)

Click *Deploy*. After ~30 s the pod is running. Use *Connect → Web Terminal* or copy the SSH command.

## 2. Get the code on the pod

If your repo is on GitHub:

```bash
cd /workspace
git clone https://github.com/<you>/algo_src.git
cd algo_src
git checkout <branch>
```

Otherwise rsync from your laptop:

```bash
# from your laptop
rsync -av --exclude='.git/objects' --exclude='outputs/' --exclude='*.rrd' \
    /path/to/algo_src/ \
    root@<pod-ip>:/workspace/algo_src/
```

## 3. Bootstrap

One time (per pod):

```bash
cd /workspace/algo_src
bash scripts/cloud/bootstrap.sh
```

This runs in ~10 minutes the first time:
- 2 min — apt-get system libs (EGL, build tools)
- 1 min — git submodule init
- 5 min — pip install + MASt3R-SLAM CUDA extension build (`mast3r_slam_backends.so`, `curope.so`)
- 2 min — checkpoint download (2.75 GB)
- 30 s — textured-room generation + import smoke test

If it ends with `bootstrap All done.`, you're ready.

## 4. Run the smoke test

```bash
bash scripts/cloud/run_smoke.sh
```

This:
1. Renders 300 frames of a scripted oval flight in the textured room (PyBullet + EGL)
2. Runs ICP-against-mesh on the depth stream
3. Runs MASt3R-SLAM on the RGB stream
4. Logs both trajectories + per-frame errors to `outputs/perception/mast3r_smoketest.rrd`
5. Starts `rerun --web-viewer --bind 0.0.0.0 --web-viewer-port 9876`

## 5. View results in your browser

In the RunPod web UI, click *Connect → HTTP* on port 9876. Or directly: `http://<runpod-public-host>:9876/`.

You should see a 4-pane layout:
- 3D world: room outline, GT (blue), ICP (green), MASt3R-SLAM (orange) trajectories
- 2D RGB: drone-eye-view at the current sim time
- Time series: ICP pos error, MASt3R pos error, ICP fitness

## 6. Iteration loop

Code changes locally, push to GitHub (or rsync), `git pull` on the pod, re-run `run_smoke.sh`. The bootstrap doesn't need to repeat.

## 7. Stop the pod when done

RunPod bills per-second while running. *Stop pod* (not *Terminate*) preserves the volume so next time you only re-pull code, no re-bootstrap. Stopped pods cost ~$0.04/hr for the volume.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `eglRendererPlugin` warning at scene setup | EGL libs not installed → `apt-get install libegl1 libgles2-mesa` |
| RGB image is solid black | EGL plugin failed to load → check `ldconfig -p | grep -i egl`; reinstall `libegl1-mesa-dev` |
| `curope not available` warning | curope build failed; will fall back to slow Python RoPE2D — works but ~2× slower. To fix: `cd external_packages/MASt3R-SLAM/thirdparty/mast3r/dust3r/croco/models/curope && pip install .` |
| `mast3r_slam_backends` import error | CUDA build failed during pip install. Check `nvcc --version` matches `python -c 'import torch; print(torch.version.cuda)'`. RunPod's PyTorch 2.5.1 image ships CUDA 12.1; mismatch causes link errors |
| MASt3R-SLAM tracking lost from frame 0 | Likely the rendered RGB has too few features. Check `outputs/perception/textured_room_preview.png` — should clearly show different textures per wall |
| Out-of-memory mid-run | Reduce `--n-frames` or pass `keyframe_buffer=32` to `Mast3rLocalizer` |

## Cost ballpark

| Phase | Pod-hours | $ |
|---|---|---|
| First bootstrap (one-time) | 0.2 h | $0.10 |
| Smoke test runs | 0.5 h × 5 iterations | $1.00 |
| Phase 2 splat experiments | 5–10 h | $2–4 |
| **Total Phase 1+2** | **~10 h** | **~$5** |

Use *community cloud* tier on RunPod for the cheapest pricing. RTX 3090 in community is typically $0.20–0.25/hr.
