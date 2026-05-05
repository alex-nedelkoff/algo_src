"""M2 — Depth Anything V2 latency + temporal-consistency benchmark.

Plan-doc reference: ``docs/superpowers/plans/2026-05-05-vision-racing-week1-plan.md``

Goal: characterise per-frame inference cost and frame-to-frame depth
flicker for Depth Anything V2 (Small / Base / Large) on host GPU. The
output decides whether we need Video Depth Anything's temporal
consistency (extra repo install, integration cost) or whether DA V2
plus optional smoothing is sufficient.

Why DA V2 only and not VDA: ``transformers==5.8.0`` does not ship a
``video_depth_anything`` model; using VDA requires cloning the
upstream repo and managing its deps separately. We defer that work
until we have evidence DA's flicker actually degrades downstream
control. This script produces that evidence (the temporal-consistency
metric).

Inputs
------
Three pre-rendered warehouse images at ``outputs/render/warehouse_*.png``.
We synthesise short "static-scene" sequences from each by adding small
horizontal pixel shifts (simulating sub-pixel camera jitter on a fixed
scene). On a perfectly stable depth model the per-pixel depth would
match between adjacent frames; the residual is the temporal flicker.

Outputs
-------
``outputs/perception/depth_anything_benchmark/``
    summary.json
    {<model_id>}/{<resolution>}/
        latency_per_frame_ms.npy      (n_frames,)
        depth_<i>.npy                 (H, W) per frame
    visualization_<model>_<res>.png   side-by-side rgb | depth heatmap

Usage::

    python -m scripts.perception.benchmark_depth_anything \\
        [--models small base large] \\
        [--resolutions 144x256 224x224 384x512] \\
        [--n-frames 20] \\
        [--out outputs/perception/depth_anything_benchmark]
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402  preload before transformers

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

DEFAULT_RESOLUTIONS = {
    "144x256": (144, 256),    # BP_FlyingPawn camera default (per COR-100)
    "224x224": (224, 224),    # ViT-friendly square
    "384x512": (384, 512),    # AirSim 'medium' resolution
}

WAREHOUSE_RENDERS = [
    "outputs/render/warehouse_top.png",
    "outputs/render/warehouse_perspective.png",
    "outputs/render/warehouse_inside.png",
]


def _make_jitter_sequence(img: Image.Image, n_frames: int, max_shift_px: int = 2) -> list[Image.Image]:
    """Synthesise a 'static-scene' sequence by adding small horizontal shifts.

    A perfectly temporally-stable depth model would produce nearly identical
    depth maps between adjacent frames (modulo the shift). The variance we
    observe is the model's intrinsic flicker.
    """
    rng = np.random.default_rng(42)
    frames = [img]
    arr = np.asarray(img)
    for _ in range(n_frames - 1):
        dx = int(rng.integers(-max_shift_px, max_shift_px + 1))
        dy = int(rng.integers(-max_shift_px, max_shift_px + 1))
        shifted = np.roll(arr, shift=(dy, dx), axis=(0, 1))
        frames.append(Image.fromarray(shifted))
    return frames


def _resize(img: Image.Image, size_hw: tuple[int, int]) -> Image.Image:
    h, w = size_hw
    return img.resize((w, h), resample=Image.BILINEAR)


def _undo_jitter(depth_map: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Reverse the synthetic shift so flicker comparison is apples-to-apples."""
    return np.roll(depth_map, shift=(-dy, -dx), axis=(0, 1))


def benchmark_one(
    pipe,
    rgb_frames: list[Image.Image],
    *,
    resolution: tuple[int, int],
    out_dir: Path,
    warmup: int = 3,
) -> dict:
    """Run inference on each frame, save depth + record latency."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_resized = [_resize(im, resolution) for im in rgb_frames]
    n = len(frames_resized)

    # Warmup
    for _ in range(warmup):
        _ = pipe(frames_resized[0])

    # Inference + timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies = np.zeros(n, dtype=np.float64)
    depths = []
    for i, img in enumerate(frames_resized):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies[i] = (t1 - t0) * 1000.0  # ms
        depth = np.asarray(out["depth"], dtype=np.float32)
        depths.append(depth)
        np.save(out_dir / f"depth_{i:03d}.npy", depth)
    np.save(out_dir / "latency_per_frame_ms.npy", latencies)

    return {
        "n_frames": n,
        "latency_mean_ms": float(latencies.mean()),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "fps_at_mean": float(1000.0 / latencies.mean()),
        "depths": depths,
    }


def temporal_consistency(depths: list[np.ndarray]) -> dict:
    """Quantify frame-to-frame variation in depth on the same scene.

    Computes per-pixel RMSE between consecutive normalized depth maps.
    Smaller = more temporally consistent. Normalised by mean depth so the
    metric is comparable across model variants that produce different
    absolute scales.
    """
    if len(depths) < 2:
        return {"adjacent_rmse_norm": 0.0, "mean_depth": 0.0}
    norm_depths = []
    for d in depths:
        d_min, d_max = float(d.min()), float(d.max())
        denom = max(d_max - d_min, 1e-9)
        norm_depths.append((d - d_min) / denom)
    sq_diffs = []
    for a, b in zip(norm_depths[:-1], norm_depths[1:]):
        sq_diffs.append(float(np.sqrt(((a - b) ** 2).mean())))
    mean_d = float(np.mean([d.mean() for d in depths]))
    return {
        "adjacent_rmse_norm": float(np.mean(sq_diffs)),
        "adjacent_rmse_norm_max": float(np.max(sq_diffs)),
        "mean_depth": mean_d,
    }


def make_visualization(rgb: Image.Image, depth: np.ndarray, out_path: Path) -> None:
    """Side-by-side RGB | depth heatmap, saved as PNG."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB input")
    axes[0].axis("off")
    im = axes[1].imshow(depth, cmap="turbo")
    axes[1].set_title("DA V2 depth")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small", "base", "large"],
                    choices=list(MODEL_IDS.keys()))
    ap.add_argument("--resolutions", nargs="+", default=["144x256", "224x224", "384x512"],
                    choices=list(DEFAULT_RESOLUTIONS.keys()))
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/perception/depth_anything_benchmark"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out.resolve()}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # Build the input sequence: load each warehouse render, synthesise a
    # small jittered sequence per render. Total = len(renders) * n_frames.
    rgb_seqs = []
    for path in WAREHOUSE_RENDERS:
        img = Image.open(path).convert("RGB")
        rgb_seqs.extend(_make_jitter_sequence(img, args.n_frames))
    print(f"Total frames: {len(rgb_seqs)} ({len(WAREHOUSE_RENDERS)} renders × {args.n_frames})")

    summary = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_frames_per_render": args.n_frames,
        "n_renders": len(WAREHOUSE_RENDERS),
        "results": {},
    }

    from transformers import pipeline

    for model_key in args.models:
        model_id = MODEL_IDS[model_key]
        print(f"\n=== {model_key.upper()} ({model_id}) ===")
        pipe = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        n_params = sum(p.numel() for p in pipe.model.parameters())
        print(f"  params: {n_params/1e6:.1f} M")

        summary["results"][model_key] = {"n_params_M": n_params / 1e6, "by_resolution": {}}

        for res_key in args.resolutions:
            resolution = DEFAULT_RESOLUTIONS[res_key]
            print(f"  -- {res_key} ({resolution[0]}x{resolution[1]}) --")
            sub_dir = args.out / model_key / res_key
            res = benchmark_one(pipe, rgb_seqs, resolution=resolution, out_dir=sub_dir)
            tc = temporal_consistency(res["depths"])
            print(f"    latency: mean {res['latency_mean_ms']:.1f} ms / "
                  f"median {res['latency_median_ms']:.1f} ms / "
                  f"p95 {res['latency_p95_ms']:.1f} ms  "
                  f"({res['fps_at_mean']:.1f} fps)")
            print(f"    temporal: adj-RMSE-norm {tc['adjacent_rmse_norm']:.4f} "
                  f"(max {tc['adjacent_rmse_norm_max']:.4f}, mean depth {tc['mean_depth']:.2f})")

            # One viz per (model, resolution) combo using the first frame.
            viz_path = args.out / f"viz_{model_key}_{res_key}.png"
            make_visualization(_resize(Image.open(WAREHOUSE_RENDERS[0]).convert("RGB"), resolution),
                               res["depths"][0], viz_path)

            summary["results"][model_key]["by_resolution"][res_key] = {
                "latency_mean_ms": res["latency_mean_ms"],
                "latency_median_ms": res["latency_median_ms"],
                "latency_p95_ms": res["latency_p95_ms"],
                "fps_at_mean": res["fps_at_mean"],
                "n_frames": res["n_frames"],
                **tc,
            }

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = args.out / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
