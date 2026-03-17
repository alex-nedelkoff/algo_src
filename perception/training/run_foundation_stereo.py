"""Batch FoundationStereo inference on rectified stereo pairs → dense depth maps.

Must be run inside the `foundation_stereo` conda env.

Usage:
  conda activate foundation_stereo
  python -m perception.training.run_foundation_stereo \
      --input-dir ~/corvidx/data/uzh-fpv/rectified/indoor_forward_3_snapdragon/ \
      --ckpt ~/corvidx/model_checkpoints/foundation_stereo/23-51-11/model_best_bp2.pth \
      --output-dir ~/corvidx/data/uzh-fpv/depth/indoor_forward_3_snapdragon/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def load_model(ckpt_path: Path, valid_iters: int = 32):
    """Load FoundationStereo model from checkpoint.

    Adds the FoundationStereo repo to sys.path for imports.
    """
    import torch
    from omegaconf import OmegaConf

    # Add FoundationStereo repo to path
    fs_repo = Path(os.environ.get(
        "FOUNDATION_STEREO_REPO",
        os.path.expanduser("~/corvidx/repos/FoundationStereo"),
    ))
    if str(fs_repo) not in sys.path:
        sys.path.insert(0, str(fs_repo))

    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(str(ckpt_path.parent / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = valid_iters

    model = FoundationStereo(cfg)
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    log.info("Checkpoint: step=%d, epoch=%d", ckpt["global_step"], ckpt["epoch"])
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    return model, cfg


def run_inference(
    model,
    input_dir: Path,
    output_dir: Path,
    scale: float = 1.0,
    valid_iters: int = 32,
    batch_size: int = 1,
) -> None:
    """Run FoundationStereo on all rectified stereo pairs."""
    import torch
    from core.utils.utils import InputPadder

    left_dir = input_dir / "left"
    right_dir = input_dir / "right"
    left_images = sorted(left_dir.glob("*.png"))
    right_images = sorted(right_dir.glob("*.png"))

    n_frames = min(len(left_images), len(right_images))
    if n_frames == 0:
        raise ValueError(f"No images found in {input_dir}")

    # Load rectification params for depth conversion
    params_path = input_dir / "rectification_params.json"
    if params_path.exists():
        with open(params_path) as f:
            rect_params = json.load(f)
        fx = rect_params["fx"]
        baseline = rect_params["baseline"]
        log.info("Depth conversion: fx=%.2f, baseline=%.4fm", fx, baseline)
    else:
        log.warning("No rectification_params.json — saving raw disparity only")
        fx = None
        baseline = None

    depth_dir = output_dir / "depth"
    vis_dir = output_dir / "vis"
    depth_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    log.info("Processing %d stereo pairs ...", n_frames)
    t0 = time.time()

    torch.autograd.set_grad_enabled(False)

    for i in range(n_frames):
        img_l = cv2.imread(str(left_images[i]), cv2.IMREAD_UNCHANGED)
        img_r = cv2.imread(str(right_images[i]), cv2.IMREAD_UNCHANGED)

        # Convert grayscale to 3-channel if needed
        if img_l.ndim == 2:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB)
        else:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        if img_r.ndim == 2:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB)
        else:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        if scale < 1.0:
            img_l = cv2.resize(img_l, fx=scale, fy=scale, dsize=None)
            img_r = cv2.resize(img_r, fx=scale, fy=scale, dsize=None)

        H, W = img_l.shape[:2]

        # To torch tensor: (1, 3, H, W)
        t_l = torch.as_tensor(img_l).cuda().float()[None].permute(0, 3, 1, 2)
        t_r = torch.as_tensor(img_r).cuda().float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
        t_l, t_r = padder.pad(t_l, t_r)

        with torch.cuda.amp.autocast(True):
            disp = model.forward(t_l, t_r, iters=valid_iters, test_mode=True)

        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)

        # Convert disparity to depth
        if fx is not None and baseline is not None:
            fx_scaled = fx * scale
            depth = np.where(disp > 0, fx_scaled * baseline / disp, 0.0).astype(np.float32)
            np.save(str(depth_dir / f"depth_{i:06d}.npy"), depth)
        else:
            np.save(str(depth_dir / f"disp_{i:06d}.npy"), disp.astype(np.float32))

        # Save visualization for a few frames
        if i < 10 or i % 100 == 0:
            # Colorize depth/disparity for visualization
            disp_vis = disp.copy()
            disp_vis[disp_vis <= 0] = 0
            if disp_vis.max() > 0:
                disp_norm = (disp_vis / disp_vis.max() * 255).astype(np.uint8)
            else:
                disp_norm = np.zeros_like(disp_vis, dtype=np.uint8)
            disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_MAGMA)
            cv2.imwrite(str(vis_dir / f"vis_{i:06d}.png"), disp_color)

        elapsed = time.time() - t0
        fps = (i + 1) / elapsed
        eta = (n_frames - i - 1) / fps if fps > 0 else 0
        if (i + 1) % 50 == 0 or i == n_frames - 1:
            log.info(
                "  Frame %d/%d | %.1f fps | ETA %.0fs",
                i + 1, n_frames, fps, eta,
            )

    elapsed = time.time() - t0
    log.info("Done: %d frames in %.1fs (%.1f fps)", n_frames, elapsed, n_frames / elapsed)

    # Copy rectification params to output
    if params_path.exists():
        import shutil
        shutil.copy2(params_path, output_dir / "rectification_params.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch FoundationStereo depth inference on rectified stereo pairs"
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory with rectified left/ and right/")
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="Path to FoundationStereo checkpoint (.pth)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory for depth maps")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Downscale factor (<=1.0)")
    parser.add_argument("--valid-iters", type=int, default=32,
                        help="Number of flow-field update iterations")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (currently only 1 supported)")

    args = parser.parse_args()
    model, cfg = load_model(args.ckpt, args.valid_iters)
    run_inference(
        model, args.input_dir, args.output_dir,
        scale=args.scale, valid_iters=args.valid_iters,
    )


if __name__ == "__main__":
    main()
