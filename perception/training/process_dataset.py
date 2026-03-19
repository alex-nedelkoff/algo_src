"""CLI entrypoint: process a dataset through the VPR covisibility pipeline.

Usage (inside Docker container):
  python -m perception.training.process_dataset \
      --dataset replica --data-dir /data/Replica \
      --scenes room0 --num-samples 10000

  python -m perception.training.process_dataset \
      --dataset replica --data-dir /data/Replica \
      --num-samples 10000 --upload
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from perception.training.data.covisibility import compute_scene_covisibility
from perception.training.data.pairs import PairTable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def _get_loader(dataset: str, data_dir: Path, scene_id: str):
    """Get a scene loader for the given dataset."""
    if dataset == "replica":
        from perception.training.data.replica.loader import ReplicaScene
        return ReplicaScene(data_dir, scene_id)
    elif dataset == "uzh-fpv":
        from perception.training.data.uzh_fpv.loader import UZHFPVScene
        return UZHFPVScene(data_dir, scene_id)
    elif dataset == "tartanair":
        from perception.training.data.tartanair.loader import TartanAirScene
        return TartanAirScene(data_dir, scene_id, frame_skip=5)
    elif dataset == "euroc":
        from perception.training.data.euroc.loader import EuRoCScene
        return EuRoCScene(data_dir, scene_id)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _get_scene_list(dataset: str, scenes: list[str] | None, data_dir: Path) -> list[str]:
    """Get list of scenes to process."""
    if scenes:
        return scenes

    if dataset == "replica":
        from perception.training.data.replica.constants import SCENES
        # Only return scenes that exist
        return [s for s in SCENES if (data_dir / s / "results").exists()]
    elif dataset == "uzh-fpv":
        from perception.training.data.uzh_fpv.loader import UZHFPVScene
        return UZHFPVScene.available_scenes(data_dir)
    elif dataset == "tartanair":
        from perception.training.data.tartanair.loader import TartanAirScene
        return TartanAirScene.available_scenes(data_dir)
    elif dataset == "euroc":
        from perception.training.data.euroc.loader import EuRoCScene
        return EuRoCScene.available_scenes(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _prepare_scene_for_upload(
    scene, scene_id: str, output_dir: Path
) -> Path:
    """Prepare a scene's data for R2 upload in the expected layout."""
    scene_out = output_dir / scene_id
    rgb_out = scene_out / "rgb"
    depth_out = scene_out / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    # Symlink or copy RGB and depth to upload layout
    K = scene.intrinsics()
    poses = scene.poses()

    # Save poses
    np.save(str(scene_out / "poses.npy"), poses)

    # Save intrinsics
    intrinsics = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": int(scene.rgb(0).shape[1]),
        "h": int(scene.rgb(0).shape[0]),
    }
    with open(scene_out / "intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)

    # Symlink RGB frames
    for i in range(scene.n_frames):
        src = scene._rgb_paths[i]
        dst = rgb_out / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # Symlink depth frames
    for i in range(scene.n_frames):
        src = scene._depth_paths[i]
        dst = depth_out / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    return scene_out


def process_scene(
    dataset: str,
    data_dir: Path,
    scene_id: str,
    num_samples: int,
    output_dir: Path,
    upload: bool = False,
) -> PairTable:
    """Process a single scene: load → covisibility → Parquet."""
    log.info("Processing %s/%s ...", dataset, scene_id)

    scene = _get_loader(dataset, data_dir, scene_id)
    K = scene.intrinsics()
    poses = scene.poses()

    log.info("  %d frames, loading depths ...", scene.n_frames)
    depths = scene.depths()

    log.info("  Computing covisibility matrix (num_samples=%d) ...", num_samples)
    overlap = compute_scene_covisibility(
        depths, poses, K, num_samples=num_samples
    )

    log.info("  Building pair table ...")
    pairs = PairTable.from_overlap_matrix(
        overlap, dataset=dataset, scene_id=scene_id, min_overlap=0.01
    )
    log.info("  %d pairs extracted.", len(pairs))

    # Save Parquet
    parquet_dir = output_dir / "pairs"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{scene_id}.parquet"
    pairs.save(parquet_path)
    log.info("  Pairs saved to %s", parquet_path)

    if upload:
        from perception.training.data.upload import upload_pairs, upload_scene

        scene_upload_dir = _prepare_scene_for_upload(scene, scene_id, output_dir / "scenes")
        upload_scene(scene_upload_dir, dataset, scene_id)
        upload_pairs(parquet_path, dataset, scene_id)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="VPR covisibility data pipeline")
    parser.add_argument(
        "--dataset", required=True, choices=["replica", "uzh-fpv", "tartanair", "euroc"],
        help="Dataset to process",
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Path to raw dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: data-dir/../processed/<dataset>)")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene IDs to process (default: all)")
    parser.add_argument("--num-samples", type=int, default=10_000, help="Pixel subsample count for covisibility")
    parser.add_argument("--upload", action="store_true", help="Upload results to R2")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir.parent / "processed" / args.dataset

    scene_list = _get_scene_list(args.dataset, args.scenes, args.data_dir)
    log.info("Will process %d scenes: %s", len(scene_list), scene_list)

    all_pairs = []
    for scene_id in scene_list:
        pairs = process_scene(
            dataset=args.dataset,
            data_dir=args.data_dir,
            scene_id=scene_id,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            upload=args.upload,
        )
        all_pairs.append(pairs)

    # Merge and save combined pair table
    if all_pairs:
        merged = PairTable.merge(all_pairs)
        merged_path = args.output_dir / "pairs" / "all.parquet"
        merged.save(merged_path)
        log.info("Merged pair table: %d total pairs → %s", len(merged), merged_path)

    # Write metadata
    metadata = {
        "dataset": args.dataset,
        "scenes": scene_list,
        "total_pairs": sum(len(p) for p in all_pairs),
        "num_samples": args.num_samples,
        "processing_date": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata written to %s", meta_path)

    if args.upload:
        from perception.training.data.upload import upload_metadata
        upload_metadata(metadata, args.dataset)

    log.info("Done.")


if __name__ == "__main__":
    main()
