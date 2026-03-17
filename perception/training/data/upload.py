"""Upload processed VPR scene data and pair tables to Cloudflare R2."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from artifacts.r2 import DEFAULT_BUCKET, make_r2_client, upload_file

log = logging.getLogger(__name__)

R2_PREFIX = "datasets/vpr"


def _get_client():
    """Create R2 client from environment variables."""
    account_id = os.environ.get("R2_ACCOUNT_ID", "")
    access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")

    if not (account_id and access_key and secret_key):
        raise RuntimeError(
            "R2 credentials not set. Export R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY."
        )

    return make_r2_client(account_id, access_key, secret_key)


def upload_scene(
    scene_dir: Path,
    dataset: str,
    scene_id: str,
    bucket: str = DEFAULT_BUCKET,
) -> None:
    """Upload a processed scene's RGB, depth, poses, and intrinsics to R2.

    Expected local layout:
      scene_dir/rgb/frame000000.jpg, ...
      scene_dir/depth/depth000000.png, ...
      scene_dir/poses.npy
      scene_dir/intrinsics.json
    """
    client = _get_client()
    scene_dir = Path(scene_dir)
    prefix = f"{R2_PREFIX}/{dataset}/scenes/{scene_id}"

    # Upload RGB images
    rgb_dir = scene_dir / "rgb"
    if rgb_dir.exists():
        for img in sorted(rgb_dir.glob("*.jpg")):
            upload_file(client, img, bucket, f"{prefix}/rgb/{img.name}")

    # Upload depth maps
    depth_dir = scene_dir / "depth"
    if depth_dir.exists():
        for dep in sorted(depth_dir.glob("*.png")):
            upload_file(client, dep, bucket, f"{prefix}/depth/{dep.name}")

    # Upload poses
    poses_file = scene_dir / "poses.npy"
    if poses_file.exists():
        upload_file(client, poses_file, bucket, f"{prefix}/poses.npy")

    # Upload intrinsics
    intrinsics_file = scene_dir / "intrinsics.json"
    if intrinsics_file.exists():
        upload_file(client, intrinsics_file, bucket, f"{prefix}/intrinsics.json")

    log.info("Uploaded scene %s/%s to R2.", dataset, scene_id)


def upload_pairs(
    parquet_path: Path,
    dataset: str,
    scene_id: str,
    bucket: str = DEFAULT_BUCKET,
) -> None:
    """Upload a scene's pair table Parquet file to R2."""
    client = _get_client()
    r2_key = f"{R2_PREFIX}/{dataset}/pairs/{scene_id}.parquet"
    upload_file(client, parquet_path, bucket, r2_key)
    log.info("Uploaded pairs %s/%s to R2.", dataset, scene_id)


def upload_metadata(
    metadata: dict,
    dataset: str,
    bucket: str = DEFAULT_BUCKET,
) -> None:
    """Upload dataset metadata.json to R2."""
    import tempfile

    client = _get_client()
    r2_key = f"{R2_PREFIX}/{dataset}/metadata.json"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(metadata, f, indent=2)
        tmp_path = Path(f.name)

    try:
        upload_file(client, tmp_path, bucket, r2_key)
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("Uploaded metadata for %s to R2.", dataset)
