"""Shared R2 upload and idempotency logic.

Extracted from ``scripts/sync_artifacts.py`` so that both the in-training
``ArtifactUploader`` and the post-hoc ``sync_artifacts.py`` share identical
upload / idempotency behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# R2 / public URL constants
# ---------------------------------------------------------------------------

R2_ENDPOINT_TEMPLATE = "https://{account_id}.r2.cloudflarestorage.com"
R2_PUBLIC_BASE = "https://pub-030397d1ed024b568ceff9508ce0bd30.r2.dev"
DEFAULT_BUCKET = "corvidx-artifacts"


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------


def make_r2_client(account_id: str, access_key: str, secret_key: str):
    """Create a boto3 S3 client configured for Cloudflare R2.

    Raises ``ImportError`` if *boto3* is not installed (the caller decides
    whether to exit, enter no-op mode, etc.).
    """
    import boto3
    from botocore.config import Config

    endpoint = R2_ENDPOINT_TEMPLATE.format(account_id=account_id)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    return client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _md5_hex(path: Path) -> str:
    """Compute MD5 hex digest of a local file."""
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _object_exists(
    client, bucket: str, key: str, local_path: Path, local_size: int
) -> bool:
    """Return True if the R2 object already exists with matching size and etag."""
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        remote_size = resp["ContentLength"]
        if remote_size != local_size:
            return False
        # Compare etag (MD5 for single-part uploads, quoted in response)
        remote_etag = resp.get("ETag", "").strip('"')
        local_md5 = _md5_hex(local_path)
        return remote_etag == local_md5
    except Exception:
        return False


def _human_size(n_bytes: int | float) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_file(
    client,
    local_path: Path,
    bucket: str,
    r2_key: str,
    *,
    skip_existing: bool = True,
) -> Optional[str]:
    """Upload a single local file to R2.

    Returns the public URL on success, or ``None`` if the upload was skipped
    because a matching object already exists on R2 (same size + etag).
    """
    size = local_path.stat().st_size

    if skip_existing and _object_exists(client, bucket, r2_key, local_path, size):
        log.info("  [skip] %s (already uploaded, same size+etag)", r2_key)
        return None

    log.info("  [upload] %s  (%s)", r2_key, _human_size(size))
    client.upload_file(str(local_path), bucket, r2_key)
    public_url = f"{R2_PUBLIC_BASE}/{r2_key}"
    return public_url


# ---------------------------------------------------------------------------
# Rerun viewer URL
# ---------------------------------------------------------------------------


def _rerun_version() -> str:
    """Return the installed ``rerun-sdk`` version, or ``'latest'``."""
    try:
        from importlib.metadata import version

        return version("rerun-sdk")
    except Exception:
        return "latest"


def rerun_viewer_url(
    r2_key: str,
    r2_public_base: str = R2_PUBLIC_BASE,
    rerun_version: Optional[str] = None,
) -> str:
    """Build an ``app.rerun.io`` viewer URL for an ``.rrd`` file on R2.

    Parameters
    ----------
    r2_key:
        Object key within the R2 bucket (e.g.
        ``runs/<run_id>/rerun/step_1000/eval_ep_0.rrd``).
    r2_public_base:
        Public base URL of the R2 bucket.  Defaults to :data:`R2_PUBLIC_BASE`.
    rerun_version:
        Rerun SDK version to embed in the viewer URL.  Defaults to the
        locally installed version (or ``'latest'``).
    """
    version = rerun_version or _rerun_version()
    file_url = f"{r2_public_base}/{r2_key}"
    return f"https://app.rerun.io/version/{version}/?url={file_url}"


# ---------------------------------------------------------------------------
# W&B reference artifact registration
# ---------------------------------------------------------------------------


def register_wandb_artifact(
    api,
    run_path: str,
    artifact_name: str,
    r2_url: str,
    artifact_type: str,
) -> None:
    """Register an R2 URL as a W&B reference artifact (best-effort).

    Uses the **public** ``wandb.Api()`` REST client, which is thread-safe
    (unlike the live ``wandb.Run`` object).

    Parameters
    ----------
    api:
        A ``wandb.Api()`` instance.
    run_path:
        Fully-qualified W&B run path, e.g. ``"entity/project/run_id"``.
    artifact_name:
        Name for the W&B artifact (e.g. ``"rerun-step_1000000"``).
    r2_url:
        Public R2 URL to register as a reference.
    artifact_type:
        W&B artifact type (e.g. ``"rerun-recording"``, ``"trajectory"``).
    """
    try:
        run = api.run(run_path)

        # Create a reference artifact pointing at the R2 URL
        import wandb

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=f"R2-hosted artifact: {r2_url}",
        )
        artifact.add_reference(r2_url)
        run.log_artifact(artifact)

        log.info(
            "W&B artifact '%s' registered for run %s.", artifact_name, run_path
        )
    except Exception as exc:
        log.warning(
            "W&B artifact registration failed (non-fatal): %s", exc
        )
