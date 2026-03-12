#!/usr/bin/env python3
"""Upload training artifacts to Cloudflare R2 and register with W&B.

Usage:
    python scripts/sync_artifacts.py outputs/2026-03-11_10-30-00/
    python scripts/sync_artifacts.py outputs/2026-03-11_10-30-00/ --run-id abc12345
    python scripts/sync_artifacts.py outputs/2026-03-11_10-30-00/ --timeout 300
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
import zipfile
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
# Env / dotenv helpers
# ---------------------------------------------------------------------------


def _load_dotenv(path: Path) -> None:
    """Parse a .env file and populate os.environ (simple implementation,
    no external dependency required)."""
    if not path.is_file():
        return
    with path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        log.error("Required environment variable %s is not set.", name)
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# W&B run ID discovery
# ---------------------------------------------------------------------------


def _find_wandb_run_id(run_dir: Path) -> Optional[str]:
    """Extract W&B run ID from wandb/latest-run/run-*.wandb metadata."""
    wandb_dir = run_dir / "wandb" / "latest-run"
    if not wandb_dir.is_dir():
        return None

    # The .wandb file is a binary protobuf — but the run ID is embedded
    # in the filename itself: run-<run_id>.wandb
    candidates = list(wandb_dir.glob("run-*.wandb"))
    if not candidates:
        # Also try looking for a wandb-metadata.json or run-*.wandb at top level
        candidates = list(wandb_dir.glob("*.wandb"))

    for candidate in candidates:
        m = re.search(r"run-([a-zA-Z0-9]+)\.wandb$", candidate.name)
        if m:
            return m.group(1)

    # Fallback: check for wandb-metadata.json
    meta_json = wandb_dir / "wandb-metadata.json"
    if meta_json.is_file():
        try:
            data = json.loads(meta_json.read_text())
            run_id = data.get("run_id") or data.get("id")
            if run_id:
                return run_id
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# R2 / boto3 helpers
# ---------------------------------------------------------------------------


def _make_s3_client(account_id: str, access_key: str, secret_key: str):
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        log.error(
            "boto3 is not installed. Install it with: pip install boto3  "
            "(or add 'artifacts' extras: pip install -e '.[artifacts]')"
        )
        sys.exit(1)

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


def _object_exists(client, bucket: str, key: str, local_size: int) -> bool:
    """Return True if the R2 object already exists with the same size."""
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
        remote_size = resp["ContentLength"]
        return remote_size == local_size
    except Exception:
        return False


def _upload_file(
    client,
    bucket: str,
    local_path: Path,
    r2_key: str,
    *,
    skip_existing: bool = True,
) -> Optional[str]:
    """Upload a single local file to R2. Returns the public URL or None on skip."""
    size = local_path.stat().st_size

    if skip_existing and _object_exists(client, bucket, r2_key, size):
        log.info("  [skip] %s (already uploaded, same size)", r2_key)
        return None

    log.info("  [upload] %s  (%s)", r2_key, _human_size(size))
    client.upload_file(str(local_path), bucket, r2_key)
    public_url = f"{R2_PUBLIC_BASE}/{r2_key}"
    return public_url


def _upload_bytes(
    client,
    bucket: str,
    data: bytes,
    r2_key: str,
    *,
    skip_existing: bool = True,
) -> Optional[str]:
    """Upload in-memory bytes to R2."""
    size = len(data)

    if skip_existing and _object_exists(client, bucket, r2_key, size):
        log.info("  [skip] %s (already uploaded, same size)", r2_key)
        return None

    log.info("  [upload] %s  (%s, in-memory zip)", r2_key, _human_size(size))
    client.put_object(Bucket=bucket, Key=r2_key, Body=data)
    return f"{R2_PUBLIC_BASE}/{r2_key}"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _zip_directory(directory: Path) -> bytes:
    """Zip an entire directory in-memory and return the bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(directory.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(directory.parent))
    return buf.getvalue()


def _step_number_from_path(p: Path) -> int:
    """Extract numeric step from paths like step_1000000."""
    m = re.search(r"step_(\d+)", p.name)
    return int(m.group(1)) if m else 0


def _checkpoint_step(p: Path) -> int:
    """Extract step number from ppo_<n>_steps.zip filename."""
    m = re.search(r"ppo_(\d+)_steps", p.name)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Rerun viewer URL
# ---------------------------------------------------------------------------


def _rerun_version() -> str:
    try:
        from importlib.metadata import version

        return version("rerun-sdk")
    except Exception:
        return "latest"


def _rerun_viewer_url(r2_key: str) -> str:
    version = _rerun_version()
    file_url = f"{R2_PUBLIC_BASE}/{r2_key}"
    return f"https://app.rerun.io/version/{version}/?url={file_url}"


# ---------------------------------------------------------------------------
# Artifact collection — build priority-ordered upload list
# ---------------------------------------------------------------------------


class ArtifactEntry:
    """A single file to be uploaded."""

    def __init__(self, local_path: Optional[Path], r2_key: str, *, is_dir_zip: bool = False):
        self.local_path = local_path  # None means in-memory (dir-zip case handled separately)
        self.r2_key = r2_key
        self.is_dir_zip = is_dir_zip  # directory that needs to be zipped first


def _collect_artifacts(run_dir: Path, run_id: str) -> list[ArtifactEntry]:
    """Return artifacts in upload priority order."""
    prefix = f"runs/{run_id}"
    entries: list[ArtifactEntry] = []

    # -----------------------------------------------------------------------
    # Priority 1: best_model/
    # -----------------------------------------------------------------------
    best_model = run_dir / "best_model" / "best_model.zip"
    if best_model.is_file():
        entries.append(ArtifactEntry(best_model, f"{prefix}/best_model/best_model.zip"))
    else:
        best_dir = run_dir / "best_model"
        if best_dir.is_dir():
            for f in sorted(best_dir.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(run_dir)
                    entries.append(ArtifactEntry(f, f"{prefix}/{rel}"))

    # -----------------------------------------------------------------------
    # Priority 2: final_model  (zip if dir, upload as final_model.zip)
    # -----------------------------------------------------------------------
    final_zip = run_dir / "final_model.zip"
    final_dir = run_dir / "final_model"
    if final_zip.is_file():
        entries.append(ArtifactEntry(final_zip, f"{prefix}/final_model.zip"))
    elif final_dir.is_dir():
        entries.append(ArtifactEntry(final_dir, f"{prefix}/final_model.zip", is_dir_zip=True))

    # -----------------------------------------------------------------------
    # Priority 3: trajectory .npz files (checkpoints/trajectories/step_*/)
    # -----------------------------------------------------------------------
    traj_root = run_dir / "checkpoints" / "trajectories"
    if traj_root.is_dir():
        step_dirs = sorted(traj_root.iterdir(), key=_step_number_from_path)
        for step_dir in step_dirs:
            if step_dir.is_dir():
                for npz in sorted(step_dir.glob("*.npz")):
                    rel_step = step_dir.relative_to(traj_root)
                    entries.append(
                        ArtifactEntry(npz, f"{prefix}/trajectories/{rel_step}/{npz.name}")
                    )

    # -----------------------------------------------------------------------
    # Priority 4: checkpoints (ppo_*_steps.zip), newest first
    # -----------------------------------------------------------------------
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        checkpoint_zips = [
            f for f in ckpt_dir.glob("ppo_*_steps.zip") if f.is_file()
        ]
        # Descending step order (newest first)
        checkpoint_zips.sort(key=_checkpoint_step, reverse=True)
        for ckpt in checkpoint_zips:
            entries.append(ArtifactEntry(ckpt, f"{prefix}/checkpoints/{ckpt.name}"))

    # -----------------------------------------------------------------------
    # Priority 5: rerun .rrd files (checkpoints/rerun/step_*/)
    # -----------------------------------------------------------------------
    rerun_root = run_dir / "checkpoints" / "rerun"
    if rerun_root.is_dir():
        step_dirs = sorted(rerun_root.iterdir(), key=_step_number_from_path)
        for step_dir in step_dirs:
            if step_dir.is_dir():
                for rrd in sorted(step_dir.glob("*.rrd")):
                    rel_step = step_dir.relative_to(rerun_root)
                    entries.append(
                        ArtifactEntry(rrd, f"{prefix}/rerun/{rel_step}/{rrd.name}")
                    )

    # -----------------------------------------------------------------------
    # Config (low priority but small — append after rerun)
    # -----------------------------------------------------------------------
    hydra_config = run_dir / ".hydra" / "config.yaml"
    if hydra_config.is_file():
        entries.append(ArtifactEntry(hydra_config, f"{prefix}/config.yaml"))

    return entries


# ---------------------------------------------------------------------------
# W&B reference artifact registration
# ---------------------------------------------------------------------------


def _register_wandb_artifacts(
    run_id: str,
    project: Optional[str],
    entity: Optional[str],
    uploaded_urls: list[str],
    rerun_viewer_urls: list[str],
) -> None:
    """Best-effort: register R2 URLs as W&B reference artifacts."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        log.info("wandb not installed — skipping W&B artifact registration.")
        return

    try:
        run = wandb.init(
            id=run_id,
            resume="allow",
            project=project or "corvidx-drone-racing",
            entity=entity,
        )

        # Log R2 artifact as a reference artifact
        artifact = wandb.Artifact(
            name=f"r2-artifacts-{run_id}",
            type="model",
            description="R2-hosted training artifacts",
        )
        for url in uploaded_urls:
            # Use the URL path as the artifact entry name
            name = url.replace(R2_PUBLIC_BASE + "/", "")
            artifact.add_reference(url, name=name)

        run.log_artifact(artifact)

        # Also surface rerun viewer URLs in the run summary
        if rerun_viewer_urls:
            run.summary["rerun_viewer_urls"] = rerun_viewer_urls

        run.finish()
        log.info("W&B artifact registration complete for run %s.", run_id)
    except Exception as exc:
        log.warning("W&B artifact registration failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main upload loop
# ---------------------------------------------------------------------------


def run_sync(
    run_dir: Path,
    *,
    run_id: Optional[str],
    timeout: Optional[float],
    bucket: str,
    account_id: str,
    access_key: str,
    secret_key: str,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> int:
    """Execute the upload. Returns exit code (0 = success)."""

    # Resolve run ID
    discovered_id = _find_wandb_run_id(run_dir)
    if discovered_id:
        if run_id and run_id != discovered_id:
            log.warning(
                "W&B run ID from filesystem (%s) differs from --run-id (%s); "
                "using filesystem value.",
                discovered_id,
                run_id,
            )
        resolved_run_id = discovered_id
    elif run_id:
        resolved_run_id = run_id
        log.info("No wandb directory found — using provided --run-id: %s", resolved_run_id)
    else:
        log.error(
            "Could not determine W&B run ID. Either make sure wandb/latest-run/ "
            "exists in the run directory, or pass --run-id explicitly."
        )
        return 1

    log.info("Run ID: %s", resolved_run_id)
    log.info("Run directory: %s", run_dir)

    client = _make_s3_client(account_id, access_key, secret_key)

    artifacts = _collect_artifacts(run_dir, resolved_run_id)
    if not artifacts:
        log.warning("No artifacts found to upload in %s", run_dir)
        return 0

    log.info("Found %d artifact(s) to consider for upload.", len(artifacts))

    deadline = time.monotonic() + timeout if timeout else None

    uploaded_urls: list[str] = []
    skipped: list[str] = []
    timed_out: list[str] = []
    failed: list[tuple[str, str]] = []
    total_bytes = 0
    rerun_viewer_urls: list[str] = []

    for entry in artifacts:
        if deadline and time.monotonic() >= deadline:
            remaining = [e.r2_key for e in artifacts[artifacts.index(entry) :]]
            timed_out.extend(remaining)
            log.warning(
                "Timeout reached. %d file(s) were not uploaded.", len(timed_out)
            )
            break

        try:
            if entry.is_dir_zip:
                # Directory must be zipped in memory
                assert entry.local_path is not None
                log.info("Zipping directory: %s", entry.local_path)
                zip_bytes = _zip_directory(entry.local_path)
                url = _upload_bytes(client, bucket, zip_bytes, entry.r2_key)
                if url is None:
                    skipped.append(entry.r2_key)
                else:
                    uploaded_urls.append(url)
                    total_bytes += len(zip_bytes)
            else:
                assert entry.local_path is not None
                url = _upload_file(client, bucket, entry.local_path, entry.r2_key)
                if url is None:
                    skipped.append(entry.r2_key)
                else:
                    uploaded_urls.append(url)
                    total_bytes += entry.local_path.stat().st_size
                    if entry.r2_key.endswith(".rrd"):
                        rerun_viewer_urls.append(_rerun_viewer_url(entry.r2_key))
        except Exception as exc:
            log.error("Failed to upload %s: %s", entry.r2_key, exc)
            failed.append((entry.r2_key, str(exc)))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Sync summary for run: {resolved_run_id}")
    print("=" * 60)
    print(f"  Uploaded : {len(uploaded_urls)} file(s)  ({_human_size(total_bytes)})")
    print(f"  Skipped  : {len(skipped)} file(s) (already on R2)")
    if timed_out:
        print(f"  Timed out: {len(timed_out)} file(s) — increase --timeout or re-run")
    if failed:
        print(f"  Failed   : {len(failed)} file(s)")
        for key, err in failed:
            print(f"    {key}: {err}")

    if uploaded_urls:
        print()
        print("Uploaded URLs:")
        for url in uploaded_urls:
            print(f"  {url}")

    if rerun_viewer_urls:
        print()
        print("Rerun viewer URLs:")
        for url in rerun_viewer_urls:
            print(f"  {url}")

    if timed_out:
        print()
        print("Files NOT uploaded (timeout):")
        for key in timed_out:
            print(f"  {key}")

    print()

    # -----------------------------------------------------------------------
    # W&B reference artifact registration (best-effort)
    # -----------------------------------------------------------------------
    if uploaded_urls:
        _register_wandb_artifacts(
            run_id=resolved_run_id,
            project=wandb_project,
            entity=wandb_entity,
            uploaded_urls=uploaded_urls,
            rerun_viewer_urls=rerun_viewer_urls,
        )

    return 1 if failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync training artifacts to Cloudflare R2 and register with W&B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the Hydra output directory (e.g. outputs/2026-03-11_10-30-00/)",
    )
    parser.add_argument(
        "--run-id",
        metavar="RUN_ID",
        help=(
            "W&B run ID to use as the R2 prefix. "
            "Auto-detected from wandb/latest-run/ if not provided."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        default=None,
        help="Stop uploading after this many seconds (useful on spot instances).",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        metavar="BUCKET",
        help="R2 bucket name (default: $R2_BUCKET or 'corvidx-artifacts').",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        metavar="PROJECT",
        help="W&B project name for artifact registration.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        metavar="ENTITY",
        help="W&B entity (team/user) for artifact registration.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    # Load .env from project root (two levels up from scripts/)
    script_dir = Path(__file__).resolve().parent
    _load_dotenv(script_dir.parent / ".env")

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        log.error("Run directory does not exist: %s", run_dir)
        return 1

    account_id = _require_env("R2_ACCOUNT_ID")
    access_key = _require_env("R2_ACCESS_KEY_ID")
    secret_key = _require_env("R2_SECRET_ACCESS_KEY")
    bucket = args.bucket or os.environ.get("R2_BUCKET", DEFAULT_BUCKET)

    return run_sync(
        run_dir,
        run_id=args.run_id,
        timeout=args.timeout,
        bucket=bucket,
        account_id=account_id,
        access_key=access_key,
        secret_key=secret_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    sys.exit(main())
