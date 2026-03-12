"""Background daemon thread for uploading trajectory artifacts to R2.

Runs alongside the training loop so that ``TrajectoryRecorderCallback`` can
hand off ``.npz`` directories without blocking.  The worker thread converts
each episode to ``.rrd`` (if *rerun-sdk* is available), uploads both formats
to Cloudflare R2, and registers W&B reference artifacts.

All failures are logged as warnings — the training loop is **never** affected.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple

from artifacts.r2 import (
    DEFAULT_BUCKET,
    make_r2_client,
    register_wandb_artifact,
    rerun_viewer_url,
    upload_file,
)

log = logging.getLogger(__name__)

# Work‐item type: (npz_directory, training_step)
_WorkItem = Tuple[Path, int]


class ArtifactUploader:
    """Daemon thread that uploads trajectory artifacts to R2 in the background.

    Parameters
    ----------
    run_id:
        W&B run ID — also used as the R2 path prefix (``runs/{run_id}/...``).
    wandb_entity:
        W&B entity (user or team) for constructing the ``run_path``.
    wandb_project:
        W&B project name for constructing the ``run_path``.
    """

    def __init__(
        self,
        run_id: str,
        wandb_entity: str,
        wandb_project: str,
    ) -> None:
        self._run_id = run_id
        self._run_path = f"{wandb_entity}/{wandb_project}/{run_id}"
        self._enabled = True

        # ----- R2 client ---------------------------------------------------
        account_id = os.environ.get("R2_ACCOUNT_ID", "")
        access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self._bucket = os.environ.get("R2_BUCKET", DEFAULT_BUCKET)

        if not (account_id and access_key and secret_key):
            log.warning(
                "R2 credentials not set (R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / "
                "R2_SECRET_ACCESS_KEY). ArtifactUploader entering no-op mode."
            )
            self._enabled = False
            return

        try:
            self._client = make_r2_client(account_id, access_key, secret_key)
        except ImportError:
            log.warning(
                "boto3 is not installed — ArtifactUploader entering no-op mode. "
                "Install with: pip install boto3"
            )
            self._enabled = False
            return

        # ----- W&B API (thread-safe) ---------------------------------------
        try:
            import wandb

            self._wandb_api = wandb.Api()
        except Exception as exc:
            log.warning(
                "Could not create wandb.Api() — artifact registration will be "
                "skipped: %s",
                exc,
            )
            self._wandb_api = None

        # ----- Queue & worker thread ---------------------------------------
        self._queue: queue.Queue[Optional[_WorkItem]] = queue.Queue(maxsize=10)
        self._thread = threading.Thread(
            target=self._worker, name="artifact-uploader", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, npz_path: Path, step: int) -> None:
        """Enqueue a directory of ``.npz`` episode files for upload.

        Non-blocking.  If the internal queue is full the **oldest** item is
        dropped so that training is never stalled.
        """
        if not self._enabled:
            return

        item: _WorkItem = (npz_path, step)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the oldest item to make room.
            try:
                dropped = self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if dropped is None:
                    # Never discard the shutdown sentinel.
                    self._queue.put_nowait(None)
                    log.warning(
                        "Upload queue full — discarding step %s.", step
                    )
                    return
                log.warning(
                    "Upload queue full — dropping oldest item (step %s) "
                    "to make room for step %s.",
                    dropped[1],
                    step,
                )
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                log.warning(
                    "Upload queue still full after drop — discarding step %s.",
                    step,
                )

    def close(self, timeout: float = 60) -> None:
        """Signal shutdown and wait for the worker to drain.

        Parameters
        ----------
        timeout:
            Seconds to wait for the worker thread to finish.
        """
        if not self._enabled:
            return

        # Send sentinel to signal shutdown.
        self._queue.put(None)
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            remaining = self._queue.qsize()
            log.warning(
                "ArtifactUploader thread did not finish within %.0fs — "
                "~%d items may not have been uploaded.",
                timeout,
                remaining,
            )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Background loop: pull work items, generate .rrd, upload, register."""
        while True:
            item = self._queue.get()
            if item is None:
                # Sentinel — shut down.
                break
            try:
                self._process(item)
            except Exception:
                log.warning(
                    "Unexpected error processing upload item %s — skipping.",
                    item,
                    exc_info=True,
                )

    def _process(self, item: _WorkItem) -> None:
        npz_dir, step = item
        npz_dir = Path(npz_dir)

        if not npz_dir.is_dir():
            log.warning("npz_path is not a directory: %s — skipping.", npz_dir)
            return

        npz_files = sorted(npz_dir.glob("*.npz"))
        if not npz_files:
            log.warning("No .npz files found in %s — skipping.", npz_dir)
            return

        for npz_file in npz_files:
            self._process_episode(npz_file, step)

    def _process_episode(self, npz_file: Path, step: int) -> None:
        stem = npz_file.stem  # e.g. "eval_ep_0"

        # --- 1. Generate .rrd from .npz -----------------------------------
        rrd_file: Optional[Path] = None
        try:
            from sim.viz.rerun_generator import generate_rrd

            rrd_file = generate_rrd(npz_file)
        except ImportError:
            log.warning(
                "rerun-sdk not available — uploading .npz only (no .rrd) "
                "for %s.",
                npz_file.name,
            )
        except Exception:
            log.warning(
                "Failed to generate .rrd for %s — uploading .npz only.",
                npz_file.name,
                exc_info=True,
            )

        # --- 2. Upload .npz to R2 -----------------------------------------
        npz_key = f"runs/{self._run_id}/trajectories/step_{step}/{stem}.npz"
        try:
            upload_file(self._client, npz_file, self._bucket, npz_key)
        except Exception:
            log.warning(
                "Failed to upload %s to R2 — skipping.", npz_key, exc_info=True
            )

        # --- 3. Upload .rrd to R2 (if generated) --------------------------
        rrd_key: Optional[str] = None
        if rrd_file is not None and rrd_file.exists():
            rrd_key = f"runs/{self._run_id}/rerun/step_{step}/{stem}.rrd"
            try:
                upload_file(self._client, rrd_file, self._bucket, rrd_key)
            except Exception:
                log.warning(
                    "Failed to upload %s to R2 — skipping.",
                    rrd_key,
                    exc_info=True,
                )
                rrd_key = None

        # --- 4. Register W&B artifacts ------------------------------------
        if self._wandb_api is None:
            return

        if rrd_key is not None:
            viewer_url = rerun_viewer_url(rrd_key)
            artifact_name = f"rerun-step_{step}-{stem}"
            try:
                register_wandb_artifact(
                    self._wandb_api,
                    self._run_path,
                    artifact_name,
                    viewer_url,
                    artifact_type="rerun-recording",
                )
            except Exception:
                log.warning(
                    "Failed to register W&B artifact '%s' — skipping.",
                    artifact_name,
                    exc_info=True,
                )
