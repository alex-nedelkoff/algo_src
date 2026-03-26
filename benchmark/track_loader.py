"""Download and cache golden set tracks from R2."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmark.track_generator import deserialize_track

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "corvidx" / "golden_set"
R2_PREFIX = "golden-set"


class GoldenSetLoader:
    """Loads golden set tracks from local directory, cache, or R2."""

    def __init__(
        self,
        version: str = "v1",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
    ) -> None:
        self.version = version
        self.cache_dir = (cache_dir or DEFAULT_CACHE_DIR / version)
        self.local_dir = local_dir

    def load_all(self) -> dict[str, dict[str, Any]]:
        """Load all tracks. Returns {layout_name: {track, arena_bounds, ...}}."""
        source = self._resolve_source()
        tracks = {}
        for npz_file in sorted(source.glob("*.npz")):
            name = npz_file.stem
            tracks[name] = deserialize_track(npz_file)
            log.info("Loaded golden set track: %s (%d gates)",
                     name, tracks[name]["track"].num_gates)
        if not tracks:
            raise FileNotFoundError(
                f"No .npz track files found in {source}. "
                f"Run 'python -m benchmark generate' first."
            )
        return tracks

    def _resolve_source(self) -> Path:
        """Find tracks: local_dir > cache > download from R2."""
        if self.local_dir and self.local_dir.exists():
            log.info("Using local golden set: %s", self.local_dir)
            return self.local_dir

        if self.cache_dir.exists():
            npz_files = list(self.cache_dir.glob("*.npz"))
            if npz_files:
                log.info("Using cached golden set: %s", self.cache_dir)
                return self.cache_dir

        log.info("Downloading golden set %s from R2...", self.version)
        self._download_from_r2()
        return self.cache_dir

    def _download_from_r2(self) -> None:
        """Download golden set from R2 to cache."""
        from artifacts.r2 import make_r2_client_from_env, DEFAULT_BUCKET

        client = make_r2_client_from_env()

        prefix = f"{R2_PREFIX}/{self.version}/"
        response = client.list_objects_v2(Bucket=DEFAULT_BUCKET, Prefix=prefix)
        contents = response.get("Contents", [])
        if not contents:
            raise FileNotFoundError(
                f"No golden set found at R2 path: {prefix}. "
                f"Run 'python -m benchmark generate --upload' first."
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for obj in contents:
            key = obj["Key"]
            filename = key.split("/")[-1]
            if not filename.endswith(".npz"):
                continue
            local_path = self.cache_dir / filename
            log.info("  Downloading %s", key)
            client.download_file(DEFAULT_BUCKET, key, str(local_path))
