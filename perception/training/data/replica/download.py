"""Download and extract the Replica dataset (NICE-SLAM version)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPLICA_URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"


def download_replica(data_dir: str | Path = "/data") -> Path:
    """Download and extract Replica to data_dir/Replica/.

    Skips download if zip already exists. Skips extraction if directory exists.

    Args:
        data_dir: Root data directory (default: /data for Docker).

    Returns:
        Path to extracted Replica directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "Replica.zip"
    extract_dir = data_dir / "Replica"

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Replica already extracted at {extract_dir}, skipping.")
        return extract_dir

    if not zip_path.exists():
        print(f"Downloading Replica from {REPLICA_URL} ...")
        subprocess.run(
            ["wget", "-c", REPLICA_URL, "-O", str(zip_path)],
            check=True,
        )
    else:
        print(f"Replica zip already exists at {zip_path}, skipping download.")

    print(f"Extracting to {extract_dir} ...")
    subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(data_dir)],
        check=True,
    )

    print(f"Replica ready at {extract_dir}")
    return extract_dir


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data"
    download_replica(data_dir)
