"""Manifest builder and train/val split creation for gate detection data.

Scans source subdirectories for samples produced by ``save_sample()``,
builds a Parquet manifest, and creates stratified train/val split files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

SOURCES: list[str] = ["tii", "blenderproc"]

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("source", pa.string()),
        ("n_gates", pa.int32()),
        ("n_drones", pa.int32()),
        ("resolution_h", pa.int32()),
        ("resolution_w", pa.int32()),
        ("has_distortion", pa.bool_()),
    ]
)


def build_manifest(data_dir: str | Path, output_path: str | Path) -> None:
    """Scan source subdirectories and build a Parquet manifest.

    Each sample directory must contain ``metadata.json`` and
    ``corner_coords.json`` (as produced by :func:`save_sample`).

    Args:
        data_dir: Root data directory containing source subdirs (e.g. "tii",
            "blenderproc").
        output_path: Path to write the output Parquet file.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list] = {
        "sample_id": [],
        "source": [],
        "n_gates": [],
        "n_drones": [],
        "resolution_h": [],
        "resolution_w": [],
        "has_distortion": [],
    }

    for source in SOURCES:
        source_dir = data_dir / source
        if not source_dir.is_dir():
            log.info("Source directory %s not found, skipping.", source_dir)
            continue

        for sample_dir in sorted(source_dir.iterdir()):
            if not sample_dir.is_dir():
                continue

            metadata_path = sample_dir / "metadata.json"
            corners_path = sample_dir / "corner_coords.json"

            if not metadata_path.exists() or not corners_path.exists():
                log.warning(
                    "Skipping %s: missing metadata.json or corner_coords.json",
                    sample_dir,
                )
                continue

            with open(metadata_path) as f:
                metadata = json.load(f)

            with open(corners_path) as f:
                corner_coords = json.load(f)

            distortion_model = metadata.get("distortion_model")
            has_distortion = distortion_model not in (None, "pinhole")

            resolution = metadata.get("resolution", [0, 0])

            rows["sample_id"].append(f"{source}/{sample_dir.name}")
            rows["source"].append(source)
            rows["n_gates"].append(len(corner_coords))
            rows["n_drones"].append(metadata.get("n_drones", 0))
            rows["resolution_h"].append(resolution[0])
            rows["resolution_w"].append(resolution[1])
            rows["has_distortion"].append(has_distortion)

    table = pa.table(rows, schema=SCHEMA)
    pq.write_table(table, str(output_path))

    log.info("Wrote manifest with %d rows to %s", len(table), output_path)


def create_splits(
    manifest_path: str | Path,
    splits_dir: str | Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Create stratified train/val splits from a manifest.

    Stratification is by source — each source contributes proportionally
    to the validation set.

    Args:
        manifest_path: Path to the Parquet manifest built by
            :func:`build_manifest`.
        splits_dir: Directory to write ``train.txt`` and ``val.txt``.
        val_ratio: Fraction of samples to place in the validation set.
        seed: Random seed for reproducibility.
    """
    manifest_path = Path(manifest_path)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(manifest_path))
    sample_ids = table.column("sample_id").to_pylist()
    sources = table.column("source").to_pylist()

    rng = np.random.default_rng(seed)

    # Group sample indices by source
    source_groups: dict[str, list[int]] = {}
    for idx, src in enumerate(sources):
        source_groups.setdefault(src, []).append(idx)

    train_ids: list[str] = []
    val_ids: list[str] = []

    for src, indices in sorted(source_groups.items()):
        indices_arr = np.array(indices)
        rng.shuffle(indices_arr)

        n_val = max(1, int(round(len(indices_arr) * val_ratio)))
        val_indices = indices_arr[:n_val]
        train_indices = indices_arr[n_val:]

        for i in val_indices:
            val_ids.append(sample_ids[i])
        for i in train_indices:
            train_ids.append(sample_ids[i])

    with open(splits_dir / "train.txt", "w") as f:
        for sid in sorted(train_ids):
            f.write(sid + "\n")

    with open(splits_dir / "val.txt", "w") as f:
        for sid in sorted(val_ids):
            f.write(sid + "\n")

    log.info(
        "Split %d samples: %d train, %d val (ratio=%.2f, seed=%d)",
        len(sample_ids),
        len(train_ids),
        len(val_ids),
        val_ratio,
        seed,
    )


def main() -> None:
    """CLI entry point for manifest building and splitting."""
    parser = argparse.ArgumentParser(
        description="Build gate detection manifest and create train/val splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory containing source subdirs.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42).",
    )

    args = parser.parse_args()

    data_dir: Path = args.data_dir
    manifest_path = data_dir / "manifest.parquet"
    splits_dir = data_dir / "splits"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    build_manifest(data_dir, manifest_path)
    create_splits(manifest_path, splits_dir, val_ratio=args.val_ratio, seed=args.seed)


if __name__ == "__main__":
    main()
