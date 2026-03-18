"""Tests for gate detection manifest builder and split creation.

Covers: build_manifest Parquet output, create_splits stratified splitting, CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from perception.training.data.gate_detection.format import (
    generate_corner_heatmaps,
    render_gate_mask,
    save_sample,
)
from perception.training.data.gate_detection.merge import (
    SOURCES,
    build_manifest,
    create_splits,
)

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

H, W = 480, 640
STRIDE = 4


def _make_sample(
    sample_dir: Path,
    *,
    n_gates: int = 1,
    n_drones: int = 0,
    resolution: tuple[int, int] = (H, W),
    distortion_model: str | None = None,
) -> None:
    """Create a minimal fake sample on disk using save_sample()."""
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 255, size=(resolution[0], resolution[1], 3), dtype=np.uint8)

    corner_coords = []
    for gate_id in range(1, n_gates + 1):
        corner_coords.append(
            {
                "gate_id": gate_id,
                "corners": [
                    [100.0, 100.0],
                    [200.0, 100.0],
                    [200.0, 200.0],
                    [100.0, 200.0],
                ],
                "confidence": 1.0,
            }
        )

    gate_mask = render_gate_mask(corner_coords, resolution[0], resolution[1])
    obstacle_mask = np.zeros((resolution[0], resolution[1]), dtype=np.uint8)
    corner_heatmaps = generate_corner_heatmaps(
        corner_coords, resolution[0], resolution[1], stride=STRIDE
    )

    metadata: dict = {
        "source": "test",
        "resolution": list(resolution),
        "n_drones": n_drones,
    }
    if distortion_model is not None:
        metadata["distortion_model"] = distortion_model

    save_sample(
        sample_dir,
        rgb=rgb,
        gate_mask=gate_mask,
        obstacle_mask=obstacle_mask,
        corner_coords=corner_coords,
        corner_heatmaps=corner_heatmaps,
        metadata=metadata,
    )


def _populate_data_dir(
    data_dir: Path,
    *,
    tii_count: int = 5,
    blenderproc_count: int = 5,
) -> None:
    """Create a fake data directory with tii and blenderproc sources."""
    for i in range(tii_count):
        _make_sample(
            data_dir / "tii" / f"sample_{i:04d}",
            n_gates=1,
            n_drones=0,
            distortion_model="pinhole",
        )
    for i in range(blenderproc_count):
        _make_sample(
            data_dir / "blenderproc" / f"sample_{i:04d}",
            n_gates=2,
            n_drones=1,
            resolution=(720, 1280),
            distortion_model="equidistant",
        )


def _column_values_for_source(table: pa.Table, col: str, source: str) -> list:
    """Extract unique column values for rows matching a given source (pure pyarrow)."""
    sources = table.column("source").to_pylist()
    values = table.column(col).to_pylist()
    return sorted(set(v for v, s in zip(values, sources) if s == source))


# --------------------------------------------------------------------- #
# SOURCES constant
# --------------------------------------------------------------------- #


class TestSourcesConstant:
    def test_sources_contains_expected(self) -> None:
        assert "tii" in SOURCES
        assert "blenderproc" in SOURCES

    def test_sources_is_list(self) -> None:
        assert isinstance(SOURCES, list)


# --------------------------------------------------------------------- #
# build_manifest
# --------------------------------------------------------------------- #


class TestBuildManifest:
    """Tests for build_manifest() Parquet output."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "data"
        _populate_data_dir(d, tii_count=5, blenderproc_count=3)
        return d

    def test_manifest_row_count(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        assert len(table) == 8  # 5 tii + 3 blenderproc

    def test_manifest_columns(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        expected_cols = {
            "sample_id",
            "source",
            "n_gates",
            "n_drones",
            "resolution_h",
            "resolution_w",
            "has_distortion",
        }
        assert set(table.column_names) == expected_cols

    def test_manifest_source_values(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        sources = set(table.column("source").to_pylist())
        assert sources == {"tii", "blenderproc"}

    def test_manifest_n_gates(self, data_dir: Path, tmp_path: Path) -> None:
        """TII samples have 1 gate, BlenderProc samples have 2."""
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        assert _column_values_for_source(table, "n_gates", "tii") == [1]
        assert _column_values_for_source(table, "n_gates", "blenderproc") == [2]

    def test_manifest_n_drones(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        assert _column_values_for_source(table, "n_drones", "tii") == [0]
        assert _column_values_for_source(table, "n_drones", "blenderproc") == [1]

    def test_manifest_resolution(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        assert _column_values_for_source(table, "resolution_h", "tii") == [480]
        assert _column_values_for_source(table, "resolution_h", "blenderproc") == [720]

    def test_manifest_has_distortion(self, data_dir: Path, tmp_path: Path) -> None:
        """pinhole -> False, equidistant -> True."""
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        assert _column_values_for_source(table, "has_distortion", "tii") == [False]
        assert _column_values_for_source(table, "has_distortion", "blenderproc") == [True]

    def test_manifest_sample_ids_unique(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        ids = table.column("sample_id").to_pylist()
        assert len(ids) == len(set(ids))

    def test_manifest_sample_id_format(self, data_dir: Path, tmp_path: Path) -> None:
        """sample_id should be 'source/dirname'."""
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        ids = table.column("sample_id").to_pylist()
        for sid in ids:
            assert "/" in sid
            source_part = sid.split("/")[0]
            assert source_part in SOURCES

    def test_manifest_missing_source_dir_is_skipped(self, tmp_path: Path) -> None:
        """If a source directory doesn't exist, it should be skipped."""
        data_dir = tmp_path / "data"
        _make_sample(data_dir / "tii" / "sample_0000", n_gates=1)
        # No blenderproc directory at all
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        assert len(table) == 1

    def test_manifest_creates_parent_dirs(self, data_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "manifest.parquet"
        build_manifest(data_dir, out)
        assert out.exists()

    def test_manifest_no_distortion_key_means_false(self, tmp_path: Path) -> None:
        """Missing distortion_model key in metadata -> has_distortion=False."""
        data_dir = tmp_path / "data"
        _make_sample(
            data_dir / "tii" / "sample_0000",
            n_gates=1,
            distortion_model=None,  # metadata will not have distortion_model key
        )
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        table = pq.read_table(str(out))
        assert table.column("has_distortion").to_pylist() == [False]


# --------------------------------------------------------------------- #
# create_splits
# --------------------------------------------------------------------- #


class TestCreateSplits:
    """Tests for create_splits() stratified splitting."""

    @pytest.fixture
    def manifest_path(self, tmp_path: Path) -> Path:
        data_dir = tmp_path / "data"
        # Use larger counts for meaningful split proportions
        _populate_data_dir(data_dir, tii_count=20, blenderproc_count=20)
        out = tmp_path / "manifest.parquet"
        build_manifest(data_dir, out)
        return out

    def test_splits_files_exist(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        splits_dir = tmp_path / "splits"
        create_splits(manifest_path, splits_dir, val_ratio=0.1, seed=42)
        assert (splits_dir / "train.txt").exists()
        assert (splits_dir / "val.txt").exists()

    def test_splits_no_overlap(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        splits_dir = tmp_path / "splits"
        create_splits(manifest_path, splits_dir, val_ratio=0.1, seed=42)

        train_ids = set((splits_dir / "train.txt").read_text().strip().splitlines())
        val_ids = set((splits_dir / "val.txt").read_text().strip().splitlines())
        assert train_ids & val_ids == set()

    def test_splits_cover_all_samples(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        splits_dir = tmp_path / "splits"
        create_splits(manifest_path, splits_dir, val_ratio=0.1, seed=42)

        train_ids = set((splits_dir / "train.txt").read_text().strip().splitlines())
        val_ids = set((splits_dir / "val.txt").read_text().strip().splitlines())

        table = pq.read_table(str(manifest_path))
        all_ids = set(table.column("sample_id").to_pylist())
        assert train_ids | val_ids == all_ids

    def test_splits_approximate_proportions(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        splits_dir = tmp_path / "splits"
        create_splits(manifest_path, splits_dir, val_ratio=0.1, seed=42)

        train_ids = (splits_dir / "train.txt").read_text().strip().splitlines()
        val_ids = (splits_dir / "val.txt").read_text().strip().splitlines()

        total = len(train_ids) + len(val_ids)
        val_frac = len(val_ids) / total
        # With 40 samples at 10% val, we expect ~4 val samples
        # Allow a tolerance since stratification rounds per-source
        assert 0.05 <= val_frac <= 0.20

    def test_splits_deterministic(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        """Same seed should produce same splits."""
        splits1 = tmp_path / "splits1"
        splits2 = tmp_path / "splits2"
        create_splits(manifest_path, splits1, val_ratio=0.1, seed=42)
        create_splits(manifest_path, splits2, val_ratio=0.1, seed=42)

        assert (splits1 / "train.txt").read_text() == (splits2 / "train.txt").read_text()
        assert (splits1 / "val.txt").read_text() == (splits2 / "val.txt").read_text()

    def test_splits_different_seed(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        """Different seeds should (very likely) produce different splits."""
        splits1 = tmp_path / "splits1"
        splits2 = tmp_path / "splits2"
        create_splits(manifest_path, splits1, val_ratio=0.1, seed=42)
        create_splits(manifest_path, splits2, val_ratio=0.1, seed=99)

        # With 40 samples, it's extremely unlikely the same 4 val samples are chosen
        val1 = set((splits1 / "val.txt").read_text().strip().splitlines())
        val2 = set((splits2 / "val.txt").read_text().strip().splitlines())
        assert val1 != val2

    def test_splits_stratified_by_source(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        """Both tii and blenderproc should contribute to val set."""
        splits_dir = tmp_path / "splits"
        create_splits(manifest_path, splits_dir, val_ratio=0.25, seed=42)

        val_ids = set((splits_dir / "val.txt").read_text().strip().splitlines())

        table = pq.read_table(str(manifest_path))
        sample_ids = table.column("sample_id").to_pylist()
        sources = table.column("source").to_pylist()

        # Build source lookup
        tii_samples = {sid for sid, src in zip(sample_ids, sources) if src == "tii"}
        bp_samples = {sid for sid, src in zip(sample_ids, sources) if src == "blenderproc"}

        assert len(val_ids & tii_samples) > 0, "No tii samples in val set"
        assert len(val_ids & bp_samples) > 0, "No blenderproc samples in val set"

    def test_splits_one_per_line(
        self, manifest_path: Path, tmp_path: Path
    ) -> None:
        """Each line should be a single sample ID with no extra whitespace."""
        splits_dir = tmp_path / "splits"
        create_splits(manifest_path, splits_dir, val_ratio=0.1, seed=42)

        for name in ("train.txt", "val.txt"):
            content = (splits_dir / name).read_text()
            assert content.endswith("\n")
            lines = content.strip().splitlines()
            for line in lines:
                assert line == line.strip()
