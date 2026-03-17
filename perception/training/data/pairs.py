"""Parquet-backed pair table with covisibility-binned sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema([
    ("dataset", pa.string()),
    ("scene_id", pa.string()),
    ("frame_i", pa.int32()),
    ("frame_j", pa.int32()),
    ("overlap_score", pa.float32()),
])


class PairTable:
    """Parquet-backed table of image pairs with covisibility overlap scores."""

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    @classmethod
    def from_overlap_matrix(
        cls,
        overlap: np.ndarray,
        dataset: str,
        scene_id: str,
        min_overlap: float = 0.0,
    ) -> PairTable:
        """Build pair table from an N x N symmetric overlap matrix.

        Only stores upper triangle (i < j) and filters out pairs below min_overlap.
        """
        N = overlap.shape[0]
        rows_i, rows_j = np.triu_indices(N, k=1)
        scores = overlap[rows_i, rows_j].astype(np.float32)

        # Filter
        mask = scores > min_overlap
        rows_i = rows_i[mask]
        rows_j = rows_j[mask]
        scores = scores[mask]

        table = pa.table({
            "dataset": pa.array([dataset] * len(scores), type=pa.string()),
            "scene_id": pa.array([scene_id] * len(scores), type=pa.string()),
            "frame_i": pa.array(rows_i.astype(np.int32)),
            "frame_j": pa.array(rows_j.astype(np.int32)),
            "overlap_score": pa.array(scores),
        }, schema=SCHEMA)

        return cls(table)

    @classmethod
    def load(cls, path: str | Path) -> PairTable:
        """Load from Parquet file."""
        return cls(pq.read_table(str(path), schema=SCHEMA))

    @classmethod
    def merge(cls, tables: list[PairTable]) -> PairTable:
        """Merge multiple pair tables into one."""
        combined = pa.concat_tables([t._table for t in tables])
        return cls(combined)

    def save(self, path: str | Path) -> None:
        """Save to Parquet file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self._table, str(path))

    @property
    def table(self) -> pa.Table:
        return self._table

    def __len__(self) -> int:
        return len(self._table)

    def sample_uniform(
        self,
        n: int,
        bins: list[tuple[float, float]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> pa.Table:
        """Sample n pairs uniformly across covisibility bins.

        If a bin has fewer pairs than its quota, surplus is redistributed
        to other bins proportionally.

        Args:
            n: Total number of pairs to sample.
            bins: List of (low, high) overlap ranges. Defaults to
                  hard (0.05-0.3), medium (0.3-0.7), easy (0.7-1.0).
            rng: Optional random generator.

        Returns:
            Sampled pa.Table subset.
        """
        if rng is None:
            rng = np.random.default_rng()

        if bins is None:
            bins = [(0.05, 0.3), (0.3, 0.7), (0.7, 1.0)]

        scores = self._table.column("overlap_score").to_numpy()

        # Assign rows to bins
        bin_indices: list[np.ndarray] = []
        for low, high in bins:
            mask = (scores >= low) & (scores < high)
            bin_indices.append(np.where(mask)[0])

        # Compute per-bin quotas with redistribution
        per_bin = n // len(bins)
        remainder = n % len(bins)
        quotas = [per_bin] * len(bins)
        quotas[-1] += remainder

        # Redistribute from under-filled bins
        sampled_indices: list[np.ndarray] = []
        deficit = 0
        filled_bins = []
        for i, (indices, quota) in enumerate(zip(bin_indices, quotas)):
            available = len(indices)
            if available <= quota:
                sampled_indices.append(indices)
                deficit += quota - available
            else:
                filled_bins.append(i)
                sampled_indices.append(None)  # placeholder

        # Redistribute deficit to filled bins
        if deficit > 0 and filled_bins:
            extra_per = deficit // len(filled_bins)
            extra_rem = deficit % len(filled_bins)
            for idx, i in enumerate(filled_bins):
                extra = extra_per + (1 if idx < extra_rem else 0)
                quota_adjusted = quotas[i] + extra
                chosen = rng.choice(bin_indices[i], size=min(quota_adjusted, len(bin_indices[i])), replace=False)
                sampled_indices[i] = chosen
        else:
            # Fill remaining bins normally
            for i in filled_bins:
                chosen = rng.choice(bin_indices[i], size=min(quotas[i], len(bin_indices[i])), replace=False)
                sampled_indices[i] = chosen

        all_sampled = np.concatenate([s for s in sampled_indices if s is not None and len(s) > 0])
        return self._table.take(all_sampled)
