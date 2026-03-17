"""Unit tests for pair table."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from perception.training.data.pairs import PairTable


def _make_overlap_matrix(N=10, seed=42):
    """Create a random symmetric overlap matrix with 1s on diagonal."""
    rng = np.random.default_rng(seed)
    raw = rng.random((N, N)).astype(np.float32)
    sym = (raw + raw.T) / 2
    np.fill_diagonal(sym, 1.0)
    return sym


class TestPairTable:
    def test_from_overlap_matrix(self):
        overlap = _make_overlap_matrix(N=10)
        pairs = PairTable.from_overlap_matrix(overlap, "test", "scene0")
        # Upper triangle of 10x10: 45 pairs max
        assert len(pairs) <= 45
        assert len(pairs) > 0

    def test_min_overlap_filter(self):
        overlap = _make_overlap_matrix(N=10)
        all_pairs = PairTable.from_overlap_matrix(overlap, "test", "scene0", min_overlap=0.0)
        filtered = PairTable.from_overlap_matrix(overlap, "test", "scene0", min_overlap=0.5)
        assert len(filtered) <= len(all_pairs)

    def test_save_load_roundtrip(self):
        overlap = _make_overlap_matrix(N=5)
        pairs = PairTable.from_overlap_matrix(overlap, "test", "scene0")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.parquet"
            pairs.save(path)
            loaded = PairTable.load(path)
            assert len(loaded) == len(pairs)

    def test_merge(self):
        o1 = _make_overlap_matrix(N=5, seed=1)
        o2 = _make_overlap_matrix(N=5, seed=2)
        p1 = PairTable.from_overlap_matrix(o1, "test", "scene0")
        p2 = PairTable.from_overlap_matrix(o2, "test", "scene1")
        merged = PairTable.merge([p1, p2])
        assert len(merged) == len(p1) + len(p2)

    def test_sample_uniform(self):
        overlap = _make_overlap_matrix(N=20)
        pairs = PairTable.from_overlap_matrix(overlap, "test", "scene0")
        n = min(10, len(pairs))
        sampled = pairs.sample_uniform(n)
        assert len(sampled) <= n
