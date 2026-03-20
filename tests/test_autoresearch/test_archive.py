"""Tests for MAP-Elites archive."""

import json
from pathlib import Path

import pytest

from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.descriptors.compute import DescriptorVector


@pytest.fixture
def archive():
    return MapElitesArchive(
        bins_per_axis=5,
        axis_ranges=[(0.25, 1.0), (0.0, 1.0), (0.0, 24.0)],
    )


@pytest.fixture
def sample_entry():
    return CellEntry(
        fitness=5.2,
        status="candidate",
        wandb_run_id="run_abc",
        git_commit="abc123",
        descriptors=DescriptorVector(0.5, 0.3, 10.0),
        constraint_results={"passed": True},
        budget_spent=5_000_000,
        hypothesis_id="hyp_001",
    )


class TestGridBinning:
    def test_descriptor_to_cell(self, archive):
        desc = DescriptorVector(0.5, 0.3, 10.0)
        cell = archive.descriptor_to_cell(desc)
        assert len(cell) == 3
        assert all(0 <= c < 5 for c in cell)

    def test_min_descriptor_maps_to_zero(self, archive):
        desc = DescriptorVector(0.25, 0.0, 0.0)
        cell = archive.descriptor_to_cell(desc)
        assert cell == (0, 0, 0)

    def test_max_descriptor_maps_to_last_bin(self, archive):
        desc = DescriptorVector(1.0, 1.0, 24.0)
        cell = archive.descriptor_to_cell(desc)
        assert cell == (4, 4, 4)

    def test_out_of_range_clamps(self, archive):
        desc = DescriptorVector(0.1, -0.5, 30.0)
        cell = archive.descriptor_to_cell(desc)
        assert cell == (0, 0, 4)


class TestInsertion:
    def test_insert_into_empty_cell(self, archive, sample_entry):
        inserted = archive.try_insert(sample_entry)
        assert inserted is True
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert archive.get(cell) == sample_entry

    def test_better_fitness_replaces(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        better = CellEntry(
            fitness=4.0, status="candidate", wandb_run_id="run_def",
            git_commit="def456", descriptors=sample_entry.descriptors,
            constraint_results={"passed": True}, budget_spent=5_000_000,
            hypothesis_id="hyp_002",
        )
        inserted = archive.try_insert(better)
        assert inserted is True
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert archive.get(cell).fitness == 4.0

    def test_worse_fitness_rejected(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        worse = CellEntry(
            fitness=10.0, status="candidate", wandb_run_id="run_ghi",
            git_commit="ghi789", descriptors=sample_entry.descriptors,
            constraint_results={"passed": True}, budget_spent=5_000_000,
            hypothesis_id="hyp_003",
        )
        inserted = archive.try_insert(worse)
        assert inserted is False

    def test_empty_cells_count(self, archive, sample_entry):
        assert archive.n_occupied == 0
        archive.try_insert(sample_entry)
        assert archive.n_occupied == 1
        assert archive.n_empty == 124


class TestPromotion:
    def test_initial_status_is_candidate(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert archive.get(cell).status == "candidate"

    def test_approve_candidate(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        archive.set_status(cell, "approved")
        assert archive.get(cell).status == "approved"

    def test_reject_candidate(self, archive, sample_entry):
        archive.try_insert(sample_entry)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        archive.set_status(cell, "rejected")
        assert archive.get(cell).status == "rejected"


class TestSerialization:
    def test_round_trip(self, archive, sample_entry, tmp_path):
        archive.try_insert(sample_entry)
        path = tmp_path / "archive.json"
        save_archive(archive, path)
        loaded = load_archive(path)
        cell = archive.descriptor_to_cell(sample_entry.descriptors)
        assert loaded.get(cell).fitness == sample_entry.fitness
        assert loaded.get(cell).wandb_run_id == sample_entry.wandb_run_id

    def test_empty_archive_round_trip(self, archive, tmp_path):
        path = tmp_path / "archive.json"
        save_archive(archive, path)
        loaded = load_archive(path)
        assert loaded.n_occupied == 0
