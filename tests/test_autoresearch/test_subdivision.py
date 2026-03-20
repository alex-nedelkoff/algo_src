"""Tests for adaptive MAP-Elites cell subdivision."""

import pytest
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.archive.subdivision import should_subdivide, subdivide_cell, SubdividedArchive
from autoresearch.archive.serialization import save_archive, load_archive
from autoresearch.descriptors.compute import DescriptorVector


def test_insertion_counting():
    archive = MapElitesArchive()
    desc = DescriptorVector(0.5, 0.3, 10.0)
    for i in range(4):
        archive.try_insert(CellEntry(
            fitness=5.0 - i * 0.3, status="approved", wandb_run_id=f"run_{i}",
            git_commit=f"c{i}", descriptors=desc, constraint_results={},
            budget_spent=5_000_000, hypothesis_id=f"h{i}",
        ))
    cell = archive.descriptor_to_cell(desc)
    assert archive.get_insertion_count(cell) == 4


def test_should_subdivide_with_enough_insertions():
    archive = MapElitesArchive()
    desc = DescriptorVector(0.5, 0.3, 10.0)
    for i in range(3):
        archive.try_insert(CellEntry(
            fitness=5.0 - i, status="approved", wandb_run_id=f"r{i}",
            git_commit=f"c{i}", descriptors=desc, constraint_results={},
            budget_spent=5_000_000, hypothesis_id=f"h{i}",
        ))
    cell = archive.descriptor_to_cell(desc)
    assert should_subdivide(archive, cell, min_insertions=3) is True


def test_empty_cell_no_subdivide():
    archive = MapElitesArchive()
    assert should_subdivide(archive, (2, 2, 2), min_insertions=3) is False


def test_already_subdivided_no_repeat():
    archive = MapElitesArchive()
    desc = DescriptorVector(0.5, 0.3, 10.0)
    for i in range(3):
        archive.try_insert(CellEntry(
            fitness=5.0 - i, status="approved", wandb_run_id=f"r{i}",
            git_commit=f"c{i}", descriptors=desc, constraint_results={},
            budget_spent=5_000_000, hypothesis_id=f"h{i}",
        ))
    cell = archive.descriptor_to_cell(desc)
    archive.mark_subdivided(cell)
    assert should_subdivide(archive, cell, min_insertions=3) is False


def test_subdivide_creates_subcells():
    archive = MapElitesArchive()
    entry = CellEntry(
        fitness=4.0, status="approved", wandb_run_id="r0", git_commit="abc",
        descriptors=DescriptorVector(0.5, 0.3, 10.0), constraint_results={},
        budget_spent=5_000_000, hypothesis_id="h0",
    )
    archive.try_insert(entry)
    cell = archive.descriptor_to_cell(entry.descriptors)
    sub = subdivide_cell(archive, cell)
    assert isinstance(sub, SubdividedArchive)
    assert sub.n_cells == 8
    assert sub.n_occupied >= 1


def test_subdivision_serialization_round_trip(tmp_path):
    archive = MapElitesArchive()
    desc = DescriptorVector(0.5, 0.3, 10.0)
    for i in range(3):
        archive.try_insert(CellEntry(
            fitness=5.0 - i, status="approved", wandb_run_id=f"r{i}",
            git_commit=f"c{i}", descriptors=desc, constraint_results={},
            budget_spent=5_000_000, hypothesis_id=f"h{i}",
        ))
    cell = archive.descriptor_to_cell(desc)
    archive.mark_subdivided(cell)

    path = tmp_path / "archive.json"
    save_archive(archive, path)
    loaded = load_archive(path)
    assert loaded.is_subdivided(cell) is True
    assert loaded.get_insertion_count(cell) == 3
