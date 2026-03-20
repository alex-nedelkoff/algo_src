"""Adaptive cell subdivision for MAP-Elites archive."""

from __future__ import annotations
from dataclasses import dataclass
from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


@dataclass
class SubdividedArchive:
    parent_cell: tuple[int, int, int]
    sub_grid: dict[tuple[int, int, int], CellEntry]
    axis_ranges: list[tuple[float, float]]

    @property
    def n_cells(self) -> int:
        return 8

    @property
    def n_occupied(self) -> int:
        return len(self.sub_grid)


def should_subdivide(
    archive: MapElitesArchive,
    cell: tuple[int, int, int],
    min_insertions: int = 3,
) -> bool:
    if archive.is_subdivided(cell):
        return False
    if archive.get(cell) is None:
        return False
    return archive.get_insertion_count(cell) >= min_insertions


def _cell_ranges(
    archive: MapElitesArchive,
    cell: tuple[int, int, int],
) -> list[tuple[float, float]]:
    ranges = []
    for axis_idx, (lo, hi) in enumerate(archive.axis_ranges):
        bin_width = (hi - lo) / archive.bins_per_axis
        cell_lo = lo + cell[axis_idx] * bin_width
        cell_hi = cell_lo + bin_width
        ranges.append((cell_lo, cell_hi))
    return ranges


def subdivide_cell(
    archive: MapElitesArchive,
    cell: tuple[int, int, int],
) -> SubdividedArchive:
    cell_ranges = _cell_ranges(archive, cell)
    sub_grid: dict[tuple[int, int, int], CellEntry] = {}

    existing = archive.get(cell)
    if existing is not None:
        desc = existing.descriptors.to_tuple()
        sub_cell = []
        for val, (lo, hi) in zip(desc, cell_ranges):
            mid = (lo + hi) / 2
            sub_cell.append(0 if val < mid else 1)
        sub_grid[tuple(sub_cell)] = existing

    return SubdividedArchive(
        parent_cell=cell, sub_grid=sub_grid, axis_ranges=cell_ranges,
    )
