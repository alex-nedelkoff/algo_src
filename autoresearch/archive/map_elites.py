"""MAP-Elites archive with fixed grid and promotion states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from autoresearch.descriptors.compute import DescriptorVector


@dataclass
class CellEntry:
    """An entry in a MAP-Elites archive cell."""

    fitness: float
    status: Literal["candidate", "approved", "rejected"]
    wandb_run_id: str
    git_commit: str
    descriptors: DescriptorVector
    constraint_results: dict
    budget_spent: int
    hypothesis_id: str


class MapElitesArchive:
    """Fixed-grid MAP-Elites archive."""

    def __init__(
        self,
        bins_per_axis: int = 5,
        axis_ranges: list[tuple[float, float]] | None = None,
    ) -> None:
        self.bins_per_axis = bins_per_axis
        self.axis_ranges = axis_ranges or [
            (0.25, 1.0), (0.0, 1.0), (0.0, 24.0),
        ]
        self._grid: dict[tuple[int, int, int], CellEntry] = {}
        self._subdivided: set[tuple[int, int, int]] = set()
        self._insertion_counts: dict[tuple[int, int, int], int] = {}

    @property
    def n_cells(self) -> int:
        return self.bins_per_axis ** 3

    @property
    def n_occupied(self) -> int:
        return len(self._grid)

    @property
    def n_empty(self) -> int:
        return self.n_cells - self.n_occupied

    def descriptor_to_cell(self, desc: DescriptorVector) -> tuple[int, int, int]:
        values = desc.to_tuple()
        indices = []
        for val, (lo, hi) in zip(values, self.axis_ranges):
            clamped = max(lo, min(hi, val))
            normalized = (clamped - lo) / (hi - lo)
            idx = min(int(normalized * self.bins_per_axis), self.bins_per_axis - 1)
            indices.append(idx)
        return (indices[0], indices[1], indices[2])

    def get(self, cell: tuple[int, int, int]) -> CellEntry | None:
        return self._grid.get(cell)

    def try_insert(self, entry: CellEntry) -> bool:
        cell = self.descriptor_to_cell(entry.descriptors)
        # Track insertion attempts
        self._insertion_counts[cell] = self._insertion_counts.get(cell, 0) + 1
        existing = self._grid.get(cell)
        if existing is None or entry.fitness < existing.fitness:
            self._grid[cell] = entry
            return True
        return False

    def set_status(self, cell: tuple[int, int, int], status: Literal["candidate", "approved", "rejected"]) -> None:
        entry = self._grid.get(cell)
        if entry is None:
            raise KeyError(f"Cell {cell} is empty")
        entry.status = status

    def mark_subdivided(self, cell: tuple[int, int, int]) -> None:
        self._subdivided.add(cell)

    def is_subdivided(self, cell: tuple[int, int, int]) -> bool:
        return cell in self._subdivided

    def get_insertion_count(self, cell: tuple[int, int, int]) -> int:
        return self._insertion_counts.get(cell, 0)

    def occupied_cells(self) -> dict[tuple[int, int, int], CellEntry]:
        return dict(self._grid)

    def approved_cells(self) -> dict[tuple[int, int, int], CellEntry]:
        return {k: v for k, v in self._grid.items() if v.status == "approved"}
