"""Serialize/deserialize MAP-Elites archive to/from JSON."""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch.archive.map_elites import MapElitesArchive, CellEntry
from autoresearch.descriptors.compute import DescriptorVector


def save_archive(archive: MapElitesArchive, path: Path | str) -> None:
    path = Path(path)
    data = {
        "bins_per_axis": archive.bins_per_axis,
        "axis_ranges": archive.axis_ranges,
        "cells": {},
    }
    for cell, entry in archive.occupied_cells().items():
        key = f"{cell[0]},{cell[1]},{cell[2]}"
        data["cells"][key] = {
            "fitness": entry.fitness,
            "status": entry.status,
            "wandb_run_id": entry.wandb_run_id,
            "git_commit": entry.git_commit,
            "descriptors": list(entry.descriptors.to_tuple()),
            "constraint_results": entry.constraint_results,
            "budget_spent": entry.budget_spent,
            "hypothesis_id": entry.hypothesis_id,
        }
    data["subdivided"] = [list(c) for c in archive._subdivided]
    data["insertion_counts"] = {
        f"{c[0]},{c[1]},{c[2]}": count
        for c, count in archive._insertion_counts.items()
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_archive(path: Path | str) -> MapElitesArchive:
    path = Path(path)
    data = json.loads(path.read_text())
    archive = MapElitesArchive(
        bins_per_axis=data["bins_per_axis"],
        axis_ranges=[tuple(r) for r in data["axis_ranges"]],
    )
    for key, entry_data in data.get("cells", {}).items():
        desc_vals = entry_data["descriptors"]
        entry = CellEntry(
            fitness=entry_data["fitness"],
            status=entry_data["status"],
            wandb_run_id=entry_data["wandb_run_id"],
            git_commit=entry_data["git_commit"],
            descriptors=DescriptorVector(*desc_vals),
            constraint_results=entry_data["constraint_results"],
            budget_spent=entry_data["budget_spent"],
            hypothesis_id=entry_data["hypothesis_id"],
        )
        archive.try_insert(entry)
    for sub_cell in data.get("subdivided", []):
        archive._subdivided.add(tuple(sub_cell))
    for key, count in data.get("insertion_counts", {}).items():
        cell_indices = tuple(int(x) for x in key.split(","))
        archive._insertion_counts[cell_indices] = count
    return archive
