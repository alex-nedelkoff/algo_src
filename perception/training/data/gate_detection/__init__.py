"""Gate detection training data format, I/O, and mesh utilities."""

from __future__ import annotations

from .format import (
    CORNER_NAMES,
    generate_corner_heatmaps,
    load_sample,
    render_gate_mask,
    save_sample,
)
from .gate_mesh import (
    generate_drone_mesh,
    generate_gate_mesh,
    get_gate_inner_corners,
)

__all__ = [
    "CORNER_NAMES",
    "generate_corner_heatmaps",
    "generate_drone_mesh",
    "generate_gate_mesh",
    "get_gate_inner_corners",
    "load_sample",
    "render_gate_mask",
    "save_sample",
]
