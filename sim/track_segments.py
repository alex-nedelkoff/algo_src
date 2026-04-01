"""Track segment primitives for composable track generation.

Each segment function takes a start position, heading, and RNG, and returns
a SegmentResult with gate positions/headings and exit state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentResult:
    """Result of generating a track segment."""
    gates: list[tuple[np.ndarray, float]]  # (position, heading) pairs
    exit_pos: np.ndarray
    exit_heading: float


def _clip_to_arena(pos: np.ndarray, arena_half_width: float) -> np.ndarray:
    """Clip XY coordinates to arena bounds, leave Z unchanged."""
    clipped = pos.copy()
    clipped[0] = float(np.clip(pos[0], -arena_half_width, arena_half_width))
    clipped[1] = float(np.clip(pos[1], -arena_half_width, arena_half_width))
    return clipped


def _clamp_elevation(z: float, elevation_min: float, elevation_max: float) -> float:
    return float(np.clip(z, elevation_min, elevation_max))


def segment_straight(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """1-2 gates placed along the current heading, 3-6m spacing."""
    n_gates = int(rng.integers(1, 3))  # 1 or 2
    gates: list[tuple[np.ndarray, float]] = []
    pos = start_pos.copy()
    heading = start_heading

    for _ in range(n_gates):
        dist = rng.uniform(3.0, 6.0)
        new_x = pos[0] + dist * math.cos(heading)
        new_y = pos[1] + dist * math.sin(heading)
        new_z = _clamp_elevation(pos[2] + rng.uniform(-0.3, 0.3), elevation_min, elevation_max)
        new_pos = _clip_to_arena(np.array([new_x, new_y, new_z]), arena_half_width)
        gates.append((new_pos, heading))
        pos = new_pos

    return SegmentResult(gates=gates, exit_pos=pos.copy(), exit_heading=heading)
