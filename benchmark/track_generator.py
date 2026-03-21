"""Golden set track generation and serialization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from sim.tracks import Track
from sim.types import GateState

log = logging.getLogger(__name__)


def serialize_track(
    track: Track,
    output_path: Path,
    arena_bounds: float,
    gate_passage_radius: float,
    max_steps: int,
) -> None:
    """Serialize a Track to .npz format."""
    positions = np.array([g.position for g in track.gates])
    orientations = np.array([g.orientation for g in track.gates])
    np.savez(
        output_path,
        positions=positions,
        orientations=orientations,
        arena_bounds=np.array(arena_bounds),
        gate_passage_radius=np.array(gate_passage_radius),
        max_steps=np.array(max_steps),
    )


def deserialize_track(npz_path: Path) -> dict[str, Any]:
    """Load a Track from .npz format. Returns dict with 'track' and metadata."""
    data = np.load(npz_path)
    gates = [
        GateState(position=pos, orientation=ori)
        for pos, ori in zip(data["positions"], data["orientations"])
    ]
    return {
        "track": Track(gates),
        "arena_bounds": float(data["arena_bounds"]),
        "gate_passage_radius": float(data["gate_passage_radius"]),
        "max_steps": int(data["max_steps"]),
    }


def generate_procedural_track(
    n_gates: int,
    seed: int,
    spread: float = 3.0,
    elevation_range: tuple[float, float] = (1.5, 3.0),
    shape: str = "loop",
    position_noise: float = 0.3,
    radius_variation: float = 0.0,
    vertical_stacking: float = 0.0,
) -> Track:
    """Generate a track procedurally from parameters and a seed.

    Args:
        n_gates: Number of gates.
        seed: Random seed for reproducibility.
        spread: Base radius of the track layout.
        elevation_range: (min_z, max_z) for gate heights.
        shape: Track shape — "loop" (circle), "figure8" (crossing loops),
               "zigzag" (alternating sides), "spiral" (ascending spiral).
        position_noise: Fraction of spread for random position perturbation.
        radius_variation: Fraction of spread for per-gate radius variation (0 = constant).
        vertical_stacking: How much elevation changes between consecutive gates
                           (0 = random within range, >0 = progressive climb/descent).
    """
    rng = np.random.default_rng(seed)
    positions = []

    if shape == "figure8":
        # Two loops crossing at center
        for i in range(n_gates):
            t = i / n_gates * 2 * np.pi
            r = spread * (1.0 + radius_variation * rng.uniform(-1, 1))
            x = r * np.sin(t)
            y = r * np.sin(2 * t) / 2  # lemniscate-like
            x += rng.normal(0, spread * position_noise)
            y += rng.normal(0, spread * position_noise)
            z = rng.uniform(*elevation_range)
            positions.append(np.array([x, y, z]))

    elif shape == "zigzag":
        # Alternating left-right with forward progression
        for i in range(n_gates):
            progress = i / n_gates
            x = (progress - 0.5) * spread * 2
            side = 1.0 if i % 2 == 0 else -1.0
            y = side * spread * (0.5 + rng.uniform(0, 0.5))
            x += rng.normal(0, spread * position_noise * 0.5)
            y += rng.normal(0, spread * position_noise * 0.5)
            z = rng.uniform(*elevation_range)
            positions.append(np.array([x, y, z]))

    elif shape == "spiral":
        # Ascending or descending spiral
        z_lo, z_hi = elevation_range
        for i in range(n_gates):
            t = i / n_gates * 2 * np.pi * 1.5  # 1.5 turns
            r = spread * (1.0 + radius_variation * rng.uniform(-1, 1))
            x = r * np.cos(t) + rng.normal(0, spread * position_noise)
            y = r * np.sin(t) + rng.normal(0, spread * position_noise)
            # Progressive elevation
            z = z_lo + (z_hi - z_lo) * (i / max(n_gates - 1, 1))
            z += rng.normal(0, 0.2)  # small noise
            positions.append(np.array([x, y, z]))

    else:  # "loop" (default)
        angles = np.linspace(0, 2 * np.pi, n_gates, endpoint=False)
        for i, angle in enumerate(angles):
            r = spread * (1.0 + radius_variation * rng.uniform(-1, 1))
            x = r * np.cos(angle) + rng.normal(0, spread * position_noise)
            y = r * np.sin(angle) + rng.normal(0, spread * position_noise)
            if vertical_stacking > 0:
                # Oscillating height
                z_lo, z_hi = elevation_range
                z_mid = (z_lo + z_hi) / 2
                z_amp = (z_hi - z_lo) / 2
                z = z_mid + z_amp * np.sin(2 * np.pi * i / n_gates * vertical_stacking)
                z += rng.normal(0, 0.15)
            else:
                z = rng.uniform(*elevation_range)
            positions.append(np.array([x, y, z]))

    # Compute orientations: each gate faces toward the next gate
    gates = []
    for i, pos in enumerate(positions):
        next_pos = positions[(i + 1) % n_gates]
        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]
        yaw = np.arctan2(dy, dx)
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        gates.append(GateState(position=pos, orientation=quat))

    return Track(gates)


def build_track_from_layout(layout_path: Path) -> tuple[Track, dict[str, Any]]:
    """Build a Track from a layout YAML file. Returns (Track, metadata dict)."""
    cfg = OmegaConf.load(layout_path)
    meta = {
        "arena_bounds": float(cfg.arena_bounds),
        "gate_passage_radius": float(cfg.gate_passage_radius),
        "max_steps": int(cfg.max_steps),
    }

    if cfg.track_type == "hand_designed":
        gates = [
            GateState(
                position=np.array(g.position, dtype=np.float64),
                orientation=np.array(g.orientation, dtype=np.float64),
            )
            for g in cfg.gates
        ]
        return Track(gates), meta

    elif cfg.track_type == "procedural":
        gp = cfg.generator_params
        track = generate_procedural_track(
            n_gates=int(gp.n_gates),
            seed=int(cfg.seed),
            spread=float(gp.get("spread", 3.0)),
            elevation_range=tuple(gp.get("elevation_range", [1.5, 3.0])),
            shape=str(gp.get("shape", "loop")),
            position_noise=float(gp.get("position_noise", 0.3)),
            radius_variation=float(gp.get("radius_variation", 0.0)),
            vertical_stacking=float(gp.get("vertical_stacking", 0.0)),
        )
        return track, meta

    else:
        raise ValueError(f"Unknown track_type: {cfg.track_type}")
