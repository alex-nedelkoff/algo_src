"""Zigzag track generator — open-ended segments with alternating turns.

Produces linear segments where turn direction alternates each gate,
creating a slalom pattern. Segments are chainable — the env spawns a
fresh segment when the drone completes the current one.
"""
from __future__ import annotations

import math

import numpy as np

from sim.tracks import Track, _yaw_to_quat
from sim.types import GateState


class ZigzagTrackGenerator:
    """Generate open-ended zigzag track segments.

    Gates are placed sequentially with forced alternating left/right turns.
    The resulting Track has ``chainable=True`` and stores exit state for
    the env to spawn the next segment.
    """

    def __init__(
        self,
        *,
        n_gates_min: int = 5,
        n_gates_max: int = 8,
        turn_angle_min: float = 100.0,
        turn_angle_max: float = 160.0,
        gate_spacing_min: float = 1.0,
        gate_spacing_max: float = 4.0,
        elevation_min: float = 0.5,
        elevation_max: float = 4.0,
        elevation_delta_max: float = 1.5,
    ) -> None:
        self.n_gates_min = n_gates_min
        self.n_gates_max = n_gates_max
        self.turn_angle_min_rad = math.radians(turn_angle_min)
        self.turn_angle_max_rad = math.radians(turn_angle_max)
        self.gate_spacing_min = gate_spacing_min
        self.gate_spacing_max = gate_spacing_max
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.elevation_delta_max = elevation_delta_max

    def generate(
        self,
        rng: np.random.Generator,
        start_pos: np.ndarray | None = None,
        start_heading: float | None = None,
    ) -> Track:
        n_gates = int(rng.integers(self.n_gates_min, self.n_gates_max + 1))

        if start_pos is None:
            start_pos = np.array([
                rng.uniform(-3.0, 3.0),
                rng.uniform(-3.0, 3.0),
                rng.uniform(self.elevation_min, self.elevation_max),
            ])
        if start_heading is None:
            start_heading = rng.uniform(-math.pi, math.pi)

        positions = []
        heading = start_heading
        pos = np.array(start_pos, dtype=np.float64)
        turn_sign = 1.0 if rng.random() < 0.5 else -1.0

        for i in range(n_gates):
            dist = rng.uniform(self.gate_spacing_min, self.gate_spacing_max)
            new_x = pos[0] + dist * math.cos(heading)
            new_y = pos[1] + dist * math.sin(heading)

            prev_z = pos[2] if positions else start_pos[2]
            z_lo = max(self.elevation_min, prev_z - self.elevation_delta_max)
            z_hi = min(self.elevation_max, prev_z + self.elevation_delta_max)
            new_z = rng.uniform(z_lo, z_hi)

            gate_pos = np.array([new_x, new_y, new_z])
            positions.append(gate_pos)
            pos = gate_pos

            turn_mag = rng.uniform(self.turn_angle_min_rad, self.turn_angle_max_rad)
            heading = heading + turn_sign * turn_mag
            turn_sign *= -1.0

        exit_dist = rng.uniform(self.gate_spacing_min, self.gate_spacing_max)
        exit_pos = np.array([
            positions[-1][0] + exit_dist * math.cos(heading),
            positions[-1][1] + exit_dist * math.sin(heading),
            positions[-1][2],
        ])

        gates: list[GateState] = []
        for i, gpos in enumerate(positions):
            if i < len(positions) - 1:
                next_pos = positions[i + 1]
            else:
                next_pos = exit_pos
            dx = next_pos[0] - gpos[0]
            dy = next_pos[1] - gpos[1]
            yaw = math.atan2(dy, dx)
            gates.append(GateState(position=gpos, orientation=_yaw_to_quat(yaw)))

        return Track(
            gates,
            chainable=True,
            exit_pos=exit_pos,
            exit_heading=heading,
            generator=self,
        )
