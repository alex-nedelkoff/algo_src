"""Composable track generator — chains segment primitives into closed-loop tracks."""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np

from sim.track_segments import SEGMENT_REGISTRY, _wrap_angle
from sim.tracks import Track, _yaw_to_quat, build_figure8_track
from sim.types import GateState

log = logging.getLogger(__name__)


@dataclass
class PerturbationConfig:
    """Post-generation noise config. Disabled by default."""
    enabled: bool = False
    position_noise: float = 0.0
    heading_noise: float = 0.0
    gate_radius_jitter: float = 0.0
    drop_gate_prob: float = 0.0


class ComposedTrackGenerator:
    """Chain 2-5 segment primitives into closed-loop racing tracks."""

    def __init__(
        self,
        segment_weights: dict[str, float],
        n_segments_min: int = 2,
        n_segments_max: int = 5,
        n_gates_max: int = 14,
        arena_half_width: float = 10.0,
        elevation_min: float = 0.5,
        elevation_max: float = 4.0,
        perturbation: PerturbationConfig | None = None,
        closure_max_retries: int = 30,
    ):
        self.n_segments_min = n_segments_min
        self.n_segments_max = n_segments_max
        self.n_gates_max = n_gates_max
        self.arena_half_width = arena_half_width
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.perturbation = perturbation
        self.closure_max_retries = closure_max_retries

        # Build weighted segment selection
        names = list(segment_weights.keys())
        weights = np.array([segment_weights[n] for n in names], dtype=np.float64)
        weights /= weights.sum()
        self._segment_names = names
        self._segment_weights = weights

    def generate(self, rng: np.random.Generator) -> Track:
        for _ in range(self.closure_max_retries):
            result = self._try_generate(rng)
            if result is not None:
                return result

        warnings.warn(
            f"ComposedTrackGenerator failed after {self.closure_max_retries} retries, "
            f"falling back to figure-8.",
            stacklevel=2,
        )
        return build_figure8_track()

    def _try_generate(self, rng: np.random.Generator) -> Track | None:
        n_segments = int(rng.integers(self.n_segments_min, self.n_segments_max + 1))

        # Random start position
        margin = self.arena_half_width * 0.3
        start_pos = np.array([
            rng.uniform(-margin, margin),
            rng.uniform(-margin, margin),
            rng.uniform(self.elevation_min, self.elevation_max),
        ])
        start_heading = rng.uniform(-math.pi, math.pi)

        all_gates: list[tuple[np.ndarray, float]] = []
        pos = start_pos.copy()
        heading = start_heading

        for _ in range(n_segments):
            if len(all_gates) >= self.n_gates_max:
                break

            # Pick a segment type
            seg_name = rng.choice(self._segment_names, p=self._segment_weights)
            seg_fn = SEGMENT_REGISTRY[seg_name]

            result = seg_fn(
                start_pos=pos,
                start_heading=heading,
                rng=rng,
                elevation_min=self.elevation_min,
                elevation_max=self.elevation_max,
                arena_half_width=self.arena_half_width,
            )

            # Add gates up to the max
            remaining = self.n_gates_max - len(all_gates)
            for gate in result.gates[:remaining]:
                all_gates.append(gate)

            pos = result.exit_pos
            heading = result.exit_heading

        if len(all_gates) < 2:
            return None

        # Attempt closure: check if last gate can reach first gate
        first_pos = all_gates[0][0]
        last_pos = all_gates[-1][0]
        close_dist = float(np.linalg.norm(first_pos - last_pos))

        if close_dist < 1.0 or close_dist > 12.0:
            return None

        # Check closure angle
        dx = first_pos[0] - last_pos[0]
        dy = first_pos[1] - last_pos[1]
        close_heading = math.atan2(dy, dx)
        close_turn = abs(_wrap_angle(close_heading - heading))
        if close_turn > math.radians(120):
            return None

        # Check min separation between non-consecutive gates
        for i in range(len(all_gates)):
            for j in range(i + 2, len(all_gates)):
                if i == 0 and j == len(all_gates) - 1:
                    continue
                sep = float(np.linalg.norm(all_gates[j][0] - all_gates[i][0]))
                if sep < 0.8:
                    return None

        # Apply perturbation (no-op when disabled)
        if self.perturbation and self.perturbation.enabled:
            all_gates = self._apply_perturbation(all_gates, rng)

        # Build Track — orient each gate toward the next gate
        gate_states: list[GateState] = []
        for i, (gate_pos, gate_heading) in enumerate(all_gates):
            next_pos = all_gates[(i + 1) % len(all_gates)][0]
            dx = next_pos[0] - gate_pos[0]
            dy = next_pos[1] - gate_pos[1]
            facing = math.atan2(dy, dx)
            gate_states.append(GateState(position=gate_pos, orientation=_yaw_to_quat(facing)))

        return Track(gate_states)

    def _apply_perturbation(
        self,
        gates: list[tuple[np.ndarray, float]],
        rng: np.random.Generator,
    ) -> list[tuple[np.ndarray, float]]:
        """Apply post-generation noise. Stub for future use."""
        return gates
