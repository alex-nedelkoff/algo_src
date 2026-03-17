"""Procedural track generation for diverse training layouts."""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np

from sim.tracks import Track, _yaw_to_quat, build_figure8_track
from sim.types import GateState

log = logging.getLogger(__name__)


class ProceduralTrackGenerator:
    """Generate closed-loop racing tracks with randomized geometry.

    Sequential gate placement: each gate is placed relative to the previous
    one by sampling turn angle, distance, and elevation. The final segment
    must close back to gate 0 within angle/distance constraints.
    """

    def __init__(
        self,
        *,
        n_gates_min: int = 4,
        n_gates_max: int = 12,
        gate_spacing_min: float = 2.0,
        gate_spacing_max: float = 8.0,
        turn_angle_min: float = -120.0,
        turn_angle_max: float = 120.0,
        elevation_min: float = 1.0,
        elevation_max: float = 4.0,
        elevation_delta_max: float = 0.8,
        arena_half_width: float = 20.0,
        closure_max_angle: float = 90.0,
        closure_max_retries: int = 50,
    ) -> None:
        if n_gates_min < 3:
            raise ValueError(f"n_gates_min must be >= 3, got {n_gates_min}")
        if n_gates_min > n_gates_max:
            raise ValueError(f"n_gates_min ({n_gates_min}) > n_gates_max ({n_gates_max})")
        if gate_spacing_min > gate_spacing_max:
            raise ValueError(f"gate_spacing_min ({gate_spacing_min}) > gate_spacing_max ({gate_spacing_max})")
        if elevation_min > elevation_max:
            raise ValueError(f"elevation_min ({elevation_min}) > elevation_max ({elevation_max})")

        self.n_gates_min = n_gates_min
        self.n_gates_max = n_gates_max
        self.gate_spacing_min = gate_spacing_min
        self.gate_spacing_max = gate_spacing_max
        self.turn_angle_min_rad = math.radians(turn_angle_min)
        self.turn_angle_max_rad = math.radians(turn_angle_max)
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.elevation_delta_max = elevation_delta_max
        self.arena_half_width = arena_half_width
        self.closure_max_angle_rad = math.radians(closure_max_angle)
        self.closure_max_retries = closure_max_retries

    def generate(self, rng: np.random.Generator) -> Track:
        for attempt in range(self.closure_max_retries):
            result = self._try_generate(rng)
            if result is not None:
                return result

        warnings.warn(
            f"Procedural track generation failed after {self.closure_max_retries} "
            f"retries, falling back to figure-8 track.",
            stacklevel=2,
        )
        return build_figure8_track()

    def _try_generate(self, rng: np.random.Generator) -> Track | None:
        n_gates = int(rng.integers(self.n_gates_min, self.n_gates_max + 1))

        margin = self.arena_half_width * 0.3
        x0 = rng.uniform(-margin, margin)
        y0 = rng.uniform(-margin, margin)
        z0 = rng.uniform(self.elevation_min, self.elevation_max)
        heading = rng.uniform(-math.pi, math.pi)

        positions = [np.array([x0, y0, z0])]
        headings = [heading]

        # Place gates 1 .. n_gates-2 freely; gate n_gates-1 is constrained to
        # allow closure back to gate 0.
        turn_std = (self.turn_angle_max_rad - self.turn_angle_min_rad) / 4.0

        for step in range(1, n_gates - 1):
            # Bias turn toward origin when running low on remaining gates
            remaining = n_gates - 1 - step
            if remaining <= 2:
                # Nudge heading toward gate 0 to improve closure odds
                to_origin = math.atan2(positions[0][1] - positions[-1][1],
                                       positions[0][0] - positions[-1][0])
                nudge = _wrap_angle(to_origin - heading)
                biased_mean = np.clip(nudge * 0.5,
                                      self.turn_angle_min_rad,
                                      self.turn_angle_max_rad)
                turn = rng.normal(biased_mean, turn_std * 0.5)
            else:
                turn = rng.normal(0.0, turn_std)
            turn = float(np.clip(turn, self.turn_angle_min_rad, self.turn_angle_max_rad))
            heading = heading + turn

            dist = rng.uniform(self.gate_spacing_min, self.gate_spacing_max)

            new_x = positions[-1][0] + dist * math.cos(heading)
            new_y = positions[-1][1] + dist * math.sin(heading)

            new_x = float(np.clip(new_x, -self.arena_half_width, self.arena_half_width))
            new_y = float(np.clip(new_y, -self.arena_half_width, self.arena_half_width))

            # Reject if arena clipping brought this gate too close to the previous one
            xy_dist = math.sqrt((new_x - positions[-1][0])**2 + (new_y - positions[-1][1])**2)
            if xy_dist < self.gate_spacing_min * 0.9:
                return None

            prev_z = positions[-1][2]
            z_lo = max(self.elevation_min, prev_z - self.elevation_delta_max)
            z_hi = min(self.elevation_max, prev_z + self.elevation_delta_max)
            new_z = rng.uniform(z_lo, z_hi)

            new_pos = np.array([new_x, new_y, new_z])
            actual_dist = float(np.linalg.norm(new_pos - positions[-1]))
            if actual_dist < self.gate_spacing_min or actual_dist > self.gate_spacing_max:
                return None

            positions.append(new_pos)
            headings.append(heading)

        # Place the last gate: find a position reachable from gate n-2 that
        # also allows an approach angle to gate 0 within closure_max_angle.
        # Sample candidate last-gate positions by steering toward gate 0.
        last_placed = False
        for _ in range(20):
            to_g0 = math.atan2(positions[0][1] - positions[-1][1],
                               positions[0][0] - positions[-1][0])
            nudge = _wrap_angle(to_g0 - headings[-1])
            biased_mean = float(np.clip(nudge * 0.6,
                                        self.turn_angle_min_rad,
                                        self.turn_angle_max_rad))
            turn = rng.normal(biased_mean, turn_std * 0.4)
            turn = float(np.clip(turn, self.turn_angle_min_rad, self.turn_angle_max_rad))
            candidate_heading = headings[-1] + turn

            dist = rng.uniform(self.gate_spacing_min, self.gate_spacing_max)
            cx = positions[-1][0] + dist * math.cos(candidate_heading)
            cy = positions[-1][1] + dist * math.sin(candidate_heading)
            cx = float(np.clip(cx, -self.arena_half_width, self.arena_half_width))
            cy = float(np.clip(cy, -self.arena_half_width, self.arena_half_width))

            # Skip if arena clipping shrank spacing below minimum
            xy_dist_last = math.sqrt((cx - positions[-1][0])**2 + (cy - positions[-1][1])**2)
            if xy_dist_last < self.gate_spacing_min * 0.9:
                continue

            prev_z = positions[-1][2]
            z_lo = max(self.elevation_min, prev_z - self.elevation_delta_max)
            z_hi = min(self.elevation_max, prev_z + self.elevation_delta_max)
            cz = rng.uniform(z_lo, z_hi)

            candidate = np.array([cx, cy, cz])

            # Check this gate is within 3D spacing bounds from the previous gate
            prev_to_cand = float(np.linalg.norm(candidate - positions[-1]))
            if prev_to_cand < self.gate_spacing_min or prev_to_cand > self.gate_spacing_max:
                continue

            # Check closure from candidate back to gate 0
            close_dist_3d = float(np.linalg.norm(positions[0] - candidate))
            if close_dist_3d > self.gate_spacing_max * 2:
                continue
            if close_dist_3d < self.gate_spacing_min:
                continue

            dx_close = positions[0][0] - cx
            dy_close = positions[0][1] - cy
            close_heading = math.atan2(dy_close, dx_close)
            close_turn = abs(_wrap_angle(close_heading - candidate_heading))
            if close_turn > self.closure_max_angle_rad:
                continue

            if abs(positions[0][2] - cz) > self.elevation_delta_max:
                continue

            # Check min separation from all non-adjacent gates
            ok = True
            for i in range(len(positions) - 1):  # skip gate n-2 (adjacent)
                if i == 0:  # gate 0 will be neighbour via closure
                    continue
                sep = float(np.linalg.norm(candidate - positions[i]))
                if sep < self.gate_spacing_min:
                    ok = False
                    break
            if not ok:
                continue

            positions.append(candidate)
            headings.append(candidate_heading)
            last_placed = True
            break

        if not last_placed:
            return None

        # Final global validation
        close_dist_final = float(np.linalg.norm(positions[0] - positions[-1]))
        if close_dist_final > self.gate_spacing_max * 2:
            return None

        dx = positions[0][0] - positions[-1][0]
        dy = positions[0][1] - positions[-1][1]
        close_heading = math.atan2(dy, dx)
        close_turn = abs(_wrap_angle(close_heading - headings[-1]))
        if close_turn > self.closure_max_angle_rad:
            return None

        if abs(positions[0][2] - positions[-1][2]) > self.elevation_delta_max:
            return None

        # Min separation between non-consecutive gates
        for i in range(len(positions)):
            for j in range(i + 2, len(positions)):
                if i == 0 and j == len(positions) - 1:
                    continue
                sep = float(np.linalg.norm(positions[j] - positions[i]))
                if sep < self.gate_spacing_min:
                    return None

        # Build Track
        gates: list[GateState] = []
        for i, pos in enumerate(positions):
            next_pos = positions[(i + 1) % n_gates]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            yaw = math.atan2(dy, dx)
            gates.append(GateState(position=pos, orientation=_yaw_to_quat(yaw)))

        return Track(gates)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
