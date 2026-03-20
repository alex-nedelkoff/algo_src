"""Randomized figure-eight track generator.

A figure-eight consists of two elliptical loops connected at a central crossing
point. The two crossing gates are offset slightly so they do not physically
overlap.

Algorithm
---------
1. Sample n_per_loop (3-5), radius (1.5-4 m), crossing_offset (0.3-0.8 m),
   base_z (1.0-3.5 m).
2. Generate n_per_loop points on an ellipse for loop A (centred at y=+radius)
   and n_per_loop points for loop B (centred at y=-radius), excluding the
   crossing endpoints so they are not duplicated.
3. Stitch: [loop_a gates] → [crossing gate at (-offset, 0)] →
           [loop_b gates] → [crossing gate at (+offset, 0)].
4. Add gentle elevation variation (±0.3 m sinusoidal) within [1.0, 3.5] m.
5. Rotate the whole track by a random yaw and translate to a random position.
6. Compute gate orientations — each gate's local x-axis (forward normal)
   points from that gate toward the next gate (wrap-around).
"""
from __future__ import annotations

import math

import numpy as np

from sim.tracks import Track, _yaw_to_quat
from sim.types import GateState


class Figure8TrackGenerator:
    """Generates randomized figure-eight tracks.

    Each call to :meth:`generate` with a seeded :class:`numpy.random.Generator`
    produces a fully deterministic but randomized figure-eight layout.
    """

    def __init__(
        self,
        *,
        loop_radius_min: float = 1.5,
        loop_radius_max: float = 4.0,
        gates_per_loop_min: int = 3,
        gates_per_loop_max: int = 5,
        crossing_offset_min: float = 0.3,
        crossing_offset_max: float = 0.8,
        elevation_min: float = 1.0,
        elevation_max: float = 3.5,
        elevation_delta_max: float = 0.6,
    ) -> None:
        self._RADIUS_MIN = loop_radius_min
        self._RADIUS_MAX = loop_radius_max
        self._GATES_PER_LOOP_MIN = gates_per_loop_min
        self._GATES_PER_LOOP_MAX = gates_per_loop_max
        self._CROSSING_OFFSET_MIN = crossing_offset_min
        self._CROSSING_OFFSET_MAX = crossing_offset_max
        self._BASE_Z_MIN = elevation_min
        self._BASE_Z_MAX = elevation_max
        self._ELEV_VARIATION = elevation_delta_max * 0.5
        self._TRANSLATE_RANGE = 3.0

    def generate(self, rng: np.random.Generator) -> Track:
        """Generate a randomized figure-eight track.

        Args:
            rng: Seeded NumPy random generator. Identical seeds yield identical
                 tracks; distinct seeds yield distinct tracks.

        Returns:
            A :class:`~sim.tracks.Track` with 6-10 gates in figure-eight order.
        """
        # --- Sample parameters -----------------------------------------------
        radius = rng.uniform(self._RADIUS_MIN, self._RADIUS_MAX)
        n_per_loop = int(rng.integers(self._GATES_PER_LOOP_MIN, self._GATES_PER_LOOP_MAX + 1))
        crossing_offset = rng.uniform(self._CROSSING_OFFSET_MIN, self._CROSSING_OFFSET_MAX)
        base_z = rng.uniform(self._BASE_Z_MIN, self._BASE_Z_MAX)
        global_yaw = rng.uniform(0.0, 2 * math.pi)
        tx = rng.uniform(-self._TRANSLATE_RANGE, self._TRANSLATE_RANGE)
        ty = rng.uniform(-self._TRANSLATE_RANGE, self._TRANSLATE_RANGE)

        # --- Build loop points in canonical (unrotated) frame ----------------
        # Loop A: ellipse centred at (0, +radius), traversed counter-clockwise.
        # We sample n_per_loop interior points (excluding the two crossing
        # endpoints at angle=π and angle=0 so we don't duplicate them).
        loop_a_pts = self._ellipse_loop_points(
            cx=0.0, cy=+radius,
            rx=radius * 0.8, ry=radius * 0.6,
            n=n_per_loop,
            start_angle=math.pi,   # starts at the crossing (left side)
            clockwise=False,
        )

        # Loop B: ellipse centred at (0, -radius), traversed clockwise
        # (so the overall figure-eight flows correctly).
        loop_b_pts = self._ellipse_loop_points(
            cx=0.0, cy=-radius,
            rx=radius * 0.8, ry=radius * 0.6,
            n=n_per_loop,
            start_angle=0.0,       # starts at the crossing (right side)
            clockwise=True,
        )

        # --- Stitch the full sequence ----------------------------------------
        # crossing A → loop_a interior → crossing B → loop_b interior
        crossing_a = np.array([-crossing_offset, 0.0])  # entering loop A
        crossing_b = np.array([+crossing_offset, 0.0])  # entering loop B

        xy_pts = np.vstack([
            crossing_a,       # [0]   crossing gate entering loop A
            loop_a_pts,       # [1..n_per_loop]  loop A interior
            crossing_b,       # [n_per_loop+1]  crossing gate entering loop B
            loop_b_pts,       # [n_per_loop+2..]  loop B interior
        ])  # total: 2 + 2*n_per_loop points, i.e. 6-10 for n_per_loop in 2-4
        # Note: n_per_loop from integers(3,6) gives [3,4,5]; each loop
        # contributes n_per_loop-1 interior points + 1 crossing = n_per_loop
        # but we build n_per_loop interior points; let me recount:
        # xy_pts shape: 1 (cross_a) + n_per_loop (loop_a) + 1 (cross_b) + n_per_loop (loop_b)
        # = 2 + 2*n_per_loop  → range [2+6, 2+10] = [8, 12]  — too many.
        # Reduce interior points to n_per_loop-1 per loop.

        # Rebuild with corrected counts: use n_per_loop-1 interior points each.
        n_interior = n_per_loop - 1  # 2-4 interior points per loop
        loop_a_pts = self._ellipse_loop_points(
            cx=0.0, cy=+radius,
            rx=radius * 0.8, ry=radius * 0.6,
            n=n_interior,
            start_angle=math.pi,
            clockwise=False,
        )
        loop_b_pts = self._ellipse_loop_points(
            cx=0.0, cy=-radius,
            rx=radius * 0.8, ry=radius * 0.6,
            n=n_interior,
            start_angle=0.0,
            clockwise=True,
        )

        xy_pts = np.vstack([
            crossing_a,    # crossing entering loop A
            loop_a_pts,    # n_interior loop A gates
            crossing_b,    # crossing entering loop B
            loop_b_pts,    # n_interior loop B gates
        ])
        # Total: 2 + 2*n_interior = 2 + 2*(n_per_loop-1)
        # n_per_loop in {3,4,5} → n_interior in {2,3,4} → total in {6,8,10}  ✓

        n_gates = len(xy_pts)

        # --- Add elevation with gentle sinusoidal variation ------------------
        elev_variation = self._ELEV_VARIATION
        z_pts = np.array([
            base_z + elev_variation * math.sin(2 * math.pi * k / n_gates)
            for k in range(n_gates)
        ])
        # Clamp to valid range
        z_pts = np.clip(z_pts, self._BASE_Z_MIN, self._BASE_Z_MAX)

        # --- Apply global rotation and translation ---------------------------
        cos_yaw = math.cos(global_yaw)
        sin_yaw = math.sin(global_yaw)
        positions: list[np.ndarray] = []
        for k in range(n_gates):
            x_local, y_local = xy_pts[k]
            x_world = cos_yaw * x_local - sin_yaw * y_local + tx
            y_world = sin_yaw * x_local + cos_yaw * y_local + ty
            positions.append(np.array([x_world, y_world, z_pts[k]]))

        # --- Compute gate orientations ---------------------------------------
        gates: list[GateState] = []
        for i, pos in enumerate(positions):
            next_pos = positions[(i + 1) % n_gates]
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            yaw = math.atan2(dy, dx)
            gates.append(GateState(position=pos, orientation=_yaw_to_quat(yaw)))

        return Track(gates)

    @staticmethod
    def _ellipse_loop_points(
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        n: int,
        start_angle: float,
        clockwise: bool,
    ) -> np.ndarray:
        """Sample n interior points on an ellipse arc.

        The arc spans one full loop (2π) but excludes the start/end crossing
        point itself, yielding n evenly-spaced interior points.

        Args:
            cx, cy: Ellipse centre in the canonical XY frame.
            rx, ry: Semi-axes.
            n: Number of interior points to sample.
            start_angle: Angle (rad) of the crossing point on this ellipse.
            clockwise: If True, traverse the ellipse clockwise (negative angle
                direction); otherwise counter-clockwise.

        Returns:
            Array of shape (n, 2) with XY coordinates.
        """
        direction = -1.0 if clockwise else 1.0
        # n+1 equal divisions; take indices 1..n (excluding 0 = start/crossing)
        angles = [
            start_angle + direction * 2 * math.pi * k / (n + 1)
            for k in range(1, n + 1)
        ]
        pts = np.array([
            [cx + rx * math.cos(a), cy + ry * math.sin(a)]
            for a in angles
        ])
        return pts
