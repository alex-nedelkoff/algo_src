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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def _arc_gates(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    total_angle: float,
    n_gates: int,
    radius: float,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """Place gates evenly along a circular arc.

    The arc center is offset perpendicular to start_heading by *radius*.
    Positive total_angle → left turn; negative → right turn.
    """
    # Perpendicular direction to heading (left = +90 deg)
    sign = 1.0 if total_angle >= 0 else -1.0
    perp = start_heading + sign * math.pi / 2.0

    # Arc center
    cx = start_pos[0] + radius * math.cos(perp)
    cy = start_pos[1] + radius * math.sin(perp)

    # Angle from center to start position
    angle_to_start = math.atan2(start_pos[1] - cy, start_pos[0] - cx)

    gates: list[tuple[np.ndarray, float]] = []
    step = total_angle / n_gates

    for i in range(1, n_gates + 1):
        arc_angle = angle_to_start + i * step
        gx = cx + radius * math.cos(arc_angle)
        gy = cy + radius * math.sin(arc_angle)
        gz = _clamp_elevation(
            start_pos[2] + rng.uniform(-0.2, 0.2), elevation_min, elevation_max
        )
        gate_pos = _clip_to_arena(np.array([gx, gy, gz]), arena_half_width)

        # Gate heading is tangent to the arc at this point.
        # For CCW (left, sign=+1): tangent = arc_angle + pi/2
        # For CW  (right, sign=-1): tangent = arc_angle - pi/2
        # Both cases: arc_angle + sign * pi/2
        gate_heading = _wrap_angle(arc_angle + sign * math.pi / 2.0)
        gates.append((gate_pos, gate_heading))

    exit_pos = gates[-1][0].copy()
    exit_heading = gates[-1][1]
    return SegmentResult(gates=gates, exit_pos=exit_pos, exit_heading=exit_heading)


# ---------------------------------------------------------------------------
# Segment primitives
# ---------------------------------------------------------------------------

def segment_turn_90(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """1-2 gates along a 90-degree arc (radius 2-4m, random left/right)."""
    n_gates = int(rng.integers(1, 3))
    radius = rng.uniform(2.0, 4.0)
    direction = rng.choice([-1, 1])
    total_angle = direction * math.pi / 2.0
    return _arc_gates(
        start_pos, start_heading, rng, total_angle, n_gates, radius,
        elevation_min, elevation_max, arena_half_width,
    )


def segment_turn_180(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """2-3 gates along a 180-degree hairpin (radius 1.5-3m)."""
    n_gates = int(rng.integers(2, 4))
    radius = rng.uniform(1.5, 3.0)
    direction = rng.choice([-1, 1])
    total_angle = direction * math.pi
    return _arc_gates(
        start_pos, start_heading, rng, total_angle, n_gates, radius,
        elevation_min, elevation_max, arena_half_width,
    )


def segment_slalom(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """3-5 gates with alternating left-right turns (angle 30-60 deg, spacing 1-3m)."""
    n_gates = int(rng.integers(3, 6))
    angle_deg = rng.uniform(30.0, 60.0)
    angle_rad = math.radians(angle_deg)
    spacing = rng.uniform(1.0, 3.0)

    fwd = np.array([math.cos(start_heading), math.sin(start_heading)])
    lat = np.array([-math.sin(start_heading), math.cos(start_heading)])
    lateral_amplitude = spacing * 0.5

    gates: list[tuple[np.ndarray, float]] = []

    for i in range(n_gates):
        # Alternate left (+) and right (-) from the entry heading.
        side = 1.0 if i % 2 == 0 else -1.0
        gate_heading = _wrap_angle(start_heading + side * angle_rad)

        # Place gate along baseline with consistent lateral offset from center line
        forward_dist = spacing * (i + 1)
        center_xy = start_pos[:2] + forward_dist * fwd
        gate_xy = center_xy + side * lateral_amplitude * lat

        gz = _clamp_elevation(start_pos[2] + rng.uniform(-0.2, 0.2), elevation_min, elevation_max)
        gate_pos = _clip_to_arena(np.array([gate_xy[0], gate_xy[1], gz]), arena_half_width)
        gates.append((gate_pos, gate_heading))

    pos = gates[-1][0].copy()
    exit_heading = gates[-1][1]
    return SegmentResult(gates=gates, exit_pos=pos, exit_heading=exit_heading)


def segment_chicane(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """2-3 gates forming a tight S-curve (exit heading ≈ entry heading)."""
    n_gates = int(rng.integers(2, 4))
    angle_deg = rng.uniform(20.0, 45.0)
    angle_rad = math.radians(angle_deg)
    spacing = rng.uniform(1.5, 3.0)

    # First half turns one way, second half turns back
    first_side = rng.choice([-1.0, 1.0])
    gates: list[tuple[np.ndarray, float]] = []
    pos = start_pos.copy()
    heading = start_heading

    half = n_gates // 2
    for i in range(n_gates):
        if i < half:
            side = first_side
        else:
            side = -first_side

        gx = pos[0] + spacing * math.cos(heading)
        gy = pos[1] + spacing * math.sin(heading)
        gz = _clamp_elevation(pos[2] + rng.uniform(-0.15, 0.15), elevation_min, elevation_max)
        gate_pos = _clip_to_arena(np.array([gx, gy, gz]), arena_half_width)

        gate_heading = _wrap_angle(heading + side * angle_rad)
        gates.append((gate_pos, gate_heading))
        pos = gate_pos
        heading = gate_heading

    return SegmentResult(gates=gates, exit_pos=pos.copy(), exit_heading=heading)


def segment_climb(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """1-2 gates with elevation gain 1-2m."""
    n_gates = int(rng.integers(1, 3))
    gates: list[tuple[np.ndarray, float]] = []
    pos = start_pos.copy()
    heading = start_heading
    total_gain = rng.uniform(1.0, 2.0)
    gain_per_gate = total_gain / n_gates

    for _ in range(n_gates):
        dist = rng.uniform(3.0, 6.0)
        gx = pos[0] + dist * math.cos(heading)
        gy = pos[1] + dist * math.sin(heading)
        gz = _clamp_elevation(pos[2] + gain_per_gate, elevation_min, elevation_max)
        gate_pos = _clip_to_arena(np.array([gx, gy, gz]), arena_half_width)
        gates.append((gate_pos, heading))
        pos = gate_pos

    return SegmentResult(gates=gates, exit_pos=pos.copy(), exit_heading=heading)


def segment_dive(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """1-2 gates with elevation loss 1-2m."""
    n_gates = int(rng.integers(1, 3))
    gates: list[tuple[np.ndarray, float]] = []
    pos = start_pos.copy()
    heading = start_heading
    total_loss = rng.uniform(1.0, 2.0)
    loss_per_gate = total_loss / n_gates

    for _ in range(n_gates):
        dist = rng.uniform(3.0, 6.0)
        gx = pos[0] + dist * math.cos(heading)
        gy = pos[1] + dist * math.sin(heading)
        gz = _clamp_elevation(pos[2] - loss_per_gate, elevation_min, elevation_max)
        gate_pos = _clip_to_arena(np.array([gx, gy, gz]), arena_half_width)
        gates.append((gate_pos, heading))
        pos = gate_pos

    return SegmentResult(gates=gates, exit_pos=pos.copy(), exit_heading=heading)


def segment_orbit(
    start_pos: np.ndarray,
    start_heading: float,
    rng: np.random.Generator,
    elevation_min: float,
    elevation_max: float,
    arena_half_width: float,
) -> SegmentResult:
    """3-5 gates along a sustained 180-270 degree arc (radius 2-4m)."""
    n_gates = int(rng.integers(3, 6))
    radius = rng.uniform(2.0, 4.0)
    direction = rng.choice([-1, 1])
    angle_deg = rng.uniform(180.0, 270.0)
    total_angle = direction * math.radians(angle_deg)
    return _arc_gates(
        start_pos, start_heading, rng, total_angle, n_gates, radius,
        elevation_min, elevation_max, arena_half_width,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SEGMENT_REGISTRY: dict[str, object] = {
    "straight": segment_straight,
    "turn_90": segment_turn_90,
    "turn_180": segment_turn_180,
    "slalom": segment_slalom,
    "chicane": segment_chicane,
    "climb": segment_climb,
    "dive": segment_dive,
    "orbit": segment_orbit,
}
