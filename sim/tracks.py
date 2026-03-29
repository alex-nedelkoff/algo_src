"""Track representation — pure geometry container."""

from __future__ import annotations

import math

import numpy as np

from sim.types import GateState


class Track:
    """Ordered sequence of gates forming a racing track.

    Pure geometry container. Gate passage tracking is handled per-env
    by GateRaceEnv._gate_indices.

    For chainable tracks (e.g. zigzag), the track stores exit state
    and a back-reference to the generator for spawning the next segment.
    """

    def __init__(
        self,
        gates: list[GateState],
        *,
        chainable: bool = False,
        exit_pos: np.ndarray | None = None,
        exit_heading: float | None = None,
        generator: object | None = None,
    ) -> None:
        if not gates:
            raise ValueError("Track must have at least one gate")
        self.gates = gates
        self.chainable = chainable
        self.exit_pos = exit_pos
        self.exit_heading = exit_heading
        self.generator = generator

    @property
    def num_gates(self) -> int:
        return len(self.gates)


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw angle to quaternion [w, x, y, z].

    Encodes a pure rotation about the Z axis so the gate's local x-axis
    (forward normal) points in the direction given by ``yaw``.

    Args:
        yaw: Yaw angle in radians.

    Returns:
        Quaternion (4,) as [w, x, y, z].
    """
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def build_figure8_track() -> Track:
    """Build a fixed 8-gate figure-8 track for MonoRace M23 training.

    Geometry: two offset loops sharing a slightly offset crossing point near
    the origin.  All gates are at z=2.0 m and within ±4 m in x/y so the
    layout fits inside the 5 m × 5 m MonoRace arena with margin for start
    position perturbations.

    Gate orientations are yaw-only quaternions computed so that each gate's
    forward normal (local x-axis after rotation) points from the current gate
    toward the next gate, wrapping from gate 7 back to gate 0.

    Returns:
        Track with 8 GateState objects in figure-8 order.
    """
    Z = 2.0

    # Gate positions: two loops with slightly offset crossing gates (3 & 7)
    # so they do not occupy the same physical location.
    positions = [
        # Right loop (counter-clockwise viewed from above)
        np.array([ 2.0,  1.5, Z]),   # 0
        np.array([ 0.0,  3.0, Z]),   # 1
        np.array([-2.0,  1.5, Z]),   # 2
        # Center crossing — approaching from the left loop
        np.array([-0.3,  0.0, Z]),   # 3
        # Left loop (clockwise viewed from above)
        np.array([-2.0, -1.5, Z]),   # 4
        np.array([ 0.0, -3.0, Z]),   # 5
        np.array([ 2.0, -1.5, Z]),   # 6
        # Center crossing back — approaching from the right loop
        np.array([ 0.3,  0.0, Z]),   # 7
    ]

    n = len(positions)
    gates: list[GateState] = []
    for i, pos in enumerate(positions):
        next_pos = positions[(i + 1) % n]
        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]
        yaw = math.atan2(dy, dx)
        gates.append(GateState(position=pos, orientation=_yaw_to_quat(yaw)))

    return Track(gates)
