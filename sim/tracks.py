"""Track representation — pure geometry container."""

from __future__ import annotations

from sim.types import GateState


class Track:
    """Ordered sequence of gates forming a racing track.

    Pure geometry container. Gate passage tracking is handled per-env
    by GateRaceEnv._gate_indices.
    """

    def __init__(self, gates: list[GateState]) -> None:
        if not gates:
            raise ValueError("Track must have at least one gate")
        self.gates = gates

    @property
    def num_gates(self) -> int:
        return len(self.gates)
