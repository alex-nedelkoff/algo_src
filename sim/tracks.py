"""Track representation and lap detection logic."""

from __future__ import annotations

import numpy as np

from sim.types import GateState, QuadState


class Track:
    """Ordered sequence of gates forming a racing track.

    Provides gate iteration and lap detection logic.
    """

    def __init__(self, gates: list[GateState]) -> None:
        """Initialize a track with an ordered list of gates.

        Args:
            gates: Ordered list of GateState defining the track layout.

        Raises:
            ValueError: If gates list is empty.
        """
        if not gates:
            raise ValueError("Track must have at least one gate")
        self.gates = gates
        self._current_gate_idx = 0
        self._laps_completed = 0

    @property
    def num_gates(self) -> int:
        """Number of gates in the track."""
        return len(self.gates)

    @property
    def current_gate_idx(self) -> int:
        """Index of the next gate to pass through."""
        return self._current_gate_idx

    @property
    def current_gate(self) -> GateState:
        """The next gate the drone should pass through."""
        return self.gates[self._current_gate_idx]

    @property
    def laps_completed(self) -> int:
        """Number of complete laps finished."""
        return self._laps_completed

    def reset(self) -> None:
        """Reset track state for a new episode."""
        self._current_gate_idx = 0
        self._laps_completed = 0
        for gate in self.gates:
            gate.passed = False

    def distance_to_current_gate(self, state: QuadState) -> float:
        """Compute distance from the drone to the current target gate.

        Args:
            state: Current quadrotor state.

        Returns:
            Euclidean distance to the current gate center.
        """
        return float(np.linalg.norm(state.pos - self.current_gate.position))
