"""Track representation and lap detection logic."""

from __future__ import annotations

import numpy as np

from core.types import GateState, QuadState


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

    def check_gate_passage(
        self,
        state: QuadState,
        passage_radius: float = 1.0,
    ) -> bool:
        """Check if the drone has passed through the current gate.

        Uses a simple distance-based check: the drone is considered to have
        passed through the gate if it is within passage_radius of the gate center.

        Args:
            state: Current quadrotor state.
            passage_radius: Distance threshold for gate passage detection.

        Returns:
            True if the gate was passed on this check.
        """
        gate = self.current_gate
        distance = float(np.linalg.norm(state.pos - gate.position))

        if distance <= passage_radius:
            self.gates[self._current_gate_idx].passed = True
            self._current_gate_idx += 1

            # Check for lap completion
            if self._current_gate_idx >= self.num_gates:
                self._laps_completed += 1
                self._current_gate_idx = 0
                # Reset passed flags for next lap
                for g in self.gates:
                    g.passed = False

            return True

        return False

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
