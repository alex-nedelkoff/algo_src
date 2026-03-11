"""Tests for Track as pure geometry container."""
import numpy as np
import pytest
from sim.tracks import Track
from sim.types import GateState


class TestTrackGeometry:
    def test_track_stores_gates(self):
        gates = [
            GateState(position=np.array([5.0, 0.0, 2.0])),
            GateState(position=np.array([10.0, 5.0, 2.0])),
        ]
        track = Track(gates)
        assert track.num_gates == 2
        np.testing.assert_array_equal(track.gates[0].position, [5.0, 0.0, 2.0])

    def test_empty_track_raises(self):
        with pytest.raises(ValueError, match="at least one gate"):
            Track([])

    def test_track_has_no_mutable_state(self):
        gates = [GateState(position=np.array([1.0, 0.0, 0.0]))]
        track = Track(gates)
        assert not hasattr(track, '_current_gate_idx')
        assert not hasattr(track, '_laps_completed')
        assert not hasattr(track, 'reset')
